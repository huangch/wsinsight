"""Composable URI-aware path abstraction with caching and remote syncing."""

# uri_path.py
# pip install fsspec s3fs requests pandas

from __future__ import annotations

import os, json, hashlib, shutil, tempfile, time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Tuple, Union, Optional
from urllib.parse import urlparse, urlunparse
import click
import fsspec
import requests
import pandas as pd
import weakref  # added for auto-cleanup

import logging
logging.getLogger("aiobotocore.credentials").setLevel(logging.WARNING)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)


class URIPath:
    """
    Path-like wrapper for:
      - Local filesystem paths
      - fsspec remotes (e.g., s3://bucket/key)
      - gdc-manifest:///abs/path/to/manifest.tsv/<filename>

    Features:
      - open(): stream (local/remote) or local-open after materialize (gdc-manifest)
      - materialize(): persistent local copy (cached under cache_dir)
      - __fspath__(): returns a real local filename (auto materialize if needed)
      - exists()/is_file()/is_dir()/iterdir() via clean scheme dispatch
      - storage_options stored once (e.g., profile="saml", client_kwargs={...})
      - GDC (open-access OK w/o token) download via requests + retries
    """

    # ------------------------ Construct ------------------------
    def __init__(
        self,
        uri: Union[str, os.PathLike[str], "URIPath"],
        *,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,        # GDC only (optional for open-access)
        token_path: Optional[str] = None,   # GDC only (optional)
        **storage_options: Any,             # fsspec options (e.g., profile="saml", client_kwargs={...})
    ) -> None:
        if isinstance(uri, URIPath):
            self.uri: str = uri.uri
            self._cache_dir: str = cache_dir or uri._cache_dir
            merged = dict(uri.storage_options); merged.update(storage_options)
            self.storage_options: Dict[str, Any] = merged
            self._materialized_path: Optional[str] = uri._materialized_path
            self._gdc_token = token if token is not None else getattr(uri, "_gdc_token", None)
            self._gdc_token_path = token_path if token_path is not None else getattr(uri, "_gdc_token_path", None)
        else:
            try:
                self.uri = os.fspath(uri)
            except TypeError as exc:
                raise TypeError("uri must be str, os.PathLike, or URIPath") from exc
            self._cache_dir = cache_dir or self._default_cache_dir()
            self.storage_options = dict(storage_options)
            self._materialized_path = None
            self._gdc_token = token
            self._gdc_token_path = token_path

        parsed = urlparse(self.uri)
        self.scheme = parsed.scheme or "file"
        self.is_local = (self.scheme in ("", "file"))
        self.bucket = parsed.netloc if not self.is_local else ""
        self.key = parsed.path.lstrip("/") if not self.is_local else ""

        if self.scheme == "gdc-manifest":
            parts = parsed.path.split("/")
            try:
                idx = max(i for i, p in enumerate(parts) if p.lower().endswith((".tsv", ".txt", ".csv")))
            except ValueError:
                raise ValueError("gdc-manifest URI must include a manifest file (.tsv/.txt/.csv)")
            manifest_abs = "/" + "/".join(parts[1:idx + 1])
            rel = parts[idx + 1:]
            self._gdc_manifest_path = manifest_abs
            self._gdc_filename_in_manifest = "/".join([p for p in rel if p])  # may be ""
            self._path = Path(self._gdc_filename_in_manifest or "manifest/")
        else:
            base = os.path.basename(self.uri.rstrip("/"))
            self._path = Path(self.uri) if self.is_local else Path(base or "remote.file")

        # Check the credentials if the target file/dir is from remote
        self._validate_credentials()
        
        # ---- auto-clean internals (added; non-breaking) ----
        self._finalizer = None  # weakref.finalize handle; set when we materialize

    # ------------------------ Path-ish ------------------------
    def __str__(self) -> str: return self.uri
    def __repr__(self) -> str: return f"URIPath({self.uri!r})"
    def __fspath__(self) -> str:
        return os.fspath(self._path) if self.is_local and self.scheme != "gdc-manifest" else self._ensure_local()

    def __truediv__(self, other: os.PathLike | str) -> "URIPath":
        other_str = os.fspath(other)
        if self.scheme == "gdc-manifest":
            base = self.uri.rstrip("/")
            return URIPath(f"{base}/{other_str}",
                           cache_dir=self._cache_dir,
                           token=self._gdc_token, token_path=self._gdc_token_path,
                           **self.storage_options)
        if self.is_local:
            return URIPath(str(Path(self._path) / other_str), cache_dir=self._cache_dir, **self.storage_options)
        base = self.uri.rstrip("/")
        return URIPath(f"{base}/{other_str}", cache_dir=self._cache_dir, **self.storage_options)

    def mkdir(
        self,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
        **overrides: Any,
    ) -> None:
        if self.scheme == "gdc-manifest":
            raise IOError("gdc-manifest URIs are read-only")

        if self.is_local:
            Path(self._path).mkdir(mode=mode, parents=parents, exist_ok=exist_ok)
            return

        fs, fs_path = self._fs_and_path(**overrides)
        target = fs_path.rstrip("/")
        if not parents:
            parent = target.rsplit("/", 1)[0] if "/" in target else ""
            if parent and not fs.exists(parent):
                raise FileNotFoundError(f"Parent directory does not exist for {self.uri}")

        mkdirs = getattr(fs, "makedirs", None)
        if mkdirs:
            mkdirs(target, exist_ok=exist_ok)
            return

        try:
            fs.mkdir(target, create_parents=parents)
        except TypeError:
            fs.mkdir(target)
        except FileExistsError:
            if not exist_ok:
                raise

    @property
    def name(self) -> str:   return self._path.name
    @property
    def stem(self) -> str:   return self._path.stem
    @property
    def suffix(self) -> str: return self._path.suffix

    @property
    def parent(self) -> "URIPath":
        if self.scheme == "gdc-manifest":
            if not getattr(self, "_gdc_filename_in_manifest", ""):
                return self
            parent_rel = "/".join(self._gdc_filename_in_manifest.split("/")[:-1])
            base = f"gdc-manifest://{self._gdc_manifest_path}"
            if parent_rel:
                base = f"{base}/{parent_rel}/"
            return URIPath(base, cache_dir=self._cache_dir,
                           token=self._gdc_token, token_path=self._gdc_token_path,
                           **self.storage_options)
        if self.is_local:
            return URIPath(str(Path(self._path).parent), cache_dir=self._cache_dir, **self.storage_options)
        if self.scheme == "s3":
            if "/" in self.key:
                parent_key = "/".join(self.key.split("/")[:-1])
                return URIPath(f"s3://{self.bucket}/{parent_key}/", cache_dir=self._cache_dir, **self.storage_options)
            return URIPath(f"s3://{self.bucket}", cache_dir=self._cache_dir, **self.storage_options)
        return URIPath(self.uri.rsplit("/", 1)[0], cache_dir=self._cache_dir, **self.storage_options)

    @property
    def parts(self) -> Tuple[str, ...]:
        if self.scheme == "gdc-manifest":
            return ("gdc-manifest", self._gdc_manifest_path, *([p for p in self._gdc_filename_in_manifest.split('/') if p]))
        if self.is_local:
            return Path(self._path).parts
        if self.scheme == "s3":
            return ("s3", self.bucket, *([p for p in self.key.split('/') if p]))
        return (self.scheme, self.bucket, self.key)

    # ------------------------ Public I/O ------------------------
    def open(self, mode: str = "rb", **overrides: Any):
        write_mode = self._is_write_mode(mode)

        if self.scheme == "gdc-manifest":
            if write_mode:
                raise IOError("gdc-manifest URIs are read-only")
            lp = self.materialize(**overrides)
            return open(lp, mode)

        if self.is_local:
            return open(os.fspath(self._path), mode)

        # Remote via fsspec
        require_existing = self._requires_existing_file(mode)
        remote_exists = self.exists(**overrides)
        if require_existing and not remote_exists:
            raise FileNotFoundError(f"Remote path not found: {self.uri}")

        if write_mode:
            if remote_exists:
                local_path = self._materialize_via_fsspec(None, **overrides)
            else:
                local_path = self._prepare_cache_path_for_remote()
            fp = open(local_path, mode)
            self._materialized_path = local_path
            self._register_finalizer(local_path)
            def _sync_back():
                self._sync_remote_from_cache(local_path, **overrides)
            return _SyncOnCloseFile(fp, _sync_back)

        # read-only remote path -> stream from cached copy
        local_path = self._materialize_via_fsspec(None, **overrides)
        return open(local_path, mode)

    def materialize(self, dest_path: Optional[str | os.PathLike] = None, **overrides: Any) -> str:
        if self.is_local and self.scheme != "gdc-manifest":
            self._materialized_path = os.fspath(self._path)
            # local file — no cache to clean, no finalizer needed
            return self._materialized_path

        if self.scheme == "gdc-manifest":
            filename = self._gdc_filename_in_manifest
            if not filename:
                raise IsADirectoryError("gdc-manifest points to the manifest root, not a file.")
            df = self._load_manifest_table(self._gdc_manifest_path)
            row = df[df["filename"] == filename]
            if row.empty:
                raise FileNotFoundError(f"{filename!r} not found in manifest {self._gdc_manifest_path}")
            uuid = str(row["id"].iloc[0])
            md5 = str(row["md5"].iloc[0]).lower() if "md5" in row and pd.notna(row["md5"].iloc[0]) else None

            if dest_path is None:
                key_for_hash = f"{self._gdc_manifest_path}::{uuid}"
                h = self._hash_key(key_for_hash)
                dest_dir = os.path.join(self._cache_dir, "gdc", h[:2]); self._ensure_dir(dest_dir)
                dest = os.path.join(dest_dir, filename)
            else:
                dest = os.fspath(dest_path); self._ensure_dir(os.path.dirname(dest))

            # if already materialized, register finalizer and return
            if self._materialized_path and os.path.exists(self._materialized_path):
                self._register_finalizer(self._materialized_path)
                return self._materialized_path
            if os.path.exists(dest):
                self._materialized_path = dest
                self._register_finalizer(dest)
                return dest

            token = overrides.get("token", None) or self._gdc_token
            token_path = overrides.get("token_path", None) or self._gdc_token_path
            if token is None and token_path:
                with open(os.path.expanduser(token_path), "r") as fp:
                    token = fp.read().strip()

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ""); tmp.close()
            try:
                self._gdc_download_requests(uuid, tmp.name, token=token)
                if md5 is not None and not self._md5_ok(tmp.name, md5):
                    raise IOError("MD5 checksum mismatch after download")
                try: os.replace(tmp.name, dest)
                except Exception: shutil.copyfile(tmp.name, dest)
            finally:
                try: os.remove(tmp.name)
                except OSError: pass

            self._materialized_path = dest
            self._register_finalizer(dest)
            return dest

        # remote via fsspec
        return self._materialize_via_fsspec(dest_path, **overrides)

    # ------------------------ Dispatcher: exists / is_file / is_dir / iterdir ------------------------
    def exists(self, **overrides: Any) -> bool:
        if self.scheme == "gdc-manifest":
            return self._exists_gdc_manifest()
        if self.is_local:
            return self._exists_local()
        return self._exists_remote(**overrides)

    def is_file(self, **overrides: Any) -> bool:
        if self.scheme == "gdc-manifest":
            return self._is_file_gdc_manifest()
        if self.is_local:
            return self._path.is_file()
        return self._is_file_remote(**overrides)

    def is_dir(self, **overrides: Any) -> bool:
        if self.scheme == "gdc-manifest":
            return self._is_dir_gdc_manifest()
        if self.is_local:
            return self._path.is_dir()
        return self._is_dir_remote(**overrides)

    def iterdir(self, recursive: bool = False, files_only: bool = True, **overrides: Any) -> Iterator["URIPath"]:
        if self.scheme == "gdc-manifest":
            yield from self._iterdir_gdc_manifest()
            return
        if self.is_local:
            yield from self._iterdir_local(recursive=recursive, files_only=files_only)
            return
        yield from self._iterdir_remote(recursive=recursive, files_only=files_only, **overrides)

    # ------------------------ Scheme helpers: LOCAL ------------------------
    def _exists_local(self) -> bool:
        return self._path.exists()

    def _iterdir_local(self, *, recursive: bool, files_only: bool) -> Iterator["URIPath"]:
        it = self._path.rglob("*") if recursive else self._path.iterdir()
        for child in it:
            if files_only and child.is_dir():
                continue
            yield URIPath(str(child), cache_dir=self._cache_dir, **self.storage_options)

    # ------------------------ Scheme helpers: GDC-MANIFEST ------------------------
    def _exists_gdc_manifest(self) -> bool:
        if not getattr(self, "_gdc_filename_in_manifest", ""):
            return os.path.exists(self._gdc_manifest_path)
        try:
            df = self._load_manifest_table(self._gdc_manifest_path)
            return any(df["filename"] == self._gdc_filename_in_manifest)
        except Exception:
            return False

    def _is_file_gdc_manifest(self) -> bool:
        return getattr(self, "_gdc_filename_in_manifest", "") != "" and self._exists_gdc_manifest()

    def _is_dir_gdc_manifest(self) -> bool:
        return getattr(self, "_gdc_filename_in_manifest", "") == ""

    def _iterdir_gdc_manifest(self) -> Iterator["URIPath"]:
        if not os.path.exists(self._gdc_manifest_path):
            return
        df = self._load_manifest_table(self._gdc_manifest_path)
        for fn in df["filename"].tolist():
            yield URIPath(f"gdc-manifest://{self._gdc_manifest_path}/{fn}",
                          cache_dir=self._cache_dir,
                          token=self._gdc_token, token_path=self._gdc_token_path,
                          **self.storage_options)

    # ------------------------ Scheme helpers: REMOTE (fsspec, e.g., S3) ------------------------
    def _exists_remote(self, **overrides: Any) -> bool:
        try:
            fs, fs_path = self._fs_and_path(**overrides)
        except Exception:
            try:
                fs, fs_path = self._fs_and_path()
            except Exception:
                return False

        try:
            return bool(fs.exists(fs_path))
        except Exception:
            # Fallback 1: info()
            try:
                fs.info(fs_path)
                return True
            except FileNotFoundError:
                return False
            except Exception:
                # Fallback 2: parent listing
                try:
                    parent = fs._parent(fs_path) if hasattr(fs, "_parent") else fs_path.rsplit("/", 1)[0]
                    for entry in fs.ls(parent, detail=True):
                        name = entry.get("name") or entry
                        if isinstance(name, str) and name.rstrip("/") == fs_path.rstrip("/"):
                            return True
                    return False
                except Exception:
                    return False

    def _is_file_remote(self, **overrides: Any) -> bool:
        try:
            fs, fs_path = self._fs_and_path(**overrides)
        except Exception:
            return False
        try:
            info = fs.info(fs_path)
            t = info.get("type")
            return (t == "file") if t is not None else ("size" in info and info["size"] is not None)
        except FileNotFoundError:
            return False
        except Exception:
            try:
                if not fs.exists(fs_path):
                    return False
                return not fs.isdir(fs_path)
            except Exception:
                return False

    def _is_dir_remote(self, **overrides: Any) -> bool:
        try:
            fs, fs_path = self._fs_and_path(**overrides)
        except Exception:
            return False
        try:
            return bool(fs.isdir(fs_path))
        except Exception:
            return False

    def _iterdir_remote(self, *, recursive: bool, files_only: bool, **overrides: Any) -> Iterator["URIPath"]:
        fs, fs_path = self._fs_and_path(**overrides)
        base = fs_path if fs_path.endswith("/") else fs_path + "/"
        if recursive:
            for name in fs.find(base):
                yield URIPath(f"{self.scheme}://{name}" if self.scheme != "s3" else f"s3://{name}",
                              cache_dir=self._cache_dir, **self.storage_options)
        else:
            try:
                for entry in fs.ls(base, detail=True):
                    if files_only and entry.get("type") == "directory":
                        continue
                    yield URIPath(f"{self.scheme}://{entry['name']}" if self.scheme != "s3" else f"s3://{entry['name']}",
                                  cache_dir=self._cache_dir, **self.storage_options)
            except FileNotFoundError:
                return

    def _validate_credentials(self) -> None:
        """
        Fail fast on bad credentials for remote targets.
    
        Local paths: no-op.
        gdc-manifest: if a token is set, do a tiny HEAD.
        S3 (and other fsspec remotes): verify AWS identity (S3) or init fs.
        """
        # Local or manifest without token -> nothing to validate
        if self.is_local and self.scheme != "gdc-manifest":
            return
    
        if self.scheme == "gdc-manifest":
            token = getattr(self, "_gdc_token", None)
            if not token:
                return
            try:
                import requests as _rq
                headers = {"Accept": "application/octet-stream", "X-Auth-Token": token}
                # tiny auth check; not downloading anything
                resp = _rq.head("https://api.gdc.cancer.gov/data/", headers=headers, timeout=8)
                if resp.status_code >= 400:
                    raise RuntimeError(f"GDC token check failed (status {resp.status_code})")
            except Exception as e:
                raise RuntimeError(f"GDC credential check failed: {e!r}") from e
            return
    
        # fsspec-backed remotes
        if self.scheme == "s3":
            # 1) Init the filesystem with current options (surfaces invalid tokens in practice)
            try:
                self._fs_and_path()
            except Exception as e:
                raise RuntimeError(f"S3 filesystem init failed for {self.uri!r}: {e!r}") from e
            return
    
        # Generic fsspec scheme (e.g., gs, abfs): try to init the FS
        try:
            self._fs_and_path()
        except Exception as e:
            raise RuntimeError(f"Remote filesystem init failed for {self.uri!r}: {e!r}") from e
    

    # ------------------------ fsspec helpers ------------------------
    def _fs_and_path(self, **overrides: Any):
        opts = self._normalize_storage_opts({**self.storage_options, **overrides})
        target = self.uri if not (self.is_local and self.scheme != "gdc-manifest") else os.fspath(self._path)
        return fsspec.url_to_fs(target, **opts)

    def _materialize_via_fsspec(self, dest_path: Optional[str | os.PathLike], **overrides: Any) -> str:
        if dest_path is None:
            h = self._hash_key(self.uri)
            dest_dir = os.path.join(self._cache_dir, "remote", h[:2]); self._ensure_dir(dest_dir)
            dest = os.path.join(dest_dir, self.name or "remote.file")
        else:
            dest = os.fspath(dest_path); self._ensure_dir(os.path.dirname(dest))

        if self._materialized_path and os.path.exists(self._materialized_path):
            self._register_finalizer(self._materialized_path)
            return self._materialized_path
        if os.path.exists(dest):
            self._materialized_path = dest
            self._register_finalizer(dest)
            return dest

        fs, fs_path = self._fs_and_path(**overrides)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(self.name)[1] or ""); tmp.close()
        try:
            fs.get(fs_path, tmp.name)
            try: os.replace(tmp.name, dest)
            except Exception: shutil.copyfile(tmp.name, dest)
        finally:
            try: os.remove(tmp.name)
            except OSError: pass
        self._materialized_path = dest
        self._register_finalizer(dest)
        return dest

    def _prepare_cache_path_for_remote(self) -> str:
        h = self._hash_key(self.uri)
        dest_dir = os.path.join(self._cache_dir, "remote", h[:2])
        self._ensure_dir(dest_dir)
        dest = os.path.join(dest_dir, self.name or "remote.file")
        self._ensure_dir(os.path.dirname(dest))
        return dest

    def _sync_remote_from_cache(self, local_path: str, **overrides: Any) -> None:
        fs, fs_path = self._fs_and_path(**overrides)
        fs.put(local_path, fs_path)

    @staticmethod
    def _is_write_mode(mode: str) -> bool:
        flags = {"w", "a", "x", "+"}
        return any(ch in mode for ch in flags)

    @staticmethod
    def _requires_existing_file(mode: str) -> bool:
        return ("r" in mode) and ("w" not in mode) and ("x" not in mode)

    # ------------------------ GDC (requests backend) ------------------------
    def _gdc_download_requests(self, uuid: str, out_path: str, *, token: Optional[str] = None) -> None:
        url = f"https://api.gdc.cancer.gov/data/{uuid}"  # fixed typo
        headers = {"Accept": "application/octet-stream"}
        if token: headers["X-Auth-Token"] = token

        backoff = 1.0
        for attempt in range(5):
            try:
                with requests.get(url, headers=headers, stream=True, timeout=120) as r:
                    if r.status_code in (429, 500, 502, 503, 504):
                        raise requests.HTTPError(f"{r.status_code} retryable")
                    r.raise_for_status()
                    with open(out_path, "wb") as fp:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk: fp.write(chunk)
                return
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
                if attempt == 4: raise
                time.sleep(backoff); backoff = min(backoff * 2, 16.0)

    # ------------------------ Utilities ------------------------
    @staticmethod
    def _normalize_storage_opts(opts: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(opts)
        for k in ("client_kwargs", "config_kwargs", "s3_additional_kwargs"):
            if k in out and isinstance(out[k], str):
                try:
                    out[k] = json.loads(out[k])
                except Exception:
                    out.pop(k, None)
            if k in out and not isinstance(out[k], dict):
                out.pop(k, None)
        return out

    @staticmethod
    def _default_cache_dir() -> str:
        base = os.path.expanduser("~/.cache")
        if os.name == "nt":
            base = os.path.join(os.environ.get("LOCALAPPDATA", base))
        return os.path.join(base, "uripath-cache")

    @staticmethod
    def _ensure_dir(p: str) -> None:
        Path(p).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash_key(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _md5_ok(path: str, expect: str) -> bool:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest().lower() == str(expect).lower()

    @staticmethod
    def _load_manifest_table(manifest_path: str) -> pd.DataFrame:
        ext = manifest_path.lower()
        if ext.endswith(".csv"):
            df = pd.read_csv(manifest_path)
        else:
            # .tsv/.txt → sep autodetect (handles tabs/spaces)
            df = pd.read_csv(manifest_path, sep=None, engine="python")
        cols = {c.lower(): c for c in df.columns}
        def pick(*names):
            for n in names:
                if n in cols: return cols[n]
            raise ValueError(f"Manifest missing required column among {names!r}")
        id_col = pick("id", "file_id")
        fn_col = pick("filename", "file_name")
        out = pd.DataFrame({"id": df[id_col].astype(str), "filename": df[fn_col].astype(str)})
        if "md5" in cols:
            out["md5"] = df[cols["md5"]].astype(str)
        return out

    # Convenience
    def as_uri(self) -> str: return self.uri
    def local_path(self) -> str: return self._ensure_local()
    def _ensure_local(self) -> str:
        if self.is_local and self.scheme != "gdc-manifest":
            return os.fspath(self._path)
        if self._materialized_path and os.path.exists(self._materialized_path):
            # already materialized: ensure finalizer is registered
            self._register_finalizer(self._materialized_path)
            return self._materialized_path
        return self.materialize()

    # ---- ordering & equality ----
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, URIPath):
            return NotImplemented
        # equality by canonical URI string
        return self.uri == other.uri

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, URIPath):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    def __hash__(self) -> int:
        # keep consistent with __eq__
        return hash(self.uri)

    def _sort_key(self):
        """
        Normalize to a tuple so ordering is predictable across schemes.
        """
        if self.scheme == "gdc-manifest":
            # manifest path first, then the filename within it
            return ("gdc-manifest",
                    os.path.abspath(self._gdc_manifest_path),
                    self._gdc_filename_in_manifest or "")
        if self.is_local:
            # absolute local path
            return ("file", os.path.abspath(os.fspath(self._path)))
        # generic fsspec remote (e.g., s3)
        return (self.scheme or "file", self.bucket or "", self.key or "")

    def with_suffix(self, suffix: str) -> "URIPath":
        """
        Return a new URIPath with the file suffix changed.
        Behaves like pathlib.Path.with_suffix:
          - suffix must start with '.' or be '' to remove
          - raises ValueError if path looks like a directory
        """
        if not isinstance(suffix, str):
            raise TypeError("suffix must be a string")
        if suffix and not suffix.startswith("."):
            raise ValueError("Invalid suffix %r (must start with '.')" % suffix)

        # Disallow directories (same as pathlib)
        # local dir:
        if self.is_local and self._path.is_dir():
            raise ValueError("Can't change suffix of a directory")
        # remote dir heuristic: URI ends with '/' or manifest root
        if not self.is_local:
            if self.uri.endswith("/"):
                raise ValueError("Can't change suffix of a directory")
            if self.scheme == "gdc-manifest" and getattr(self, "_gdc_filename_in_manifest", "") == "":
                raise ValueError("Can't change suffix of a directory (manifest root)")

        # ----- gdc-manifest -----
        if self.scheme == "gdc-manifest":
            fname = self._gdc_filename_in_manifest
            stem, _ = os.path.splitext(fname)
            new_name = (stem + suffix) if suffix else stem  # '' removes suffix
            base = f"gdc-manifest://{self._gdc_manifest_path}"
            new_uri = f"{base}/{new_name}"
            return URIPath(
                new_uri,
                cache_dir=self._cache_dir,
                token=self._gdc_token, token_path=self._gdc_token_path,
                **self.storage_options
            )

        # ----- local -----
        if self.is_local:
            new_local = str(self._path.with_suffix(suffix))
            return URIPath(new_local, cache_dir=self._cache_dir, **self.storage_options)

        # ----- generic remote (e.g., s3) -----
        # Replace suffix only in the last path segment
        last = os.path.basename(self.key) if self.key else self.name
        stem, _ = os.path.splitext(last)
        new_last = (stem + suffix) if suffix else stem
        if self.scheme == "s3":
            # rebuild s3 URI
            prefix = os.path.dirname(self.key) if self.key else ""
            new_key = f"{prefix}/{new_last}" if prefix else new_last
            new_uri = f"s3://{self.bucket}/{new_key}"
        else:
            # fallback: replace only the trailing segment in self.uri
            # handle URIs that may not have bucket semantics
            prefix = self.uri.rstrip("/").rsplit("/", 1)[0]
            new_uri = f"{prefix}/{new_last}"
        return URIPath(new_uri, cache_dir=self._cache_dir, **self.storage_options)

    def with_name(self, new_name):
        """
        Return a new URIPath with the final component (filename) replaced by `new_name`,
        preserving the original storage_options.

        Args:
            path (URIPath | str | pathlib.Path): input path/URI (URIPath preferred).
            new_name (str): replacement for the last path component (no '/' or '\\').

        Returns:
            URIPath: same scheme/authority/parent, same storage_options, new name.
        """

        if any(sep in new_name for sep in ("/", "\\")):
            raise ValueError(f"Invalid new_name '{new_name}': must not contain path separators")

        # Capture storage_options if `path` is a URIPath; fall back to empty dict
        is_uriobj = isinstance(self, URIPath)
        storage_options = getattr(self, "storage_options", None)

        # Get the string URI/path to manipulate
        s = str(self.uri if is_uriobj and hasattr(self, "uri") else self)

        parsed = urlparse(s)
        is_uri = bool(parsed.scheme) and not (len(parsed.scheme) == 1 and parsed.scheme.isalpha())

        if not is_uri:
            # Local FS: rebuild name via pathlib, then convert back to URI if original was URIPath with 'file' scheme
            p = Path(s).with_name(new_name)
            # Build a URI-like string for uniformity
            new_uri = p.as_posix()
            if is_uriobj and getattr(self, "scheme", "file") == "file":
                # ensure file:// prefix when original URIPath represented a local file URI
                new_uri = f"file://{new_uri.lstrip('/')}"
        else:
            # True URI: replace only the last segment of the path
            pth = parsed.path or ""
            if pth.endswith("/"):
                new_path = pth + new_name
            else:
                head, _, _ = pth.rpartition("/")
                new_path = (head + "/" if head else ("/" if pth.startswith("/") else "")) + new_name
            new_uri = urlunparse((
                parsed.scheme, parsed.netloc, new_path,
                parsed.params, parsed.query, parsed.fragment
            ))

        # Always return a URIPath, preserving storage_options
        return URIPath(new_uri, cache_dir=self._cache_dir, **(storage_options or {}))

    # ------------------------ Auto-cache cleanup helpers (inside class) ------------------------
    def _register_finalizer(self, path: Optional[str]) -> None:
        """
        Register a best-effort deletion of the given cached file when this object
        is garbage-collected. Idempotent: re-registers if needed.
        """
        if not path:
            return
        # If there was a previous finalizer, replace it (keeps latest cache file)
        if getattr(self, "_finalizer", None):
            try:
                # don't run earlier finalizer here; just detach to avoid double-delete
                self._finalizer.detach()
            except Exception:
                pass
        self._finalizer = weakref.finalize(self, self._cleanup_cached_file, str(path))

    @staticmethod
    def _cleanup_cached_file(path_str: str) -> None:
        """Best-effort remove of a cached file; swallow errors."""
        try:
            if path_str and os.path.exists(path_str):
                os.remove(path_str)
        except Exception:
            pass

    def close(self) -> None:
        """
        Manually delete the materialized cache file (if any), immediately.
        Safe to call multiple times; does not change public usage anywhere else.
        """
        p = getattr(self, "_materialized_path", None)
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
        # run & detach finalizer if present
        if getattr(self, "_finalizer", None):
            try:
                if self._finalizer.alive:
                    self._finalizer()
            except Exception:
                pass
            self._finalizer = None

    def __del__(self):
        # Fallback if user never called close(); best effort
        try:
            if getattr(self, "_finalizer", None) and self._finalizer.alive:
                self._finalizer()
        except Exception:
            pass


class URIPathType(click.ParamType):
    """Click parameter type that parses CLI args into ``URIPath`` objects."""
    name = "UniversalPath"
    def __init__(self, exists: bool = False, **storage_options: Any,):
        self.exists = exists
        self.storage_options = dict(storage_options)
    def convert(self, value, param, ctx):
        obj = URIPath(value, **self.storage_options)

        if self.exists:
            # Pull storage_options (e.g., {"profile":"saml"}) from Click ctx, if present
            # so = (ctx.params.get("storage_options") or
            #       ctx.obj.get("storage_options") if ctx and ctx.obj else {})  # optional
            try:
                if not obj.exists():
                    self.fail(f"Path not found: {value}", param, ctx)
            except Exception as e:
                self.fail(f"Failed to validate existence for {value}: {e}", param, ctx)
        return obj


class _SyncOnCloseFile:
    """Proxy that syncs cache back to remote storage after close."""

    def __init__(self, fp, on_close: Callable[[], None]):
        self._fp = fp
        self._on_close = on_close

    def __getattr__(self, item):
        return getattr(self._fp, item)

    def __enter__(self):
        self._fp.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = self._fp.__exit__(exc_type, exc_val, exc_tb)
        self.close()
        return result

    def close(self):
        if self._fp and not self._fp.closed:
            self._fp.close()
        if self._on_close:
            self._on_close()
            self._on_close = None

    @property
    def closed(self):
        return self._fp.closed
