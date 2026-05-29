"""Unit tests for the URIPath / URIPathType hardening work.

These tests focus on behavior that historically was wrong or silently
swallowed errors:

* the on-disk cache is shared across ``URIPath`` instances and must not be
  deleted when one of them is garbage-collected unless the user explicitly
  opted in via ``auto_cleanup=True``;
* malformed JSON in ``client_kwargs`` / ``config_kwargs`` /
  ``s3_additional_kwargs`` must raise instead of being silently dropped (which
  used to demote authenticated S3 requests to anonymous ones);
* ``URIPathType.convert`` must surface clean Click errors;
* image-list manifests must accept a UTF-8 BOM and ignore blank/commented
  lines.
"""

from __future__ import annotations

import codecs
import gc
import json
import os
import tempfile
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from wsinsight.uri_path import URIPath, URIPathType


# --------------------------------------------------------------------------
# storage_options normalization
# --------------------------------------------------------------------------

def test_normalize_storage_opts_rejects_malformed_json():
    with pytest.raises(ValueError, match="not valid JSON"):
        URIPath._normalize_storage_opts({"client_kwargs": "{not json}"})


def test_normalize_storage_opts_parses_valid_json():
    out = URIPath._normalize_storage_opts(
        {"client_kwargs": json.dumps({"region_name": "us-east-1"})}
    )
    assert out["client_kwargs"] == {"region_name": "us-east-1"}


def test_normalize_storage_opts_rejects_non_object_json():
    with pytest.raises(ValueError, match="must decode to a JSON object"):
        URIPath._normalize_storage_opts({"client_kwargs": "[1, 2, 3]"})


def test_normalize_storage_opts_rejects_wrong_type():
    with pytest.raises(TypeError, match="must be a dict or JSON string"):
        URIPath._normalize_storage_opts({"client_kwargs": 42})


# --------------------------------------------------------------------------
# gs:// (Google Cloud Storage) URI manipulation -- parity with s3://
# (credential init skipped: these only exercise pure URI string handling)
# --------------------------------------------------------------------------

def test_gs_parent_nested():
    p = URIPath("gs://my-bucket/a/b/c.svs", _skip_validation=True)
    assert str(p.parent) == "gs://my-bucket/a/b/"


def test_gs_parent_top_level():
    p = URIPath("gs://my-bucket/c.svs", _skip_validation=True)
    assert str(p.parent) == "gs://my-bucket"


def test_gs_parts():
    p = URIPath("gs://my-bucket/a/b/c.svs", _skip_validation=True)
    assert p.parts == ("gs", "my-bucket", "a", "b", "c.svs")


def test_gs_with_suffix():
    p = URIPath("gs://my-bucket/a/b/c.svs", _skip_validation=True)
    assert str(p.with_suffix(".png")) == "gs://my-bucket/a/b/c.png"


def test_gs_truediv_joins_key():
    p = URIPath("gs://my-bucket/a", _skip_validation=True)
    assert str(p / "b.svs") == "gs://my-bucket/a/b.svs"


# --------------------------------------------------------------------------
# GS_STORAGE_OPTIONS env var parsing (parity with S3_STORAGE_OPTIONS)
# --------------------------------------------------------------------------

def test_gs_storage_options_parsed(monkeypatch):
    from wsinsight.cli._paths import default_storage_kwargs

    monkeypatch.delenv("S3_STORAGE_OPTIONS", raising=False)
    monkeypatch.setenv("GS_STORAGE_OPTIONS", json.dumps({"token": "/tmp/sa.json"}))
    out = default_storage_kwargs()
    assert out["token"] == "/tmp/sa.json"


def test_gs_storage_options_rejects_malformed(monkeypatch):
    from wsinsight.cli._paths import default_storage_kwargs

    monkeypatch.setenv("GS_STORAGE_OPTIONS", "{not json}")
    with pytest.raises(RuntimeError, match="GS_STORAGE_OPTIONS must contain valid JSON"):
        default_storage_kwargs()


def test_gs_storage_options_rejects_non_object(monkeypatch):
    from wsinsight.cli._paths import default_storage_kwargs

    monkeypatch.setenv("GS_STORAGE_OPTIONS", "[1, 2, 3]")
    with pytest.raises(RuntimeError, match="GS_STORAGE_OPTIONS must be a JSON object"):
        default_storage_kwargs()


# --------------------------------------------------------------------------
# Click ParamType
# --------------------------------------------------------------------------

def test_uri_path_type_reports_missing_path_via_click():
    @click.command()
    @click.argument("p", type=URIPathType(exists=True))
    def cmd(p):  # pragma: no cover - never reached
        click.echo(str(p))

    result = CliRunner().invoke(cmd, ["/definitely/not/a/real/path/xyzzy"])
    assert result.exit_code == 2
    assert "Path not found" in result.output


def test_uri_path_type_accepts_existing(tmp_path: Path):
    target = tmp_path / "real.txt"
    target.write_text("hi")

    @click.command()
    @click.argument("p", type=URIPathType(exists=True))
    def cmd(p):
        click.echo(str(p))

    result = CliRunner().invoke(cmd, [str(target)])
    assert result.exit_code == 0, result.output
    assert str(target) in result.output


# --------------------------------------------------------------------------
# Cache lifecycle
# --------------------------------------------------------------------------

def test_default_cache_survives_gc(tmp_path: Path):
    """Two URIPaths can share a cache file; GC of one must not nuke it."""
    cache = tmp_path / "cache"
    payload = cache / "remote" / "ab" / "shared.svs"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"slide-bytes")

    a = URIPath(f"file://{payload}", cache_dir=str(cache))
    a._materialized_path = str(payload)
    a._register_finalizer(str(payload))  # default auto_cleanup=False -> no-op

    assert a._finalizer is None, "default mode must not register a finalizer"
    del a
    gc.collect()
    assert payload.exists(), "shared cache must survive GC under default policy"


def test_auto_cleanup_deletes_cache(tmp_path: Path):
    payload = tmp_path / "ephemeral.svs"
    payload.write_bytes(b"x")

    obj = URIPath(f"file://{payload}", cache_dir=str(tmp_path), auto_cleanup=True)
    obj._materialized_path = str(payload)
    obj._register_finalizer(str(payload))
    assert obj._finalizer is not None

    obj._finalizer()  # simulate GC firing
    assert not payload.exists()


def test_child_inherits_auto_cleanup(tmp_path: Path):
    parent = URIPath(str(tmp_path), auto_cleanup=True)
    child = parent / "x.txt"
    assert child._auto_cleanup is True


def test_child_inherits_storage_options(tmp_path: Path):
    parent = URIPath(str(tmp_path), profile="saml")
    child = parent / "subdir"
    assert child.storage_options.get("profile") == "saml"


# --------------------------------------------------------------------------
# image-list scheme
# --------------------------------------------------------------------------

def test_image_list_handles_bom_blank_and_comments(tmp_path: Path):
    list_path = tmp_path / "slides.txt"
    body = codecs.BOM_UTF8 + b"/data/a.svs\n# comment\n\n  /data/b.svs  \n"
    list_path.write_bytes(body)

    ul = URIPath(f"image-list://{list_path}")
    entries = [str(e) for e in ul.iterdir()]
    assert entries == ["/data/a.svs", "/data/b.svs"]


def test_image_list_missing_file_yields_nothing(tmp_path: Path):
    ul = URIPath(f"image-list://{tmp_path / 'missing.txt'}")
    assert list(ul.iterdir()) == []
    assert ul.exists() is False


def test_coerce_image_list_rejects_plain_text_file(tmp_path: Path):
    bogus = tmp_path / "README.txt"
    bogus.write_text("not a slide")
    p = URIPath(str(bogus))
    with pytest.raises(ValueError, match="image-list://"):
        p.coerce_image_list()


# --------------------------------------------------------------------------
# Error propagation: bad URI types
# --------------------------------------------------------------------------

def test_uri_path_rejects_non_pathlike():
    with pytest.raises(TypeError, match="uri must be"):
        URIPath(12345)


# --------------------------------------------------------------------------
# GDC manifest table cache
# --------------------------------------------------------------------------

def test_manifest_cache_hits_on_repeated_load(tmp_path: Path, monkeypatch):
    mf = tmp_path / "manifest.tsv"
    mf.write_text("id\tfilename\nu1\ta.svs\nu2\tb.svs\n")
    URIPath._MANIFEST_CACHE.clear()

    calls = {"n": 0}
    real_read_csv = __import__("pandas").read_csv

    def counting_read_csv(*args, **kwargs):
        calls["n"] += 1
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr("wsinsight.uri_path.pd.read_csv", counting_read_csv)

    URIPath._load_manifest_table(str(mf))
    URIPath._load_manifest_table(str(mf))
    URIPath._load_manifest_table(str(mf))
    assert calls["n"] == 1, "manifest should be parsed once and cached"


def test_manifest_cache_invalidates_on_mtime_change(tmp_path: Path):
    import time as _time
    mf = tmp_path / "manifest.tsv"
    mf.write_text("id\tfilename\nu1\ta.svs\n")
    URIPath._MANIFEST_CACHE.clear()

    df1 = URIPath._load_manifest_table(str(mf))
    assert list(df1["filename"]) == ["a.svs"]

    # Force a different mtime/size by rewriting.
    _time.sleep(0.01)
    mf.write_text("id\tfilename\nu2\tb.svs\n")
    df2 = URIPath._load_manifest_table(str(mf))
    assert list(df2["filename"]) == ["b.svs"]

