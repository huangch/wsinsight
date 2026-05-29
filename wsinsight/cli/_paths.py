"""Shared path/URI helpers for WSInsight CLI subcommands.

Centralises:
- ``default_storage_kwargs()``: fsspec storage options (cache_dir + S3
  options) read once from environment variables.
- ``ensure_input_directory()``: validate that an ``-i`` style argument
  points at an existing directory. Used for ``--wsi-dir`` and similar.
- ``ensure_output_directory()``: validate / create an ``-o`` style argument.
  Works for both local paths and remote (e.g. ``s3://``) prefixes that don't
  yet exist. ``--results-dir`` should always pass through this helper so a
  brand-new S3 prefix is auto-created.

Historically each CLI module duplicated ``_storage_kwargs`` and
``_assert_directory``; consolidating them prevents drift.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
from platformdirs import user_cache_dir

from ..uri_path import URIPath


def default_storage_kwargs() -> dict[str, object]:
    """Storage options applied to every ``URIPathType`` in the CLI.

    Reads two env vars:
    - ``WSINSIGHT_REMOTE_CACHE_DIR``: where to materialize remote files.
      Defaults to the user platformdirs cache dir.
    - ``S3_STORAGE_OPTIONS``: JSON object of fsspec storage options
      (e.g. ``{"profile": "saml", "client_kwargs": {"endpoint_url": "..."}}``).
      Malformed JSON or a non-object payload raises ``RuntimeError`` instead of
      being silently dropped.
    - ``GS_STORAGE_OPTIONS``: JSON object of fsspec/gcsfs storage options for
      Google Cloud Storage (``gs://``). Auth defaults to Application Default
      Credentials (``GOOGLE_APPLICATION_CREDENTIALS``); use this to override,
      e.g. ``{"token": "/path/to/service-account.json"}``. Same validation as
      ``S3_STORAGE_OPTIONS``.
    """
    cache_dir = os.getenv("WSINSIGHT_REMOTE_CACHE_DIR")
    if cache_dir is None:
        cache_dir = Path(user_cache_dir(appname="wsinsight", appauthor=False))
    storage: dict[str, object] = {"cache_dir": cache_dir}
    for env_name in ("S3_STORAGE_OPTIONS", "GS_STORAGE_OPTIONS"):
        raw = os.getenv(env_name)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{env_name} must contain valid JSON."
            ) from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{env_name} must be a JSON object.")
        storage.update(parsed)
    return storage


def ensure_input_directory(path: URIPath, option_name: str) -> None:
    """Require ``path`` to be an existing directory, raising ``ClickException``.

    Use for ``--wsi-dir`` and any other read-only directory inputs.
    """
    if not path.exists():
        raise click.ClickException(
            f"{option_name} directory not found: {path}"
        )
    if not path.is_dir():
        raise click.ClickException(f"{option_name} must be a directory: {path}")


def ensure_output_directory(path: URIPath, option_name: str) -> None:
    """Ensure ``path`` is a writable directory, creating it if needed.

    For local paths this is ``mkdir(parents=True, exist_ok=True)``. For remote
    prefixes (e.g. ``s3://bucket/new-prefix/``) ``mkdir`` is a best-effort no-op
    on object stores; the helper still calls it so the abstraction stays
    uniform, then verifies the parent (if local) is a directory.

    The function never fails because the prefix didn't pre-exist on S3 — the
    Critical bug that bit users running fresh outputs into S3 buckets.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, OSError) as exc:
        # ``mkdir`` may raise if the URI points at an existing *file*; surface a
        # clean Click error.
        raise click.ClickException(
            f"{option_name} could not be created at {path}: {exc}"
        ) from exc

    # For local destinations we can still sanity-check the result. For remote
    # destinations (object stores) ``is_dir`` is unreliable on freshly created
    # prefixes and we deliberately skip the assertion.
    if path.is_local and not path.is_dir():
        raise click.ClickException(
            f"{option_name} exists but is not a directory: {path}"
        )
