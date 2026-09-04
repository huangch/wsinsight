"""Shared run-metadata helpers for WSInsight CLI subcommands.

Each WSInsight CLI subcommand writes a timestamped ``<command>_metadata_<TS>.json``
file into ``results_dir`` at the end of a successful run. ``patch`` and ``infer``
predate this module and emit a richer record that also captures the model
object (see their local ``_get_info_for_save`` helpers); the analytics
subcommands (``hplot``, ``hplot-finalize``, ``ncomp``, ``ecomp``, ``tcomp``,
``niche``) have no model object and use the model-less helper here.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Mapping

import click

from .. import __version__
from ..uri_path import URIPath


def _inside_container() -> str:
    """Return a coarse container indicator (docker/singularity) for logging."""
    if Path("/.dockerenv").exists():
        return "yes, docker"
    if (
        Path("/singularity").exists()
        or Path("/singularity.d").exists()
        or Path("/.singularity.d").exists()
    ):
        return "yes, apptainer/singularity"
    return "no"


def _get_timestamp_human() -> str:
    """Timezone-aware human-readable timestamp used inside the JSON record."""
    return datetime.now().astimezone().strftime("%c %Z")


def _get_timestamp_filename() -> str:
    """Filesystem-safe timestamp matching ``patch``/``infer`` filenames."""
    return datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")


def _get_git_info() -> dict[str, Any] | None:
    """Best-effort git metadata for the wsinsight working tree, or ``None``."""
    here = Path(__file__).parent.resolve()
    git_program = shutil.which("git")
    if git_program is None:
        return None
    probe = subprocess.run(
        [git_program, "branch"],
        cwd=here,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode != 0:
        return None

    def _stdout(args: list[str]) -> str:
        proc = subprocess.run(args, capture_output=True, cwd=here)
        return "" if proc.returncode != 0 else proc.stdout.decode().strip()

    diff = subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"], cwd=here)
    return {
        "git_remote": _stdout(["git", "config", "--get", "remote.origin.url"]),
        "git_branch": _stdout(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit": _stdout(["git", "rev-parse", "HEAD"]),
        "uncommitted_changes": diff.returncode != 0,
    }


def _jsonable(value: Any) -> Any:
    """Coerce click parameter values into JSON-serialisable primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (URIPath, Path)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    return str(value)


def get_runtime_metadata(
    command: str,
    params: Mapping[str, Any] | None = None,
    *,
    extra: Mapping[str, Any] | None = None,
    runtime_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the metadata record written by every WSInsight subcommand.

    All commands share the same base schema ``{command, params, runtime,
    timestamp}``.  ``runtime_extra`` is merged into the ``runtime`` block
    (e.g. ``pytorch_version`` from ``patch`` / ``infer``); ``extra`` is merged
    at the top level (e.g. a ``model`` block).  Both are JSON-coerced.
    """
    runtime: dict[str, Any] = {
        "version": __version__,
        "working_dir": os.getcwd(),
        # A list, not a joined string: values containing spaces stay unambiguous.
        "args": list(sys.argv),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "in_container": _inside_container(),
        "git": _get_git_info(),
    }
    if runtime_extra:
        runtime.update({str(k): _jsonable(v) for k, v in runtime_extra.items()})
    record: dict[str, Any] = {
        "command": command,
        "params": {k: _jsonable(v) for k, v in (params or {}).items()},
        "runtime": runtime,
        "timestamp": _get_timestamp_human(),
    }
    if extra:
        record.update({str(k): _jsonable(v) for k, v in extra.items()})
    return record


def write_runtime_metadata(
    results_dir: URIPath | Path,
    command: str,
    params: Mapping[str, Any] | None = None,
    *,
    extra: Mapping[str, Any] | None = None,
    runtime_extra: Mapping[str, Any] | None = None,
) -> Path:
    """Serialize the metadata record to ``results_dir/<command>_metadata_<TS>.json``.

    ``command`` is also used as the filename prefix; dashes are replaced with
    underscores so the filename matches existing ``patch_metadata_*.json`` /
    ``infer_metadata_*.json`` conventions.  ``extra`` / ``runtime_extra`` let
    ``patch`` and ``infer`` attach their model and framework details while
    keeping the shared base schema.
    """
    record = get_runtime_metadata(
        command, params, extra=extra, runtime_extra=runtime_extra
    )
    safe = command.replace("-", "_")
    out = results_dir / f"{safe}_metadata_{_get_timestamp_filename()}.json"
    click.echo(f"\nSaving metadata about run to {out}\n")
    with out.open("w") as f:
        json.dump(record, f, indent=2)
    return out
