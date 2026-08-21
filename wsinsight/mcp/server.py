"""FastMCP server exposing WSInsight subcommands as MCP tools.

Tools are auto-registered from the bundled CLI schema. Long-running
commands (``run``, ``patch``, ``infer``, ``ncomp`` …) return a
``job_id`` immediately and the agent polls ``job_status`` /
``job_logs`` / ``cancel_job``. Short commands (``export``, ``reg``)
run synchronously and return the subprocess exit code plus a tail of
its stdout/stderr.

Each subcommand is exposed with the exact Click parameter names and
types from :file:`wsinsight/cli/cli_schema.json`, so the MCP tool's
input schema is a faithful per-parameter mirror of the CLI's
``--help`` rather than a generic ``args: dict`` blob.

Usage::

    from wsinsight.mcp.server import build_server

    mcp = build_server(max_concurrent=2, experimental=False)
    mcp.run()                      # stdio
    mcp.run(transport="http",      # streamable HTTP
            host="127.0.0.1", port=8765)
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated
from typing import Any

try:
    from fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "wsinsight.mcp requires the 'fastmcp' package. "
        "Install it with: pip install 'wsinsight[mcp]'"
    ) from exc

from pydantic import Field

from wsinsight.mcp.adapters import args_to_argv
from wsinsight.mcp.jobs import JobManager
from wsinsight.mcp.schema import (
    command_to_input_schema,  # noqa: F401 - re-exported for tests
)
from wsinsight.mcp.schema import discover_commands
from wsinsight.mcp.schema import is_long_running
from wsinsight.mcp.schema import load_schema

# -- Click kind -> Python type mapping -------------------------------------

_KIND_TO_PY: dict[str, type] = {
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "number": float,
    "bool": bool,
    "boolean": bool,
    "path": str,
    "choice": str,
}


def _build_signature(
    command: dict[str, Any],
) -> tuple[inspect.Signature, dict[str, Any]]:
    """Return ``(signature, annotations_dict)`` for one CLI command."""
    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    for p in command.get("params", []):
        if p.get("param_type") not in ("option", "argument"):
            continue
        pname = p["name"]
        kind = str(p.get("kind", "string")).lower()
        py_type: Any = _KIND_TO_PY.get(kind, str)
        if p.get("multiple"):
            py_type = list[py_type]  # type: ignore[valid-type]
        help_text = " ".join(str(p.get("help", "")).split())

        if p.get("required"):
            annotation = (
                Annotated[py_type, Field(description=help_text)]
                if help_text
                else py_type
            )
            default: Any = inspect.Parameter.empty
        else:
            # Allow None as the absence sentinel so adapters can drop it.
            wide = py_type | None  # type: ignore[operator]
            annotation = (
                Annotated[wide, Field(description=help_text)] if help_text else wide
            )
            raw_default = p.get("default", None)
            default = raw_default
        parameters.append(
            inspect.Parameter(
                name=pname,
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=annotation,
            )
        )
        annotations[pname] = annotation
    return inspect.Signature(parameters), annotations


# -- subprocess runners ----------------------------------------------------


def _run_sync(argv_tail: list[str], timeout_s: float = 600.0) -> dict[str, Any]:
    """Run ``python -m wsinsight <argv_tail>`` synchronously and capture output."""
    started = time.time()
    try:
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "wsinsight"] + argv_tail,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "argv": argv_tail,
            "duration_s": round(time.time() - started, 3),
            "error": f"command exceeded {timeout_s}s synchronous timeout",
        }
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = out.splitlines()[-50:]
    return {
        "status": "done" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "argv": argv_tail,
        "duration_s": round(time.time() - started, 3),
        "log_tail": tail,
    }


def _make_long_tool(jobs: JobManager, name: str, command: dict[str, Any]):
    """Build a per-command tool function whose signature mirrors the CLI."""

    sig, ann = _build_signature(command)

    def _impl(**kwargs: Any) -> dict[str, Any]:
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        argv_tail = args_to_argv(command, cleaned)
        job_id = jobs.submit(name, argv_tail)
        return {
            "job_id": job_id,
            "status": "started",
            "argv": argv_tail,
            "hint": (
                f"Poll job_status(job_id={job_id!r}) and "
                f"job_logs(job_id={job_id!r}). "
                f"Cancel with cancel_job(job_id={job_id!r})."
            ),
        }

    _impl.__signature__ = sig  # type: ignore[attr-defined]
    _impl.__annotations__ = dict(ann)
    _impl.__name__ = name.replace("-", "_")
    return _impl


def _make_short_tool(name: str, command: dict[str, Any]):
    sig, ann = _build_signature(command)

    def _impl(**kwargs: Any) -> dict[str, Any]:
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        argv_tail = args_to_argv(command, cleaned)
        return _run_sync(argv_tail)

    _impl.__signature__ = sig  # type: ignore[attr-defined]
    _impl.__annotations__ = dict(ann)
    _impl.__name__ = name.replace("-", "_")
    return _impl


def _walk_results_dir(root: Path, max_entries: int = 500) -> list[dict]:
    out: list[dict] = []
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if len(out) >= max_entries:
            out.append({"path": "...", "note": f"truncated at {max_entries} entries"})
            break
        rel = str(p.relative_to(root))
        try:
            size = p.stat().st_size if p.is_file() else None
        except OSError:
            size = None
        out.append(
            {
                "path": rel,
                "type": "dir" if p.is_dir() else "file",
                "size_bytes": size,
            }
        )
    return out


# -- builder ---------------------------------------------------------------


def build_server(
    *,
    max_concurrent: int | None = None,
    experimental: bool = False,
    server_name: str = "wsinsight",
) -> "FastMCP":
    """Build and return a configured (but not-yet-running) :class:`FastMCP` server."""
    mcp = FastMCP(server_name)
    jobs = JobManager(max_concurrent=max_concurrent, experimental=experimental)

    # 1. Per-subcommand tools.
    for name, cmd in discover_commands(experimental=experimental).items():
        long_running = is_long_running(name)
        fn = (
            _make_long_tool(jobs, name, cmd)
            if long_running
            else _make_short_tool(name, cmd)
        )
        help_text = " ".join(str(cmd.get("help", "")).split())
        if long_running:
            description = (
                help_text
                + "\n\n[long-running] Returns a job_id; poll job_status / job_logs "
                "and stop early with cancel_job."
            )
        else:
            description = help_text
        mcp.tool(name=name.replace("-", "_"), description=description)(fn)

    # 2. Job-management meta-tools.
    @mcp.tool(
        name="job_status",
        description=(
            "Return a snapshot of one job (status, pid, GPU, duration, "
            "returncode, total log lines). Use job_logs to read output."
        ),
    )
    def job_status(job_id: str) -> dict | None:
        return jobs.status(job_id)

    @mcp.tool(
        name="job_logs",
        description=(
            "Return the next chunk of stdout/stderr lines for a job. Pass "
            "since_line from a previous response's next_line to paginate."
        ),
    )
    def job_logs(job_id: str, since_line: int = 0, max_lines: int = 500) -> dict | None:
        return jobs.logs(job_id, since_line=since_line, max_lines=max_lines)

    @mcp.tool(
        name="cancel_job",
        description=(
            "Request graceful cancellation (SIGINT) of a running job. Calling "
            "cancel_job a second time on the same job escalates to SIGTERM."
        ),
    )
    def cancel_job(job_id: str) -> dict | None:
        return jobs.cancel(job_id)

    @mcp.tool(
        name="list_jobs",
        description="List all jobs (running and completed) known to this server.",
    )
    def list_jobs() -> list[dict]:
        return jobs.list()

    # 3. Resources.
    @mcp.resource(
        "wsinsight://schema",
        name="cli_schema",
        description="The full WSInsight CLI JSON schema (single source of truth).",
        mime_type="application/json",
    )
    def schema_resource() -> str:
        return json.dumps(load_schema(), indent=2)

    @mcp.resource(
        "wsinsight://models",
        name="model_zoo",
        description="Names of WSInfer/WSInsight Model Zoo entries discovered locally.",
        mime_type="application/json",
    )
    def models_resource() -> str:
        out: dict[str, Any] = {"source": "local-cache", "models": []}
        try:
            from wsinfer_zoo.client import load_registry  # type: ignore

            reg = load_registry()
            out["models"] = sorted(reg.models.keys())
        except Exception as exc:  # pragma: no cover
            out["error"] = repr(exc)
        return json.dumps(out, indent=2)

    @mcp.resource(
        "wsinsight://results/{results_dir}/layout",
        name="results_layout",
        description=(
            "Recursive listing of a results directory (file paths + sizes). "
            "Pass the results_dir as a URL-encoded absolute or relative path."
        ),
        mime_type="application/json",
    )
    def results_layout(results_dir: str) -> str:
        root = Path(results_dir).expanduser()
        return json.dumps(
            {"root": str(root.resolve()), "entries": _walk_results_dir(root)},
            indent=2,
        )

    # 4. Prompt.
    @mcp.prompt(
        name="reproduce_tcga_crc",
        description=(
            "Walk through reproducing the manuscript's TCGA-CRC analysis "
            "end-to-end using the wsinsight tools exposed by this server."
        ),
    )
    def reproduce_tcga_crc() -> str:
        return (
            "You are an analyst reproducing the TCGA-CRC analysis described "
            "in the WSInsight manuscript. Use the tools exposed by this MCP "
            "server to:\n"
            "1. Call `infer` (long-running) on the cohort with the PanNuke "
            "CellViT-SAM-H-x40 model. The wsi_dir argument may be a "
            "directory of local WSIs, an s3:// URI, or a GDC manifest path.\n"
            "2. After the job completes (poll `job_status`), call `ncomp` "
            "with default ncomp_k=2 and ncomp_max_neighbor_distance=25 (microns).\n"
            "3. Call `export` with --geojson and/or --omecsv enabled.\n"
            "4. Read the `wsinsight://results/<results-dir>/layout` resource "
            "to confirm GeoJSON, OME-CSV and ncomp CSVs were produced.\n"
            "Use job_logs to surface progress to the user; cancel with "
            "cancel_job if asked."
        )

    return mcp


__all__ = ["build_server"]
