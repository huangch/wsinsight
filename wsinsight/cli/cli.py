"""Top-level Click group wiring wsinsight's patch, infer, and run commands."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import click

from ..wsi import set_backend
from .run import run
from .infer import infer
from .patch import patch
# from .convert_csv_to_sbubmi import tosbu
from .export import export
from .hplot import hplot, hplot_finalize_cmd
from .reg import reg
from .ncomp import ncomp
from .ecomp import ecomp
from .tcomp import tcomp
from .cme import cme

_logging_levels = ["debug", "info", "warning", "error", "critical"]

# Subcommands hidden unless the user opts into experimental features by
# setting the WSINSIGHT_EXPERIMENTAL environment variable. Keep this list in
# sync with qupath-extension-wsinsight/commands/WSInsightCommands.EXPERIMENTAL.
_EXPERIMENTAL_COMMANDS = ("hplot", "hplot-finalize", "cme", "tcomp", "ecomp")


def _experimental_enabled() -> bool:
    v = os.environ.get("WSINSIGHT_EXPERIMENTAL", "").strip().lower()
    return v in {"1", "true", "yes", "on"}


# We use invoke_without_command=True so that 'wsinsight' on its own can be used for
# inference on slides.
@click.group()
@click.option(
    "--backend",
    default=None,
    help="Backend for loading whole slide images.",
    type=click.Choice(["openslide", "tiffslide"]),
)
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(_logging_levels),
    help="Set the loudness of logging.",
)
@click.version_option()
@click.pass_context
def cli(
    ctx: click.Context,
    backend: Literal["openslide"] | Literal["tiffslide"] | None,
    log_level: str,
) -> None:
    """Configure logging/backends and expose the core WSInsight subcommands."""

    # Configure logger.
    levels = {level: getattr(logging, level.upper()) for level in _logging_levels}
    level = levels[log_level]
    logging.basicConfig(level=level)

    if backend is not None:
        set_backend(backend)

    # Block invocation of experimental subcommands unless opted in. They remain
    # registered so `wsinsight describe` emits a full schema for the QuPath
    # extension, but cannot be executed without WSINSIGHT_EXPERIMENTAL=1.
    sub = ctx.invoked_subcommand
    if sub in _EXPERIMENTAL_COMMANDS and not _experimental_enabled():
        raise click.UsageError(
            f"'{sub}' is an experimental WSInsight subcommand. "
            "Set WSINSIGHT_EXPERIMENTAL=1 to enable it."
        )


cli.add_command(run)
cli.add_command(patch)
cli.add_command(infer)
# cli.add_command(tosbu)
cli.add_command(export)
cli.add_command(hplot)
cli.add_command(hplot_finalize_cmd)
cli.add_command(reg)
cli.add_command(ncomp)
cli.add_command(ecomp)
cli.add_command(tcomp)
cli.add_command(cme)

# Hide experimental commands from --help unless WSINSIGHT_EXPERIMENTAL is set.
# They remain registered so `describe` can emit the full schema; invocation is
# blocked in the group callback above.
if not _experimental_enabled():
    for _name in _EXPERIMENTAL_COMMANDS:
        _cmd = cli.commands.get(_name)
        if _cmd is not None:
            _cmd.hidden = True


def _describe_param(param: click.Parameter) -> dict[str, Any]:
    """Serialise one Click option/argument into a JSON-friendly dict."""
    kind: str
    choices: list[str] = []
    t = param.type
    if isinstance(t, click.Choice):
        kind = "choice"
        choices = list(t.choices)
    elif isinstance(t, click.Path):
        kind = "path"
    elif isinstance(t, click.types.BoolParamType):
        kind = "bool"
    elif isinstance(t, click.types.IntParamType):
        kind = "int"
    elif isinstance(t, click.types.FloatParamType):
        kind = "float"
    else:
        kind = "string"

    default = param.default
    if callable(default):
        default = None
    if isinstance(default, Path):
        default = str(default)
    # Make sure the default is JSON-serialisable; Click sometimes uses sentinel
    # objects (e.g. UNSET) to mark "no default" for required options.
    try:
        json.dumps(default)
    except TypeError:
        default = None

    entry: dict[str, Any] = {
        "name": param.name,
        "kind": kind,
        "required": bool(param.required),
        "default": default,
        "help": (param.help if isinstance(param, click.Option) else "") or "",
        "multiple": bool(getattr(param, "multiple", False)),
        "is_flag": bool(getattr(param, "is_flag", False)),
    }
    if isinstance(param, click.Option):
        entry["param_type"] = "option"
        # First declaration is typically the long option (e.g. "--wsi-dir")
        entry["flags"] = list(param.opts) + list(param.secondary_opts)
    else:
        entry["param_type"] = "argument"
        entry["flags"] = []
    if choices:
        entry["choices"] = choices
    if kind == "path":
        entry["path_file_okay"] = bool(getattr(t, "file_okay", True))
        entry["path_dir_okay"] = bool(getattr(t, "dir_okay", True))
        entry["path_exists"] = bool(getattr(t, "exists", False))
    return entry


def _describe_command(name: str, cmd: click.Command) -> dict[str, Any]:
    ctx = click.Context(cmd, info_name=name)
    params: list[dict[str, Any]] = []
    for p in cmd.get_params(ctx):
        # Skip auto-added --help
        if p.name == "help":
            continue
        params.append(_describe_param(p))
    return {
        "name": name,
        "help": (cmd.help or cmd.short_help or "").strip(),
        "params": params,
    }


@cli.command(name="describe")
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write the schema JSON to this file instead of stdout.",
)
def describe_cmd(output_path: str | None) -> None:
    """Emit a machine-readable JSON schema of every wsinsight subcommand.

    Intended for downstream tools (e.g. the QuPath extension) that want to
    auto-generate forms without hard-coding the CLI surface. The output is a
    JSON object with a ``commands`` dict keyed by subcommand name.
    """
    schema: dict[str, Any] = {"schema_version": 1, "commands": {}}
    for name, cmd in cli.commands.items():
        if name == "describe":
            continue
        schema["commands"][name] = _describe_command(name, cmd)
    payload = json.dumps(schema, indent=2, sort_keys=True)
    if output_path:
        Path(output_path).write_text(payload + "\n", encoding="utf-8")
    else:
        click.echo(payload)