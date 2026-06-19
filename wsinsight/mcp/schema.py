"""Schema-driven generation of MCP tool definitions from the WSInsight CLI.

The single source of truth is :file:`wsinsight/cli/cli_schema.json`,
produced by ``wsinsight describe``. This module loads it, classifies
commands as stable / experimental and short / long-running, and converts
each Click parameter into a JSON-schema property suitable for use as an
MCP tool input schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Subcommands that are exposed by default. Mirrors the gating in
# wsinsight/cli/cli.py (experimental commands are hidden unless the
# WSINSIGHT_EXPERIMENTAL env var is set).
STABLE_COMMANDS: frozenset[str] = frozenset(
    {"run", "patch", "infer", "ncomp", "export", "reg"}
)
EXPERIMENTAL_COMMANDS: frozenset[str] = frozenset(
    {"hplot", "hplot-finalize", "ecomp", "tcomp", "cme", "cme-profile"}
)

# Commands that may run for many minutes or hours. These are exposed as
# background-job tools in the MCP server (the tool returns a job_id and
# the agent polls job_status / job_logs / cancel_job). All other stable
# commands run synchronously.
LONG_RUNNING_COMMANDS: frozenset[str] = frozenset(
    {"run", "patch", "infer", "ncomp", "hplot", "ecomp", "tcomp", "cme"}
)

_KIND_TO_JSON_TYPE: dict[str, str] = {
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "path": "string",
    "choice": "string",
}


def schema_path() -> Path:
    """Return the path to the bundled CLI JSON schema."""
    return Path(__file__).resolve().parent.parent / "cli" / "cli_schema.json"


def load_schema() -> dict[str, Any]:
    """Load and return the bundled CLI JSON schema."""
    return json.loads(schema_path().read_text(encoding="utf-8"))


def _param_to_json_property(param: dict[str, Any]) -> dict[str, Any]:
    """Convert one Click parameter entry into a JSON-schema property dict."""
    kind = str(param.get("kind", "string")).lower()
    json_type = _KIND_TO_JSON_TYPE.get(kind, "string")
    prop: dict[str, Any] = {"type": json_type}
    help_text = param.get("help")
    if help_text:
        prop["description"] = " ".join(str(help_text).split())
    if param.get("multiple"):
        prop = {"type": "array", "items": prop}
        if help_text:
            prop["description"] = " ".join(str(help_text).split())
    default = param.get("default")
    if (
        default is not None
        and not param.get("required", False)
        and not param.get("multiple", False)
    ):
        prop["default"] = default
    return prop


def command_to_input_schema(command: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-schema ``object`` describing one command's parameters.

    The schema's keys are the canonical Click parameter names (snake_case),
    which is what :mod:`wsinsight.mcp.adapters` will translate back to the
    CLI's flag names.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in command.get("params", []):
        if param.get("param_type") == "argument":
            # Positional arguments are rare in WSInsight CLI; treat the
            # same way as required options.
            pass
        name = str(param["name"])
        properties[name] = _param_to_json_property(param)
        if param.get("required"):
            required.append(name)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def discover_commands(experimental: bool = False) -> dict[str, dict[str, Any]]:
    """Return ``{name: command_dict}`` for the commands the server should expose."""
    raw = load_schema().get("commands", {})
    allowed = set(STABLE_COMMANDS)
    if experimental:
        allowed |= set(EXPERIMENTAL_COMMANDS)
    return {name: cmd for name, cmd in raw.items() if name in allowed}


def is_long_running(name: str) -> bool:
    """Return True if the named command should be exposed as a background job."""
    return name in LONG_RUNNING_COMMANDS


__all__ = [
    "STABLE_COMMANDS",
    "EXPERIMENTAL_COMMANDS",
    "LONG_RUNNING_COMMANDS",
    "schema_path",
    "load_schema",
    "command_to_input_schema",
    "discover_commands",
    "is_long_running",
]
