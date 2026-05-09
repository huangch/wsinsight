"""Translate MCP-supplied dict args into a WSInsight CLI argv list.

The MCP tool input schema (see :mod:`wsinsight.mcp.schema`) uses Click's
canonical parameter names (snake_case). The CLI itself accepts kebab-case
flags. This module looks up each parameter in the bundled CLI schema and
emits the correct ``--flag value`` (or bare ``--flag`` for boolean flags)
sequence, so that a single generic adapter can drive every subcommand
without per-command boilerplate.
"""

from __future__ import annotations

from typing import Any


class AdapterError(ValueError):
    """Raised when MCP-supplied args cannot be translated into a valid argv."""


def _flag_for(param: dict[str, Any]) -> str:
    """Pick the long flag (``--foo-bar``) for a Click param."""
    flags = param.get("flags") or []
    for f in flags:
        if f.startswith("--"):
            return f
    if flags:
        return flags[0]
    raise AdapterError(f"Parameter has no flags: {param.get('name')!r}")


def args_to_argv(command: dict[str, Any], args: dict[str, Any]) -> list[str]:
    """Return ``[<subcommand>, --flag, value, ...]`` for ``wsinsight ...``.

    * Boolean flags are emitted as bare flags only when truthy.
    * Repeatable flags (``multiple=True``) are emitted once per value.
    * Unknown keys raise :class:`AdapterError`.
    * Required parameters missing from ``args`` raise :class:`AdapterError`.
    """
    name = command["name"]
    params = {p["name"]: p for p in command.get("params", [])}

    unknown = set(args) - set(params)
    if unknown:
        raise AdapterError(
            f"Unknown parameter(s) for command {name!r}: {sorted(unknown)}. "
            f"Allowed: {sorted(params)}"
        )

    missing = [
        n
        for n, p in params.items()
        if p.get("required") and (n not in args or args[n] is None)
    ]
    if missing:
        raise AdapterError(
            f"Missing required parameter(s) for command {name!r}: {missing}"
        )

    argv: list[str] = [name]
    for n, p in params.items():
        if n not in args or args[n] is None:
            continue
        value = args[n]
        flag = _flag_for(p)
        if p.get("is_flag"):
            if bool(value):
                argv.append(flag)
            continue
        if p.get("multiple"):
            if not isinstance(value, (list, tuple)):
                raise AdapterError(
                    f"Parameter {n!r} of command {name!r} expects a list "
                    f"(multiple=True); got {type(value).__name__}."
                )
            for item in value:
                argv.extend([flag, str(item)])
            continue
        argv.extend([flag, str(value)])
    return argv


__all__ = ["AdapterError", "args_to_argv"]
