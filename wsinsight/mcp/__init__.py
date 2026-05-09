"""WSInsight MCP (Model Context Protocol) server.

This subpackage exposes the WSInsight CLI as MCP tools so that AI agents
(e.g. Claude Desktop, VS Code Copilot) can invoke WSInsight subcommands
through the same surface as human users and viewer plugins.

Entry point: ``wsinsight-mcp`` (see :mod:`wsinsight.mcp.__main__`).
"""

from __future__ import annotations

__all__ = ["build_server"]


def build_server(*args, **kwargs):  # pragma: no cover - thin re-export
    from wsinsight.mcp.server import build_server as _b

    return _b(*args, **kwargs)
