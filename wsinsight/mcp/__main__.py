"""Entry point for the ``wsinsight-mcp`` console script.

By default starts the FastMCP server over stdio (suitable for Claude
Desktop, VS Code Copilot's MCP integration, Cursor, and any other MCP
client that spawns local tool servers as child processes).

Examples::

    wsinsight-mcp                       # stdio (default)
    wsinsight-mcp --http 127.0.0.1:8765 # streamable HTTP, loopback only
    wsinsight-mcp --max-concurrent 1    # serialise GPU jobs
    wsinsight-mcp --experimental        # expose hplot/ecomp/tcomp/niche tools

For multi-user / remote deployments, run behind a reverse proxy that
adds authentication; the server itself binds to the supplied host
verbatim and provides no auth layer.
"""

from __future__ import annotations

import argparse
import logging
import sys


def _parse_http(spec: str) -> tuple[str, int]:
    if ":" not in spec:
        raise SystemExit(f"--http expects HOST:PORT, got {spec!r}")
    host, port_s = spec.rsplit(":", 1)
    try:
        port = int(port_s)
    except ValueError as exc:
        raise SystemExit(f"--http port must be an integer, got {port_s!r}") from exc
    return host or "127.0.0.1", port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wsinsight-mcp",
        description="Run the WSInsight MCP server (Model Context Protocol).",
    )
    parser.add_argument(
        "--http",
        metavar="HOST:PORT",
        default=None,
        help="Serve over Streamable HTTP on HOST:PORT instead of stdio. "
        "Bind to 127.0.0.1 unless you have a reverse proxy with auth in front.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent jobs. Default = number of visible GPUs "
        "(CUDA_VISIBLE_DEVICES, else torch.cuda.device_count(), else 1).",
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Expose experimental subcommands (hplot/hplot-finalize/ecomp/tcomp/niche).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level for the MCP server (default: INFO).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("wsinsight.mcp")

    from wsinsight.mcp.server import build_server

    mcp = build_server(
        max_concurrent=args.max_concurrent,
        experimental=args.experimental,
    )

    if args.http:
        host, port = _parse_http(args.http)
        log.info("starting MCP server on http://%s:%d (Streamable HTTP)", host, port)
        mcp.run(transport="http", host=host, port=port)
    else:
        log.info("starting MCP server on stdio")
        mcp.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
