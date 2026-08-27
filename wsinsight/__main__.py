"""Executable entry point that configures multiprocessing then dispatches CLI."""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback

import click

from .cancel import install_sigint_handler
from .cli.cli import cli


def main() -> None:
    """Initialize runtime knobs and invoke the Click CLI."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    mp.set_start_method("spawn", force=True)
    # Keep lightweight commands (for example `--help` / `describe`) usable in
    # environments where heavy ML deps are not installed yet.
    try:
        import torch
    except Exception:  # noqa: BLE001
        torch = None
    if torch is not None:
        torch.multiprocessing.set_sharing_strategy("file_system")
    install_sigint_handler()

    try:
        cli()
    except (click.Abort, KeyboardInterrupt):
        click.secho("\nWSInsight: aborted by user.", fg="yellow", err=True)
        raise SystemExit(130) from None
    except Exception:
        # Print the full Python traceback so the user can diagnose without
        # re-running with --stack-trace.  ``traceback.format_exc()`` walks
        # ``sys.exc_info()`` for the currently-handled exception.
        click.secho(
            f"WSInsight failed. Traceback:\n{traceback.format_exc()}",
            fg="yellow",
            err=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
