"""Executable entry point that configures multiprocessing then dispatches CLI."""

from __future__ import annotations

import multiprocessing as mp
import os

import click
import torch

from .cli.cli import cli


def main() -> None:
    """Initialize runtime knobs and invoke the Click CLI."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    mp.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")

    try:
        cli()
    except Exception as e:
        click.secho(f"WSInsight failed. Error message:\n{e}", fg="yellow")


if __name__ == "__main__":
    main()
