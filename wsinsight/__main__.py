"""Executable entry point that configures multiprocessing then dispatches CLI."""

from __future__ import annotations

import multiprocessing as mp
import os

# Silence TensorFlow info / oneDNN notices emitted at first TF import (some
# transitive deps pull TF in even though wsinsight itself uses PyTorch).
# These vars must be set BEFORE any TF import happens, so they live above the
# ``torch`` import too — torch's CUDA stack can otherwise pull in TF helpers.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")        # 3 = ERROR only
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import click
import torch

from .cancel import install_sigint_handler
from .cli.cli import cli


def main() -> None:
    """Initialize runtime knobs and invoke the Click CLI."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    mp.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    install_sigint_handler()

    try:
        cli()
    except (click.Abort, KeyboardInterrupt):
        click.secho("\nWSInsight: aborted by user.", fg="yellow", err=True)
        raise SystemExit(130)
    except Exception as e:
        click.secho(f"WSInsight failed. Error message:\n{e}", fg="yellow")


if __name__ == "__main__":
    main()
