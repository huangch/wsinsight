"""Standalone CLI for neighborhood composition (ncomp) analysis over WSInsight outputs.

For each cell, ``wsinsight ncomp``
builds a Delaunay graph, computes k-hop neighbors, and records the cell-type
composition of each cell's local neighborhood.  The same graph construction
is used by ``wsinsight hplot``; ``ncomp`` differs in that analysis is per-cell
rather than per-layer.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import List

import click
from platformdirs import user_cache_dir

from ..insightlib.ncomp_generation import ncomp_generation
from ..uri_path import URIPath, URIPathType
from ._meta import write_runtime_metadata


# ---------------------------------------------------------------------------
# Shared CLI helpers (mirrored from cli/hplot.py)
# ---------------------------------------------------------------------------

def _coerce_number(token: str) -> int | float | str:
    """Convert CLI token to int/float when possible, otherwise lowercase text."""
    t = token.strip()
    if re.fullmatch(r"[+-]?\d+", t):
        try:
            return int(t)
        except ValueError:
            pass
    try:
        x = float(t)
        if math.isfinite(x):
            return x
    except ValueError:
        pass
    return t.lower()


def _csv_to_list(
    _: click.Context, __: click.Parameter, value: str | list[str] | None
) -> list[int | float | str]:
    if value is None:
        return []
    tokens = (
        value
        if isinstance(value, list)
        else [x for x in re.split(r"[,\s]+", str(value).strip()) if x]
    )
    return [_coerce_number(str(x)) for x in tokens]


def _assert_directory(path: URIPath, option_name: str) -> None:
    """Ensure the provided ``URIPath`` exists and points to a directory."""
    if not path.exists():
        raise click.ClickException(f"{option_name} directory not found: {path}")
    if not path.is_dir():
        raise click.ClickException(f"{option_name} must be a directory")


def _storage_kwargs() -> dict[str, object]:
    cache_dir = os.getenv("WSINSIGHT_REMOTE_CACHE_DIR")
    if cache_dir is None:
        cache_dir = Path(user_cache_dir(appname="wsinsight", appauthor=False))
    storage: dict[str, object] = {"cache_dir": cache_dir}
    s3_options = os.getenv("S3_STORAGE_OPTIONS")
    if s3_options:
        try:
            parsed = json.loads(s3_options)
        except json.JSONDecodeError as exc:
            raise RuntimeError("S3_STORAGE_OPTIONS must contain valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("S3_STORAGE_OPTIONS must be a JSON object.")
        storage.update(parsed)
    return storage


_STORAGE_KWARGS = _storage_kwargs()


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "-i",
    "--wsi-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help=(
        "Directory containing whole slide images.  Used only to enumerate slides "
        "and to resolve µm-per-pixel spacing; images are not fully read during ncomp."
    ),
)
@click.option(
    "-o",
    "--results-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help=(
        "Directory holding WSInsight inference outputs.  Must contain a "
        "``model-outputs-csv/`` subdirectory produced by ``wsinsight infer`` or "
        "``wsinsight run``."
    ),
)
@click.option(
    "--ncomp-max-neighbor-distance",
    default=25.0,
    type=click.FloatRange(min=0),
    show_default=True,
    help="Maximum distance (µm) between neighboring cells in the Delaunay graph.",
)
@click.option(
    "--ncomp-k",
    default=2,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of hops that define the neighborhood radius (k-hop).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="Recompute and overwrite existing per-slide ncomp outputs.",
)
@click.option(
    "--num-workers",
    default=8,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of slides to process concurrently.",
)
def ncomp(
    *,
    wsi_dir: URIPath,
    results_dir: URIPath,
    ncomp_max_neighbor_distance: float = 25.0,
    ncomp_k: int = 2,
    overwrite: bool = False,
    num_workers: int = 8,
) -> None:
    """Compute neighborhood composition for each cell in WSInsight inference outputs.

    For every cell (cell type determined by the highest ``prob_*`` score),
    the k-hop Delaunay graph neighbors are collected and the proportion of each
    cell type in that neighborhood is recorded.

    \b
    Outputs written to <results-dir>/:
      ncomp-outputs-csv/<slide_id>.csv   per-cell neighborhood composition
    """

    wsi_dir = wsi_dir.coerce_image_list()
    _assert_directory(wsi_dir, "--wsi-dir")
    _assert_directory(results_dir, "--results-dir")

    slide_paths = sorted([p for p in wsi_dir.iterdir() if wsi_dir.scheme == "image-list" or p.is_file()])
    if not slide_paths:
        raise click.ClickException(f"No files found in slide directory: {wsi_dir}")

    model_output_dir = results_dir / "model-outputs-csv"
    if not model_output_dir.exists():
        raise click.ClickException(
            "The 'model-outputs-csv' directory was not found in the results directory. "
            "Run 'wsinsight infer' or 'wsinsight run' first."
        )

    click.secho("\nRunning neighborhood composition (ncomp) analysis.\n", fg="green")

    failed = ncomp_generation(
        wsi_dir=wsi_dir,
        slide_paths=slide_paths,
        results_dir=results_dir,
        max_neighbor_distance_um=ncomp_max_neighbor_distance,
        ncomp_k=ncomp_k,
        num_workers=num_workers,
        overwrite=overwrite,
    )

    if failed:
        click.secho(
            f"\nncomp failed for {len(failed)} slide(s):", fg="yellow"
        )
        click.secho("\n".join(failed), fg="yellow")
    else:
        click.secho("\nncomp completed successfully.\n", fg="green")

    write_runtime_metadata(
        results_dir,
        "ncomp",
        params=click.get_current_context().params,
    )
