"""Standalone CLI for triad composition (tcomp) analysis over WSInsight outputs.

For each Delaunay triad (triangle), ``wsinsight tcomp`` builds the dual graph
of the Delaunay triangulation (vertex-shared adjacency), computes k-hop
neighbors at the triad level, and records the triad-type composition of each
triad's local neighborhood, alongside per-triad geometric features.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
from platformdirs import user_cache_dir

from ..insightlib.tcomp_generation import tcomp_generation
from ..uri_path import URIPath, URIPathType


def _assert_directory(path: URIPath, option_name: str) -> None:
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


@click.command()
@click.option(
    "-i",
    "--wsi-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help=(
        "Directory containing whole slide images.  Used only to enumerate slides "
        "and to resolve µm-per-pixel spacing; images are not fully read during tcomp."
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
    "--tcomp-max-edge",
    default=25.0,
    type=click.FloatRange(min=0),
    show_default=True,
    help="Maximum triad edge length (µm); triads whose longest edge exceeds this are dropped.",
)
@click.option(
    "--tcomp-k",
    default=2,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of hops that define the triad-level neighborhood radius (k-hop on the dual graph).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="Recompute and overwrite existing per-slide tcomp outputs.",
)
@click.option(
    "--num-workers",
    default=8,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of slides to process concurrently.",
)
@click.option(
    "--no-neighborhood",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Skip dual-graph construction and k-hop aggregation.  Output one row per "
        "Delaunay triad with geometry, cell types, triad type, and a ``centroid_region`` "
        "column (derived from per-vertex ``region_prob_*`` columns).  Much faster; "
        "no ``neighborhood_*`` columns are written."
    ),
)
def tcomp(
    *,
    wsi_dir: URIPath,
    results_dir: URIPath,
    tcomp_max_edge: float = 25.0,
    tcomp_k: int = 2,
    overwrite: bool = False,
    num_workers: int = 8,
    no_neighborhood: bool = False,
) -> None:
    """Compute triad-level neighborhood composition for WSInsight inference outputs.

    For every Delaunay triad (triangle of three cells), the k-hop neighboring
    triads are collected from the dual graph and the proportion of each triad
    type in that neighborhood is recorded, together with per-triad geometric
    features (area, perimeter, regularity, max edge length).

    \b
    Outputs written to <results-dir>/:
      tcomp-outputs-csv/<slide_id>.csv   per-triad neighborhood composition
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

    click.secho("\nRunning triad composition (tcomp) analysis.\n", fg="green")

    failed = tcomp_generation(
        wsi_dir=wsi_dir,
        slide_paths=slide_paths,
        results_dir=results_dir,
        max_edge_um=tcomp_max_edge,
        tcomp_k=tcomp_k,
        num_workers=num_workers,
        overwrite=overwrite,
        no_neighborhood=no_neighborhood,
    )

    if failed:
        click.secho(
            f"\ntcomp failed for {len(failed)} slide(s):", fg="yellow"
        )
        click.secho("\n".join(failed), fg="yellow")
    else:
        click.secho("\ntcomp completed successfully.\n", fg="green")
