"""Standalone CLI for edge composition (ecomp) analysis over WSInsight outputs.

For each Delaunay edge, ``wsinsight ecomp`` builds the line graph of the
Delaunay triangulation, computes k-hop neighbors at the edge level, and
records the edge-type composition of each edge's local neighborhood.
"""

from __future__ import annotations

import click

from ..insightlib.ecomp_generation import _gpu_available
from ..insightlib.ecomp_generation import ecomp_generation
from ..uri_path import URIPath
from ..uri_path import URIPathType
from ._meta import write_runtime_metadata
from ._paths import default_storage_kwargs
from ._paths import ensure_input_directory

_STORAGE_KWARGS = default_storage_kwargs()


@click.command()
@click.option(
    "-i",
    "--wsi-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help=(
        "Directory containing whole slide images.  Used only to enumerate slides "
        "and to resolve µm-per-pixel spacing; images are not fully read during ecomp."
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
    "--max-edge",
    "ecomp_max_edge",
    default=25.0,
    type=click.FloatRange(min=0),
    show_default=True,
    help="Maximum Delaunay edge length (µm); longer edges are pruned before building the line graph.",
)
@click.option(
    "--k",
    "ecomp_k",
    default=2,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of hops that define the edge-level neighborhood radius (k-hop on the line graph).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="Recompute and overwrite existing per-slide ecomp outputs.",
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
    "no_neighborhood",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Skip line-graph construction and k-hop aggregation.  Output one row per "
        "Delaunay edge with geometry, cell types, edge type, and a ``center_region`` "
        "column (derived from per-vertex ``region_prob_*`` columns).  Much faster; "
        "no ``neighborhood_*`` columns are written."
    ),
)
def ecomp(
    *,
    wsi_dir: URIPath,
    results_dir: URIPath,
    ecomp_max_edge: float = 25.0,
    ecomp_k: int = 2,
    overwrite: bool = False,
    num_workers: int = 8,
    no_neighborhood: bool = False,
) -> None:
    """Compute edge-level neighborhood composition for WSInsight inference outputs.

    For every Delaunay edge (two cells connected in the triangulation), the
    k-hop neighboring edges are collected from the line graph and the
    proportion of each edge type in that neighborhood is recorded.

    \b
    Outputs written to <results-dir>/:
      ecomp-outputs-csv/<slide_id>.csv   per-edge neighborhood composition
    """

    wsi_dir = wsi_dir.coerce_image_list()
    ensure_input_directory(wsi_dir, "--wsi-dir")
    ensure_input_directory(results_dir, "--results-dir")

    from ..wsi import list_slide_paths

    slide_paths = list_slide_paths(wsi_dir)
    if not slide_paths:
        raise click.ClickException(f"No files found in slide directory: {wsi_dir}")

    model_output_dir = results_dir / "model-outputs-csv"
    if not model_output_dir.exists():
        raise click.ClickException(
            "The 'model-outputs-csv' directory was not found in the results directory. "
            "Run 'wsinsight infer' or 'wsinsight run' first."
        )

    click.secho("\nRunning edge composition (ecomp) analysis.\n", fg="green")
    backend = "cuda" if _gpu_available() else "cpu"
    click.secho(f"ecomp backend: {backend}", fg="cyan")

    failed = ecomp_generation(
        wsi_dir=wsi_dir,
        slide_paths=slide_paths,
        results_dir=results_dir,
        max_edge_um=ecomp_max_edge,
        ecomp_k=ecomp_k,
        num_workers=num_workers,
        overwrite=overwrite,
        no_neighborhood=no_neighborhood,
    )

    if failed:
        click.secho(f"\necomp failed for {len(failed)} slide(s):", fg="yellow")
        click.secho("\n".join(failed), fg="yellow")
    else:
        click.secho("\necomp completed successfully.\n", fg="green")

    write_runtime_metadata(
        results_dir,
        "ecomp",
        params=click.get_current_context().params,
    )
