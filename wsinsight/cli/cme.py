"""Standalone CLI for cellular microenvironment (CME) analysis.

``wsinsight cme`` discovers recurring cellular microenvironments across a cohort
of whole slide images.  It builds per-slide Delaunay graphs, trains a global DGI
encoder, clusters the resulting embeddings, and writes per-cell CME labels plus
optional annotation-level region merges.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import pandas as pd

from ..insightlib.cme_generation import cme_generation
from ..uri_path import URIPath, URIPathType
from ..write_geojson import write_geojsons
from ._meta import write_runtime_metadata
from ._paths import (
    default_storage_kwargs,
    ensure_input_directory,
)

_STORAGE_KWARGS = default_storage_kwargs()


def _num_cpus() -> int:
    """Get number of CPUs on the system."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 0


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
        "Directory containing whole slide images.  Used for slide enumeration "
        "and um-per-pixel resolution; images are read only when --cme-hoptimus "
        "is set."
    ),
)
@click.option(
    "-o",
    "--results-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help=(
        "Directory holding WSInsight inference outputs.  Must contain a "
        "model-outputs-csv/ subdirectory produced by wsinsight infer or "
        "wsinsight run."
    ),
)
@click.option(
    "--cme-hoptimus",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Enable H-Optimus tissue morphology features in addition to k-hop "
        "cell-type composition.  Requires a GPU and the timm package."
    ),
)
@click.option(
    "--cme-clusters",
    default=None,
    type=click.IntRange(min=2),
    help=(
        "Number of microenvironment clusters (KMeans).  When omitted, the "
        "optimal number is determined automatically via a Leiden "
        "community-detection sweep."
    ),
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="Delete cached checkpoints and recompute all CME outputs from scratch.",
)
@click.option(
    "--num-workers",
    default=min(_num_cpus(), 8),
    show_default=True,
    type=click.IntRange(min=0),
    help="Number of workers for GeoJSON export at the end of the run.",
)
def cme(
    *,
    wsi_dir: URIPath,
    results_dir: URIPath,
    cme_hoptimus: bool = False,
    cme_clusters: int | None = None,
    overwrite: bool = False,
    num_workers: int = 8,
) -> None:
    """Discover cellular microenvironments (CMEs) across a cohort of slides.

    Builds per-slide Delaunay cell graphs, trains a global Deep Graph Infomax
    (DGI) encoder across all slides, clusters the resulting embeddings, and
    writes per-cell CME labels and annotation-level region merges.

    \b
    Outputs written to <results-dir>/:
      cme-outputs-csv/cells/<slide>.csv   per-cell CME labels + features
      cme-outputs-csv/cmes/<slide>.csv    annotation-level merged CME regions
      cme-outputs-geojson/cells/          GeoJSON cell detections with CME labels
      cme-outputs-geojson/cmes/           GeoJSON CME region annotations
    """

    wsi_dir = wsi_dir.coerce_image_list()
    ensure_input_directory(wsi_dir, "--wsi-dir")
    ensure_input_directory(results_dir, "--results-dir")

    slide_paths = sorted(
        p for p in wsi_dir.iterdir()
        if wsi_dir.scheme == "image-list" or p.is_file()
    )
    if not slide_paths:
        raise click.ClickException(
            f"No files found in slide directory: {wsi_dir}"
        )

    model_output_dir = results_dir / "model-outputs-csv"
    if not model_output_dir.exists():
        raise click.ClickException(
            "The 'model-outputs-csv' directory was not found in the results "
            "directory.  Run 'wsinsight infer' or 'wsinsight run' first."
        )

    click.secho(
        "\nRunning cellular microenvironment (CME) analysis.\n", fg="green"
    )

    cme_generation(
        wsi_dir=wsi_dir,
        wsi_paths=slide_paths,
        results_dir=results_dir,
        max_edge_len_um=25.0,
        max_cell_radius_um=15.0,
        k_hops=2,
        alpha=1.0,
        use_hoptimus=cme_hoptimus,
        hidden=64,
        out_dim=32,
        epochs=300,
        cme_cellular=True,
        cme_annotation=True,
        cme_clustering_k=cme_clusters,
        cme_clustering_resolutions=[0.5, 1.0, 2.0],
        overwrite=overwrite,
    )

    # --- GeoJSON: cell-level detections with CME labels ----------------------
    cme_cells_dir = Path(str(results_dir)) / "cme-outputs-csv" / "cells"
    if cme_cells_dir.exists():
        cme_cell_csvs = sorted(cme_cells_dir.glob("*.csv"))
        if cme_cell_csvs:
            click.echo(
                "\nWriting CME cell detections to GeoJSON files...\n"
            )
            write_geojsons(
                csvs=cme_cell_csvs,
                overlap=0,
                results_dir=results_dir,
                output_dir=Path("cme-outputs-geojson") / "cells",
                prefix="cme",
                num_workers=num_workers,
                object_type="detection",
                set_classification=True,
                annotation_shape="box",
            )

    # --- GeoJSON: annotation-level CME regions -------------------------------
    cme_cmes_dir = Path(str(results_dir)) / "cme-outputs-csv" / "cmes"
    if cme_cmes_dir.exists():
        cme_cme_csvs = sorted(cme_cmes_dir.glob("*.csv"))
        if cme_cme_csvs:
            click.echo(
                "\nWriting CME annotation regions to GeoJSON files...\n"
            )
            write_geojsons(
                csvs=cme_cme_csvs,
                overlap=0,
                results_dir=results_dir,
                output_dir=Path("cme-outputs-geojson") / "cmes",
                prefix="cme",
                num_workers=num_workers,
                object_type="annotation",
                set_classification=True,
                annotation_shape="polygon",
            )

    write_runtime_metadata(
        results_dir,
        "cme",
        params=click.get_current_context().params,
    )

    click.secho("\nCME analysis completed.\n", fg="green")


# ---------------------------------------------------------------------------
# wsinsight cme-profile
# ---------------------------------------------------------------------------

@click.command(name="cme-profile")
@click.option(
    "-o",
    "--results-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help="Results directory containing cme-outputs-csv/cells/ from `wsinsight cme`.",
)
@click.option(
    "--top-genes",
    default=10,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of top enriched marker genes to report per CME (if expr_ columns exist).",
)
@click.option(
    "--top-types",
    default=5,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of top cell types to summarise per CME.",
)
def cme_profile_cmd(*, results_dir: URIPath, top_genes: int, top_types: int) -> None:
    """Summarise each CME's cell composition (and marker genes, if any) to help name niches."""
    # Imported lazily: cme_profile is torch-free, so this keeps the command
    # usable even without the deep-learning stack.
    from ..insightlib.cme_profile import cme_profile

    comp, markers = cme_profile(
        str(results_dir), top_genes=top_genes, top_types=top_types, write=True,
    )

    click.secho("\nCME composition (mean cell-type fractions):\n", fg="green")
    cols = [c for c in ("n_cells", "frac", "top_types") if c in comp.columns]
    with pd.option_context("display.max_colwidth", 80, "display.width", 200):
        click.echo(comp[cols].to_string())

    if markers is not None:
        click.secho("\nTop enriched marker genes per CME:\n", fg="green")
        for cme_id, grp in markers.groupby("cme", sort=False):
            top = ", ".join(f"{r.gene}({r.log2_enrichment:+.1f})" for r in grp.itertuples())
            click.echo(f"  {cme_id}: {top}")
    else:
        click.secho(
            "\n(No expr_ columns found; this is expected for whole-slide H&E "
            "cohorts. Composition fingerprints are sufficient to name niches.)\n",
            fg="yellow",
        )

    click.secho(
        f"\nWrote cme-profile-composition.csv (and markers, if any) to {results_dir}\n",
        fg="green",
    )
