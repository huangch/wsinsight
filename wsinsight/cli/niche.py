"""Standalone CLI for niche analysis.

``wsinsight niche`` discovers recurring niches across a cohort
of whole slide images.  It builds per-slide Delaunay graphs, trains a global DGI
encoder, clusters the resulting embeddings, and writes per-cell niche labels plus
optional annotation-level region merges.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import pandas as pd

from ..insightlib.niche_generation import niche_generation
from ..uri_path import URIPath, URIPathType
from ..write_geojson import write_geojsons
from ._meta import write_runtime_metadata
from ._paths import (
    default_storage_kwargs,
    ensure_input_directory,
)

_STORAGE_KWARGS = default_storage_kwargs()

_DEFAULT_LEIDEN_RESOLUTIONS = "0.5,1.0,2.0"


class _FloatListParamType(click.ParamType):
    """Comma-separated list of positive floats (Leiden resolutions)."""

    name = "floats"

    def convert(self, value, param, ctx):
        if isinstance(value, (list, tuple)):
            return list(value)
        try:
            out = [float(tok) for tok in str(value).split(",") if tok.strip()]
        except ValueError:
            self.fail(
                f"{value!r} is not a comma-separated list of numbers "
                "(e.g. '0.5,1.0,2.0').",
                param, ctx,
            )
        if not out:
            self.fail("at least one resolution is required.", param, ctx)
        if any(r <= 0 for r in out):
            self.fail("all resolutions must be positive.", param, ctx)
        return out


FLOAT_LIST = _FloatListParamType()


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
        "and um-per-pixel resolution; images are read only when --niche-hoptimus "
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
    "--niche-hoptimus",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Enable H-Optimus tissue morphology features in addition to k-hop "
        "cell-type composition.  Requires a GPU and the timm package."
    ),
)
@click.option(
    "--niche-clusters",
    default=None,
    type=click.IntRange(min=2),
    help=(
        "Number of microenvironment clusters (KMeans).  When omitted, the "
        "optimal number is determined automatically via a Leiden "
        "community-detection sweep."
    ),
)
@click.option(
    "--niche-leiden-res",
    default=_DEFAULT_LEIDEN_RESOLUTIONS,
    show_default=True,
    type=FLOAT_LIST,
    help=(
        "Comma-separated Leiden resolutions to sweep when choosing the number "
        "of niches automatically (e.g. '0.2,0.5,1.0,2.0').  Higher resolutions "
        "yield more, smaller clusters.  Ignored when --niche-clusters is set."
    ),
)
@click.option(
    "--niche-embed-dim",
    default=32,
    show_default=True,
    type=click.IntRange(min=8, max=256),
    help=(
        "Dimensionality of the DGI cell embedding.  Higher values retain more "
        "detail but increase computation.  Typical range: 16-64."
    ),
)
@click.option(
    "--niche-epochs",
    "epochs",
    default=300,
    show_default=True,
    type=click.IntRange(min=1),
    help=(
        "Upper bound on DGI encoder training epochs.  Early stopping is always "
        "active, so training may finish sooner (see --niche-patience, "
        "--niche-min-delta and --niche-min-epochs)."
    ),
)
@click.option(
    "--niche-patience",
    "early_stop_patience",
    default=20,
    show_default=True,
    type=click.IntRange(min=1),
    help=(
        "Early-stopping patience: stop after this many consecutive epochs "
        "without a mean-loss improvement greater than --niche-min-delta."
    ),
)
@click.option(
    "--niche-min-delta",
    "early_stop_min_delta",
    default=1e-4,
    show_default=True,
    type=click.FloatRange(min=0),
    help=(
        "Minimum relative mean-loss improvement required to reset the "
        "early-stopping patience counter."
    ),
)
@click.option(
    "--niche-min-epochs",
    "early_stop_min_epochs",
    default=50,
    show_default=True,
    type=click.IntRange(min=1),
    help=(
        "Never trigger early stopping before this many epochs have elapsed."
    ),
)
@click.option(
    "--niche-amp",
    "amp",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Enable CUDA automatic mixed precision for DGI training (faster, lower "
        "GPU memory).  Off by default.  No effect on CPU/MPS.  Note: FP16 math "
        "changes results very slightly versus full FP32."
    ),
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="Delete cached checkpoints and recompute all niche outputs from scratch.",
)
@click.option(
    "--export-geojson",
    is_flag=True,
    default=False,
    show_default=True,
    help="Export niche results to GeoJSON files (niche-outputs-geojson/).",
)
@click.option(
    "--num-workers",
    default=min(_num_cpus(), 8),
    show_default=True,
    type=click.IntRange(min=0),
    help="Number of workers for GeoJSON export.",
)
@click.option(
    "--niche-seed",
    "seed",
    default=0,
    show_default=True,
    type=int,
    help=(
        "Random seed for the full niche pipeline (DGI encoder training, Leiden "
        "sweep and KMeans clustering), making the discovered niche ids "
        "reproducible across runs."
    ),
)
def niche(
    *,
    wsi_dir: URIPath,
    results_dir: URIPath,
    niche_hoptimus: bool = False,
    niche_clusters: int | None = None,
    niche_leiden_res: list[float] | None = None,
    niche_embed_dim: int = 32,
    epochs: int = 300,
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 1e-4,
    early_stop_min_epochs: int = 50,
    amp: bool = False,
    overwrite: bool = False,
    export_geojson: bool = False,
    num_workers: int = 8,
    seed: int = 0,
) -> None:
    """Discover niches across a cohort of slides.

    Builds per-slide Delaunay cell graphs, trains a global Deep Graph Infomax
    (DGI) encoder across all slides, clusters the resulting embeddings, and
    writes per-cell niche labels and annotation-level region merges.

    \b
    Outputs written to <results-dir>/:
      niche-outputs-csv/cells/<slide>.csv   per-cell niche labels + features
      niche-outputs-csv/niches/<slide>.csv    annotation-level merged niche regions
      niche-outputs-geojson/cells/          GeoJSON cell detections with niche labels
      niche-outputs-geojson/niches/           GeoJSON niche region annotations
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
        "\nRunning niche analysis.\n", fg="green"
    )

    niche_generation(
        wsi_dir=wsi_dir,
        wsi_paths=slide_paths,
        results_dir=results_dir,
        max_edge_len_um=25.0,
        max_cell_radius_um=15.0,
        k_hops=2,
        alpha=1.0,
        use_hoptimus=niche_hoptimus,
        hidden=64,
        out_dim=niche_embed_dim,
        epochs=epochs,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        early_stop_min_epochs=early_stop_min_epochs,
        niche_cellular=True,
        niche_annotation=True,
        niche_clustering_k=niche_clusters,
        niche_clustering_resolutions=niche_leiden_res,
        overwrite=overwrite,
        amp=amp,
        seed=seed,
    )

    # --- GeoJSON: cell-level detections with niche labels ----------------------
    if export_geojson:
        niche_cells_dir = Path(str(results_dir)) / "niche-outputs-csv" / "cells"
        if niche_cells_dir.exists():
            niche_cell_csvs = sorted(niche_cells_dir.glob("*.csv"))
            if niche_cell_csvs:
                click.echo(
                    "\nWriting niche cell detections to GeoJSON files...\n"
                )
                write_geojsons(
                    csvs=niche_cell_csvs,
                    overlap=0,
                    results_dir=results_dir,
                    output_dir=Path("niche-outputs-geojson") / "cells",
                    prefix="niche",
                    num_workers=num_workers,
                    object_type="detection",
                    set_classification=True,
                    annotation_shape="box",
                    overwrite=overwrite,
                )

        # --- GeoJSON: annotation-level niche regions ---------------------------
        niche_niches_dir = Path(str(results_dir)) / "niche-outputs-csv" / "niches"
        if niche_niches_dir.exists():
            niche_csvs = sorted(niche_niches_dir.glob("*.csv"))
            if niche_csvs:
                click.echo(
                    "\nWriting niche annotation regions to GeoJSON files...\n"
                )
                write_geojsons(
                    csvs=niche_csvs,
                    overlap=0,
                    results_dir=results_dir,
                    output_dir=Path("niche-outputs-geojson") / "niches",
                    prefix="niche",
                    num_workers=num_workers,
                    object_type="annotation",
                    set_classification=True,
                    annotation_shape="polygon",
                    overwrite=overwrite,
                )

    write_runtime_metadata(
        results_dir,
        "niche",
        params=click.get_current_context().params,
    )

    click.secho("\nniche analysis completed.\n", fg="green")


# ---------------------------------------------------------------------------
# wsinsight niche-profile
# ---------------------------------------------------------------------------

@click.command(name="niche-profile")
@click.option(
    "-o",
    "--results-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    required=True,
    help="Results directory containing niche-outputs-csv/cells/ from `wsinsight niche`.",
)
@click.option(
    "--top-genes",
    default=10,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of top enriched marker genes to report per niche (if expr_ columns exist).",
)
@click.option(
    "--top-types",
    default=5,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of top cell types to summarise per niche.",
)
def niche_profile_cmd(*, results_dir: URIPath, top_genes: int, top_types: int) -> None:
    """Summarise each niche's cell composition (and marker genes, if any) to help name niches."""
    # Imported lazily: niche_profile is torch-free, so this keeps the command
    # usable even without the deep-learning stack.
    from ..insightlib.niche_profile import niche_profile

    comp, markers = niche_profile(
        str(results_dir), top_genes=top_genes, top_types=top_types, write=True,
    )

    click.secho("\nniche composition (mean cell-type fractions):\n", fg="green")
    cols = [c for c in ("n_cells", "frac", "top_types") if c in comp.columns]
    with pd.option_context("display.max_colwidth", 80, "display.width", 200):
        click.echo(comp[cols].to_string())

    if markers is not None:
        click.secho("\nTop enriched marker genes per niche:\n", fg="green")
        for niche_id, grp in markers.groupby("niche", sort=False):
            top = ", ".join(f"{r.gene}({r.log2_enrichment:+.1f})" for r in grp.itertuples())
            click.echo(f"  {niche_id}: {top}")
    else:
        click.secho(
            "\n(No expr_ columns found; this is expected for whole-slide H&E "
            "cohorts. Composition fingerprints are sufficient to name niches.)\n",
            fg="yellow",
        )

    click.secho(
        f"\nWrote niche-profile-composition.csv (and markers, if any) to {results_dir}\n",
        fg="green",
    )

    write_runtime_metadata(
        results_dir,
        "niche-profile",
        params=click.get_current_context().params,
    )
