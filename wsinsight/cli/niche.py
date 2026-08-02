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

from ..uri_path import URIPath, URIPathType
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
        "and um-per-pixel resolution; images are read only when --hoptimus "
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
    "--hoptimus",
    "niche_hoptimus",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Enable H-Optimus tissue morphology features in addition to k-hop "
        "cell-type composition.  Requires a GPU and the timm package."
    ),
)
@click.option(
    "--hoptimus-only",
    "niche_hoptimus_only",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Use only H-Optimus features for niche (skip k-hop composition features). "
        "Requires --hoptimus."
    ),
)
@click.option(
    "--hoptimus-pca-dim",
    "niche_hoptimus_pca_dim",
    default=None,
    show_default=False,
    type=click.IntRange(min=8),
    help=(
        "Optional PCA components used to reduce H-Optimus embeddings (1536-d) "
        "before concatenation with k-hop composition features. If omitted, "
        "raw H-Optimus features are used (no PCA). Ignored unless --hoptimus "
        "is set."
    ),
)
@click.option(
    "--hoptimus-model-dir",
    "niche_hoptimus_model_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help=(
        "Path to a local directory containing H-Optimus model weights "
        "(pytorch_model.bin or model.safetensors). When omitted, weights are "
        "downloaded from HuggingFace Hub. Ignored unless --hoptimus is set."
    ),
)
@click.option(
    "--hoptimus-batch-size",
    "niche_hoptimus_batch_size",
    default=None,
    type=click.IntRange(min=1),
    help=(
        "Number of image patches per H-Optimus forward pass. When omitted "
        "(recommended), the batch size is auto-calibrated from available GPU "
        "memory at runtime and adapts via binary search if OOM occurs. Set "
        "explicitly only to override the automatic sizing (e.g. to cap memory "
        "use when sharing the GPU with other workloads). Ignored unless "
        "--hoptimus is set."
    ),
)
@click.option(
    "--clusters",
    "niche_clusters",
    default=None,
    type=click.IntRange(min=2),
    help=(
        "Number of microenvironment clusters (KMeans).  When omitted, the "
        "optimal number is determined automatically via a Leiden "
        "community-detection sweep."
    ),
)
@click.option(
    "--leiden-res",
    "niche_leiden_res",
    default=_DEFAULT_LEIDEN_RESOLUTIONS,
    show_default=True,
    type=FLOAT_LIST,
    help=(
        "Comma-separated Leiden resolutions to sweep when choosing the number "
        "of niches automatically (e.g. '0.2,0.5,1.0,2.0').  Higher resolutions "
        "yield more, smaller clusters.  Ignored when --clusters is set."
    ),
)
@click.option(
    "--embed-dim",
    "niche_embed_dim",
    default=32,
    show_default=True,
    type=click.IntRange(min=8, max=256),
    help=(
        "Dimensionality of the DGI cell embedding.  Higher values retain more "
        "detail but increase computation.  Typical range: 16-64."
    ),
)
@click.option(
    "--epochs",
    "epochs",
    default=300,
    show_default=True,
    type=click.IntRange(min=1),
    help=(
        "Upper bound on DGI encoder training epochs.  Early stopping is always "
        "active, so training may finish sooner (see --patience, "
        "--min-delta and --min-epochs)."
    ),
)
@click.option(
    "--patience",
    "early_stop_patience",
    default=20,
    show_default=True,
    type=click.IntRange(min=1),
    help=(
        "Early-stopping patience: stop after this many consecutive epochs "
        "without a mean-loss improvement greater than --min-delta."
    ),
)
@click.option(
    "--min-delta",
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
    "--min-epochs",
    "early_stop_min_epochs",
    default=50,
    show_default=True,
    type=click.IntRange(min=1),
    help=(
        "Never trigger early stopping before this many epochs have elapsed."
    ),
)
@click.option(
    "--amp",
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
    "--k-hops",
    "niche_k_hops",
    default=2,
    show_default=True,
    type=click.IntRange(min=0),
    help=(
        "Number of neighborhood hops for the composition features.  Each hop "
        "adds one ring of the Delaunay graph, so the spatial extent a niche "
        "summarises grows roughly as k-hops x the typical cell spacing.  "
        "Raising this also grows the feature dimension linearly and the "
        "breadth-first traversal cost superlinearly."
    ),
)
@click.option(
    "--max-edge-len-um",
    "niche_max_edge_len_um",
    default=25.0,
    show_default=True,
    type=click.FloatRange(min=0),
    help=(
        "Maximal Delaunay edge length (um) when building the cell graph.  "
        "Longer edges are pruned, so this caps how far a single hop can reach "
        "and stops sparse tissue from being wired into spurious neighbourhoods."
    ),
)
@click.option(
    "--max-cell-radius-um",
    "niche_max_cell_radius_um",
    default=15.0,
    show_default=True,
    type=click.FloatRange(min=0),
    help=(
        "Maximal cell radius (um) used when merging annotation-level regions."
    ),
)
@click.option(
    "--soft",
    "niche_soft",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Use soft (probability) composition features instead of hard argmax "
        "labels.  Soft mode keeps the classifier's uncertainty, which helps "
        "when many cells have ambiguous class calls."
    ),
)
@click.option(
    "--alpha",
    "niche_alpha",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0),
    help=(
        "Dirichlet/Laplace smoothing strength for the k-hop composition "
        "features: out = (p + alpha / n_classes) / (1 + alpha).  Larger values "
        "pull sparse neighbourhoods toward a uniform composition; 0 disables "
        "smoothing and allows exact zeros."
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
    "--seed",
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
    niche_hoptimus_only: bool = False,
    niche_hoptimus_pca_dim: int | None = None,
    niche_hoptimus_model_dir: Path | None = None,
    niche_hoptimus_batch_size: int | None = None,
    niche_clusters: int | None = None,
    niche_leiden_res: list[float] | None = None,
    niche_embed_dim: int = 32,
    niche_k_hops: int = 2,
    niche_max_edge_len_um: float = 25.0,
    niche_max_cell_radius_um: float = 15.0,
    niche_soft: bool = False,
    niche_alpha: float = 1.0,
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

    if niche_hoptimus_only and not niche_hoptimus:
        raise click.UsageError("--hoptimus-only requires --hoptimus.")

    try:
        from ..insightlib.niche_generation import niche_generation
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(
            "Failed to load niche runtime dependencies. "
            "Install torch/torch-geometric stack before running `wsinsight niche`."
        ) from exc

    niche_generation(
        wsi_dir=wsi_dir,
        wsi_paths=slide_paths,
        results_dir=results_dir,
        max_edge_len_um=niche_max_edge_len_um,
        max_cell_radius_um=niche_max_cell_radius_um,
        k_hops=niche_k_hops,
        alpha=niche_alpha,
        use_hoptimus=niche_hoptimus,
        hoptimus_only=niche_hoptimus_only,
        hoptimus_model_dir=niche_hoptimus_model_dir,
        hoptimus_batch_size=niche_hoptimus_batch_size,
        pca_dim=niche_hoptimus_pca_dim,
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
        niche_soft_mode=niche_soft,
        overwrite=overwrite,
        amp=amp,
        seed=seed,
    )

    # --- GeoJSON: cell-level detections with niche labels ----------------------
    if export_geojson:
        from ..write_geojson import write_geojsons

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
