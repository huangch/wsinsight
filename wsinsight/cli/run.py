"""Combined CLI entry point for WSInsight patch extraction and inference runs.

`wsinsight run` enumerates slides once, launches the patch stage, then funnels the
same arguments into the inference/export stage so users can process cohorts with a
single command.

**Completeness guarantee**: On rerun, if any requested slide is missing a patch or
inference output, the missing artifact is automatically recovered. Progress counts
reflect the requested work, not just existing files.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import click
import tqdm

import wsinfer_zoo.client
from ..modellib.models import resolve_zoo_registry_path
from .infer import infer as infer_command
from .cme import cme as cme_command
from .ecomp import ecomp as ecomp_command
from .hplot import hplot as hplot_command
from .ncomp import ncomp as ncomp_command
from .patch import patch as patch_command
from .tcomp import tcomp as tcomp_command
from ..export_helpers import build_export_csvs
from ..qupath import make_qupath_project
from ..cancel import raise_if_cancelled
from ..uri_path import URIPath, URIPathType
from ..write_geojson import write_geojsons
from ..write_h5ad import write_h5ads
from ..write_omecsv import write_omecsvs
from ._paths import default_storage_kwargs

_STORAGE_KWARGS = default_storage_kwargs()


# --- Completeness reconciliation helpers -------------------------------------


@dataclass
class SlideStatus:
    """Tracks per-slide completion status across pipeline stages."""

    requested: set[str] = field(default_factory=set)
    existing_patches: set[str] = field(default_factory=set)
    existing_csvs: set[str] = field(default_factory=set)
    patched_this_run: set[str] = field(default_factory=set)
    inferred_this_run: set[str] = field(default_factory=set)

    @property
    def missing_patches(self) -> set[str]:
        return self.requested - self.existing_patches

    @property
    def needs_inference(self) -> set[str]:
        """Slides that have patches but no CSV."""
        return self.existing_patches - self.existing_csvs

    @property
    def completed(self) -> set[str]:
        """Slides that have both patch and CSV."""
        return self.existing_patches & self.existing_csvs

    @property
    def failed_patch(self) -> set[str]:
        """Slides we tried to patch but still have no patch file."""
        return self.patched_this_run - self.existing_patches

    @property
    def failed_infer(self) -> set[str]:
        """Slides we tried to infer but still have no CSV."""
        return self.inferred_this_run - self.existing_csvs


def _scan_existing_artifacts(
    slide_paths: list[URIPath],
    results_dir: URIPath,
) -> SlideStatus:
    """Scan results_dir and determine per-slide completion status.

    Returns a SlideStatus with requested stems and existing patch/csv stems.
    """
    status = SlideStatus()
    status.requested = {p.stem for p in slide_paths}

    patches_dir = results_dir / "patches"
    if patches_dir.exists():
        status.existing_patches = {
            p.stem for p in patches_dir.iterdir(files_only=True)
            if p.suffix.lower() == ".h5"
        }

    csv_dir = results_dir / "model-outputs-csv"
    if csv_dir.exists():
        status.existing_csvs = {
            p.stem for p in csv_dir.iterdir(files_only=True)
            if p.suffix.lower() == ".csv"
        }

    return status


def _log_reconciliation_summary(status: SlideStatus, stage: str = "final") -> None:
    """Log a clear summary of pipeline completeness."""
    n_requested = len(status.requested)
    n_completed = len(status.completed)
    missing_patch = status.missing_patches
    missing_csv = status.needs_inference

    if stage == "pre-patch":
        click.echo("\n" + "=" * 60)
        click.secho("Pipeline Status (before patching)", fg="cyan", bold=True)
        click.echo("=" * 60)
        click.echo(f"  Requested slides:    {n_requested}")
        click.echo(f"  Existing patches:    {len(status.existing_patches)}")
        click.echo(f"  Existing CSVs:       {len(status.existing_csvs)}")
        if missing_patch:
            click.secho(f"  Need patching:       {len(missing_patch)}", fg="yellow")
        if missing_csv:
            click.echo(f"  Need inference:      {len(missing_csv)}")
        click.echo("=" * 60 + "\n")

    elif stage == "post-patch":
        if status.patched_this_run:
            click.echo(f"\n  Patched this run: {len(status.patched_this_run)}")
        failed = status.failed_patch
        if failed:
            click.secho(f"\n  WARNING: {len(failed)} slide(s) failed patching:", fg="red")
            for stem in sorted(failed)[:10]:
                click.echo(f"    - {stem}")
            if len(failed) > 10:
                click.echo(f"    ... and {len(failed) - 10} more")

    elif stage == "post-infer":
        if status.inferred_this_run:
            click.echo(f"\n  Inferred this run: {len(status.inferred_this_run)}")
        failed = status.failed_infer
        if failed:
            click.secho(f"\n  WARNING: {len(failed)} slide(s) failed inference:", fg="red")
            for stem in sorted(failed)[:10]:
                click.echo(f"    - {stem}")
            if len(failed) > 10:
                click.echo(f"    ... and {len(failed) - 10} more")

    elif stage == "final":
        click.echo("\n" + "=" * 60)
        click.secho("Pipeline Summary", fg="cyan", bold=True)
        click.echo("=" * 60)
        click.echo(f"  Requested slides:    {n_requested}")
        click.echo(f"  Completed (patch+csv): {n_completed}")

        if status.patched_this_run:
            click.echo(f"  Patched this run:    {len(status.patched_this_run)}")
        if status.inferred_this_run:
            click.echo(f"  Inferred this run:   {len(status.inferred_this_run)}")

        all_failed = status.missing_patches | (status.existing_patches - status.existing_csvs)
        if all_failed:
            click.secho(f"  Still incomplete:    {len(all_failed)}", fg="red")
            for stem in sorted(all_failed)[:10]:
                has_patch = stem in status.existing_patches
                click.echo(f"    - {stem} (patch={'yes' if has_patch else 'NO'}, csv=NO)")
            if len(all_failed) > 10:
                click.echo(f"    ... and {len(all_failed) - 10} more")
        else:
            click.secho("  All slides completed successfully!", fg="green")
        click.echo("=" * 60 + "\n")


# --- System helpers ----------------------------------------------------------


def _num_cpus() -> int:
    """Get number of CPUs on the system."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - platform dependent
        return os.cpu_count() or 0


def _coerce_number(token: str):
    """Interpret comma-delimited CLI fragments as ints/floats when possible."""
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


def _csv_to_list(_: click.Context, __: click.Parameter, value: Any) -> list[Any]:
    """Parse CLI comma/space separated tokens and coerce numeric substrings."""
    if value is None:
        return []
    if isinstance(value, list):
        tokens = value
    else:
        tokens = [x for x in re.split(r"[,\s]+", str(value).strip()) if x]
    return [_coerce_number(str(x)) for x in tokens]


def _enumerate_slide_paths(wsi_dir: URIPath) -> list[URIPath]:
    """List slide files once so patch + infer reuse the same ordering."""
    wsi_dir = wsi_dir.coerce_image_list()
    if not wsi_dir.exists():
        raise FileNotFoundError(f"Whole slide image directory not found: {wsi_dir}")

    slide_paths = sorted(
        [
            path
            for path in tqdm.tqdm(
                wsi_dir.iterdir(), desc="Count files in slide directory"
            )
            if wsi_dir.scheme == "image-list" or path.is_file()
        ]
    )
    return slide_paths


_PATCH_PARAM_NAMES: tuple[str, ...] = (
    "wsi_dir",
    "slide_paths",
    "results_dir",
    "region_inference_dir",
    "qupath_measurement_detection_dir",
    "qupath_geojson_detection_dir",
    "qupath_geojson_annotation_dir",
    "qupath_detection_patch_size",
    "qupath_spacing_um_px",
    "qupath_annotation_patch_size",
    "qupath_name_as_class",
    "model_name",
    "config",
    "model_path",
    "cache_image_patches",
    "histoqc_dir",
    "seg_thumbsize",
    "seg_median_filter_size",
    "seg_binary_threshold",
    "seg_closing_kernel_size",
    "seg_min_object_size_um2",
    "seg_min_hole_size_um2",
    "patch_overlap_ratio",
    "patch_size_um",
    "patch_size_px",
    "spacing_um_px",
    "overwrite",
)

_INFER_PARAM_NAMES: tuple[str, ...] = (
    "wsi_dir",
    "slide_paths",
    "results_dir",
    "region_inference_dir",
    "qupath_measurement_detection_dir",
    "qupath_geojson_detection_dir",
    "qupath_geojson_annotation_dir",
    "qupath_detection_patch_size",
    "qupath_spacing_um_px",
    "qupath_annotation_patch_size",
    "qupath_name_as_class",
    "model_name",
    "config",
    "model_path",
    "batch_size",
    "num_workers",
    "pin_memory",
    # "speedup",
    "patch_overlap_ratio",
    "patch_size_um",
    "patch_size_px",
    "overwrite",
)

_HPLOT_PARAM_NAMES: tuple[str, ...] = (
    "wsi_dir",
    "results_dir",
    "hplot_max_neighbor_distance",
    "hplot_base_types",
    "hplot_target_types",
    "hplot_k",
    "hplot_n",
    "hplot_r",
    "hplot_range_max",
    "hplot_range_min",
    "hplot_samples_with_valid_range_only",
    "overwrite",
    "num_workers",
)

_NCOMP_PARAM_NAMES: tuple[str, ...] = (
    "wsi_dir",
    "results_dir",
    "ncomp_max_neighbor_distance",
    "ncomp_k",
    "overwrite",
    "num_workers",
)

_ECOMP_PARAM_NAMES: tuple[str, ...] = (
    "wsi_dir",
    "results_dir",
    "ecomp_max_edge",
    "ecomp_k",
    "overwrite",
    "num_workers",
)

_TCOMP_PARAM_NAMES: tuple[str, ...] = (
    "wsi_dir",
    "results_dir",
    "tcomp_max_edge",
    "tcomp_k",
    "overwrite",
    "num_workers",
)

_CME_PARAM_NAMES: tuple[str, ...] = (
    "wsi_dir",
    "results_dir",
    "cme_hoptimus",
    "cme_clusters",
    "overwrite",
    "export_geojson",
    "num_workers",
)


def _select_kwargs(values: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Subset a locals() dict to the parameters expected by a downstream command."""
    return {name: values[name] for name in keys}


@click.command()
@click.pass_context
@click.option(
    "-i",
    "--wsi-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),               
    required=True,
    help="Directory containing whole slide images, or an image-list:///path/to/filelist.txt"
    " URI pointing to a text file with one slide path per line (blank lines and # comments ignored).",
)
@click.option(
    "-o",
    "--results-dir",
    type=URIPathType(exists=False, **_STORAGE_KWARGS),
    required=True,
    help="Directory to store results. If directory exists, will skip"
    " whole slides for which outputs exist.",
)
@click.option(
    "-r",
    "--region-inference-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    default=None,
    help="Results directory from a prior region-level (patch-based) wsinsight run containing a model-outputs-csv/ folder. Requires --object-based: each detected object is matched to its enclosing region and the region's class probabilities are added as region_prob_* columns in the output.",
)
@click.option(
    "--qupath-measurement-detection-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    default=None,
    help="Directory of QuPath TSV detection-measurement exports (one <slide>.txt per slide"
    " containing 'Centroid X µm', 'Centroid Y µm', classification columns). Patches are"
    " extracted as fixed-size squares centered on each centroid. For GeoJSON detection"
    " exports, use --qupath-geojson-detection-dir instead.",
)
@click.option(
    "--qupath-geojson-detection-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    default=None,
    help="Directory containing geojson files generated by QuPath."
    " The detection in the geojson files will be used.",
)
@click.option(
    "--qupath-geojson-annotation-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    default=None,
    help="Directory containing QuPath annotation geojson files; their region labels seed the pseudo-model.",
)
@click.option(
    "--qupath-detection-patch-size",
    default=56,
    type=click.IntRange(min=1),
    help="The patch size of the pseudo model generated using QuPath detection files.",
)
@click.option(
    "--qupath-annotation-patch-size",
    default=224,
    type=click.IntRange(min=1),
    help="The patch size of the pseudo model generated using QuPath annotation files.",
)
@click.option(
    "--qupath-spacing-um-px",
    default=0.5,
    type=click.FloatRange(min=0),
    help="The spacing um/px of the pseudo model generated using QuPath annotation/detection files.",
)
@click.option(
    "--qupath-name-as-class",
    is_flag=True,
    default=False,
    show_default=True,
    help="When operating QuPath geojson/detection data, using name as class.",
)
@click.option(
    "-m",
    "--model",
    "model_name",
    type=click.Choice(sorted(wsinfer_zoo.client.load_registry(
        registry_file=resolve_zoo_registry_path(),
    ).models.keys())),
    help="Name of the model to use from WSInsight Model Zoo. Mutually exclusive with"
    " --config.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to configuration for the trained model. Use this option if the"
        " model weights are not registered in wsinsight. Mutually exclusive with"
        "--model"
    ),
)
@click.option(
    "-p",
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Path to the pretrained model. Use only when --config is passed. Mutually "
        "exclusive with --model."
    ),
)
@click.option(
    "-z",
    "--zoo-model-dir",
    "zoo_model_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help=(
        "Path to a folder containing config.json and torchscript_model.pt. "
        "Shorthand for --config + --model-path. Mutually exclusive with --model, "
        "--config, and --model-path."
    ),
)
@click.option(
    "-b",
    "--batch-size",
    type=click.IntRange(min=1),
    default=32,
    show_default=True,
    help="Batch size during model inference. If using multiple GPUs, increase the"
    " batch size.",
)
@click.option(
    "-n",
    "--num-workers",
    default=min(_num_cpus(), 8),  # Use at most 8 workers by default.
    show_default=True,
    type=click.IntRange(min=0),
    help="Number of workers to use for data loading during model inference (n=0 for"
    " single thread). Set this to the number of cores on your machine or lower.",
)
@click.option(
    "--pin-memory/--no-pin-memory",
    default=True,
    show_default=True,
    help=(
        "Pin DataLoader tensors to CUDA memory for faster host-to-device transfer.  "
        "Disable (--no-pin-memory) in memory-constrained environments where DataLoader "
        "workers are being killed by the system OOM killer."
    ),
)
# @click.option(
#     "--speedup/--no-speedup",
#     default=False,
#     show_default=True,
#     help="JIT-compile the model and apply inference optimizations. This imposes a"
#     " startup cost but may improve performance overall.",
# )
@click.option(
    "--cache-image-patches",
    is_flag=True,
    default=False,
    show_default=True,
    help="Extract image patches and save to hdf5.",
)
@click.option(
    "--qupath",
    is_flag=True,
    default=False,
    show_default=True,
    help="Create a QuPath project containing the inference results",
)
# Options for segmentation.
@click.option(
    "--histoqc-dir",
    type=URIPathType(exists=True, **_STORAGE_KWARGS),
    help="Directory containing histoqc outcomes.",
)
@click.option(
    "--seg-thumbsize",
    default=(2048, 2048),
    type=(int, int),
    help="The size of the slide thumbnail (in pixels) used for tissue segmentation."
    " The aspect ratio is preserved, and the longest side will have length"
    " max(thumbsize).",
)
@click.option(
    "--seg-median-filter-size",
    default=7,
    type=click.IntRange(min=3),
    help="The kernel size for median filtering. Must be greater than 1 and odd.",
)
@click.option(
    "--seg-binary-threshold",
    default=7,
    type=click.IntRange(min=1),
    help="The threshold for image binarization.",
)
@click.option(
    "--seg-closing-kernel-size",
    default=6,
    type=click.IntRange(min=1),
    help="The kernel size for binary closing (morphological operation).",
)
@click.option(
    "--seg-min-object-size-um2",
    default=200**2,
    type=click.FloatRange(min=0),
    help="The minimum size of an object to keep during tissue detection. If a"
    " contiguous object is smaller than this area, it replaced with background."
    " The default is 200um x 200um. The units of this argument are microns squared.",
)
@click.option(
    "--seg-min-hole-size-um2",
    default=190**2,
    type=click.FloatRange(min=0),
    help="The minimum size of a hole to keep as a hole. If a hole is smaller than this"
    " area, it is filled with foreground. The default is 190um x 190um. The units of"
    " this argument are microns squared.",
)
@click.option(
    "--patch-overlap-ratio",
    default=0.0,
    type=click.FloatRange(min=None, max=1, max_open=True),
    help="The ratio of overlap among patches. The default value of 0 produces"
    " non-overlapping patches. A value in (0, 1) will produce overlapping patches."
    " Negative values will add space between patches. A value of -1 would skip"
    " every other patch. A value of 0.5 will provide 50%% of overlap between patches."
    " Values must be in (-inf, 1).",
)
@click.option(
    "--patch-size-um",
    default=0.0,
    type=click.FloatRange(min=0.0),
    help="The size of patch in um. The default value of 0 produces"
    " full patch size of the chosen model.",
)
@click.option(
    "--patch-size-px",
    default=0,
    type=click.FloatRange(min=0),
    help="The size of patch in pixel. The default value of 0 produces"
    " full patch size of the chosen model.",
)
@click.option(
    "--spacing-um-px",
    default=0.0,
    type=click.FloatRange(min=0.0),
    help="Fallback slide resolution in micrometres-per-pixel (MPP), used ONLY for"
    " slides whose MPP cannot be read from the WSI metadata. Slide metadata is"
    " always preferred. The default of 0 disables the fallback.",
)
# @click.option(
#     "--patch-overlap-median-filter-size",
#     default=3,
#     type=click.IntRange(min=3),
#     help="The kernel size for median filtering when patch overlapping. Must be greater than 1 and odd.",
# )
# @click.option(
#     "--red-threshold",
#     default=0,
#     type=click.IntRange(min=0, max=255),
#     help="The threshold for the red channel of the image. If the red channel is greater"
#     " than this value, the pixel is considered to be tissue. (Default: 0 no filter) (Range: 0-255)",
# )
@click.option(
    "--hplot",
    is_flag=True,
    default=False,
    show_default=True,
    help="Run H-Plot analysis.",
)
@click.option(
    "--hplot-max-neighbor-distance",
    default=25.0,
    type=click.FloatRange(min=0),
    help="A parameter of float value determining maximal distance (um) to a neighboring cell.",
)
@click.option(
    "--hplot-base-types",
    callback=_csv_to_list,
    default=None,
    help="Base cell type or cell type list that form(s) the cell cluster(s), e.g., tumor cells.",
)
@click.option(
    "--hplot-target-types",
    callback=_csv_to_list,
    default=None,
    help="Target cell type or cell type list whose layer-wise proportion is computed, e.g., lymphocytes.",
)
@click.option(
    "--hplot-k",
    default=2,
    type=click.IntRange(min=0),
    help="The maximal edge distance for defining the neighborhood of a cell.",
)
@click.option(
    "--hplot-n",
    default=8,
    type=click.IntRange(min=0),
    help="The minimal neighborhood size for a cell to be computed for determining tumor regions.",
)
@click.option(
    "--hplot-r",
    default=0.5,
    type=click.FloatRange(min=0, max=1),
    help="The minimal ratio of tumor cells in the neighborhood of a cell, determining "
        "is this cell included in a tumor region.",
)
@click.option(
    "--hplot-range-max",
    default=None,
    type=click.IntRange(min=1),
    help="The maximal layer index toward OUTSIDE of tumors for the range window of H-Plot.",
)
@click.option(
    "--hplot-range-min",
    default=None,
    type=click.IntRange(max=0),
    help="The minimal layer index toward INSIDE of tumors for the range window of H-Plot.",
)
@click.option(
    "--hplot-samples-with-valid-range-only",
    is_flag=True,
    default=False,
    show_default=True,
    help="H-Plot computing uses only samples with valid range of cellular-wise layers.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    show_default=True,
    help="Overwrite existing outputs in all stages instead of skipping slides that already have results.",
)
@click.option(
    "--ncomp",
    is_flag=True,
    default=False,
    show_default=True,
    help="Run neighborhood composition (ncomp) analysis after inference.",
)
@click.option(
    "--ncomp-max-neighbor-distance",
    default=25.0,
    type=click.FloatRange(min=0),
    help="Maximum distance (µm) between neighboring cells in the Delaunay graph for ncomp.",
)
@click.option(
    "--ncomp-k",
    default=2,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of hops defining the ncomp neighborhood radius.",
)
@click.option(
    "--ecomp",
    is_flag=True,
    default=False,
    show_default=True,
    help="Run edge-level composition (ecomp) analysis after ncomp.",
)
@click.option(
    "--ecomp-max-edge",
    default=25.0,
    type=click.FloatRange(min=0),
    show_default=True,
    help="Maximum Delaunay edge length (µm) for ecomp; longer edges are pruned.",
)
@click.option(
    "--ecomp-k",
    default=2,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of hops defining the ecomp neighborhood radius (k-hop on the line graph).",
)
@click.option(
    "--tcomp",
    is_flag=True,
    default=False,
    show_default=True,
    help="Run triad-level composition (tcomp) analysis after ecomp.",
)
@click.option(
    "--tcomp-max-edge",
    default=25.0,
    type=click.FloatRange(min=0),
    show_default=True,
    help="Maximum Delaunay edge length (µm) for tcomp; triads with any longer edge are pruned.",
)
@click.option(
    "--tcomp-k",
    default=2,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of hops defining the tcomp neighborhood radius (k-hop on the dual graph).",
)
@click.option(
    "--cme",
    is_flag=True,
    default=False,
    show_default=True,
    help="Run cellular microenvironment (CME) analysis after inference.",
)
@click.option(
    "--cme-hoptimus",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Enable H-Optimus tissue morphology features for CME.  "
        "Requires a GPU and the timm package.  Ignored unless --cme is set."
    ),
)
@click.option(
    "--cme-clusters",
    default=None,
    type=click.IntRange(min=2),
    help=(
        "Number of CME clusters (KMeans).  When omitted, determined "
        "automatically via Leiden community detection.  Ignored unless --cme is set."
    ),
)
@click.option(
    "--export-geojson",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "After inference, merge all per-cell analytics and export to GeoJSON files "
        "(export-geojson/).  Equivalent to running 'wsinsight export --geojson'."
    ),
)
@click.option(
    "--export-omecsv",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "After inference, merge all per-cell analytics and export to compressed "
        "OME-CSV files (export-omecsv/).  Equivalent to running "
        "'wsinsight export --omecsv'."
    ),
)
@click.option(
    "--export-h5ad",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "After inference, merge all per-cell analytics and export to AnnData "
        ".h5ad files (export-h5ad/).  Equivalent to running "
        "'wsinsight export --h5ad'."
    ),
)
def run(
    ctx: click.Context,
    *,
    wsi_dir: URIPath,
    results_dir: URIPath,
    region_inference_dir: URIPath | None,
    qupath_measurement_detection_dir: URIPath | None,
    qupath_geojson_detection_dir: URIPath | None,
    qupath_geojson_annotation_dir: URIPath | None,
    qupath_detection_patch_size: int,
    qupath_spacing_um_px: float,
    qupath_annotation_patch_size: int,
    qupath_name_as_class: bool,
    model_name: str | None,
    config: Path | None,
    model_path: Path | None,
    zoo_model_dir: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    # speedup: bool = False,
    cache_image_patches: bool = False,
    qupath: bool = False,
    histoqc_dir: URIPath | None,
    seg_thumbsize: tuple[int, int],
    seg_median_filter_size: int,
    seg_binary_threshold: int,
    seg_closing_kernel_size: int,
    seg_min_object_size_um2: float,
    seg_min_hole_size_um2: float,
    patch_overlap_ratio: float = 0.0,
    patch_size_um: float = 0.0,
    patch_size_px: int = 0,
    spacing_um_px: float = 0.0,
    hplot: bool = False,
    hplot_max_neighbor_distance: float = 25.0,
    hplot_base_types: List | None = None,
    hplot_target_types: List | None = None,
    hplot_k: int = 2,
    hplot_n: int = 8,
    hplot_r: float = 0.5,
    hplot_range_max: int | None = None,
    hplot_range_min: int | None = None,
    hplot_samples_with_valid_range_only: bool = False,
    overwrite: bool = False,
    ncomp: bool = False,
    ncomp_max_neighbor_distance: float = 25.0,
    ncomp_k: int = 2,
    ecomp: bool = False,
    ecomp_max_edge: float = 25.0,
    ecomp_k: int = 2,
    tcomp: bool = False,
    tcomp_max_edge: float = 25.0,
    tcomp_k: int = 2,
    cme: bool = False,
    cme_hoptimus: bool = False,
    cme_clusters: int | None = None,
    export_geojson: bool = False,
    export_omecsv: bool = False,
    export_h5ad: bool = False,
) -> None:
    """Run both patch extraction and inference workflows for a slide directory.

    The command enumerates slides once, caches the list, and feeds identical
    arguments into the standalone `patch` and `infer` commands. Optional QuPath
    project generation reuses the combined results directory.
    """

    # --- Resolve --zoo-model-dir shorthand into --config + --model-path ------
    if zoo_model_dir is not None:
        if model_name is not None:
            raise click.UsageError(
                "--zoo-model-dir is mutually exclusive with --model."
            )
        if config is not None or model_path is not None:
            raise click.UsageError(
                "--zoo-model-dir is mutually exclusive with --config and --model-path."
            )
        config = zoo_model_dir / "config.json"
        model_path = zoo_model_dir / "torchscript_model.pt"
        if not config.exists():
            raise click.UsageError(
                f"--zoo-model-dir folder does not contain config.json: {zoo_model_dir}"
            )
        if not model_path.exists():
            raise click.UsageError(
                f"--zoo-model-dir folder does not contain torchscript_model.pt: {zoo_model_dir}"
            )

    # --- Coerce a plain txt file into an image-list:// virtual directory ----
    wsi_dir = wsi_dir.coerce_image_list()

    params = locals().copy()
    params.pop("ctx", None)

    slide_paths = _enumerate_slide_paths(wsi_dir)
    params["slide_paths"] = slide_paths

    # --- Preflight reconciliation: check existing artifacts ------------------
    status = _scan_existing_artifacts(slide_paths, results_dir)
    _log_reconciliation_summary(status, stage="pre-patch")

    # --- Stage 1: segmentation + patch extraction ----------------------------
    # Only patch slides that don't already have patches (unless overwrite)
    slides_needing_patch = status.missing_patches if not overwrite else status.requested
    if slides_needing_patch:
        click.secho(f"\nPatching {len(slides_needing_patch)} slide(s)...\n", fg="green")
        patch_params = params.copy()
        # Filter slide_paths to only those needing patches
        patch_params["slide_paths"] = [
            p for p in slide_paths if p.stem in slides_needing_patch
        ]
        ctx.invoke(patch_command, **_select_kwargs(patch_params, _PATCH_PARAM_NAMES))
        status.patched_this_run = slides_needing_patch.copy()
    else:
        click.echo("\nAll slides already have patches — skipping patch stage.\n")
    raise_if_cancelled()

    # --- Post-patch reconciliation: update status ----------------------------
    status_after_patch = _scan_existing_artifacts(slide_paths, results_dir)
    status.existing_patches = status_after_patch.existing_patches
    _log_reconciliation_summary(status, stage="post-patch")

    # --- Stage 2: inference --------------------------------------------------
    # Only infer slides that have patches but no CSV (unless overwrite)
    slides_needing_infer = status.needs_inference if not overwrite else status.existing_patches
    if slides_needing_infer:
        click.secho(f"\nRunning inference on {len(slides_needing_infer)} slide(s)...\n", fg="green")
        infer_params = params.copy()
        # Filter slide_paths to only those needing inference
        infer_params["slide_paths"] = [
            p for p in slide_paths if p.stem in slides_needing_infer
        ]
        ctx.invoke(infer_command, **_select_kwargs(infer_params, _INFER_PARAM_NAMES))
        status.inferred_this_run = slides_needing_infer.copy()
    else:
        click.echo("\nAll slides already have inference outputs — skipping inference stage.\n")
    raise_if_cancelled()

    # --- Post-inference reconciliation: update status ------------------------
    status_after_infer = _scan_existing_artifacts(slide_paths, results_dir)
    status.existing_csvs = status_after_infer.existing_csvs

    # --- Stage 3 (optional): H-Plot spatial analytics ------------------------
    if hplot:
        ctx.invoke(hplot_command, **_select_kwargs(params, _HPLOT_PARAM_NAMES))
        raise_if_cancelled()

    # --- Stage 4 (optional): node-level (cell) composition analytics ---------
    if ncomp:
        ctx.invoke(ncomp_command, **_select_kwargs(params, _NCOMP_PARAM_NAMES))
        raise_if_cancelled()

    # --- Stage 4b (optional): edge-level composition analytics ---------------
    if ecomp:
        ctx.invoke(ecomp_command, **_select_kwargs(params, _ECOMP_PARAM_NAMES))
        raise_if_cancelled()

    # --- Stage 4c (optional): triad-level composition analytics --------------
    if tcomp:
        ctx.invoke(tcomp_command, **_select_kwargs(params, _TCOMP_PARAM_NAMES))
        raise_if_cancelled()

    # --- Stage 5 (optional): cellular microenvironment (CME) analysis --------
    if cme:
        ctx.invoke(cme_command, **_select_kwargs(params, _CME_PARAM_NAMES))
        raise_if_cancelled()

    # --- Stage 6 (optional): merged export to GeoJSON / OME-CSV / h5ad -------
    if export_geojson or export_omecsv or export_h5ad:
        click.echo("\nMerging per-cell analytics into export CSVs...\n")
        build_export_csvs(results_dir, overwrite=True)

        export_dir = results_dir / "export-csv"
        export_candidates = list(export_dir.iterdir(files_only=True))
        export_csvs = [
            Path(p.materialize()) if isinstance(p, URIPath) else Path(p)
            for p in export_candidates
            if p.suffix == ".csv"
        ]

        if export_csvs:
            click.echo(f"  {len(export_csvs)} slide(s) ready for export.")
            num_export_workers = min(4, _num_cpus() or 4)

            if export_geojson:
                click.echo("\nWriting results to GeoJSON files...\n")
                write_geojsons(
                    csvs=export_csvs,
                    overlap=patch_overlap_ratio,
                    results_dir=results_dir,
                    output_dir="export-geojson",
                    prefix="prob",
                    num_workers=num_export_workers,
                    object_type="detection",
                    set_classification=True,
                    overwrite=True,
                )

            if export_omecsv:
                click.echo("\nWriting results to OME-CSV files...\n")
                h5s: list[Path] = []
                patches_dir = results_dir / "patches"
                if patches_dir.exists():
                    h5s = [
                        Path(p.materialize()) if isinstance(p, URIPath) else Path(p)
                        for p in patches_dir.iterdir(files_only=True)
                        if p.suffix == ".h5"
                    ]
                write_omecsvs(
                    csvs=export_csvs,
                    h5s=h5s,
                    overlap=patch_overlap_ratio,
                    results_dir=results_dir,
                    output_dir=URIPath("export-omecsv"),
                    prefix="prob",
                    num_workers=num_export_workers,
                    overwrite=True,
                )

            if export_h5ad:
                click.echo("\nWriting results to AnnData .h5ad files...\n")
                write_h5ads(
                    csvs=export_csvs,
                    results_dir=results_dir,
                    output_dir="export-h5ad",
                    prefix="prob",
                    object_type="detection",
                    overwrite=True,
                )

            click.echo("\nExport complete.")
        else:
            click.echo("\nNo export CSVs were produced — skipping export.")

    if qupath:
        click.echo("Creating QuPath project with results")
        make_qupath_project(wsi_dir, results_dir)

    # --- Final reconciliation summary ----------------------------------------
    _log_reconciliation_summary(status, stage="final")

    click.secho("\nWSInsight run completed.\n", fg="green")
