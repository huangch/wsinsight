"""Combined CLI entry point for WSInsight patch extraction and inference runs.

`wsinsight run` enumerates slides once, launches the patch stage, then funnels the
same arguments into the inference/export stage so users can process cohorts with a
single command.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, List

import click
import tqdm
from platformdirs import user_cache_dir

import wsinfer_zoo.client
from .infer import infer as infer_command
from .patch import patch as patch_command
from ..qupath import make_qupath_project
from ..uri_path import URIPath, URIPathType


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


def _cache_dir() -> Path | str:
    """Resolve the cache directory honoring optional cloud/remote overrides."""
    cache_env = os.getenv("WSINSIGHT_REMOTE_CACHE_DIR")
    if cache_env:
        return cache_env
    return Path(user_cache_dir(appname="wsinsight", appauthor=False))


def _enumerate_slide_paths(wsi_dir: URIPath) -> list[URIPath]:
    """List slide files once so patch + infer reuse the same ordering."""
    if not wsi_dir.exists():
        raise FileNotFoundError(f"Whole slide image directory not found: {wsi_dir}")

    slide_paths = sorted(
        [
            path
            for path in tqdm.tqdm(
                wsi_dir.iterdir(), desc="Count files in slide directory"
            )
            if path.is_file()
        ]
    )
    return slide_paths


_PATCH_PARAM_NAMES: tuple[str, ...] = (
    "wsi_dir",
    "slide_paths",
    "results_dir",
    "references_dir",
    "qupath_detection_dir",
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
)

_INFER_PARAM_NAMES: tuple[str, ...] = (
    "wsi_dir",
    "slide_paths",
    "results_dir",
    "references_dir",
    "qupath_detection_dir",
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
    # "speedup",
    "geojson",
    "omecsv",
    "patch_overlap_ratio",
    "patch_size_um",
    "patch_size_px",
    "hplot",
    "hplot_max_neighbor_distance",
    "hplot_base_types",
    "hplot_target_types",
    "hplot_k",
    "hplot_n",
    "hplot_r",
    "hplot_range_max",
    "hplot_range_min",
    "hplot_samples_with_valid_range_only",
    "cme_cellular",
    "cme_annotation",
    "cme_soft_mode",
    "cme_clustering_k",
    "cme_clustering_resolutions",
)


def _select_kwargs(values: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Subset a locals() dict to the parameters expected by a downstream command."""
    return {name: values[name] for name in keys}


@click.command()
@click.pass_context
@click.option(
    "-i",
    "--wsi-dir",
    type=(
        URIPathType(
            exists=True, 
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
            **json.loads(os.getenv("S3_STORAGE_OPTIONS", default=None)),
        )
        if os.getenv("S3_STORAGE_OPTIONS", default=None) is not None
        else URIPathType(
            exists=True,
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
        )
    ),               
    required=True,
    help="Directory containing whole slide images. This directory can *only* contain"
    " whole slide images.",
)
@click.option(
    "-o",
    "--results-dir",
    type=(
        URIPathType(
            exists=False, 
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
            **json.loads(os.getenv("S3_STORAGE_OPTIONS", default=None)),
        )
        if os.getenv("S3_STORAGE_OPTIONS", default=None) is not None
        else URIPathType(
            exists=False,
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
        )
    ),
    required=True,
    help="Directory to store results. If directory exists, will skip"
    " whole slides for which outputs exist.",
)
@click.option(
    "-r",
    "--references-dir",
    type=(
        URIPathType(
            exists=True, 
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
            **json.loads(os.getenv("S3_STORAGE_OPTIONS", default=None)),
        )
        if os.getenv("S3_STORAGE_OPTIONS", default=None) is not None
        else URIPathType(
            exists=True,
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
        )
    ),
    default=None,
    help="Directory containing outputs from a prior wsinsight run (e.g., model-outputs-* folders) used as reference results.",
)
@click.option(
    "--qupath-detection-dir",
    type=(
        URIPathType(
            exists=True, 
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
            **json.loads(os.getenv("S3_STORAGE_OPTIONS", default=None)),
        )
        if os.getenv("S3_STORAGE_OPTIONS", default=None) is not None
        else URIPathType(
            exists=True,
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
        )
    ),
    default=None,
    help="Directory containing detection files generated by QuPath."
    " The classification in the detection files will be used.",
)
@click.option(
    "--qupath-geojson-detection-dir",
    type=(
        URIPathType(
            exists=True,
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
            **json.loads(os.getenv("S3_STORAGE_OPTIONS", default=None)),
        )
        if os.getenv("S3_STORAGE_OPTIONS", default=None) is not None
        else URIPathType(
            exists=True,
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
        )
    ),
    default=None,
    help="Directory containing geojson files generated by QuPath."
    " The detection in the geojson files will be used.",
)
@click.option(
    "--qupath-geojson-annotation-dir",
    type=(
        URIPathType(
            exists=True,
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
            **json.loads(os.getenv("S3_STORAGE_OPTIONS", default=None)),
        )
        if os.getenv("S3_STORAGE_OPTIONS", default=None) is not None
        else URIPathType(
            exists=True,
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
        )
    ),
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
        registry_file=Path(os.getenv("WSINFER_ZOO_REGISTRY_PATH", default=None)) \
            if os.getenv("WSINFER_ZOO_REGISTRY_PATH", default=None) is not None and Path(os.getenv("WSINFER_ZOO_REGISTRY_PATH", default=None)).exists() \
            else Path(wsinfer_zoo.client.WSINFER_ZOO_REGISTRY_DEFAULT_PATH) \
            if Path(wsinfer_zoo.client.WSINFER_ZOO_REGISTRY_DEFAULT_PATH).exists() \
            else None
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
@click.option(
    "--geojson",
    is_flag=True,
    default=False,
    show_default=True,
    help="Create a GeoJSON directory containing the inference results",
)
@click.option(
    "--omecsv",
    is_flag=True,
    default=False,
    show_default=True,
    help="Create a OMECSV directory containing the inference results",
)
# Options for segmentation.
@click.option(
    "--histoqc-dir",
    type=(
        URIPathType(
            exists=True, 
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
            **json.loads(os.getenv("S3_STORAGE_OPTIONS", default=None)),
        )
        if os.getenv("S3_STORAGE_OPTIONS", default=None) is not None
        else URIPathType(
            exists=True,
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
        )
    ),
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
    help="Target cell type cell type list for computing layer-wise proportion, e.g., lymphocytes.",
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
    "--cme-cellular",
    is_flag=True,
    default=False,
    show_default=True,
    help="Run cellular-level cme analysis.",
)
@click.option(
    "--cme-annotation",
    is_flag=True,
    default=False,
    show_default=True,
    help="Run cellular-level cme analysis.",
)
@click.option(
    "--cme-soft-mode",
    is_flag=True,
    default=False,
    show_default=True,
    help="CME clustering is computed based on the probability of classification."
         "Otherwise, the categorization of classification.",
)
@click.option(
    "--cme-clustering-k",
    default=None,
    type=click.IntRange(min=0),
    help="the n-neighbors parameter using in clustering for cmes.",
)
@click.option(
    "--cme-clustering-resolutions",
    callback=_csv_to_list,
    default="0.5,1.0,2.0",
    help="Resolution parameter using in clustering for cmes.",
)
def run(
    ctx: click.Context,
    *,
    wsi_dir: URIPath,
    results_dir: URIPath,
    references_dir: URIPath | None,
    qupath_detection_dir: URIPath | None,
    qupath_geojson_detection_dir: URIPath | None,
    qupath_geojson_annotation_dir: URIPath | None,
    qupath_detection_patch_size: int,
    qupath_spacing_um_px: float,
    qupath_annotation_patch_size: int,
    qupath_name_as_class: bool,
    model_name: str | None,
    config: Path | None,
    model_path: Path | None,
    batch_size: int,
    num_workers: int = 4,
    # speedup: bool = False,
    cache_image_patches: bool = False,
    qupath: bool = False,
    geojson: bool = False,
    omecsv: bool = False,
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
    cme_cellular: bool = False,
    cme_annotation: bool = False,
    cme_soft_mode: bool = False,
    cme_clustering_k: int | None = None,
    cme_clustering_resolutions: List | None = None,
) -> None:
    """Run both patch extraction and inference workflows for a slide directory.

    The command enumerates slides once, caches the list, and feeds identical
    arguments into the standalone `patch` and `infer` commands. Optional QuPath
    project generation reuses the combined results directory.
    """

    params = locals().copy()
    params.pop("ctx", None)

    params["slide_paths"] = _enumerate_slide_paths(wsi_dir)

    # Stage 1: segmentation + patch extraction.
    ctx.invoke(patch_command, **_select_kwargs(params, _PATCH_PARAM_NAMES))

    # Stage 2: inference + downstream analytics/exports.
    ctx.invoke(infer_command, **_select_kwargs(params, _INFER_PARAM_NAMES))

    if qupath:
        click.echo("Creating QuPath project with results")
        make_qupath_project(wsi_dir, results_dir)

    click.secho("\nWSInsight run completed.\n", fg="green")
