"""WSInsight inference CLI for running models and downstream analytics on WSIs."""

from __future__ import annotations

import dataclasses
import getpass
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List

import click
import geopandas as gpd
import pandas as pd
import tqdm
from platformdirs import user_cache_dir

import wsinfer_zoo.client
from wsinfer_zoo.client import HFModel, Model, ModelConfiguration

from .. import errors
from ..insightlib.cme_generation import cme_generation
from ..insightlib.hplot_generation import hplot_generation
from ..modellib import models
from ..modellib.run_inference import run_inference
# QuPath project export relies on optional dependencies; import remains disabled until re-enabled.
from ..uri_path import URIPath, URIPathType
from ..write_geojson import write_geojsons
from ..write_omecsv import write_omecsvs


# --- System inspection helpers -------------------------------------------------

def _num_cpus() -> int:
    """Get number of CPUs on the system."""
    try:
        return len(os.sched_getaffinity(0))
    # os.sched_getaffinity seems to be linux only.
    except AttributeError:
        count = os.cpu_count()  # potentially None
        return count or 0


def _num_gpus() -> int:
    """Best-effort GPU count for sizing dataloader workers."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


def _default_infer_workers_value() -> int:
    cpu = max(_num_cpus(), 0)
    gpu = _num_gpus()
    if gpu > 0 and cpu > 0:
        return max(0, min(cpu, gpu * 2))
    return min(cpu, 4) if cpu else 0


def _default_export_workers_value() -> int:
    """Reserve part of the CPU budget for inference, cap exports at 16 workers."""
    cpu = max(_num_cpus(), 1)
    if cpu <= 2:
        return 1
    reserve = max(2, cpu // 4)
    return max(1, min(cpu - reserve, 16))


def _default_stitch_workers_value() -> int:
    """Keep TileFuse CPU usage gentle by using half the cores, capped at eight."""
    cpu = max(_num_cpus(), 1)
    if cpu <= 2:
        return 1
    return max(1, min(8, cpu // 2))


DEFAULT_INFER_WORKERS = _default_infer_workers_value()
DEFAULT_EXPORT_WORKERS = _default_export_workers_value()
DEFAULT_STITCH_WORKERS = _default_stitch_workers_value()


def _inside_container() -> str:
    """Return a coarse container indicator (docker/singularity) for logging."""
    if Path("/.dockerenv").exists():
        return "yes, docker"
    elif (
        Path("/singularity").exists()
        or Path("/singularity.d").exists()
        or Path("/.singularity.d").exists()
    ):
        # Apptainer/Singularity may rename these paths in future releases.
        return "yes, apptainer/singularity"
    return "no"


def _get_timestamp() -> str:
    """Return a timezone-aware human-readable string for metadata files."""
    dt = datetime.now().astimezone()
    # Thu Aug 25 23:32:17 2022 EDT
    return dt.strftime("%c %Z")


def _print_system_info() -> None:
    """Print information about the system."""
    import torch
    import torchvision

    from .. import __version__

    click.secho(f"\nRunning WSInsight version {__version__}", fg="green")
    print("\nIf you run into issues, please submit a new issue at")
    print("https://github.com/huangch/wsinsight/issues/new")
    print("\nSystem information")
    print("------------------")
    print(f"Timestamp: {_get_timestamp()}")
    print(f"{platform.platform()}")
    try:
        print(f"User: {getpass.getuser()}")
    except KeyError:
        # If /etc/passwd does not include the username of the current user ID, a
        # KeyError is thrown. This could happen in a Docker image running as a different
        # user with `--user $(id -u):$(id -g)` but that does not bind mount the
        # /etc/passwd file.
        print("User: UNKNOWN")
    print(f"Hostname: {platform.node()}")
    print(f"Working directory: {os.getcwd()}")
    print(f"In container: {_inside_container()}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"  Torch version: {torch.__version__}")
    print(f"  Torchvision version: {torchvision.__version__}")
    cuda_is_available = torch.cuda.is_available()
    if cuda_is_available:
        click.secho("GPU available", fg="green")
        click.secho(f"  Using {torch.cuda.device_count()} GPU(s)", fg="green")
        cuda_ver = torch.version.cuda or "NOT FOUND"
        print(f"  CUDA version: {cuda_ver}")
    else:
        click.secho("GPU not available", bg="red", fg="black")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")
    print(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    if torch.version.cuda is None:
        click.secho("\n*******************************************", fg="yellow")
        click.secho("GPU WILL NOT BE USED", fg="yellow")
        if torch.version.cuda is None:
            click.secho("  CPU-only version of PyTorch is installed", fg="yellow")
        click.secho("*******************************************", fg="yellow")
    elif not cuda_is_available:
        click.secho("\n*******************************************", fg="yellow")
        click.secho("GPU WILL NOT BE USED", fg="yellow")
        if torch.version.cuda is None:
            click.secho("  CUDA DEVICES NOT AVAILABLE", fg="yellow")
        click.secho("*******************************************", fg="yellow")


def _get_info_for_save(
    model_obj: models.LocalModelTorchScript | HFModel,
) -> dict[str, Any]:
    """Get dictionary with information about the run. To save as JSON in output dir."""

    import torch

    from .. import __version__

    here = Path(__file__).parent.resolve()

    def get_git_info() -> dict[str, str | bool]:
        here = Path(__file__).parent.resolve()

        def get_stdout(args: list[str]) -> str:
            proc = subprocess.run(args, capture_output=True, cwd=here)
            return "" if proc.returncode != 0 else proc.stdout.decode().strip()

        git_remote = get_stdout("git config --get remote.origin.url".split())
        git_branch = get_stdout("git rev-parse --abbrev-ref HEAD".split())
        git_commit = get_stdout("git rev-parse HEAD".split())

        # https://stackoverflow.com/a/3879077/5666087
        cmd = subprocess.run("git diff-index --quiet HEAD --".split(), cwd=here)
        uncommitted_changes = cmd.returncode != 0
        return {
            "git_remote": git_remote,
            "git_branch": git_branch,
            "git_commit": git_commit,
            "uncommitted_changes": uncommitted_changes,
        }

    # Test if we are in a git repo. If we are, then get git info.
    git_program = shutil.which("git")
    git_installed = git_program is not None
    is_git_repo = False
    if git_installed:
        cmd = subprocess.run(
            [str(git_program), "branch"],
            cwd=here,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        is_git_repo = cmd.returncode == 0
    git_info = None
    if git_installed and is_git_repo:
        git_info = get_git_info()

    hf_info = None
    if hasattr(model_obj, "hf_info"):
        hf_info = dataclasses.asdict(model_obj.hf_info)

    return {
        "model": {
            "config": dataclasses.asdict(model_obj.config),
            "huggingface_location": hf_info,
            "path": str(model_obj.model_path),
        },
        "runtime": {
            "version": __version__,
            "working_dir": os.getcwd(),
            "args": " ".join(sys.argv),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "in_container": _inside_container(),
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "git": git_info,
            "wsinfer_zoo_version": wsinfer_zoo.__version__,
        },
        "timestamp": _get_timestamp(),
    }


# --- CLI input parsing helpers -------------------------------------------------
def _coerce_number(token: str):
    """Convert CLI tokens to ints/floats when possible, otherwise lowercase text."""
    t = token.strip()
    # Try parsing integers before falling back to floats/strings.
    if re.fullmatch(r'[+-]?\d+', t):
        try:
            return int(t)
        except ValueError:
            pass
    # Then attempt to parse floats, allowing scientific notation.
    try:
        x = float(t)
        # Only keep finite floats; infinities/NaNs remain strings.
        if math.isfinite(x):
            return x
    except ValueError:
        pass
    # Finally treat the token as a normalized lowercase string.
    return t.lower()

def _csv_to_list(ctx, param, value):
    """Normalize comma/space separated inputs into typed Python lists."""
    if value is None:
        return []
    # Accept either explicit lists or comma/space-delimited strings.
    if isinstance(value, list):
        tokens = value
    else:
        # Split by commas or whitespace and drop empty fragments.
        tokens = [x for x in re.split(r'[,\s]+', str(value).strip()) if x]

    return [_coerce_number(str(x)) for x in tokens]


# --- URI helpers ---------------------------------------------------------------
def _materialize_local_file(path: URIPath | Path | str) -> Path:
    """Return a local filesystem Path for downstream libraries."""
    if isinstance(path, URIPath):
        return Path(path.materialize())
    return Path(path)


def _materialize_local_files(paths: list[URIPath | Path | str]) -> list[Path]:
    """Materialize a series of URIPath objects while preserving order."""
    return [_materialize_local_file(p) for p in paths]


def _optional_uri_paths(ctx: click.Context, param: click.Option, value):
    """Normalize multi-value URI options so hidden flags can remain optional."""
    if not value:
        return None
    return list(value)


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
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None) is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
            **json.loads(os.getenv("S3_STORAGE_OPTIONS", default=None)),
        )
        if os.getenv("S3_STORAGE_OPTIONS", default=None) is not None
        else URIPathType(
            exists=True,
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None) is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
        )
    ),
    required=False,
    default=None,
    hidden=True,
    help="Optional directory containing whole slide images; inferred patches remain the primary input.",
)
@click.option(
    "--slide-path",
    "slide_paths",
    type=(
        URIPathType(
            exists=True,
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None) is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
            **json.loads(os.getenv("S3_STORAGE_OPTIONS", default=None)),
        )
        if os.getenv("S3_STORAGE_OPTIONS", default=None) is not None
        else URIPathType(
            exists=True,
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None) is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
        )
    ),
    multiple=True,
    default=None,
    callback=_optional_uri_paths,
    hidden=True,
    help="Hidden argument for orchestrators to pass explicit slide paths alongside cached patches.",
)
@click.option(
    "-o",
    "--results-dir",
    type=(
        URIPathType(
            # exists=True,
            cache_dir=
                os.getenv("WSINSIGHT_REMOTE_CACHE_DIR", default=None)
                if os.getenv("WSINSIGHT_REMOTE_CACHE_DIR") is not None
                else Path(user_cache_dir(appname="wsinsight", appauthor=False)),
            **json.loads(os.getenv("S3_STORAGE_OPTIONS", default=None)),
        )
        if os.getenv("S3_STORAGE_OPTIONS", default=None) is not None
        else URIPathType(
            # exists=True,
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
    help="Use QuPath 'name' fields as classes instead of the default Classification column when ingesting detection/annotation data.",
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
    default=DEFAULT_INFER_WORKERS,
    show_default=True,
    type=click.IntRange(min=0),
    help="Dataloader workers that feed patches to PyTorch (n=0 runs single-threaded)."
    " Default heuristics use min(2 × GPU count, CPU count).",
)
@click.option(
    "--export-workers",
    default=DEFAULT_EXPORT_WORKERS,
    show_default=True,
    type=click.IntRange(min=1),
    help="Worker processes for GeoJSON/OME-CSV export; leave headroom for the OS.",
)
@click.option(
    "--stitch-workers",
    default=DEFAULT_STITCH_WORKERS,
    show_default=True,
    type=click.IntRange(min=1),
    help="Thread pool size used when TileFuse stitches object-based detections.",
)
# @click.option(
#     "--speedup/--no-speedup",
#     default=False,
#     show_default=True,
#     help="JIT-compile the model and apply inference optimizations. This imposes a"
#     " startup cost but may improve performance overall.",
# )
# Legacy CLI options retained for potential reintroduction.
# Additional segmentation CLI options kept disabled until they are fully supported.
# @click.option(
#     "--qupath",
#     is_flag=True,
#     default=False,
#     show_default=True,
#     help="Create a QuPath project containing the inference results",
# )
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
# Segmentation knobs for regenerating patches when inference needs different tiling.
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
    "--hplot",
    is_flag=True,
    default=False,
    show_default=True,
    help="Run H-Plot analysis to compute per-cell layer metrics and optional exports.",
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
    help="Skip samples lacking both inner/outer layer bounds so only valid H-Plot ranges contribute to stats.",
)
@click.option(
    "--cme-cellular",
    is_flag=True,
    default=False,
    show_default=True,
    help="Run cellular-level CME analysis to produce per-cell embeddings/labels.",
)
@click.option(
    "--cme-annotation",
    is_flag=True,
    default=False,
    show_default=True,
    help="Run annotation-level CME analysis to cluster higher-level regions.",
)
@click.option(
    "--cme-soft-mode",
    is_flag=True,
    default=False,
    show_default=True,
    help="CME clustering weights class probabilities instead of hard labels; otherwise categorical assignments are used.",
)
@click.option(
    "--cme-clustering-k",
    default=None,
    type=click.IntRange(min=0),
    help="k-nearest-neighbor count used when building CME clustering graphs.",
)
@click.option(
    "--cme-clustering-resolutions",
    callback=_csv_to_list,
    default="0.5,1.0,2.0",
    help="Resolution parameter using in clustering for cmes.",
)


# --- CLI command --------------------------------------------------------------
def infer(
    ctx: click.Context,
    *,
    wsi_dir: URIPath | None,
    slide_paths: List[URIPath] | None,
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
    num_workers: int = DEFAULT_INFER_WORKERS,
    export_workers: int = DEFAULT_EXPORT_WORKERS,
    stitch_workers: int = DEFAULT_STITCH_WORKERS,
    # speedup: bool = False,
    # qupath: bool = False,
    geojson: bool = False,
    omecsv: bool = False,
    patch_overlap_ratio: float = 0.0,
    patch_size_um: float = 0.0,
    patch_size_px: int = 0,
    # patch_overlap_median_filter_size: int = 0,
    # red_threshold: int = 0,
    # stardist_normalization_pmin: float = 1.0,
    # stardist_normalization_pmax: float = 99.8,
    hplot: bool = False,
    hplot_max_neighbor_distance: float = 25.0,
    hplot_base_types: List | None = None,
    hplot_target_types: List | None = None,
    hplot_k: int = 2,
    hplot_n: int = 8,
    hplot_r: float = 0.5,
    hplot_range_max: int = None,
    hplot_range_min: int = None,
    hplot_samples_with_valid_range_only: bool = False,
    cme_cellular: bool = False,
    cme_annotation: bool = False,
    cme_soft_mode: bool = False,
    cme_clustering_k: int | None = None,
    cme_clustering_resolutions = [0.5,1.0,2.0],
) -> None:
    """Execute WSInsight inference and optional post-processing on prepared patches.

    The command consumes an existing `results_dir` (typically produced by `wsinsight
    patch` or `wsinsight run`) that already contains patch-level data plus a
    `wsi_list.csv`. Depending on the arguments, it either loads a registered model,
    a custom config/model-path pair, or synthesizes a pseudo-model from QuPath
    detections/annotations. After validating the requested patch geometry, the
    selected model is run via `run_inference`, producing CSV outputs under
    `model-outputs-csv`.

    Optional switches then:
        • export CSV predictions to GeoJSON/OME-CSV formats (`--geojson`, `--omecsv`)
        • drive downstream spatial analytics such as H-Plot and CME generation (when
            their respective flags are supplied)
        • synchronize metadata about the full session to `infer_metadata_*.json`.

    The function treats all input/output paths as `URIPath` values, materializing
    local copies only when dependencies such as pandas/geopandas require filesystem
    paths. Usage mirrors the CLI: `wsinsight infer --results-dir ... [options]`.
    """

    # --- Validate CLI combinations -----------------------------------------
    if (
        model_name is None
        and config is None
        and model_path is None
        and qupath_detection_dir is None
        and qupath_geojson_detection_dir is None
        and qupath_geojson_annotation_dir is None
    ):
        raise click.UsageError(
            "one of --model or (--config and --model-path) or --qupath_detection_dir "
            "or --qupath_geojson_detection_dir or --qupath_geojson_annotation_dir is required."
        )
    elif (
        (config is not None or model_path is not None)
        and model_name is not None
        and (
            qupath_detection_dir is not None
            or qupath_geojson_detection_dir is not None
            or qupath_geojson_annotation_dir is not None
        )
    ):
        raise click.UsageError(
            "--config and --model-path are mutually exclusive with --model. "
            "Both --qupath_detection_dir and --qupath_geojson_detection_dir and --qupath_geojson_annotation_dir "
            "are mutually exclusive with --model, --config and --model-path."
        )
    elif (config is not None) ^ (model_path is not None):  # XOR
        raise click.UsageError(
            "--config and --model-path must both be set if one is set."
        )

    _print_system_info()

    print("\nCommand line arguments")
    print("----------------------")
    for key, value in ctx.params.items():
        print(f"{key} = {value}")
    print("----------------------\n")

    # --- Resolve model or pseudo-model -------------------------------------
    # Track runtime flags populated once we know which model/config we're using.
    stain_normalization = False 
    object_detection = None
    
    # Resolve the model configuration up-front so downstream stages know spacing,
    # patch sizes, and class metadata.
    model_obj: HFModel | models.LocalModelTorchScript
    if model_name is not None:
        model_obj = models.get_registered_model(name=model_name)
        
    elif config is not None:
        with Path(config).open("r", encoding="utf-8") as f:
            _config_dict = json.load(f)
        model_config = ModelConfiguration.from_dict(_config_dict)
        model_obj = models.LocalModelTorchScript(
            config=model_config, model_path=str(model_path)
        )
        
        object_based = True if 'object_based' in _config_dict.keys() and _config_dict['object_based'] else False
        mixed_precision = True if 'mixed_precision' in _config_dict.keys() and _config_dict['mixed_precision'] else False
        stain_normalization = True if 'stain_normalization' in _config_dict.keys() and _config_dict['stain_normalization'] else False
        object_detection = _config_dict['object_detection']["name"] if object_based and 'object_detection' in _config_dict.keys() and isinstance(_config_dict['object_detection'], dict) else None
        halo_size_px = _config_dict['halo_size_pixels'] if object_based and 'halo_size_pixels' in _config_dict.keys() and _config_dict['halo_size_pixels'] else 0
        del _config_dict, model_config
        
        
    elif qupath_detection_dir is not None:
        try:
            with (results_dir / "wsi_list.csv").open("r", encoding="utf-8") as manifest_fp:
                manifest_df = pd.read_csv(manifest_fp)
        except Exception as exc:  # pragma: no cover - IO error surface for users
            raise click.ClickException(f"Unable to read wsi_list.csv") from exc
    
        if "wsi_path" not in manifest_df.columns:
            raise click.ClickException(
                f"wsi_paths must contain a 'wsi_path' column."
            )
    
        wsi_paths = [URIPath(str(path)) for path in manifest_df["wsi_path"].dropna().tolist()]
        
        if not wsi_paths:
            raise errors.WholeSlideImagesNotFound("wsi_list.csv")
        
        if not results_dir.exists():
            raise errors.ResultsDirectoryNotFound(results_dir)

        class_names = []
            
        print("\nLoading pseudo model data...\n")
        for wsi_path in tqdm.tqdm(wsi_paths):
            slide_det_name = _materialize_local_file(wsi_path).with_suffix(".txt").name
            slide_det = qupath_detection_dir / slide_det_name
            if slide_det.exists():
                # Load detections once per slide to collect unique class labels.
                with slide_det.open("r", encoding="utf-8") as det_fp:
                    qpdet_df = pd.read_csv(det_fp, delimiter='\t')
                qpdet_class_names = (
                    qpdet_df["Name"].str.strip().str.replace(" ", "_", regex=False).str.lower().unique().tolist()
                    if qupath_name_as_class
                    else qpdet_df["Classification"].str.strip().str.replace(" ", "_", regex=False).str.lower().unique().tolist()
                )
                class_names.extend(qpdet_class_names)
                class_names = list(set(class_names))
                    
        model_obj = Model(
            ModelConfiguration(
                architecture='qupath.detection',
                num_classes=len(class_names),
                class_names=class_names,
                patch_size_pixels=qupath_detection_patch_size,
                spacing_um_px=qupath_spacing_um_px,
                transform=None,
            ),
            "",
        )
        
        patch_size_px=model_obj.config.patch_size_pixels
        object_based = True
        mixed_precision = False
        stain_normalization = None
        object_detection = None
        halo_size_px = 0
   
        
    elif qupath_geojson_detection_dir is not None:
        try:
            with (results_dir / "wsi_list.csv").open("r", encoding="utf-8") as manifest_fp:
                manifest_df = pd.read_csv(manifest_fp)
        except Exception as exc:  # pragma: no cover - IO error surface for users
            raise click.ClickException(f"Unable to read wsi_list.csv") from exc
    
        if "wsi_path" not in manifest_df.columns:
            raise click.ClickException(
                f"wsi_paths must contain a 'wsi_path' column."
            )
    
        wsi_paths = [URIPath(str(path)) for path in manifest_df["wsi_path"].dropna().tolist()]
        
        if not wsi_paths:
            raise errors.WholeSlideImagesNotFound("wsi_list.csv")
        
        if not results_dir.exists():
            raise errors.ResultsDirectoryNotFound(results_dir)
        
        class_names = []
            
        print("\nLoading pseudo model data...\n")
        for wsi_path in tqdm.tqdm(wsi_paths):
            slide_geojson_name = _materialize_local_file(wsi_path).with_suffix(".geojson").name
            slide_geojson = qupath_geojson_detection_dir / slide_geojson_name
            if slide_geojson.exists():
                # Load the geojson to derive the distinct detection classes.
                qpgeojson_gdf = gpd.read_file(slide_geojson.materialize())
                qpgeojson_gdf.set_crs(None, allow_override=True)
                qpgeojson_class_names = (
                    qpgeojson_gdf.name.str.strip().str.replace(" ", "_", regex=False).str.lower().unique().tolist()
                    if qupath_name_as_class
                    else qpgeojson_gdf.classification.str.strip().str.replace(" ", "_", regex=False).str.lower().unique().tolist()
                )
                class_names.extend(qpgeojson_class_names)
                class_names = list(set(class_names))                
                    
        model_obj = Model(
            ModelConfiguration(
                architecture='qupath.geojson',
                num_classes=len(class_names),
                class_names=class_names,
                patch_size_pixels=qupath_detection_patch_size,
                spacing_um_px=qupath_spacing_um_px,
                transform=None,
            ),
            "",
        )
        
        patch_size_px=model_obj.config.patch_size_pixels
        object_based = True
        mixed_precision = False
        stain_normalization = None
        object_detection = None
        halo_size_px = 0        
          
          
    elif qupath_geojson_annotation_dir is not None:
        try:
            with (results_dir / "wsi_list.csv").open("r", encoding="utf-8") as manifest_fp:
                manifest_df = pd.read_csv(manifest_fp)
        except Exception as exc:  # pragma: no cover - IO error surface for users
            raise click.ClickException(f"Unable to read wsi_list.csv") from exc
    
        if "wsi_path" not in manifest_df.columns:
            raise click.ClickException(
                f"wsi_paths must contain a 'wsi_path' column."
            )
    
        wsi_paths = [URIPath(str(path)) for path in manifest_df["wsi_path"].dropna().tolist()]
        
        if not wsi_paths:
            raise errors.WholeSlideImagesNotFound("wsi_list.csv")
        
        if not results_dir.exists():
            raise errors.ResultsDirectoryNotFound(results_dir)

        class_names = []
            
        print("\nLoading pseudo model data...\n")
        for wsi_path in tqdm.tqdm(wsi_paths):
            slide_geojson_name = _materialize_local_file(wsi_path).with_suffix(".geojson").name
            slide_geojson = qupath_geojson_annotation_dir / slide_geojson_name
            if slide_geojson.exists():
                # Load the annotation geojson to derive distinct region labels.
                qpgeojson_gdf = gpd.read_file(slide_geojson.materialize())
                qpgeojson_gdf.set_crs(None, allow_override=True)
                qpgeojson_class_names = (
                    qpgeojson_gdf.name.str.strip().str.replace(" ", "_", regex=False).str.lower().unique().tolist()
                    if qupath_name_as_class
                    else qpgeojson_gdf.classification.str.strip().str.replace(" ", "_", regex=False).str.lower().unique().tolist()
                )
                class_names.extend(qpgeojson_class_names)
                class_names = list(set(class_names))                
                    
        model_obj = Model(
            ModelConfiguration(
                architecture='qupath.geojson',
                num_classes=len(class_names),
                class_names=class_names,
                patch_size_pixels=qupath_annotation_patch_size,
                spacing_um_px=qupath_spacing_um_px,
                transform=None,
            ),
            "",
        )
        
        patch_size_px=model_obj.config.patch_size_pixels
        object_based = False
        mixed_precision = False
        stain_normalization = None
        object_detection = None
        halo_size_px = 0        
          
          
    else:
        raise click.ClickException("Neither of --config and --model was passed")

    if references_dir is not None and not object_based:
        raise click.ClickException("--annotaions-dir only works with object based model.")
    
    # Validating all overlap options
    nonzero_count = sum([
        0 if d == 0 else 1 
        for d in [patch_overlap_ratio, patch_size_um, patch_size_px]
    ])
    
    if nonzero_count > 1 and qupath_detection_dir is None and qupath_geojson_detection_dir is None:
        raise click.ClickException("Only one of --patch-overlap-ratio, --patch-size-um, --patch-size-px is allowed")
    elif nonzero_count == 1 and object_based and qupath_detection_dir is None and qupath_geojson_detection_dir is None:
        raise click.ClickException("--object-based doesn't work with variational patch size")
    
    if patch_overlap_ratio != 0.0:
        overlap = patch_overlap_ratio

    elif patch_size_um != 0.0:
        if patch_size_um > (model_obj.config.patch_size_pixels*model_obj.config.spacing_um_px):
            raise click.ClickException("--patch-size-um has to be smaller than patch size")
         
        overlap = 1.0-patch_size_um/(model_obj.config.patch_size_pixels*model_obj.config.spacing_um_px)
        
    elif patch_size_px != 0:
        if patch_size_px > model_obj.config.patch_size_pixels:
            raise click.ClickException("--patch-size-px must not be larger than patch size")
        
        overlap = 1.0-float(patch_size_px)/float(model_obj.config.patch_size_pixels)
        
    else:
        overlap = 0.0
    

    # --- Validate dependent artifacts --------------------------------------
    if not (results_dir / "patches").exists():
        raise click.ClickException(
            "No patches were created. Please see the logs above and check for errors. "
            "It is possible that no tissue was detected in the slides. If that is the case,"
            " please try to use different --seg-* parameters, which will change how the"
            " segmentation is done. For example, a lower binary threshold may be set."
        )
    
    if wsi_dir is not None and slide_paths is None:
        slide_paths = sorted(
            [
                p
                for p in tqdm.tqdm(
                    wsi_dir.iterdir(), desc="Count files in slide directory"
                )
                if p.is_file()
            ]
        )

    if wsi_dir is not None and (slide_paths is None or not slide_paths):
        raise FileNotFoundError(f"no files exist in the slide directory: {wsi_dir}")

    # --- Execute model inference -------------------------------------------
    click.secho("\nRunning model inference.\n", fg="green")
    
    failed_patching, failed_inference = run_inference(
        wsi_dir=wsi_dir,
        slide_paths=slide_paths,
        results_dir=results_dir,
        references_dir=references_dir,
        qupath_detection_dir=qupath_detection_dir,
        qupath_geojson_detection_dir=qupath_geojson_detection_dir,
        qupath_geojson_annotation_dir=qupath_geojson_annotation_dir,
        qupath_name_as_class=qupath_name_as_class,
        model_info=model_obj,
        halo_size_px=halo_size_px,
        batch_size=batch_size,
        num_workers=num_workers,
        # speedup=speedup,
        stain_normalization=stain_normalization,
        object_based=object_based,
        object_detection=object_detection,
        mixed_precision=mixed_precision,
        stitch_workers=stitch_workers,
    )

    # --- Optional exports: GeoJSON / OME-CSV -------------------------------
    csv_exports: list[Path] | None = None
    if geojson or omecsv:
        csv_candidates = list((results_dir / "model-outputs-csv").iterdir(files_only=True))
        csv_exports = _materialize_local_files([p for p in csv_candidates if p.suffix == ".csv"])

    if geojson:
        click.echo("\nWriting inference results to GeoJSON files\n")
        # Convert inference CSV outputs to GeoJSON for downstream visualization.
        write_geojsons(csvs=csv_exports or [], 
                   overlap=overlap, 
                   results_dir=results_dir, 
                   output_dir="model-outputs-geojson", 
                   prefix="prob", 
                   num_workers=export_workers,
                   object_type="detection" if object_based else "tile", 
                   set_classification=True if object_based else False)

    if omecsv:
        click.echo("\nWriting inference results to OMECSV files\n")
        h5_candidates = list((results_dir / "patches").iterdir(files_only=True))
        h5s = _materialize_local_files([p for p in h5_candidates if p.suffix == ".h5"])
        # Pair CSV probabilities with the cached HDF5 patches for OME exports.
        write_omecsvs(csvs=csv_exports or [], 
                  h5s=h5s, 
                  overlap=overlap, 
                  results_dir=results_dir, 
                  output_dir=URIPath("model-outputs-omecsv"), 
                  prefix="prob",
                  num_workers=export_workers,
                  )
            
    if failed_patching:
        click.secho(f"\nPatching failed for {len(failed_patching)} slides", fg="yellow")
        click.secho("\n".join(failed_patching), fg="yellow")
    if failed_inference:
        click.secho(
            f"\nInference failed for {len(failed_inference)} slides", fg="yellow"
        )
        click.secho("\n".join(failed_inference), fg="yellow")

    # --- H-Plot analytics ---------------------------------------------------
    
    if hplot and (len(hplot_base_types) != 0 and len(hplot_target_types) != 0):
        target_type_list = [c.strip().replace(' ', '_').lower() for c in hplot_target_types]
        base_type_list = [c.strip().replace(' ', '_').lower() for c in hplot_base_types]
        
        for tp in base_type_list+target_type_list: 
            if tp not in model_obj.config.class_names:
                raise click.ClickException(f"\n--hplot-target-types and --hplot-base-types should include in the chosen model.")
                click.secho("\n".join(failed_inference), fg="yellow")
                        
        click.secho("\nRunning H-Plot generation.\n", fg="green")
        
        failed_hplot_generation = hplot_generation(
            wsi_dir=None,
            wsi_paths=wsi_paths,
            results_dir=results_dir,
            base_type_list=base_type_list,
            target_type_list=target_type_list,
            max_neighbor_distance_um=hplot_max_neighbor_distance,
            hplot_k=hplot_k,
            hplot_N=hplot_n,
            hplot_R=hplot_r, 
            hplot_range_max=hplot_range_max,
            hplot_range_min=hplot_range_min,
            hplot_samples_with_valid_range_only=hplot_samples_with_valid_range_only,
            num_workers=1 if num_workers == 0 else num_workers,
        )
        
        if failed_hplot_generation:
            click.secho(
                f"\nH-Plot generation failed for {len(failed_hplot_generation)}"
                " slides", fg="yellow"
            )
            
            click.secho("\n".join(failed_hplot_generation), fg="yellow")
            
        if geojson:
            click.echo("\nWriting H-Plot cellular results to GeoJSON files\n")
            # Persist cellular layer metrics for visualization clients.
            hplot_cell_csv_uris = list((results_dir / "hplot-outputs-csv" / "cells").iterdir(files_only=True))
            hplot_cell_csvs = _materialize_local_files([p for p in hplot_cell_csv_uris if p.suffix == ".csv"])
            write_geojsons(csvs=hplot_cell_csvs, 
                           overlap=overlap, 
                           results_dir=results_dir, 
                           output_dir=URIPath("hplot-outputs-geojson"), 
                           prefix="hplot",
                           num_workers=export_workers,
                           object_type="detection",
                           set_classification=True,
                           annotation_shape="box")
        
        if omecsv:
            click.echo("\nWriting H-Plot cellular results to OMECSV files\n")
            h5s = _materialize_local_files([p for p in (results_dir / "patches").iterdir(files_only=True) if p.suffix == ".h5"])
            # Write per-cell embeddings in a format QuPath/OME tooling can ingest.
            write_omecsvs(csvs=hplot_cell_csvs, 
                          h5s=h5s, 
                          overlap=overlap, 
                          results_dir=results_dir, 
                          output_dir=URIPath("hplot-outputs-omecsv"), 
                          prefix="hplot",
                          num_workers=export_workers,
                          )
    
    elif hplot and (len(hplot_base_types) == 0 or len(hplot_target_types) == 0):
        raise click.ClickException(f"\nH-Plot requires both --hplot-base-types and hplot-target-types.")
        click.secho("\n".join(failed_inference), fg="yellow")
        
            
    # --- CME analytics ------------------------------------------------------
    if cme_cellular or cme_annotation:      
        click.secho("\nRunning cme generation.\n", fg="green")
        # Default flow: run CME with the graph-based pipeline (H-Optimus disabled).
        cme_generation(
            wsi_dir=None,
            wsi_paths=wsi_paths,
            results_dir=results_dir,
            max_edge_len_um=25,
            max_cell_radius_um=15,
            k_hops=2, alpha=1.0,
            use_hoptimus=False,                 # ← off
            hidden=64, out_dim=32, epochs=300,
            cme_cellular=cme_cellular, 
            cme_annotation=cme_annotation,
            cme_clustering_k=cme_clustering_k, 
            cme_clustering_resolutions=cme_clustering_resolutions,
            cme_soft_mode=cme_soft_mode,
            # seed=0,
        )
         
        # # Option 2: Example wiring for H-Optimus when dedicated datasets are available.
        # patch_datasets = [DummyPatchDataset(num_cells=len(slides_inputs[0][0])),
        #                   DummyPatchDataset(num_cells=len(slides_inputs[1][0]))]
        #
        # res_h0 = cme_generation(
        #     slides_inputs,
        #     max_edge_len_um=70.0,
        #     k_hops=2, alpha=1.0,
        #     use_hoptimus=True,                  # ← ON
        #     patch_datasets=patch_datasets,      # replace with your real datasets later
        #     sample_frac=0.2, pca_dim=128, knn_k=3, knn_sigma_um=60.0,
        #     hidden=64, out_dim=32, epochs=300,
        #     clusters_k=5, seed=0
        # )
        #
        # # Each result contains per-slide embeddings and labels:
        # Z_slide0 = res_h0["embeddings"][0]        # [N0, 32]
        # y_slide0 = res_h0["labels"][0]            # [N0]
        # kept_idx0 = res_h0["kept_idx"][0]         # map to original rows in slideA_cells.csv
     
     
        if geojson:
            click.echo("\nWriting CME detection cellular results to GeoJSON files\n")
            # Export CME cell-level outputs for quick map overlays.
            cme_cell_csvs = _materialize_local_files(
                [
                    p
                    for p in (results_dir / "cme-outputs-csv" / "cells").iterdir(files_only=True)
                    if p.suffix == ".csv"
                ]
            )
            write_geojsons(
                csvs=cme_cell_csvs,
                overlap=overlap,
                results_dir=results_dir,
                output_dir=URIPath("cme-outputs-geojson") / "cells",
                prefix="cme",
                num_workers=1 if export_workers == 0 else export_workers,
                object_type="detection",
                set_classification=True,
                annotation_shape="box",
            )
     
        
        # Example annotation-level exports can be re-enabled when CME polygons are finalized.
        # cme_cme_csvs = list((results_dir / "cme-outputs-csv" / "cmes").glob("*.csv"))
        # write_geojsons(
        #     csvs=cme_cme_csvs,
        #     overlap=overlap,
        #     results_dir=results_dir,
        #     output_dir=Path("cme-outputs-geojson") / "cmes",
        #     prefix="cme",
        #     num_workers=1 if num_workers == 0 else num_workers,
        #     object_type="annotation",
        #     set_classification=True,
        #     annotation_shape="polygon",
        # )
        
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    run_metadata_outpath = results_dir / f"infer_metadata_{timestamp}.json"
    click.echo(f"\nSaving metadata about run to {run_metadata_outpath}\n")
    run_metadata = _get_info_for_save(model_obj)
    with run_metadata_outpath.open("w") as f:
        json.dump(run_metadata, f, indent=2)

    # QuPath project export can be re-enabled once optional dependencies return.
    # if qupath:
    #     click.echo("Creating QuPath project with results")
    #     make_qupath_project(wsi_dir, results_dir)

    click.secho("\nWSInsight-infer tasks are all finished.\n", fg="green")
