"""WSInsight patch extraction CLI for whole slide images.

The command builds tissue masks, extracts patches, and persists intermediate
metadata so downstream inference/analytics can reuse the outputs.
"""

from __future__ import annotations
import os
import re
import dataclasses
import getpass
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from platformdirs import user_cache_dir
import click
import math
import pandas as pd
import tqdm 
import geopandas as gpd

import wsinfer_zoo.client
from wsinfer_zoo.client import HFModel
from wsinfer_zoo.client import Model, ModelConfiguration

from .. import errors
from ..modellib import models
from ..patchlib import segment_and_patch_directory_of_slides
from ..uri_path import URIPathType, URIPath
from ..wsi import _validate_wsi_directory


def _num_cpus() -> int:
    """Get number of CPUs on the system."""
    try:
        return len(os.sched_getaffinity(0))
    # os.sched_getaffinity seems to be linux only.
    except AttributeError:
        count = os.cpu_count()  # potentially None
        return count or 0


def _inside_container() -> str:
    """Best-effort detection of container runtimes for logging/metadata."""
    if Path("/.dockerenv").exists():
        return "yes, docker"
    elif (
        Path("/singularity").exists()
        or Path("/singularity.d").exists()
        or Path("/.singularity.d").exists()
    ):
        # TODO: apptainer might change the name of this directory.
        return "yes, apptainer/singularity"
    return "no"


def _get_timestamp() -> str:
    """Return a human-readable timestamp with timezone for saved metadata."""
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


def _coerce_number(token: str):
    """Convert CLI token to int/float when possible, otherwise lowercase string."""
    t = token.strip()
    # int?
    if re.fullmatch(r'[+-]?\d+', t):
        try:
            return int(t)
        except ValueError:
            pass
    # float / scientific?
    try:
        x = float(t)
        # keep only finite floats; otherwise leave as string
        if math.isfinite(x):
            return x
    except ValueError:
        pass
    # not a number → return lower-cased string
    return t.lower()

def _csv_to_list(ctx, param, value):
    """Parse comma/space separated CLI values into a normalized list."""
    if value is None:
        return []
    # Accept list or string input (keep usage unchanged)
    if isinstance(value, list):
        tokens = value
    else:
        # split by commas or whitespace; drop empties
        tokens = [x for x in re.split(r'[,\s]+', str(value).strip()) if x]

    return [_coerce_number(str(x)) for x in tokens]


def _optional_uri_paths(ctx: click.Context, param: click.Option, value):
    """Normalize optional multi-value URIPath inputs for hidden flags."""
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
    help="Hidden argument allowing orchestrators to pass explicit slide paths.",
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
    help="Directory that contains the previous wsinfer/wsinsight output as the reference for this analysis.",
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
    help="Directory containing geojson files generated by QuPath."
    " The annotation in the geojson files will be used.",
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
    "--cache-image-patches",
    is_flag=True,
    default=False,
    show_default=True,
    help="Extract image patches and save to hdf5.",
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
def patch(
    ctx: click.Context,
    *,
    wsi_dir: URIPath,
    slide_paths: list[URIPath] | None,
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
    cache_image_patches: bool = False,
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
) -> None:
    """Segment tissue and generate patch coordinates for a WSI directory.

    The command validates slide availability, loads either a registered model,
    user-supplied weights, or QuPath-derived pseudo-models, then runs tissue
    segmentation and patch enumeration. Outputs land under
    `RESULTS_DIR/model-outputs-*` for downstream inference.
    """

    if model_name is None and config is None and model_path is None and qupath_detection_dir is None and qupath_geojson_detection_dir is None and qupath_geojson_annotation_dir is None:
        raise click.UsageError(
            "one of --model or (--config and --model-path) or --qupath_detection_dir or --qupath_geojson_detection_dir or --qupath_geojson_annotation_dir is required."
        )
    elif (config is not None or model_path is not None) and model_name is not None and (qupath_detection_dir is not None or qupath_geojson_detection_dir is not None or qupath_geojson_annotation_dir is not None ):
        raise click.UsageError(
            "--config and --model-path are mutually exclusive with --model."
            "Both --qupath_detection_dir and --qupath_geojson_detection_dir and --qupath_geojson_annotation_dir are mutually exclusive with --model, --config and --model-path."
        )
    elif (config is not None) ^ (model_path is not None):  # XOR
        raise click.UsageError(
            "--config and --model-path must both be set if one is set."
        )
        
    # remote_cache_dir = remote_cache_dir / ("s3" if str(wsi_dir).startswith("s3") else "gdc-manifest" if str(wsi_dir).startswith("gdc-manifest") else "")
    # wsi_dir = URIPath(wsi_dir, cache_dir=remote_cache_dir)
    
    if not (wsi_dir.exists()):
        raise FileNotFoundError(f"Whole slide image directory not found: {wsi_dir}")

    # Test that wsi dir actually includes files. This is here for an interesting edge
    # case. When using a Linux container and if the data directory is symlinked from a
    # different directory, both directories need to be bind mounted onto the container.
    # If only the symlinked directory is included, then the patching script will fail,
    # even though it looks like there are files in the wsi_dir directory.

    # files_in_wsi_dir = list(slide_paths) if slide_paths is not None else [
    #     p
    #     for p in tqdm.tqdm(
    #         wsi_dir.iterdir(), desc="Count files in slide directory"
    #     )
    #     if p.is_file()
    # ]
    
    # if not files_in_wsi_dir:
    #     raise FileNotFoundError(f"no files exist in the slide directory: {wsi_dir}")

    if slide_paths is None:
        slide_paths = sorted(
            [
                p
                for p in tqdm.tqdm(
                    wsi_dir.iterdir(), desc="Count files in slide directory"
                )
                if p.is_file()
            ]
        )

    if slide_paths is None or not slide_paths:
        raise FileNotFoundError(f"no files exist in the slide directory: {wsi_dir}")

    _print_system_info()

    print("\nCommand line arguments")
    print("----------------------")
    for key, value in ctx.params.items():
        print(f"{key} = {value}")
    print("----------------------\n")

    # Space holder for variable stain_normalization
    # if config is used, it may be assigned.
    object_detection = None
    
    # Get weights object before running the patching script because we need to get the
    # necessary spacing and patch size.
    model_obj: HFModel | models.LocalModelTorchScript
    if model_name is not None:
        model_obj = models.get_registered_model(name=model_name)
        
    elif config is not None:
        with open(config) as f:
            _config_dict = json.load(f)
        model_config = ModelConfiguration.from_dict(_config_dict)
        model_obj = models.LocalModelTorchScript(
            config=model_config, model_path=str(model_path)
        )
        
        object_based = True if 'object_based' in _config_dict.keys() and _config_dict['object_based'] else False
        object_detection = _config_dict['object_detection']["name"] if object_based and 'object_detection' in _config_dict.keys() and isinstance(_config_dict['object_detection'], dict) else None
        stardist_normalization_pmin = _config_dict['object_detection']["normalization_pmin"] if object_detection == "stardist" else None
        stardist_normalization_pmax = _config_dict['object_detection']["normalization_pmax"] if object_detection == "stardist" else None
        halo_size_px = _config_dict['halo_size_pixels'] if object_based and 'halo_size_pixels' in _config_dict.keys() and _config_dict['halo_size_pixels'] else 0
        del _config_dict, model_config
        
        
    elif qupath_detection_dir is not None:
        if not wsi_dir.exists():
            raise errors.WholeSlideImageDirectoryNotFound(f"directory not found: {wsi_dir}")
        
        wsi_paths = [p for p in wsi_dir.iterdir() if p.is_file()]
        
        if not wsi_paths:
            raise errors.WholeSlideImagesNotFound(wsi_dir)
        
        if not results_dir.exists():
            raise errors.ResultsDirectoryNotFound(results_dir)

        _validate_wsi_directory(wsi_dir)
        
        class_names = []
            
        print("\nLoading pseudo model data...\n")
        for wsi_path in tqdm.tqdm(wsi_paths):
            slide_det_name = wsi_path.with_suffix(".txt").name
            slide_det = qupath_detection_dir / slide_det_name
            if slide_det.exists():
                # read & prefilter once (ensure remote paths are materialized)
                with slide_det.open("r", encoding="utf-8") as det_fp:
                    qpdet_df = pd.read_csv(det_fp, delimiter='\t')
                qpdet_class_names = qpdet_df["Name"].str.strip().str.replace(" ", "_", regex=False).str.lower().unique().tolist() \
                    if qupath_name_as_class else \
                    qpdet_df["Classification"].str.strip().str.replace(" ", "_", regex=False).str.lower().unique().tolist()
                class_names.extend(qpdet_class_names)
                class_names = list(set(class_names))
                    
        model_obj = Model(ModelConfiguration(architecture='qupath.detection', 
                                             num_classes=len(class_names), 
                                             class_names=class_names, 
                                             patch_size_pixels=qupath_detection_patch_size, 
                                             spacing_um_px=qupath_spacing_um_px, 
                                             transform=None), "")
        
        patch_size_px=model_obj.config.patch_size_pixels
        object_based = True
        object_detection = None
        stardist_normalization_pmin = None
        stardist_normalization_pmax = None
        halo_size_px = 0
   
        
    elif qupath_geojson_detection_dir is not None:
        if not wsi_dir.exists():
            raise errors.WholeSlideImageDirectoryNotFound(f"directory not found: {wsi_dir}")
        
        wsi_paths = [p for p in wsi_dir.iterdir() if p.is_file()]
        
        if not wsi_paths:
            raise errors.WholeSlideImagesNotFound(wsi_dir)
        
        _validate_wsi_directory(wsi_dir)
        
        class_names = []
            
        print("\nLoading pseudo model data...\n")
        for wsi_path in tqdm.tqdm(wsi_paths):
            slide_geojson_name = wsi_path.with_suffix(".geojson").name
            slide_geojson = qupath_geojson_detection_dir / slide_geojson_name
            if slide_geojson.exists():
                # read & prefilter once (GeoPandas expects a filesystem path)
                qpgeojson_gdf = gpd.read_file(slide_geojson.materialize())
                qpgeojson_gdf.set_crs(None, allow_override=True)
                qpgeojson_class_names = \
                    qpgeojson_gdf.name.str.strip().str.replace(" ", "_", regex=False).str.lower().unique().tolist() \
                    if qupath_name_as_class else \
                    qpgeojson_gdf.classification.str.strip().str.replace(" ", "_", regex=False).str.lower().unique().tolist()
                class_names.extend(qpgeojson_class_names)
                class_names = list(set(class_names))                
                    
        model_obj = Model(ModelConfiguration(architecture='qupath.geojson', 
                                             num_classes=len(class_names), 
                                             class_names=class_names, 
                                             patch_size_pixels=qupath_detection_patch_size, 
                                             spacing_um_px=qupath_spacing_um_px, 
                                             transform=None), "")
        
        patch_size_px=model_obj.config.patch_size_pixels
        object_based = True
        object_detection = None
        stardist_normalization_pmin = None
        stardist_normalization_pmax = None
        halo_size_px = 0        
          
          
    elif qupath_geojson_annotation_dir is not None:
        if not wsi_dir.exists():
            raise errors.WholeSlideImageDirectoryNotFound(f"directory not found: {wsi_dir}")
        
        wsi_paths = [p for p in wsi_dir.iterdir() if p.is_file()]
        
        if not wsi_paths:
            raise errors.WholeSlideImagesNotFound(wsi_dir)
        
        _validate_wsi_directory(wsi_dir)
        
        class_names = []
            
        print("\nLoading pseudo model data...\n")
        for wsi_path in tqdm.tqdm(wsi_paths):
            slide_geojson_name = wsi_path.with_suffix(".geojson").name
            slide_geojson = qupath_geojson_annotation_dir / slide_geojson_name
            if slide_geojson.exists():
                # read & prefilter once (GeoPandas expects a filesystem path)
                qpgeojson_gdf = gpd.read_file(slide_geojson.materialize())
                qpgeojson_gdf.set_crs(None, allow_override=True)
                qpgeojson_class_names = \
                    qpgeojson_gdf.name.str.strip().str.replace(" ", "_", regex=False).str.lower().unique().tolist() \
                    if qupath_name_as_class else \
                    qpgeojson_gdf.classification.str.strip().str.replace(" ", "_", regex=False).str.lower().unique().tolist()
                class_names.extend(qpgeojson_class_names)
                class_names = list(set(class_names))                
                    
        model_obj = Model(ModelConfiguration(architecture='qupath.geojson', 
                                             num_classes=len(class_names), 
                                             class_names=class_names, 
                                             patch_size_pixels=qupath_annotation_patch_size, 
                                             spacing_um_px=qupath_spacing_um_px, 
                                             transform=None), "")
        
        patch_size_px=model_obj.config.patch_size_pixels
        object_based = False
        object_detection = None
        stardist_normalization_pmin = None
        stardist_normalization_pmax = None
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
    
    click.secho("\nFinding patch coordinates...\n", fg="green")
    
    segment_and_patch_directory_of_slides(
        wsi_dir=wsi_dir,
        slide_paths=slide_paths,
        save_dir=results_dir,
        qupath_detection_dir=qupath_detection_dir,
        qupath_geojson_detection_dir=qupath_geojson_detection_dir,
        qupath_geojson_annotation_dir=qupath_geojson_annotation_dir,
        patch_size_px=model_obj.config.patch_size_pixels,
        patch_spacing_um_px=model_obj.config.spacing_um_px,
        halo_size_px=halo_size_px,
        histoqc_dir=histoqc_dir,
        thumbsize=seg_thumbsize,
        median_filter_size=seg_median_filter_size,
        binary_threshold=seg_binary_threshold,
        closing_kernel_size=seg_closing_kernel_size,
        min_object_size_um2=seg_min_object_size_um2,
        min_hole_size_um2=seg_min_hole_size_um2,
        overlap=overlap,
        object_based=object_based,
        object_detection=object_detection,
        stardist_normalization_pmin=stardist_normalization_pmin,
        stardist_normalization_pmax=stardist_normalization_pmax,
        cache_image_patches=cache_image_patches,
    )

    if not (results_dir / "patches").exists():
        raise click.ClickException(
            "No patches were created. Please see the logs above and check for errors."
            " It is possible that no tissue was detected in the slides. If that is the"
            " case, please try to use different --seg-* parameters, which will change"
            " how the segmentation is done. For example, a lower binary threshold may"
            " be set."
        )
    
    timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    run_metadata_outpath = results_dir / f"patch_metadata_{timestamp}.json"
    click.echo(f"\nSaving metadata about run to {run_metadata_outpath}\n")
    run_metadata = _get_info_for_save(model_obj)
    with run_metadata_outpath.open("w") as f:
        json.dump(run_metadata, f, indent=2)

    click.secho("\nWSInsight-patch tasks are all finished.\n", fg="green")
