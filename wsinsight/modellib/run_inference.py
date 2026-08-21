"""Run inference.

From the original paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7369575/):
> In the prediction (test) phase, no data augmentation was applied except for the
> normalization of the color channels.
"""

from __future__ import annotations

# from pathlib import Path
import gc
import logging
import os
from typing import List

import geopandas as gpd
import h5py
import histomicstk as htk
import numpy as np
import numpy.typing as npt

# from scipy.signal import medfilt2d
import pandas as pd
import torch
import tqdm
import wsinfer_zoo.client

from wsinsight.modellib.tilefuse import TileRemapStitcher

from .. import errors
from ..cancel import critical_section
from ..cancel import is_cancelled
from ..cancel import raise_if_cancelled
from ..insightlib.region_registration import register_objects_to_regions

# from ..wsi import get_avg_mpp
# from ..wsi import get_wsi_cls
from ..uri_path import URIPath
from ..wsi import _validate_wsi_directory
from .data import WholeSlideImagePatches
from .models import LocalModelTorchScript
from .models import get_pretrained_torch_module
from .transforms import make_compose_from_transform_config

logger = logging.getLogger(__name__)

EPSILON = 1e-8
I_0 = 255


def _as_output_tensor(pred: object) -> torch.Tensor:
    """Normalise a model's forward output to a single logits tensor.

    Zoo TorchScript models are not all bare-tensor classifiers: some return a
    ``dict`` (e.g. ``{"logits": ...}`` / ``{"out": ...}``), a ``tuple``/``list``
    (logits first), or an object exposing a ``.logits`` attribute. Calling
    ``.detach()`` on those raised ``'dict' object has no attribute 'detach'``.
    """
    if torch.is_tensor(pred):
        return pred
    if isinstance(pred, dict):
        for key in ("logits", "out", "output", "y", "pred", "predictions"):
            if key in pred and torch.is_tensor(pred[key]):
                return pred[key]
        tensor_vals = [v for v in pred.values() if torch.is_tensor(v)]
        if len(tensor_vals) == 1:
            return tensor_vals[0]
        raise TypeError(
            "model returned a dict with no unambiguous logits tensor; "
            f"keys={list(pred)}"
        )
    if isinstance(pred, (tuple, list)):
        for v in pred:
            if torch.is_tensor(v):
                return v
        raise TypeError("model returned a sequence with no tensor element.")
    if hasattr(pred, "logits") and torch.is_tensor(pred.logits):
        return pred.logits
    raise TypeError(f"unsupported model output type: {type(pred)}")


def _advance_batch_search(
    current: int,
    lo: int,
    hi: int,
    max_bs: int,
    *,
    oom: bool,
    probe_factor: float = 2.0,
) -> tuple[int, int, int]:
    """Update the OOM binary-search state and propose the next batch size.

    Maintains brackets ``(lo, hi)`` where:
      - ``lo`` is the largest batch size confirmed to fit (``0`` = none yet)
      - ``hi`` is the smallest batch size confirmed to OOM
        (``max_bs + 1`` = no ceiling seen yet)

    On OOM (``oom=True``) we tighten ``hi`` *and* invalidate any stale ``lo``
    by lowering it to ``min(lo, current - 1)``. Without that invalidation, a
    sequence like ``lo=20, hi=21, current=20`` followed by an OOM at 20 would
    deadlock at ``(20+20)//2 == 20`` forever.

    On success (``oom=False``) we raise ``lo`` to ``current``, then either
    bisect upward (if a ceiling is known) or probe upward by *probe_factor*
    (default 2.0 / doubling — calibration for tile inference is accurate so
    doubling reaches the true ceiling in one step).

    Returns
    -------
    next_current, lo, hi : int
        Proposed batch size for the next attempt and the updated brackets.
        ``next_current`` is always in ``[1, max_bs]``.
    """
    if oom:
        hi = current
        lo = min(lo, current - 1)
        next_current = max(1, (lo + hi) // 2)
    else:
        lo = current
        if hi <= max_bs:
            cand = (lo + hi) // 2
            next_current = cand if cand > lo else lo
        else:
            next_current = min(int(lo * probe_factor), max_bs) if lo > 0 else max_bs
        next_current = max(1, min(next_current, max_bs))
    return next_current, lo, hi


def _is_bs_converged(current: int, lo: int, hi: int) -> bool:
    """Return True when the OOM bisection has tightly bracketed the ceiling.

    "Converged" means we have both a confirmed-safe value (``lo``) and a
    confirmed-OOM value (``hi``) one step apart, and we are running at the
    safe value. Any OOM observed in this state is unexpected — most likely
    caused by allocator fragmentation or a transient external memory
    pressure event — and is handled with an in-place ``empty_cache`` retry
    before falling back to a bisection reset.
    """
    return lo > 0 and hi == lo + 1 and current == lo


def _calibrate_inference_batch_size(
    model: torch.nn.Module,
    transform,
    patch_size_px: int,
    device: torch.device,
    safety: float = 0.95,
    min_batch: int = 1,
    max_batch: int = 65536,
) -> tuple[int, int]:
    """Auto-calibrate an initial batch size from available GPU memory.

    Uses two-point calibration (the same technique as H-optimus calibration) to
    measure the marginal per-patch GPU memory cost, then divides available VRAM
    by that cost.

    Returns
    -------
    batch_size : int
        Recommended starting batch size (also used as the binary-search ceiling).
    bytes_per_image : int
        Marginal GPU bytes per image from calibration.  Store this and pass to
        ``_batch_size_for_available_vram`` to get a slide-specific batch-size
        estimate after slide-level allocations (e.g. TileRemapStitcher) are on
        GPU, without rerunning the expensive forward-pass calibration.

    Falls back to (32, 0) on any error.
    """
    if not torch.cuda.is_available():
        return 32, 0
    try:
        from PIL import Image as _Image

        # Determine model input tensor shape from the transform.
        dummy_pil = _Image.new("RGB", (patch_size_px, patch_size_px))
        dummy_tensor = transform(dummy_pil)  # e.g. [3, H, W]
        C, H, W = dummy_tensor.shape

        # Available VRAM: free + allocator-reserved-but-idle cache.
        def _available() -> int:
            free, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
            reserved = torch.cuda.memory_reserved(torch.cuda.current_device())
            allocated = torch.cuda.memory_allocated(torch.cuda.current_device())
            return free + max(0, reserved - allocated)

        # Two-point calibration: slope cancels fixed cuDNN workspace overhead.
        # Larger batch sizes (16→64) extrapolate more accurately to production
        # batch sizes than the old 4→16 range.
        def _measure(n: int) -> int:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline = torch.cuda.memory_allocated()
            dummy = torch.zeros(n, C, H, W, device=device)
            with torch.no_grad():
                model(dummy)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            del dummy
            torch.cuda.empty_cache()
            return peak - baseline

        marginal = 0
        for b1, b2 in ((16, 64), (4, 16)):
            try:
                m1 = _measure(b1)
                m2 = _measure(b2)
                if m2 > m1 and (b2 - b1) > 0:
                    marginal = (m2 - m1) // (b2 - b1)
                    break
            except Exception:
                torch.cuda.empty_cache()
        if marginal <= 0:
            return 32, 0

        n_gpu = max(1, torch.cuda.device_count())
        usable = _available()
        per_gpu = max(min_batch, int(usable * safety) // marginal)
        total = per_gpu * n_gpu
        batch_size = max(min_batch, min(max_batch, (total // n_gpu) * n_gpu))
        return batch_size, int(marginal)
    except Exception:
        torch.cuda.empty_cache()
        return 32, 0


def _batch_size_for_available_vram(
    bytes_per_image: int,
    n_gpu: int,
    safety: float = 0.95,
    min_batch: int = 1,
    max_batch: int = 65536,
) -> int:
    """Estimate batch size from current GPU availability and a known per-image cost.

    Call this after slide-level allocations (TileRemapStitcher, etc.) are on
    GPU to get a batch-size ceiling that reflects the actual remaining VRAM,
    without rerunning the expensive forward-pass calibration.
    """
    if not torch.cuda.is_available() or bytes_per_image <= 0:
        return min_batch
    try:
        free, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        reserved = torch.cuda.memory_reserved(torch.cuda.current_device())
        allocated = torch.cuda.memory_allocated(torch.cuda.current_device())
        usable = free + max(0, reserved - allocated)
        per_gpu = max(min_batch, int(usable * safety) // bytes_per_image)
        total = per_gpu * n_gpu
        return max(min_batch, min(max_batch, (total // n_gpu) * n_gpu))
    except Exception:
        return min_batch


def run_inference(
    wsi_dir: URIPath | None,
    slide_paths: List[URIPath] | None,
    results_dir: URIPath,
    region_inference_dir: str | URIPath | None,
    qupath_measurement_detection_dir: str | URIPath | None,
    qupath_geojson_detection_dir: str | URIPath | None,
    qupath_geojson_annotation_dir: str | URIPath | None,
    qupath_name_as_class: bool,
    model_info: wsinfer_zoo.client.HFModelTorchScript | LocalModelTorchScript,
    halo_size_px: int = 46,
    batch_size: int | None = None,
    num_workers: int = 4,
    # speedup: bool = False,
    # patch_overlap_median_filter_size: int = 3,
    stain_normalization: bool = False,
    object_based: bool = False,
    object_detection: str = None,
    mixed_precision: bool = False,
    stitch_workers: int | None = None,
    overwrite: bool = False,
    region_overwrite: bool = False,
    pin_memory: bool = True,
) -> tuple[list[str], list[str]]:
    """Run batched model inference on precomputed patches and emit CSV outputs.

    The function expects `results_dir` to contain `patches/*.h5` files generated by
    `wsinsight patch`. Each slide is loaded, optionally stain-normalized, passed
    through the requested model (registered, custom, or pseudo-model), and written to
    `{results_dir}/model-outputs-csv/<slide>.csv`. When QuPath-derived pseudo models
    are used, detection metadata is remapped into the standard probability columns.

    Parameters
    ----------
    wsi_dir
        Optional directory containing original WSIs; only needed when validating
        filesystem inputs for legacy workflows.
    slide_paths
        Optional explicit list of slides to process; when provided, patch HDF5 files
        are filtered to these stems.
    results_dir
        Directory that holds patch artifacts plus the destination for inference CSVs.
    region_inference_dir
        Results directory from a prior region-level (patch-based) wsinsight run
        containing a model-outputs-csv/ folder. Requires ``object_based=True``:
        each detected object is spatially matched to its enclosing region and the
        region's class probabilities are added as ``region_prob_*`` columns in
        the output CSV.
    qupath_* arguments
        Directories containing QuPath detections/geojsons when synthesizing pseudo
        models; mutually exclusive with standard model inputs.
    model_info
        Loaded model metadata (registered HF model or local TorchScript bundle).
        stitch_workers
            Optional thread pool size for TileRemapStitcher when stitching instance
            polygons in object-based end-to-end runs.
    halo_size_px, batch_size, num_workers, speedup, stain_normalization,
    object_based, object_detection, mixed_precision
        Execution knobs controlling tile geometry, dataloader behavior, inference
        optimizations, and object-detection specific flows.

    Returns
    -------
    tuple[list[str], list[str]]
        Two lists containing slide identifiers whose patch ingestion failed and whose
        inference phase failed, respectively.
    """
    # Make sure required directories exist.

    if wsi_dir:
        if not wsi_dir.exists():
            raise errors.WholeSlideImageDirectoryNotFound(
                f"directory not found: {wsi_dir}"
            )
        # wsi_paths = [p for p in tqdm.tqdm(wsi_dir.iterdir(), desc="") if p.is_file()]

        # if not wsi_paths:
        #     raise errors.WholeSlideImagesNotFound(wsi_dir)

        _validate_wsi_directory(wsi_dir)

    if not results_dir.exists():
        raise errors.ResultsDirectoryNotFound(results_dir)

    # Check patches directory.
    patch_dir = results_dir / "patches"
    if not patch_dir.exists():
        raise errors.PatchDirectoryNotFound(
            "The 'patches' directory was not found in results directory. This can"
            " happen for a few reasons: 1) no tissue was detected in the slides,"
            " 2) the physical spacing (MPP) could not be read from any of the slides"
            ", or 3) something else... Please read the logs above for potential errors."
        )
    # Create the patch paths based on the whole slide image paths. In effect, only
    # create patch paths if the whole slide image patch exists.

    # patch_paths = [patch_dir / p.with_suffix(".h5").name for p in wsi_paths]
    patch_paths = [p for p in patch_dir.iterdir() if p.is_file()]

    if slide_paths:
        patch_paths = [
            p for p in patch_paths if p.stem in {s.stem for s in slide_paths}
        ]

    model_output_dir = results_dir / "model-outputs-csv"
    model_output_dir.mkdir(exist_ok=True)

    # model = get_pretrained_torch_module(model=model_info)
    # model.eval()

    # Set the device.
    # Use CPU if env var specifies. Prefer checking if this is false, because
    # if the var is set to almost anything (other than 0, False, f), then it
    # should be true.
    # This env var was introduced mainly for continuous integration tests that
    # failed on apple silicon in github actions. Forcing to cpu avoids failures.
    if os.getenv("WSINFER_FORCE_CPU", "0").lower() not in {"0", "f", "false"}:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        # if torch.cuda.device_count() > 1 and not end_to_end:
        #     model = torch.nn.DataParallel(model)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f'Using device "{device}"')

    if (
        qupath_measurement_detection_dir is None
        and qupath_geojson_detection_dir is None
        and qupath_geojson_annotation_dir is None
    ):
        # model.to(device)
        model = get_pretrained_torch_module(
            model=model_info,
            # device=device,
        )

        model.eval()

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        # if speedup:
        #     if TYPE_CHECKING:
        #         model = type_cast(torch.nn.Module, jit_compile(model))
        #     else:
        #         model = jit_compile(model)

        transform = make_compose_from_transform_config(model_info.config.transform)

    else:
        transform = None

    failed_patching = [p.stem for p in patch_paths if not p.exists()]
    failed_inference: list[str] = []

    # Determine starting batch size and binary-search ceiling (_bs_max).
    # When batch_size is None, auto-calibrate from VRAM using two-point slope
    # calibration (cancels fixed cuDNN workspace overhead).  When explicitly
    # provided, use that value as both the start and the ceiling — preserving
    # the previous behaviour for users who pin a specific batch size.
    # _bytes_per_image is stored so per-slide recalibration (after stitcher
    # allocation) can re-estimate the batch ceiling without extra forward passes.
    if batch_size is None and transform is not None and torch.cuda.is_available():
        _bs_max, _bytes_per_image = _calibrate_inference_batch_size(
            model, transform, model_info.config.patch_size_pixels, device
        )
    else:
        _bs_max = batch_size if batch_size is not None else 32
        _bytes_per_image = 0

    # Binary-search brackets shared across slides.
    # _bs_lo: largest batch size confirmed safe (0 = not yet known).
    # _bs_hi: smallest batch size confirmed OOM (_bs_max+1 = none seen yet).
    current_batch_size: int = _bs_max
    _bs_lo: int = 0
    _bs_hi: int = _bs_max + 1

    # Worker-death recovery state (persists across slides).
    # When DataLoader workers are killed by the OOM killer, we first disable
    # pin_memory, then halve num_workers on subsequent failures.
    _effective_pin_memory = pin_memory
    _effective_num_workers = num_workers

    def _is_worker_killed(err: Exception) -> bool:
        """Detect if a RuntimeError is due to DataLoader worker being killed."""
        msg = str(err).lower()
        return "worker" in msg and "killed" in msg

    # for i, (wsi_path, patch_path) in enumerate(zip(wsi_paths, patch_paths)):
    #     slide_csv_name = Path(wsi_path).with_suffix(".csv").name
    #     slide_csv = model_output_dir / slide_csv_name
    #     if slide_csv.exists():
    #         print("Output CSV exists... skipping.")
    #         print(slide_csv)
    #         continue
    #
    #     if not patch_path.exists():
    #         print(f"Skipping because patch file not found: {patch_path}")
    #         continue

    # with tqdm.tqdm(total=len(wsi_paths), desc="Images", position=0) as pbar:
    #     for _, (wsi_path, patch_path) in enumerate(zip(wsi_paths, patch_paths)):

    with tqdm.tqdm(total=len(patch_paths), desc="Images", position=0) as pbar:
        for _, patch_path in enumerate(patch_paths):
            raise_if_cancelled()
            with h5py.File(patch_path, "r") as f:
                use_hdf5_images = "/images" in f
                g_slide = f["/slide"]
                wsi_path = URIPath(g_slide.attrs["slide_path"])
                mpp = g_slide.attrs["slide_mpp"]
                slide_width = g_slide.attrs["slide_width"]
                slide_height = g_slide.attrs["slide_height"]

            # print(f"Slide {i+1} of {len(wsi_paths)}")
            # print(f" Slide path: {wsi_path}")
            # print(f" Patch path: {patch_path}")

            slide_csv_name = wsi_path.with_suffix(".csv").name
            slide_csv = model_output_dir / slide_csv_name
            if not overwrite and slide_csv.exists():
                print("Output CSV exists... skipping (use --overwrite to regenerate).")
                print(slide_csv)
                pbar.update(1)
                continue

            if not patch_path.exists():
                print(f"Skipping because patch file not found: {patch_path}")
                pbar.update(1)
                continue

            if stain_normalization:
                try:
                    stain_normalization_dset = WholeSlideImagePatches(
                        wsi_path=wsi_path,
                        patch_path=patch_path,
                        use_hdf5_images=use_hdf5_images,
                    )
                except Exception as exc:
                    logger.warning(
                        "Stain normalization failed for %s: %s", wsi_path.stem, exc
                    )
                    failed_inference.append(wsi_path.stem)
                    pbar.update(1)
                    continue

                # The worker_init_fn does not seem to be used when num_workers=0
                # so we call it manually to finish setting up the dataset.

                if num_workers == 0:
                    stain_normalization_dset.worker_init()

                stain_normalization_loader = torch.utils.data.DataLoader(
                    stain_normalization_dset,
                    batch_size=256,
                    shuffle=True,
                    num_workers=1,
                    worker_init_fn=stain_normalization_dset.worker_init,
                    # multiprocessing_context="spawn",
                )

                stain_normalization_batch_imgs, _ = next(
                    iter(stain_normalization_loader)
                )
                stain_normalization_batch_imgs = (
                    stain_normalization_batch_imgs.numpy()
                    .transpose((0, 2, 3, 1))
                    .reshape(-1, 3)
                )
                W_est = htk.preprocessing.color_deconvolution.rgb_separate_stains_macenko_pca(
                    stain_normalization_batch_imgs + EPSILON, I_0
                )
                stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
                stains = ["eosin", "hematoxylin", "null"]
                W_def = np.array([stain_color_map[st] for st in stains]).T
            else:
                W_est = W_def = None

            try:
                dset = WholeSlideImagePatches(
                    wsi_path=wsi_path,
                    patch_path=patch_path,
                    use_hdf5_images=use_hdf5_images,
                    transform=transform,
                    W_est=W_est,
                    W_def=W_def,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to create dataset for %s: %s", wsi_path.stem, exc
                )
                failed_inference.append(wsi_path.stem)
                pbar.update(1)
                continue

            # The worker_init_fn does not seem to be used when num_workers=0
            # so we call it manually to finish setting up the dataset.

            if num_workers == 0:
                dset.worker_init()

            loader = torch.utils.data.DataLoader(
                dset,
                batch_size=current_batch_size,
                shuffle=False,
                num_workers=_effective_num_workers,
                worker_init_fn=dset.worker_init,
                pin_memory=_effective_pin_memory
                and (torch.cuda.is_available() or torch.backends.mps.is_available()),
                persistent_workers=_effective_num_workers > 0,
                prefetch_factor=2 if _effective_num_workers > 0 else None,
                multiprocessing_context="spawn",
            )

            # slide_path = Path(wsi_path)
            #
            # if wsi_dir:
            #     mpp = get_avg_mpp(slide_path)
            #     slide = get_wsi_cls()(wsi_path)
            #     slide_width, slide_height = slide.dimensions

            # target_step_px = int(round(model_info.config.patch_size_pixels * model_info.config.spacing_um_px / mpp))
            model_output_size_px = (
                model_info.config.patch_size_pixels - 2 * halo_size_px
            )
            slide_patch_size = int(
                round(model_output_size_px * model_info.config.spacing_um_px / mpp)
            )
            slide_halo_size = int(
                round(halo_size_px * model_info.config.spacing_um_px / mpp)
            )

            # Store the coordinates and model probabiltiies of each patch in this slide.
            # This lets us know where the probabiltiies map to in the slide.
            slide_coords: list[npt.NDArray[np.integer]] = []
            slide_probs: list[npt.NDArray[np.floating]] = []
            slide_superior_structure = None  # initialized here; set per-branch below

            if (
                object_based
                and qupath_measurement_detection_dir is not None
                and qupath_geojson_detection_dir is None
                and qupath_geojson_annotation_dir is None
            ):
                slide_det_name = wsi_path.with_suffix(".txt").name
                slide_det = qupath_measurement_detection_dir / slide_det_name

                if not slide_det.exists():
                    failed_inference.append(wsi_path.stem)
                    continue

                qpdet_df = pd.read_csv(slide_det, delimiter="\t")
                # Keep only detection/cell objects so coords, labels and N stay aligned.
                qpdet_df = qpdet_df[
                    (qpdet_df["Object type"] == "Detection")
                    | (qpdet_df["Object type"] == "Cell")
                ].reset_index(drop=True)
                width = model_info.config.patch_size_pixels
                height = model_info.config.patch_size_pixels
                half_patch_size = round(model_info.config.patch_size_pixels / 2)

                x = np.rint(qpdet_df["Centroid X µm"] / mpp - half_patch_size).astype(
                    np.int32
                )
                y = np.rint(qpdet_df["Centroid Y µm"] / mpp - half_patch_size).astype(
                    np.int32
                )

                # shape: (N, 4)
                coords = np.column_stack(
                    [x, y, np.full_like(x, width), np.full_like(y, height)]
                )
                slide_coords = [row[np.newaxis, :] for row in coords]

                indexer = (
                    pd.Index(model_info.config.class_names).get_indexer(
                        qpdet_df["Name"].str.strip().str.replace(" ", "_").str.lower()
                    )
                    if qupath_name_as_class
                    else pd.Index(model_info.config.class_names).get_indexer(
                        qpdet_df["Classification"]
                        .str.strip()
                        .str.replace(" ", "_")
                        .str.lower()
                    )
                )

                N = len(qpdet_df)
                K = len(model_info.config.class_names)
                probs = np.zeros((N, K), dtype=np.float32)

                valid = indexer >= 0
                rows = np.nonzero(valid)[0]
                cols = indexer[valid]
                probs[rows, cols] = 1.0

                slide_probs = [row[np.newaxis, :] for row in probs]

                slide_superior_structure = qpdet_df["Parent"]

            elif (
                object_based
                and qupath_measurement_detection_dir is None
                and qupath_geojson_detection_dir is not None
                and qupath_geojson_annotation_dir is None
            ):
                patch_size = model_info.config.patch_size_pixels
                half_patch_size = round(patch_size / 2)

                slide_geojson_name = wsi_path.with_suffix(".geojson").name
                slide_geojson = qupath_geojson_detection_dir / slide_geojson_name

                if not slide_geojson.exists():
                    failed_inference.append(wsi_path.stem)
                    continue

                gdf = gpd.read_file(slide_geojson)
                gdf.set_crs(None, allow_override=True)
                # Keep only detection/cell objects so coords, labels and N stay aligned.
                gdf = gdf[
                    (gdf["objectType"] == "detection") | (gdf["objectType"] == "cell")
                ].reset_index(drop=True)
                # model_info.config.class_names = \
                #     gdf.name.str.strip().str.replace(" ", "_", regex=False).str.lower().unique().tolist() \
                #     if qupath_name_as_class else \
                #     gdf.classification.str.strip().str.replace(" ", "_", regex=False).str.lower().unique().tolist()

                # prob_cols = [col for col in gdf.name.unique().tolist()] \
                #     if qupath_name_as_class else \
                #     [col for col in gdf.classification.unique().tolist()]

                # --- coords (vectorized) ---
                # constants
                width = model_info.config.patch_size_pixels
                height = model_info.config.patch_size_pixels
                half_patch_size = round(model_info.config.patch_size_pixels / 2)

                x = (gdf.geometry.centroid.x / mpp) - half_patch_size
                y = (gdf.geometry.centroid.y / mpp) - half_patch_size

                x = x.to_numpy().round().astype(np.int32)
                y = y.to_numpy().round().astype(np.int32)

                # shape: (N, 4)
                coords = np.column_stack(
                    [x, y, np.full_like(x, width), np.full_like(y, height)]
                )
                slide_coords = [row[np.newaxis, :] for row in coords]

                indexer = (
                    pd.Index(model_info.config.class_names).get_indexer(
                        gdf["name"].str.strip().str.replace(" ", "_").str.lower()
                    )
                    if qupath_name_as_class
                    else pd.Index(model_info.config.class_names).get_indexer(
                        gdf["classification"]
                        .str.strip()
                        .str.replace(" ", "_")
                        .str.lower()
                    )
                )

                N = len(gdf)
                K = len(model_info.config.class_names)
                probs = np.zeros((N, K), dtype=np.float32)

                valid = indexer >= 0
                rows = np.nonzero(valid)[0]
                cols = indexer[valid]
                probs[rows, cols] = 1.0

                slide_probs = [row[np.newaxis, :] for row in probs]

                slide_superior_structure = None

            elif not object_based and qupath_geojson_annotation_dir:
                patch_size = model_info.config.patch_size_pixels
                half_patch_size = round(patch_size / 2)

                slide_geojson_name = wsi_path.with_suffix(".geojson").name
                slide_geojson = qupath_geojson_annotation_dir / slide_geojson_name

                if not slide_geojson.exists():
                    failed_inference.append(wsi_path.stem)
                    pbar.update(1)
                    continue

                # TODO: annotation-guided inference is not yet implemented.
                # When the annotation GeoJSON exists, inference results should be
                # filtered to patches that intersect the annotated regions.
                # For now we record the slide as failed so it is not silently skipped.
                logger.warning(
                    f"Annotation-guided inference is not implemented; "
                    f"skipping slide {wsi_path.stem}"
                )
                failed_inference.append(wsi_path.stem)
                pbar.update(1)
                continue

            elif object_based and object_detection == "end2end":
                _oom_skip = False
                # Counts OOMs at the *current* batch size for *this slide*.
                # The first OOM at a converged batch size is treated as a
                # likely fragmentation event and retried in place after
                # `empty_cache()`. Only the second one triggers a bisection
                # reset.
                _oom_retries_at_current = 0
                while True:
                    stitcher = TileRemapStitcher(
                        n_classes=len(model_info.config.class_names),
                        slide_width=slide_width,
                        slide_height=slide_height,
                        slide_patch_size=slide_patch_size,
                        slide_halo_size=slide_halo_size,
                        slide_mpp=mpp,
                        model_mpp=model_info.config.spacing_um_px,
                        min_object_size=20,
                        device=device,
                    )
                    # Re-estimate batch ceiling NOW that the stitcher is on GPU.
                    # The initial calibration measured VRAM before stitcher
                    # allocation; re-reading available VRAM here gives the true
                    # remaining capacity for batch activations.  No extra forward
                    # passes needed — reuse the stored bytes_per_image slope.
                    if _bytes_per_image > 0:
                        torch.cuda.empty_cache()
                        _n_gpu = max(1, torch.cuda.device_count())
                        _slide_bs_max = _batch_size_for_available_vram(
                            _bytes_per_image, _n_gpu
                        )
                        if _slide_bs_max < current_batch_size:
                            current_batch_size = _slide_bs_max
                            _bs_lo = 0
                            _bs_hi = _slide_bs_max + 1
                    try:
                        with tqdm.tqdm(
                            total=len(loader), desc="Inference", position=1, leave=False
                        ) as qbar:
                            for batch_imgs, batch_coords in loader:
                                if is_cancelled():
                                    break
                                assert (
                                    batch_imgs.shape[0] == batch_coords.shape[0]
                                ), "length mismatch"
                                if mixed_precision:
                                    with torch.no_grad():
                                        with torch.autocast(
                                            device_type=device.type, dtype=torch.float16
                                        ):
                                            pred_dict = model(
                                                batch_imgs.to(device, non_blocking=True)
                                            )
                                else:
                                    with torch.no_grad():
                                        pred_dict = model(
                                            batch_imgs.to(device, non_blocking=True)
                                        )
                                stitcher.accumulate_batch_torch(
                                    pred_dict, batch_coords.to(device)
                                )
                                qbar.update(1)
                                gc.collect()
                        raise_if_cancelled()
                        with tqdm.tqdm(
                            desc="Stitching",
                            mininterval=0,
                            miniters=1,
                            smoothing=0,
                            dynamic_ncols=True,
                            position=1,
                            leave=False,
                        ) as qbar:
                            slide_coords, slide_probs, slide_polys = stitcher.finalize(
                                pbar=qbar,
                                num_workers=stitch_workers,
                            )
                            qbar.close()
                        gc.collect()
                        current_batch_size, _bs_lo, _bs_hi = _advance_batch_search(
                            current_batch_size, _bs_lo, _bs_hi, _bs_max, oom=False
                        )
                        break
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as _oom_err:
                        _is_cuda_oom = (
                            isinstance(_oom_err, torch.cuda.OutOfMemoryError)
                            or "out of memory" in str(_oom_err).lower()
                        )
                        _is_worker_death = _is_worker_killed(_oom_err)
                        if (
                            isinstance(_oom_err, RuntimeError)
                            and not _is_cuda_oom
                            and not _is_worker_death
                        ):
                            raise
                        # Handle worker death (system OOM killer) separately.
                        if _is_worker_death:
                            if _effective_pin_memory:
                                _effective_pin_memory = False
                                tqdm.tqdm.write(
                                    f"[WorkerKilled] Disabling pin_memory — retrying {wsi_path.stem}"
                                )
                            elif _effective_num_workers > 0:
                                _effective_num_workers = max(
                                    0, _effective_num_workers // 2
                                )
                                tqdm.tqdm.write(
                                    f"[WorkerKilled] Reducing num_workers → {_effective_num_workers} — retrying {wsi_path.stem}"
                                )
                            else:
                                tqdm.tqdm.write(
                                    f"[WorkerKilled] Already at num_workers=0, pin_memory=False — skipping {wsi_path.stem}"
                                )
                                failed_inference.append(wsi_path.stem)
                                _oom_skip = True
                                break
                            loader = torch.utils.data.DataLoader(
                                dset,
                                batch_size=current_batch_size,
                                shuffle=False,
                                num_workers=_effective_num_workers,
                                worker_init_fn=dset.worker_init,
                                pin_memory=_effective_pin_memory
                                and (
                                    torch.cuda.is_available()
                                    or torch.backends.mps.is_available()
                                ),
                                persistent_workers=_effective_num_workers > 0,
                                prefetch_factor=2
                                if _effective_num_workers > 0
                                else None,
                                multiprocessing_context="spawn",
                            )
                            continue
                        # CUDA OOM handling (existing logic).
                        torch.cuda.empty_cache()
                        gc.collect()
                        if current_batch_size <= 1:
                            tqdm.tqdm.write(
                                f"[OOM] batch_size=1 still OOM — skipping {wsi_path.stem}"
                            )
                            failed_inference.append(wsi_path.stem)
                            _oom_skip = True
                            break
                        if (
                            _is_bs_converged(current_batch_size, _bs_lo, _bs_hi)
                            and _oom_retries_at_current == 0
                        ):
                            # Likely fragmentation. Retry in place at the same
                            # batch size after the cache flush above.
                            _oom_retries_at_current = 1
                            tqdm.tqdm.write(
                                f"[OOM] empty_cache retry at bs={current_batch_size} — retrying {wsi_path.stem}"
                            )
                        elif _is_bs_converged(current_batch_size, _bs_lo, _bs_hi):
                            # Second OOM at the converged value: fragmentation
                            # was not the culprit. Reset the bracket so the
                            # next attempt drops by 1/φ instead of 1/2, then
                            # re-bisects upward.
                            _old_bs = current_batch_size
                            _bs_hi = current_batch_size
                            _bs_lo = 0
                            _PHI = (1.0 + 5.0**0.5) / 2.0
                            current_batch_size = max(1, int(_bs_hi / _PHI))
                            _oom_retries_at_current = 0
                            tqdm.tqdm.write(
                                f"[OOM] bisection reset (lo=0 hi={_old_bs}) → bs={current_batch_size} — retrying {wsi_path.stem}"
                            )
                        else:
                            current_batch_size, _bs_lo, _bs_hi = _advance_batch_search(
                                current_batch_size, _bs_lo, _bs_hi, _bs_max, oom=True
                            )
                            _oom_retries_at_current = 0
                            tqdm.tqdm.write(
                                f"[OOM] batch_size → {current_batch_size} (lo={_bs_lo} hi={_bs_hi}) — retrying {wsi_path.stem}"
                            )
                        loader = torch.utils.data.DataLoader(
                            dset,
                            batch_size=current_batch_size,
                            shuffle=False,
                            num_workers=_effective_num_workers,
                            worker_init_fn=dset.worker_init,
                            pin_memory=_effective_pin_memory
                            and (
                                torch.cuda.is_available()
                                or torch.backends.mps.is_available()
                            ),
                            persistent_workers=_effective_num_workers > 0,
                            prefetch_factor=2 if _effective_num_workers > 0 else None,
                            multiprocessing_context="spawn",
                        )
                if _oom_skip:
                    pbar.update(1)
                    continue

                #     if slide_polys is not None and len(slide_polys) > 0:
                #         with h5py.File(patch_path, "a") as f:
                #             # f.create_dataset("/polygons", data=polygons, compression=compression)
                #             # object_xy_list: List[np.ndarray], each arr shape (Ni, 2), dtype float32 (or castable)
                #             lengths = np.array([xy.shape[0] for xy in slide_polys], dtype=np.int64)
                #             offsets = np.concatenate(([0], np.cumsum(lengths)))
                #             coords  = np.vstack(slide_polys).astype(np.float32) if lengths.sum() > 0 \
                #                       else np.zeros((0, 2), np.float32)
                #
                #             if "/polygons" in f:
                #                 del f["/polygons"]
                #
                #             g = f.create_group("/polygons")
                #
                #             d_coords = g.create_dataset(
                #                 "coords", data=coords, dtype="float32",
                #                 compression=f["/coords"].compression, shuffle=True, chunks=True  # good defaults
                #             )
                # #            d_offsets = g.create_dataset("offsets", data=offsets, dtype="int64")
                #             g.create_dataset("offsets", data=offsets, dtype="int64")
                #
                #             # Helpful metadata
                #             g.attrs["layout"] = "ragged_offsets"
                #             d_coords.attrs["columns"] = np.array(["x", "y"], dtype="S1")

                if slide_polys is not None and len(slide_polys) > 0:
                    # slide_polys: list of (Ni, 2) arrays (x, y) in slide coords
                    lengths = np.array(
                        [xy.shape[0] for xy in slide_polys], dtype=np.int64
                    )
                    total_len = int(lengths.sum())
                    offsets = np.concatenate(([0], np.cumsum(lengths)))

                    poly_coords = (
                        np.vstack(slide_polys).astype(np.float32)
                        if total_len > 0
                        else np.zeros((0, 2), np.float32)
                    )

                    with patch_path.open(
                        "rb+" if patch_path.exists() else "wb+"
                    ) as fh:  # URIPath 提供本地缓存并在关闭时同步回远端
                        with h5py.File(fh, "a") as f:
                            # Try to mirror coords compression; fall back to None if missing
                            coords_dset = f["/coords"]
                            poly_compression = coords_dset.compression

                            # Replace any existing polygons group
                            if "/polygons" in f:
                                del f["/polygons"]

                            g = f.create_group("/polygons")

                            d_coords = g.create_dataset(
                                "coords",
                                data=poly_coords,
                                dtype="float32",
                                compression=poly_compression,
                                shuffle=True,
                                chunks=True,
                            )
                            g.create_dataset("offsets", data=offsets, dtype="int64")

                            # Keep metadata consistent with save_hdf5()
                            g.attrs["layout"] = "ragged_offsets"
                            d_coords.attrs["columns"] = np.array(["x", "y"], dtype="S1")

                slide_superior_structure = None

            else:
                _oom_skip = False
                # See end2end branch above for rationale: first OOM at a
                # converged batch size is treated as fragmentation and retried
                # in place; second OOM triggers a bisection reset.
                _oom_retries_at_current = 0
                while True:
                    slide_coords = []
                    slide_probs = []
                    try:
                        with tqdm.tqdm(
                            total=len(loader), position=1, leave=False
                        ) as qbar:
                            for batch_imgs, batch_coords in loader:
                                if is_cancelled():
                                    break
                                assert (
                                    batch_imgs.shape[0] == batch_coords.shape[0]
                                ), "length mismatch"
                                with torch.no_grad():
                                    logits: torch.Tensor = (
                                        _as_output_tensor(
                                            model(
                                                batch_imgs.to(device, non_blocking=True)
                                            )
                                        )
                                        .detach()
                                        .cpu()
                                    )
                                # probs has shape (batch_size, num_classes) or (batch_size,)
                                if len(logits.shape) > 1 and logits.shape[1] > 1:
                                    probs = torch.nn.functional.softmax(logits, dim=1)
                                else:
                                    probs = torch.sigmoid(logits.squeeze(1))
                                # Cloning prevents memory accumulation / "Too many open files" on WSL.
                                slide_coords.append(batch_coords.clone().numpy())
                                slide_probs.append(probs.numpy())
                                qbar.update(1)
                        raise_if_cancelled()
                        current_batch_size, _bs_lo, _bs_hi = _advance_batch_search(
                            current_batch_size, _bs_lo, _bs_hi, _bs_max, oom=False
                        )
                        break
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as _oom_err:
                        _is_cuda_oom = (
                            isinstance(_oom_err, torch.cuda.OutOfMemoryError)
                            or "out of memory" in str(_oom_err).lower()
                        )
                        _is_worker_death = _is_worker_killed(_oom_err)
                        if (
                            isinstance(_oom_err, RuntimeError)
                            and not _is_cuda_oom
                            and not _is_worker_death
                        ):
                            raise
                        # Handle worker death (system OOM killer) separately.
                        if _is_worker_death:
                            if _effective_pin_memory:
                                _effective_pin_memory = False
                                tqdm.tqdm.write(
                                    f"[WorkerKilled] Disabling pin_memory — retrying {wsi_path.stem}"
                                )
                            elif _effective_num_workers > 0:
                                _effective_num_workers = max(
                                    0, _effective_num_workers // 2
                                )
                                tqdm.tqdm.write(
                                    f"[WorkerKilled] Reducing num_workers → {_effective_num_workers} — retrying {wsi_path.stem}"
                                )
                            else:
                                tqdm.tqdm.write(
                                    f"[WorkerKilled] Already at num_workers=0, pin_memory=False — skipping {wsi_path.stem}"
                                )
                                failed_inference.append(wsi_path.stem)
                                _oom_skip = True
                                break
                            loader = torch.utils.data.DataLoader(
                                dset,
                                batch_size=current_batch_size,
                                shuffle=False,
                                num_workers=_effective_num_workers,
                                worker_init_fn=dset.worker_init,
                                pin_memory=_effective_pin_memory
                                and (
                                    torch.cuda.is_available()
                                    or torch.backends.mps.is_available()
                                ),
                                persistent_workers=_effective_num_workers > 0,
                                prefetch_factor=2
                                if _effective_num_workers > 0
                                else None,
                                multiprocessing_context="spawn",
                            )
                            continue
                        # CUDA OOM handling (existing logic).
                        torch.cuda.empty_cache()
                        gc.collect()
                        if current_batch_size <= 1:
                            tqdm.tqdm.write(
                                f"[OOM] batch_size=1 still OOM — skipping {wsi_path.stem}"
                            )
                            failed_inference.append(wsi_path.stem)
                            _oom_skip = True
                            break
                        if (
                            _is_bs_converged(current_batch_size, _bs_lo, _bs_hi)
                            and _oom_retries_at_current == 0
                        ):
                            _oom_retries_at_current = 1
                            tqdm.tqdm.write(
                                f"[OOM] empty_cache retry at bs={current_batch_size} — retrying {wsi_path.stem}"
                            )
                        elif _is_bs_converged(current_batch_size, _bs_lo, _bs_hi):
                            # Second OOM at the converged value: fragmentation
                            # was not the culprit. Reset the bracket so the
                            # next attempt drops by 1/φ instead of 1/2, then
                            # re-bisects upward.
                            _old_bs = current_batch_size
                            _bs_hi = current_batch_size
                            _bs_lo = 0
                            _PHI = (1.0 + 5.0**0.5) / 2.0
                            current_batch_size = max(1, int(_bs_hi / _PHI))
                            _oom_retries_at_current = 0
                            tqdm.tqdm.write(
                                f"[OOM] bisection reset (lo=0 hi={_old_bs}) → bs={current_batch_size} — retrying {wsi_path.stem}"
                            )
                        else:
                            current_batch_size, _bs_lo, _bs_hi = _advance_batch_search(
                                current_batch_size, _bs_lo, _bs_hi, _bs_max, oom=True
                            )
                            _oom_retries_at_current = 0
                            tqdm.tqdm.write(
                                f"[OOM] batch_size → {current_batch_size} (lo={_bs_lo} hi={_bs_hi}) — retrying {wsi_path.stem}"
                            )
                        loader = torch.utils.data.DataLoader(
                            dset,
                            batch_size=current_batch_size,
                            shuffle=False,
                            num_workers=_effective_num_workers,
                            worker_init_fn=dset.worker_init,
                            pin_memory=_effective_pin_memory
                            and (
                                torch.cuda.is_available()
                                or torch.backends.mps.is_available()
                            ),
                            persistent_workers=_effective_num_workers > 0,
                            prefetch_factor=2 if _effective_num_workers > 0 else None,
                            multiprocessing_context="spawn",
                        )
                if _oom_skip:
                    pbar.update(1)
                    continue
                slide_superior_structure = None
            #
            # Until here, we've obtained both slide coords and slide_probs
            #

            if len(slide_coords) == 0:  # No coords are registered
                continue

            slide_coords_arr = np.concatenate(slide_coords, axis=0)
            slide_df = pd.DataFrame(
                dict(
                    minx=slide_coords_arr[:, 0],
                    miny=slide_coords_arr[:, 1],
                    width=slide_coords_arr[:, 2],
                    height=slide_coords_arr[:, 3],
                )
            )
            slide_probs_arr = np.concatenate(slide_probs, axis=0)

            # # I did below for filtering but do not remember if this is actually a good idea.
            # # The goal of this section is to perform a median pass fildering
            # # So that the noises generated from patch prediction can be canceld.
            # # However, by now I believe this possible doesn't provide much information.
            # # In addition, it requires an additional step_size parameter passing through
            # # Loader, that seems not a good idea by now.
            #
            # if not object_based:
            #     tile_indices_arr = slide_coords_arr[:, :2].astype(np.int32)
            #     prob_map = 1.0/float(slide_probs_arr.shape[1])*np.ones((slide_probs_arr.shape[1], dset.tile_dim[0], dset.tile_dim[1]),dtype=np.float32)
            #     # kernel_size = int(round(dset.patch_size / dset.step_size))
            #
            #     for s in range(slide_probs_arr.shape[1]):
            #         for i, (x, y) in enumerate(tile_indices_arr):
            #             prob_map[s, x, y] = slide_probs_arr[i, s]
            #
            #         prob_map[s, :, :] = medfilt2d(prob_map[s, :, :], kernel_size=patch_overlap_median_filter_size)
            #
            #     for s in range(slide_probs_arr.shape[1]):
            #         for i, (x, y) in enumerate(tile_indices_arr):
            #             slide_probs_arr[i, s] = prob_map[s, x, y]
            #
            # # I did above for filtering but do not remember if this u=is actually a good idea.

            # Use 'prob-' prefix for all classes. This should make it clearer that the
            # column has probabilities for the class. It also makes it easier for us to
            # identify columns associated with probabilities.
            prob_colnames = [f"prob_{c}" for c in model_info.config.class_names]
            slide_df.loc[:, prob_colnames] = slide_probs_arr

            # Add a 'classification' column: the argmax class name for each detection.
            # This mirrors QuPath's classification concept and makes results immediately
            # usable without recomputing argmax from the prob_* columns.
            _argmax_idx = slide_probs_arr.argmax(axis=1)
            slide_df["classification"] = np.array(model_info.config.class_names)[
                _argmax_idx
            ]

            if slide_superior_structure is not None:
                slide_df.loc[:, "qupath_detection_parent"] = slide_superior_structure

            if region_inference_dir is not None and object_based:
                annot_csv = region_inference_dir / "model-outputs-csv" / slide_csv_name
                annot_df = pd.read_csv(
                    annot_csv,
                    engine="c",
                    memory_map=True,
                    low_memory=False,
                )
                if not region_overwrite:
                    would_add = {"region_" + c for c in annot_df.columns}
                    already_present = would_add & set(slide_df.columns)
                    if already_present:
                        print(
                            f"WARNING: skipping region registration for {wsi_path.stem} "
                            f"— region_* columns already present. "
                            f"Use --overwrite to replace."
                        )
                        with slide_csv.open("wb") as fh:
                            with critical_section(
                                f"saving inference output for {wsi_path.stem}"
                            ):
                                slide_df.to_csv(fh, index=False)
                        pbar.update(1)
                        continue
                slide_df, _ = register_objects_to_regions(slide_df, annot_df)

            with slide_csv.open("wb") as fh:  # local cache, auto-upload on close
                with critical_section(f"saving inference output for {wsi_path.stem}"):
                    slide_df.to_csv(fh, index=False)
            # print("-" * 40)
            pbar.update(1)

    return failed_patching, failed_inference
