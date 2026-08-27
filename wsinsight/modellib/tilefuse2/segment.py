"""Watershed nucleus post-processing (ported verbatim from ``tilefuse``).

Kept numerically identical to v1 on purpose: the v2 rewrite changes *which*
region a tile owns, not how a tile is segmented.  Any GPU port lands here later
and must be validated against this implementation.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

try:
    cv2.setNumThreads(1)
except Exception:
    pass


def proc_np_hv(
    np_map: np.ndarray, hv_map: np.ndarray, min_object_size: int
) -> np.ndarray:
    """Return an int32 instance map for one tile."""
    H, W = np_map.shape[:2]

    # 1) foreground
    blb_bin = (np_map >= 0.5).astype(np.uint8)
    if blb_bin.sum() == 0:
        return np.zeros((H, W), dtype=np.int32)

    labeled, num = ndi.label(blb_bin)
    if num > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            labeled = remove_small_objects(labeled, min_size=int(min_object_size))
    blb = (labeled > 0).astype(np.uint8)
    if blb.sum() == 0:
        return np.zeros((H, W), dtype=np.int32)

    # 2) HV normalize + edges
    h_dir = cv2.normalize(
        hv_map[:, :, 0],
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    v_dir = cv2.normalize(
        hv_map[:, :, 1],
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1.0 - cv2.normalize(
        sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    sobelv = 1.0 - cv2.normalize(
        sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1.0 - blb)  # suppress background
    overall[overall < 0] = 0

    # 3) distance (basins)
    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    # 4) markers
    ridge = (overall >= 0.4).astype(np.uint8)
    marker = blb.astype(np.int16) - ridge.astype(np.int16)
    marker = np.clip(marker, 0, 1).astype(np.uint8)
    if marker.any():
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker, mnum = ndi.label(marker)
        if mnum > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                marker = remove_small_objects(marker, min_size=int(min_object_size))
    else:
        marker, _ = ndi.label(blb)

    # 5) watershed
    proced_pred = watershed(dist, markers=marker, mask=blb.astype(bool))
    return proced_pred.astype(np.int32)
