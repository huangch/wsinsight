"""Patch extraction and tissue segmentation routines for wsinsight."""

from __future__ import annotations

from .io import draw_contours_on_thumbnail, extract_patches_from_slide, save_hdf5
from .pipeline import (
    MASKS_DIR,
    PATCHES_DIR,
    segment_and_patch_directory_of_slides,
    segment_and_patch_one_slide,
)

__all__ = [
    "MASKS_DIR",
    "PATCHES_DIR",
    "segment_and_patch_one_slide",
    "segment_and_patch_directory_of_slides",
    "extract_patches_from_slide",
    "save_hdf5",
    "draw_contours_on_thumbnail",
]
