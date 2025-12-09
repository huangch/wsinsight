"""Patch extraction and persistence helpers used by the patch pipeline."""

from __future__ import annotations

import logging
from typing import List, Sequence

import cv2 as cv
import h5py
import numpy as np
import numpy.typing as npt
from PIL import Image

from ..uri_path import URIPath

logger = logging.getLogger(__name__)


def extract_patches_from_slide(
    slide,
    coords: npt.NDArray[np.int_],
    patch_size: int,
) -> npt.NDArray[np.uint8]:
    """Extract RGB patches from a WSI given top-left coordinates at level 0."""

    coords = np.asarray(coords, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be (N, 2), got {coords.shape}")

    n = coords.shape[0]
    images = np.empty((n, patch_size, patch_size, 3), dtype=np.uint8)

    for i, (x, y) in enumerate(coords):
        x_int = int(x)
        y_int = int(y)

        region = slide.read_region(
            location=(x_int, y_int),
            level=0,
            size=(patch_size, patch_size),
        )

        if region.mode != "RGB":
            region = region.convert("RGB")

        images[i] = np.asarray(region, dtype=np.uint8)

    return images


def save_hdf5(
    path: str | URIPath,
    coords: npt.NDArray[np.int_],
    polygons: List[np.ndarray] | None,
    tile_dim: npt.NDArray[np.int_] | None,
    patch_size: int,
    patch_spacing_um_px: float,
    compression: str | None = "gzip",
    images: npt.NDArray[np.uint8] | None = None,
    slide_path: str | None = None,
    slide_mpp: float | None = None,
    slide_width: float | None = None,
    slide_height: float | None = None,
) -> None:
    """Write patch coordinates, optional polygons, and optional image patches to HDF5."""

    logger.info(f"Writing coordinates to disk: {path}")
    logger.info(f"Coordinates have shape {coords.shape}")

    coords = np.asarray(coords, dtype=np.int32)

    if coords.ndim != 2:
        raise ValueError(f"coords must have 2 dimensions but got {coords.ndim}")
    if coords.shape[1] != 2:
        raise ValueError(
            f"length of coords second axis must be 2 but got {coords.shape[1]}"
        )

    if tile_dim is not None and tile_dim.shape != (2,):
        raise ValueError(f"tile_dim must be (2,) but got {tile_dim.shape}")

    if images is not None:
        images = np.asarray(images, dtype=np.uint8)
        if images.shape[0] != coords.shape[0]:
            raise ValueError(
                f"images and coords must have same length; "
                f"got {images.shape[0]} vs {coords.shape[0]}"
            )

    with URIPath(path).open("w+b") as fh:
        with h5py.File(fh, "w") as f:
            g_slide = f.create_group("slide")
            if slide_path is not None:
                g_slide.attrs.create(
                    "slide_path",
                    slide_path,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
            if slide_mpp is not None:
                g_slide.attrs["slide_mpp"] = slide_mpp
            if slide_width is not None:
                g_slide.attrs["slide_width"] = slide_width
            if slide_height is not None:
                g_slide.attrs["slide_height"] = slide_height

            d_coords = f.create_dataset("/coords", data=coords, compression=compression)
            d_coords.attrs["patch_size"] = patch_size
            d_coords.attrs["patch_level"] = 0
            d_coords.attrs["patch_spacing_um_px"] = patch_spacing_um_px
            if tile_dim is not None:
                d_coords.attrs["tile_dim"] = tile_dim

            if images is not None:
                logger.info(f"Writing images dataset with shape {images.shape}")
                f.create_dataset(
                    "/images",
                    data=images,
                    compression=compression,
                    chunks=True,
                )

            if polygons is not None and len(polygons) > 0:
                lengths = np.array([xy.shape[0] for xy in polygons], dtype=np.int64)
                offsets = np.concatenate(([0], np.cumsum(lengths)))
                poly_coords = (
                    np.vstack(polygons).astype(np.float32)
                    if lengths.sum() > 0
                    else np.zeros((0, 2), np.float32)
                )

                g = f.create_group("/polygons")
                d_poly = g.create_dataset(
                    "coords",
                    data=poly_coords,
                    dtype="float32",
                    compression=compression,
                    shuffle=True,
                    chunks=True,
                )
                g.create_dataset("offsets", data=offsets, dtype="int64")

                g.attrs["layout"] = "ragged_offsets"
                d_poly.attrs["columns"] = np.array(["x", "y"], dtype="S1")


def draw_contours_on_thumbnail(
    thumb: Image.Image,
    contours: Sequence[npt.NDArray[np.int_]],
    hierarchy: npt.NDArray[np.int_],
) -> Image.Image:
    """Draw contours onto an image."""

    assert hierarchy.ndim == 3
    assert hierarchy.shape[0] == 1
    assert hierarchy.shape[2] == 4
    assert len(contours) == hierarchy.shape[1]

    contour_is_external = (hierarchy[0, :, 3] < 0).tolist()
    external = [c for c, external in zip(contours, contour_is_external) if external]
    hole = [c for c, external in zip(contours, contour_is_external) if not external]

    img = np.array(thumb)
    cv.drawContours(img, external, -1, (0, 255, 255), 7)
    cv.drawContours(img, hole, -1, (255, 255, 0), 7)

    return Image.fromarray(img).convert("RGB")
