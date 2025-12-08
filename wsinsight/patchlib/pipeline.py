"""High-level segmentation and patch extraction pipeline for whole-slide images."""

from __future__ import annotations

import itertools
import logging
import os
import os.path
from typing import List, Tuple

import cv2 as cv
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
import tqdm
from PIL import Image
from absl import logging as absl_logging
from csbdeep.utils import normalize
from shapely.geometry import Polygon, MultiPolygon
from stardist.models import StarDist2D
from tifffile import imread

from ..wsi import _validate_wsi_directory, get_avg_mpp, get_wsi_cls
from ..uri_path import URIPath
from .io import draw_contours_on_thumbnail, extract_patches_from_slide, save_hdf5
from .patch import (
    get_multipolygon_from_binary_arr,
    get_object_coordinates_within_polygon,
    get_patch_coordinates_within_polygon,
)
from .segment import segment_tissue

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logger = logging.getLogger(__name__)
tf.get_logger().setLevel(logging.ERROR)
absl_logging.set_verbosity(absl_logging.ERROR)

MASKS_DIR = "masks"
PATCHES_DIR = "patches"


def segment_and_patch_one_slide(
    slide_path: URIPath,
    save_dir: URIPath,
    qupath_detection_dir: URIPath,
    qupath_geojson_detection_dir: URIPath,
    qupath_geojson_annotation_dir: URIPath,
    patch_size_px: int,
    patch_spacing_um_px: float,
    halo_size_px: int = 0,
    histoqc_dir: str | URIPath | None = None,
    thumbsize: tuple[int, int] = (2048, 2048),
    median_filter_size: int = 7,
    binary_threshold: int = 7,
    closing_kernel_size: int = 6,
    min_object_size_um2: float = 200 ** 2,
    min_hole_size_um2: float = 190 ** 2,
    overlap: float = 0.0,
    object_based: bool = False,
    object_detection: str | None = None,
    stardist_normalization_pmin: float = 1.0,
    stardist_normalization_pmax: float = 99.8,
    cache_image_patches: bool = False,
) -> None:
    """Get non-overlapping patch coordinates in tissue regions for a slide."""

    slide_prefix = slide_path.stem

    logger.info(f"Segmenting and patching slide {slide_path}")
    logger.info(f"Using prefix as slide ID: {slide_prefix}")

    patch_path = save_dir / PATCHES_DIR / f"{slide_prefix}.h5"
    mask_path = save_dir / MASKS_DIR / f"{slide_prefix}.jpg"

    if patch_path.exists() and mask_path.exists():
        logger.info("Patch output and mask output files already exist")
        logger.info(f"patch_path={patch_path}")
        logger.info(f"mask_path={mask_path}")
        return None

    slide = get_wsi_cls()(slide_path)
    mpp = get_avg_mpp(slide_path)
    logger.info(f"Slide has WxH {slide.dimensions} and MPP={mpp}")

    logger.info(
        f"Requested patch size of {patch_size_px} px at {patch_spacing_um_px} um/px"
    )
    logger.info(
        f"Scaling patch size by {patch_spacing_um_px / mpp} for patch coordinates at"
        f" level 0 (MPP={mpp}) | patch_spacing_um_px / mpp"
        f" ({patch_spacing_um_px} / {mpp})"
    )
    patch_size = int(round(patch_size_px * patch_spacing_um_px / mpp))

    logger.info(f"Final patch size is {patch_size}")

    if len(thumbsize) != 2:
        raise ValueError(f"Length of 'thumbsize' must be 2 but got {len(thumbsize)}")
    thumb: Image.Image = slide.get_thumbnail(thumbsize)
    if thumb.mode != "RGB":
        logger.warning(f"Converting mode of thumbnail from {thumb.mode} to RGB")
        thumb = thumb.convert("RGB")

    thumb_mpp = (mpp * (np.array(slide.dimensions) / thumb.size)).mean()
    logger.info(f"Thumbnail has WxH {thumb.size} and MPP={thumb_mpp}")
    thumb_mpp_squared: float = thumb_mpp ** 2

    min_object_size_px: int = round(min_object_size_um2 / thumb_mpp_squared)
    min_hole_size_px: int = round(min_hole_size_um2 / thumb_mpp_squared)

    logger.info(
        f"Transformed minimum object size to {min_object_size_px} pixel area in"
        " thumbnail"
    )
    logger.info(
        f"Transformed minimum hole size to {min_hole_size_px} pixel area in thumbnail"
    )

    if histoqc_dir:
        histoqc_mask_use_file_path = (
            histoqc_dir / slide_path.name / f"{slide_path.name}_mask_use.png"
        )

        histoqc_mask_use = Image.open(histoqc_mask_use_file_path)

        thumb_ratio = min(
            thumbsize[0] / histoqc_mask_use.size[0],
            thumbsize[1] / histoqc_mask_use.size[1],
        )

        histoqc_thumb_size = (
            int(np.round(thumb_ratio * histoqc_mask_use.size[0])),
            int(np.round(thumb_ratio * histoqc_mask_use.size[1])),
        )

        histoqc_thumb = histoqc_mask_use.resize(
            histoqc_thumb_size,
            Image.Resampling.NEAREST,
        )

        arr = np.array(np.asarray(histoqc_thumb), dtype=bool)
    else:
        arr = segment_tissue(
            np.asarray(thumb),
            median_filter_size=median_filter_size,
            binary_threshold=binary_threshold,
            closing_kernel_size=closing_kernel_size,
            min_object_size_px=min_object_size_px,
            min_hole_size_px=min_hole_size_px,
        )

    if not np.issubdtype(arr.dtype, np.bool_):
        raise TypeError(
            f"expected the segmentation array to be boolean dtype but got {arr.dtype}"
        )

    scale: tuple[float, float] = (
        slide.dimensions[0] / thumb.size[0],
        slide.dimensions[1] / thumb.size[1],
    )
    _res = get_multipolygon_from_binary_arr(arr.astype("uint8") * 255, scale=scale)
    if _res is None:
        logger.warning(f"No tissue was found in slide {slide_path}")
        return None
    polygon, contours, hierarchy = _res

    if (
        object_based
        and qupath_detection_dir is not None
        and qupath_geojson_detection_dir is None
        and qupath_geojson_annotation_dir is None
    ):
        patch_size = patch_size_px
        half_patch_size = round(patch_size / 2)

        slide_det = qupath_detection_dir / f"{slide_prefix}.txt"

        if not slide_det.exists():
            logger.info(f"Skipping because detection file not found: {slide_det}")
            coords = np.zeros((0, 2), dtype=np.int32)
            polygons = None
            tile_dim = None
        else:
            qpdet_df = pd.read_csv(slide_det, delimiter="\t")

            xs = np.rint(qpdet_df["Centroid X µm"] / mpp - half_patch_size).astype(np.int32)
            ys = np.rint(qpdet_df["Centroid Y µm"] / mpp - half_patch_size).astype(np.int32)

            coords = np.column_stack([xs, ys])

            polygons = np.asarray([
                [
                    [x - half_patch_size, y - half_patch_size],
                    [x - half_patch_size, y + half_patch_size],
                    [x + half_patch_size, y + half_patch_size],
                    [x + half_patch_size, y - half_patch_size],
                    [x - half_patch_size, y - half_patch_size],
                ]
                for x, y in zip(xs, ys)
            ])

            tile_dim = None

    elif (
        object_based
        and qupath_detection_dir is None
        and qupath_geojson_detection_dir is not None
        and qupath_geojson_annotation_dir is None
    ):
        patch_size = patch_size_px
        half_patch_size = round(patch_size / 2)

        slide_geojson = qupath_geojson_detection_dir / f"{slide_prefix}.geojson"

        if not slide_geojson.exists():
            logger.info(f"Skipping because geojson file not found: {slide_geojson}")
            coords = np.zeros((0, 2), dtype=np.int32)
            polygons = None
            tile_dim = None
        else:
            gdf = gpd.read_file(slide_geojson)
            gdf.set_crs(None, allow_override=True)

            x = (gdf.geometry.centroid.x / mpp) - half_patch_size
            y = (gdf.geometry.centroid.y / mpp) - half_patch_size

            x = x.to_numpy().round().astype(np.int32)
            y = y.to_numpy().round().astype(np.int32)

            coords = np.column_stack([x, y])

            gdf = gdf[gdf.geometry.notnull()]
            gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
            if gdf.empty:
                return None

            gdf = gdf.explode(index_parts=False, ignore_index=True)

            polygons_list: List[np.ndarray] = []
            for geom in gdf.geometry:
                if geom.is_empty:
                    continue
                if isinstance(geom, Polygon):
                    polys = [geom]
                elif isinstance(geom, MultiPolygon):
                    polys = [p for p in geom.geoms if not p.is_empty]
                else:
                    continue

                for poly in polys:
                    ex = np.asarray(poly.exterior.coords, dtype=np.float64)
                    xy = np.column_stack([ex[:, 0], ex[:, 1]]).astype(np.float32)
                    polygons_list.append(xy)

            polygons = polygons_list
            tile_dim = None

    elif (
        object_based
        and qupath_detection_dir is None
        and qupath_geojson_detection_dir is None
        and qupath_geojson_annotation_dir is None
        and object_detection == "end2end"
    ):
        slide_width, slide_height = slide.dimensions
        half_patch_size = round(patch_size / 2)

        overlap = (2 * halo_size_px / patch_size_px)

        coords = get_patch_coordinates_within_polygon(
            slide_width=slide_width,
            slide_height=slide_height,
            patch_size=patch_size,
            half_patch_size=half_patch_size,
            polygon=polygon,
            overlap=overlap,
        )

        step_size = round((1 - overlap) * patch_size)
        tile_centroids_arr: npt.NDArray[np.int_] = np.array(
            list(
                itertools.product(
                    range(0 + half_patch_size, slide_width, step_size),
                    range(0 + half_patch_size, slide_height, step_size),
                )
            )
        )

        tile_dim = (
            (tile_centroids_arr - half_patch_size) / step_size
        ).max(axis=0).astype(np.int32) + 1
        polygons = None

        logger.info(f"Found {len(coords)} patches within tissue")

    elif (
        object_based
        and qupath_detection_dir is None
        and qupath_geojson_detection_dir is None
        and qupath_geojson_annotation_dir is None
        and object_detection != "end2end"
    ):
        img = imread(slide_path)
        img = normalize(
            img,
            stardist_normalization_pmin,
            stardist_normalization_pmax,
            axis=(0, 1),
        )

        model = StarDist2D.from_pretrained("2D_versatile_he")
        _, polys = model.predict_instances_big(
            img,
            axes="YXC",
            block_size=4096,
            min_overlap=128,
            context=128,
            n_tiles=(4, 4, 1),
        )

        N = len(polys["coord"])

        object_centroids_arr = np.zeros((N, 2), dtype=np.int32)
        polygons: list[np.ndarray] = []

        for n in range(N):
            ys = np.asarray(polys["coord"][n][0], dtype=np.float32)
            xs = np.asarray(polys["coord"][n][1], dtype=np.float32)
            xy = np.column_stack([xs, ys])

            if xy.shape[0] > 0 and not np.allclose(xy[0], xy[-1]):
                xy = np.vstack([xy, xy[0]])

            polygons.append(xy)

            poly = Polygon(xy)
            if not poly.is_valid:
                poly = poly.buffer(0)

            cx, cy = poly.centroid.coords[0]
            object_centroids_arr[n] = np.rint([cx, cy]).astype(np.int32)

        slide_width, slide_height = slide.dimensions
        half_patch_size = int(round(patch_size / 2))

        coords = get_object_coordinates_within_polygon(
            object_centroids_arr=object_centroids_arr,
            half_patch_size=half_patch_size,
            polygon=polygon,
        )

        tile_dim = None

    else:
        slide_width, slide_height = slide.dimensions
        half_patch_size = round(patch_size / 2)

        coords = get_patch_coordinates_within_polygon(
            slide_width=slide_width,
            slide_height=slide_height,
            patch_size=patch_size,
            half_patch_size=half_patch_size,
            polygon=polygon,
            overlap=overlap,
        )

        step_size = round((1 - overlap) * patch_size)
        tile_centroids_arr: npt.NDArray[np.int_] = np.array(
            list(
                itertools.product(
                    range(0 + half_patch_size, slide_width, step_size),
                    range(0 + half_patch_size, slide_height, step_size),
                )
            )
        )

        tile_dim = (
            (tile_centroids_arr - half_patch_size) / step_size
        ).max(axis=0).astype(np.int32) + 1

        polygons = []

        for c in range(len(coords)):
            tile_minx = coords[c][0]
            tile_miny = coords[c][1]
            tile_maxx = tile_minx + patch_size - 1
            tile_maxy = tile_miny + patch_size - 1
            tile_polygon = np.asarray(
                [
                    [tile_minx, tile_miny],
                    [tile_maxx, tile_miny],
                    [tile_maxx, tile_maxy],
                    [tile_minx, tile_maxy],
                    [tile_minx, tile_miny],
                ]
            )
            polygons.append(tile_polygon)

        logger.info(f"Found {len(coords)} patches within tissue")

    patch_path.parent.mkdir(exist_ok=True, parents=True)
    if coords.size > 0:
        logger.info(
            f"Extracting {coords.shape[0]} patches of size {patch_size}x{patch_size} "
            f"for slide {slide_prefix}"
        )

        images = (
            extract_patches_from_slide(slide, coords, patch_size)
            if cache_image_patches
            else None
        )

        mpp = get_avg_mpp(slide_path)
        slide = get_wsi_cls()(slide_path)
        slide_width, slide_height = slide.dimensions

        logger.info(f"Writing coordinates and images to {patch_path}")
        save_hdf5(
            path=patch_path,
            coords=coords,
            polygons=polygons,
            tile_dim=tile_dim,
            patch_size=patch_size,
            patch_spacing_um_px=patch_spacing_um_px,
            compression="gzip",
            images=images,
            slide_path=str(slide_path),
            slide_mpp=mpp,
            slide_width=slide_width,
            slide_height=slide_height,
        )
    else:
        logger.warning(f"No patches found for slide {slide_path}")

    logger.info(f"Writing tissue thumbnail with contours to disk: {mask_path}")
    mask_path.parent.mkdir(exist_ok=True, parents=True)
    img = draw_contours_on_thumbnail(thumb, contours=contours, hierarchy=hierarchy)
    img.thumbnail((1024, 1024), resample=Image.Resampling.LANCZOS)
    with mask_path.open("wb") as fh:
        img.save(fh)

    return None


def segment_and_patch_directory_of_slides(
    wsi_dir: URIPath,
    slide_paths: List[URIPath],
    save_dir: URIPath,
    qupath_detection_dir: str | URIPath,
    qupath_geojson_detection_dir: str | URIPath,
    qupath_geojson_annotation_dir: str | URIPath,
    patch_size_px: int,
    patch_spacing_um_px: float,
    halo_size_px: int = 0,
    histoqc_dir: str | URIPath | None = None,
    thumbsize: tuple[int, int] = (2048, 2048),
    median_filter_size: int = 7,
    binary_threshold: int = 7,
    closing_kernel_size: int = 6,
    min_object_size_um2: float = 200 ** 2,
    min_hole_size_um2: float = 190 ** 2,
    overlap: float = 0.0,
    object_based: bool = False,
    object_detection: str | None = None,
    stardist_normalization_pmin: float = 1.0,
    stardist_normalization_pmax: float = 99.8,
    cache_image_patches: bool = False,
) -> None:
    """Batch segment and patch a directory of slides."""

    wsi_dir = URIPath(wsi_dir)

    _validate_wsi_directory(wsi_dir)

    for i, slide_path in enumerate(slide_paths):
        logger.info(f"Slide {i+1} of {len(slide_paths)} ({(i+1)/len(slide_paths):.2%})")
        try:
            segment_and_patch_one_slide(
                slide_path=slide_path,
                save_dir=save_dir,
                qupath_detection_dir=qupath_detection_dir,
                qupath_geojson_detection_dir=qupath_geojson_detection_dir,
                qupath_geojson_annotation_dir=qupath_geojson_annotation_dir,
                patch_size_px=patch_size_px,
                patch_spacing_um_px=patch_spacing_um_px,
                halo_size_px=halo_size_px,
                histoqc_dir=histoqc_dir,
                thumbsize=thumbsize,
                median_filter_size=median_filter_size,
                binary_threshold=binary_threshold,
                closing_kernel_size=closing_kernel_size,
                min_object_size_um2=min_object_size_um2,
                min_hole_size_um2=min_hole_size_um2,
                overlap=overlap,
                object_based=object_based,
                object_detection=object_detection,
                stardist_normalization_pmin=stardist_normalization_pmin,
                stardist_normalization_pmax=stardist_normalization_pmax,
                cache_image_patches=cache_image_patches,
            )
        except Exception as e:  # pragma: no cover - logged for operators
            logger.error(f"Failed to segment and patch slide\n{slide_path}", exc_info=e)

    return None
