"""Spatial object-to-region registration utilities for WSInsight."""

from __future__ import annotations

import os
from math import ceil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import tqdm as tqdm_module

from ..num_worker_optimizer import pick_workers_safe, throttle_when_busy


def register_objects_to_regions(
    slide_df: pd.DataFrame,
    annot_df: pd.DataFrame,
    tie_break: str = "largest_area",
    mem_budget_mb: int = 256,
    max_workers: int | None = None,
    pbar: Optional["tqdm_module.tqdm"] = None,
    out_prefix: str = "region_",
) -> tuple[pd.DataFrame, float]:
    """Match each object in *slide_df* to its enclosing region in *annot_df*.

    The centre point of each object is tested against the bounding boxes of
    all regions.  The matching region's columns are written to *slide_df*
    under ``out_prefix`` (default ``"region_"``).  Objects with no enclosing
    region receive NaN for those columns.

    Parameters
    ----------
    slide_df:
        Object-level DataFrame with at least ``minx``, ``miny``, ``width``,
        and ``height`` columns (pixel coordinates).
    annot_df:
        Region-level DataFrame with the same spatial columns plus ``prob_*``
        columns.
    tie_break:
        Strategy when an object centre falls inside multiple regions.
        ``"largest_area"`` (default) picks the biggest enclosing region;
        ``"first"`` picks the region with the lowest index.
    mem_budget_mb:
        Per-worker memory limit that governs the adaptive chunk size.
    max_workers:
        Thread-pool size.  Defaults to ``pick_workers_safe(os.cpu_count()-8, 8)``.
    out_prefix:
        Bare prefix prepended to every copied region column name.  Must end
        in ``"_"``.  Default ``"region_"`` reproduces the back-compat naming
        scheme; ``wsinsight reg --tag foo`` passes ``"region_foo_"``.

    Returns
    -------
    tuple[pd.DataFrame, float]
        *slide_df* (mutated in place) with ``<out_prefix><region_col>``
        columns added or overwritten, plus the per-slide match rate
        (fraction of objects that found an enclosing region in [0, 1]).
    """
    if max_workers is None:
        max_workers = pick_workers_safe(max_workers=(os.cpu_count() or 16) - 8, min_workers=8)

    # 1) object centres (local arrays — never written into slide_df)
    cx_all = (slide_df["minx"] + slide_df["width"] * 0.5).to_numpy()
    cy_all = (slide_df["miny"] + slide_df["height"] * 0.5).to_numpy()

    # 2) region bounding-box arrays
    ax0 = annot_df["minx"].to_numpy()
    ay0 = annot_df["miny"].to_numpy()
    ax1 = (annot_df["minx"] + annot_df["width"]).to_numpy()
    ay1 = (annot_df["miny"] + annot_df["height"]).to_numpy()
    area = (annot_df["width"] * annot_df["height"]).to_numpy()

    # All columns from the region CSV are copied with a "region_" prefix.
    # This includes spatial columns (minx/miny/width/height) as region_minx etc.
    # as well as all prob_* columns as region_prob_*.
    copy_cols = list(annot_df.columns)
    copy_mat = annot_df[copy_cols].to_numpy(dtype=np.float64) if copy_cols else None

    # 3) containment predicate
    def contains(cx_col, cy_col):
        return (
            (cx_col[:, None] >= ax0[None, :]) &
            (cx_col[:, None] <= ax1[None, :]) &
            (cy_col[:, None] >= ay0[None, :]) &
            (cy_col[:, None] <= ay1[None, :])
        )

    # 4) adaptive chunk sizing
    n_points = len(slide_df)
    n_annots = len(annot_df)
    # Rough bytes per point in a worker: boolean mask (n_annots * 1 byte)
    # + a float score matrix view (n_annots * 4 bytes) → ~5 bytes/annot per point
    bytes_per_point = max(1, 5 * n_annots)
    target_bytes_per_worker = int(mem_budget_mb * 1024 ** 2)
    # points per chunk so that mask+scores fit in mem budget (with a safety factor)
    points_per_chunk_mem = max(1000, target_bytes_per_worker // bytes_per_point)
    # also ensure enough chunks to keep workers busy
    min_chunks = max_workers * 4
    points_per_chunk_busy = max(1000, ceil(n_points / max(1, min_chunks)))
    # final adaptive chunk size = conservative min of both constraints
    points_per_chunk = int(max(1000, min(points_per_chunk_mem, points_per_chunk_busy)))

    # 5) initialise output columns (add or overwrite for all region columns)
    for c in copy_cols:
        slide_df[out_prefix + c] = np.nan

    # 6) chunk worker
    def process_chunk(s: int, e: int):
        cx = cx_all[s:e]
        cy = cy_all[s:e]

        mask = contains(cx, cy)  # shape (B, A)
        has_hit = mask.any(axis=1)

        best = np.full(len(cx), -1, dtype=np.int64)
        if has_hit.any():
            if tie_break == "largest_area":
                cand_scores = np.where(mask, area[None, :], -np.inf)
                best_ix = cand_scores.argmax(axis=1)
            else:  # "first"
                idxs = np.tile(np.arange(mask.shape[1]), (mask.shape[0], 1))
                cand_scores = np.where(mask, -idxs, -np.inf)
                best_ix = cand_scores.argmax(axis=1)
            best[has_hit] = best_ix[has_hit]

        results = {}
        n_hit = int((best >= 0).sum())
        if copy_cols:
            hit_rows = best >= 0
            for j, c in enumerate(copy_cols):
                vals = np.full(len(cx), np.nan, dtype=np.float64)
                if hit_rows.any() and copy_mat is not None:
                    vals[hit_rows] = copy_mat[best[hit_rows], j]
                results[out_prefix + c] = vals

        return s, e, results, n_hit

    # 7) schedule work
    indices = list(range(0, n_points, points_per_chunk))
    if pbar is not None:
        pbar.reset(total=len(indices))
    total_hits = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(process_chunk, s, min(n_points, s + points_per_chunk))
            for s in indices
        ]
        for fut in as_completed(futures):
            throttle_when_busy()
            s, e, res, n_hit = fut.result()
            total_hits += n_hit
            for col, vals in res.items():
                slide_df.loc[slide_df.index[s:e], col] = vals
            if pbar is not None:
                pbar.update(1)

    match_rate = (total_hits / n_points) if n_points > 0 else 0.0
    return slide_df, match_rate


def register_objects_to_objects(
    slide_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    radius_um: float,
    spacing_um_px: float,
    out_prefix: str = "object_",
    pbar: Optional["tqdm_module.tqdm"] = None,
) -> tuple[pd.DataFrame, float]:
    """Match each object in *slide_df* to its nearest neighbour in *secondary_df*.

    Object centres are computed from ``minx + width/2`` and ``miny + height/2``
    in both frames (pixel coordinates).  A KD-tree is built on the secondary
    centres and queried for each primary centre with ``k=1`` and a maximum
    search distance of ``radius_um / spacing_um_px`` pixels.  Matched primary
    rows receive the entire secondary row's columns under ``out_prefix``;
    unmatched rows receive NaN.

    Parameters
    ----------
    slide_df:
        Primary object-level DataFrame with ``minx``, ``miny``, ``width``,
        ``height`` columns.
    secondary_df:
        Secondary object-level DataFrame with the same spatial columns plus
        ``prob_*`` columns to copy across.
    radius_um:
        Maximum match radius in micrometres; matches with centroid-to-centroid
        distance above this threshold are dropped.
    spacing_um_px:
        Pixel size in micrometres-per-pixel used to convert *radius_um* into
        the pixel space of *slide_df* / *secondary_df*.
    out_prefix:
        Bare prefix prepended to every copied secondary column name.  Default
        ``"object_"`` matches ``wsinsight reg -c`` with no tag.
    pbar:
        Optional ``tqdm`` progress bar for the chunked write loop.

    Returns
    -------
    tuple[pd.DataFrame, float]
        *slide_df* (mutated in place) with ``<out_prefix><sec_col>`` columns
        added, plus the per-slide match rate (fraction of primary objects
        with a neighbour within radius) in [0, 1].
    """
    from scipy.spatial import cKDTree  # local import: scipy is heavy

    if radius_um <= 0:
        raise ValueError(f"radius_um must be positive (got {radius_um})")
    if spacing_um_px <= 0:
        raise ValueError(f"spacing_um_px must be positive (got {spacing_um_px})")

    n_primary = len(slide_df)
    radius_px = float(radius_um) / float(spacing_um_px)

    copy_cols = list(secondary_df.columns)

    # Initialise output columns with NaN.
    for c in copy_cols:
        slide_df[out_prefix + c] = np.nan

    if n_primary == 0 or len(secondary_df) == 0 or not copy_cols:
        if pbar is not None:
            pbar.reset(total=1)
            pbar.update(1)
        return slide_df, 0.0

    # Centroids.
    cx_p = (slide_df["minx"] + slide_df["width"] * 0.5).to_numpy(dtype=np.float64)
    cy_p = (slide_df["miny"] + slide_df["height"] * 0.5).to_numpy(dtype=np.float64)
    cx_s = (secondary_df["minx"] + secondary_df["width"] * 0.5).to_numpy(dtype=np.float64)
    cy_s = (secondary_df["miny"] + secondary_df["height"] * 0.5).to_numpy(dtype=np.float64)

    tree = cKDTree(np.stack([cx_s, cy_s], axis=1))
    dists, idxs = tree.query(
        np.stack([cx_p, cy_p], axis=1),
        k=1,
        distance_upper_bound=radius_px,
    )
    matched = np.isfinite(dists)
    n_matched = int(matched.sum())

    # Chunked write to keep memory steady on huge frames.
    copy_mat = secondary_df[copy_cols].to_numpy(dtype=np.float64)
    chunk = max(10_000, n_primary // 64 if n_primary > 64 else n_primary)
    starts = list(range(0, n_primary, chunk))
    if pbar is not None:
        pbar.reset(total=max(1, len(starts)))
    for s in starts:
        e = min(n_primary, s + chunk)
        local_match = matched[s:e]
        if local_match.any():
            local_idx = idxs[s:e][local_match]
            for j, c in enumerate(copy_cols):
                vals = np.full(e - s, np.nan, dtype=np.float64)
                vals[local_match] = copy_mat[local_idx, j]
                slide_df.loc[slide_df.index[s:e], out_prefix + c] = vals
        if pbar is not None:
            pbar.update(1)

    match_rate = n_matched / n_primary
    return slide_df, match_rate
