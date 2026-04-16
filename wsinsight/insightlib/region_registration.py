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
) -> pd.DataFrame:
    """Match each object in *slide_df* to its enclosing region in *annot_df*.

    The centre point of each object is tested against the bounding boxes of
    all regions.  The matching region's ``prob_*`` values are written to
    ``region_prob_*`` columns in *slide_df*.  Objects with no enclosing region
    receive NaN for those columns.  Pre-existing ``region_prob_*`` columns that
    correspond to different class names are preserved.

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

    Returns
    -------
    pd.DataFrame
        *slide_df* (mutated in place) with ``region_prob_*`` columns added or
        overwritten for the classes present in *annot_df*.
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
        slide_df["region_" + c] = np.nan

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
        if copy_cols:
            hit_rows = best >= 0
            for j, c in enumerate(copy_cols):
                vals = np.full(len(cx), np.nan, dtype=np.float64)
                if hit_rows.any() and copy_mat is not None:
                    vals[hit_rows] = copy_mat[best[hit_rows], j]
                results["region_" + c] = vals

        return s, e, results

    # 7) schedule work
    indices = list(range(0, n_points, points_per_chunk))
    if pbar is not None:
        pbar.reset(total=len(indices))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(process_chunk, s, min(n_points, s + points_per_chunk))
            for s in indices
        ]
        for fut in as_completed(futures):
            throttle_when_busy()
            s, e, res = fut.result()
            for col, vals in res.items():
                slide_df.loc[slide_df.index[s:e], col] = vals
            if pbar is not None:
                pbar.update(1)

    return slide_df
