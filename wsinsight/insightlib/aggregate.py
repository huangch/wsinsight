"""Cell-type aggregate detection and quotient-graph contraction for WSInsight.

An *aggregate* is a connected, density-gated cluster of a chosen set of cell
types over the Delaunay graph — for example T- and B-cells condensing into a
tertiary lymphoid structure (TLS).  Aggregates are detected with the same
``(k, N, R)`` neighborhood-enrichment gate used by H-Plot's tumor-region
detection, then split into connected components and contracted into a quotient
graph whose super-nodes are the aggregates.

This module is intentionally free of any disease- or structure-specific
identifier (no hard-coded "TLS"): the caller supplies the ingredient cell
types and a product name through ``wsinsight agg``.

Public API
----------
``identify_aggregates``  per-cell aggregate id (``-1`` = not a member)
``contract_to_quotient`` aggregate super-nodes + boundary-crossing meta-edges
``aggregate_features``   one row per aggregate (size, area, centroid, composition)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .insight_helpers import identify_region_by_cell_function_enrichment
from .insight_helpers import k_hop_neighbors

# ---------------------------------------------------------------------------
# Aggregate detection
# ---------------------------------------------------------------------------


def _induced_region_components(
    is_region: np.ndarray,
    edges_df: pd.DataFrame,
    min_size: int,
) -> np.ndarray:
    """Connected components of the subgraph induced by region cells.

    Parameters
    ----------
    is_region:
        Boolean array (N,) marking cells inside an enriched region.
    edges_df:
        Pruned Delaunay edges with ``source`` / ``target`` columns.
    min_size:
        Minimum number of cells for a component to survive.

    Returns
    -------
    np.ndarray
        Raw component label per cell (N,); ``-1`` for cells that are not in a
        surviving component.  Labels are arbitrary at this stage; they are
        re-ordered deterministically by the caller.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    region = np.asarray(is_region, dtype=bool)
    n = region.shape[0]
    raw = np.full(n, -1, dtype=np.int64)
    if n == 0 or not region.any():
        return raw

    src = edges_df["source"].to_numpy(dtype=np.int64, copy=False)
    dst = edges_df["target"].to_numpy(dtype=np.int64, copy=False)

    # Keep only edges whose endpoints are BOTH region cells (the induced graph).
    keep = region[src] & region[dst]
    isrc = src[keep]
    idst = dst[keep]

    # Symmetric adjacency over all N nodes; non-region nodes have no edges and
    # become singleton components which we ignore below.
    rows = np.concatenate([isrc, idst])
    cols = np.concatenate([idst, isrc])
    data = np.ones(rows.shape[0], dtype=np.uint8)
    adj = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    _n_comp, labels = connected_components(adj, directed=False)

    # Size of each component, counting region cells only.
    region_labels = labels[region]
    uniq, counts = np.unique(region_labels, return_counts=True)
    big_enough = set(uniq[counts >= min_size].tolist())
    if not big_enough:
        return raw

    keep_cell = region & np.isin(labels, list(big_enough))
    raw[keep_cell] = labels[keep_cell]
    return raw


def _relabel_by_centroid(
    raw_labels: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Map arbitrary component labels to contiguous ids ordered by centroid.

    Ordering by ``(centroid_y, centroid_x)`` makes aggregate ids reproducible
    across runs regardless of scipy's internal labelling.
    """
    out = np.full(raw_labels.shape[0], -1, dtype=np.int64)
    member = raw_labels >= 0
    if not member.any():
        return out

    labels = raw_labels[member]
    uniq = np.unique(labels)
    centroids = []
    for lab in uniq:
        pts = centers[raw_labels == lab]
        centroids.append((float(pts[:, 1].mean()), float(pts[:, 0].mean()), lab))
    # Sort by (centroid_y, centroid_x) for a stable, geometry-based order.
    centroids.sort()
    remap = {lab: new_id for new_id, (_, _, lab) in enumerate(centroids)}
    for lab, new_id in remap.items():
        out[raw_labels == lab] = new_id
    return out


def identify_aggregates(
    nodes_df: pd.DataFrame,
    base_targets: Sequence[str],
    edges_df: pd.DataFrame,
    *,
    k: int,
    N: int,
    R: float,
    min_size: int,
    predicted_labels: pd.Series | None = None,
) -> np.ndarray:
    """Assign each cell an aggregate id (``-1`` if it is in no aggregate).

    The pipeline mirrors H-Plot's region detection: a cell is an *ingredient*
    if its predicted label (argmax over ``prob_*``) is in *base_targets*; the
    ``(k, N, R)`` enrichment gate marks dense ingredient neighborhoods as a
    region; connected components of that region become aggregates; components
    smaller than *min_size* cells are discarded.

    Parameters
    ----------
    nodes_df:
        Per-cell DataFrame with ``center_x`` / ``center_y`` and ``prob_*``
        columns (or an explicit *predicted_labels*).
    base_targets:
        Ingredient cell-type names (without the ``prob_`` prefix).
    edges_df:
        Pruned Delaunay edges (``source`` / ``target`` / ``length``).
    k, N, R:
        Neighborhood radius, minimum neighborhood size, minimum ingredient
        fraction — the same knobs as ``--hplot-k/-n/-r``.
    min_size:
        Minimum number of cells per aggregate.
    predicted_labels:
        Optional precomputed per-cell label Series (values like ``prob_tumor``
        or plain ``tumor``).  When ``None`` it is derived from ``prob_*``.

    Returns
    -------
    np.ndarray
        ``(N,)`` int64 aggregate id per cell; ``-1`` for non-members.  Ids are
        contiguous ``0..F-1`` ordered by aggregate centroid.
    """
    n_cells = len(nodes_df)
    if n_cells == 0:
        return np.empty(0, dtype=np.int64)

    if predicted_labels is None:
        prob_columns = [c for c in nodes_df.columns if c.startswith("prob_")]
        if not prob_columns:
            raise ValueError(
                "nodes_df has no 'prob_*' columns and no predicted_labels were given."
            )
        predicted_labels = nodes_df[prob_columns].idxmax(axis=1)

    # Accept both "prob_tumor" and "tumor" spellings for the ingredient set.
    wanted = set()
    for t in base_targets:
        t = str(t)
        wanted.add(t)
        wanted.add(f"prob_{t}")
    is_base_type = predicted_labels.isin(wanted).to_numpy(dtype=bool)

    work = nodes_df.copy()
    work["is_base_type"] = is_base_type

    if not is_base_type.any():
        return np.full(n_cells, -1, dtype=np.int64)

    _neighbors, _A, Mk = k_hop_neighbors(n_cells, edges_df, k)
    work = identify_region_by_cell_function_enrichment(
        _neighbors, work, N, R, Mk_sparse=Mk
    )
    is_region = work["is_base_region"].to_numpy(dtype=bool)

    raw = _induced_region_components(is_region, edges_df, min_size)
    centers = nodes_df[["center_x", "center_y"]].to_numpy()
    return _relabel_by_centroid(raw, centers)


# ---------------------------------------------------------------------------
# Quotient-graph contraction
# ---------------------------------------------------------------------------


def contract_to_quotient(
    aggregate_id: np.ndarray,
    edges_df: pd.DataFrame,
    centers: np.ndarray,
) -> dict:
    """Contract aggregates into super-nodes and collect meta-edges.

    Super-nodes are the aggregates only (non-member cells are not represented
    in v1).  A meta-edge connects two aggregates when at least one original
    Delaunay edge joins a cell of one aggregate to a cell of the other.

    Returns
    -------
    dict with keys
        ``aggregate_centers``       (F, 2) float64 centroid per aggregate
        ``aggregate_sizes``         (F,)   int64 cell count per aggregate
        ``cell_to_aggregate``       (N,)   int64 aggregate id per cell (-1 none)
        ``quotient_edges_source``   (Q,)   int64 aggregate id
        ``quotient_edges_target``   (Q,)   int64 aggregate id (source < target)
    """
    aggregate_id = np.asarray(aggregate_id, dtype=np.int64)
    member = aggregate_id >= 0
    n_agg = int(aggregate_id.max()) + 1 if member.any() else 0

    centers = np.asarray(centers, dtype=np.float64)
    agg_centers = np.zeros((n_agg, 2), dtype=np.float64)
    agg_sizes = np.zeros(n_agg, dtype=np.int64)
    for a in range(n_agg):
        pts = centers[aggregate_id == a]
        agg_sizes[a] = len(pts)
        if len(pts):
            agg_centers[a] = pts.mean(axis=0)

    if n_agg and len(edges_df):
        src = edges_df["source"].to_numpy(dtype=np.int64, copy=False)
        dst = edges_df["target"].to_numpy(dtype=np.int64, copy=False)
        a_src = aggregate_id[src]
        a_dst = aggregate_id[dst]
        cross = (a_src >= 0) & (a_dst >= 0) & (a_src != a_dst)
        lo = np.minimum(a_src[cross], a_dst[cross])
        hi = np.maximum(a_src[cross], a_dst[cross])
        if lo.size:
            pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
            q_src = pairs[:, 0]
            q_dst = pairs[:, 1]
        else:
            q_src = np.empty(0, dtype=np.int64)
            q_dst = np.empty(0, dtype=np.int64)
    else:
        q_src = np.empty(0, dtype=np.int64)
        q_dst = np.empty(0, dtype=np.int64)

    return {
        "aggregate_centers": agg_centers,
        "aggregate_sizes": agg_sizes,
        "cell_to_aggregate": aggregate_id,
        "quotient_edges_source": q_src,
        "quotient_edges_target": q_dst,
    }


# ---------------------------------------------------------------------------
# Per-aggregate feature table
# ---------------------------------------------------------------------------


def _convex_hull_area_px2(points: np.ndarray) -> float:
    """Convex-hull area of a point set in pixel^2 (0 for < 3 distinct points)."""
    if len(points) < 3:
        return 0.0
    uniq = np.unique(points, axis=0)
    if len(uniq) < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull

        # For 2-D inputs scipy's ConvexHull.volume is the polygon area.
        return float(ConvexHull(uniq).volume)
    except Exception:
        return 0.0


def aggregate_features(
    nodes_df: pd.DataFrame,
    aggregate_id: np.ndarray,
    *,
    slide_id: str,
    mpp: float,
    predicted_labels: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a per-aggregate table (one row per aggregate).

    Columns
    -------
    aggregate_id, slide_id, n_cells, area_um2, center_x, center_y,
    composition_<type>_frac (one per cell type present among members).

    There is intentionally **no distance column**: distance-to-tumor-border is
    H-Plot's responsibility and is computed downstream.
    """
    aggregate_id = np.asarray(aggregate_id, dtype=np.int64)
    member = aggregate_id >= 0
    n_agg = int(aggregate_id.max()) + 1 if member.any() else 0

    if predicted_labels is None:
        prob_columns = [c for c in nodes_df.columns if c.startswith("prob_")]
        if prob_columns:
            predicted_labels = (
                nodes_df[prob_columns].idxmax(axis=1).str.removeprefix("prob_")
            )
        else:
            predicted_labels = pd.Series(
                ["unknown"] * len(nodes_df), index=nodes_df.index
            )
    else:
        predicted_labels = predicted_labels.astype(str).str.removeprefix("prob_")

    all_types = sorted(predicted_labels.unique().tolist())
    centers = nodes_df[["center_x", "center_y"]].to_numpy(dtype=np.float64)
    labels_arr = predicted_labels.to_numpy()
    mpp2 = float(mpp) * float(mpp)

    rows: list[dict] = []
    for a in range(n_agg):
        mask = aggregate_id == a
        pts = centers[mask]
        n = int(mask.sum())
        if n == 0:
            continue
        cx, cy = pts.mean(axis=0)
        row: dict = {
            "aggregate_id": a,
            "slide_id": slide_id,
            "n_cells": n,
            "area_um2": _convex_hull_area_px2(pts) * mpp2,
            "center_x": int(round(cx)),
            "center_y": int(round(cy)),
        }
        member_labels = labels_arr[mask]
        for t in all_types:
            row[f"composition_{t}_frac"] = float(np.mean(member_labels == t))
        rows.append(row)

    columns = [
        "aggregate_id",
        "slide_id",
        "n_cells",
        "area_um2",
        "center_x",
        "center_y",
    ] + [f"composition_{t}_frac" for t in all_types]
    return pd.DataFrame(rows, columns=columns)
