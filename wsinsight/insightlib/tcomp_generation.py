"""Triad-level composition (tcomp) generation for WSInsight.

For each Delaunay triad (triangle), compute the triad-type composition of its
k-hop neighborhood in the *dual graph* of the Delaunay triangulation (two
triads are neighbors iff they share ≥ 1 vertex).  This is the 2-simplex
counterpart of ``ncomp`` (cells) and ``ecomp`` (edges).

Outputs
-------
Per slide
    ``<results_dir>/tcomp-outputs-csv/<slide_id>.csv``
    One row per Delaunay triad with columns:
        triad_id, vertex_{1,2,3}_id,
        centroid_x, centroid_y,
        triad_max_edge_um, triad_area_um2,
        triad_perimeter_um, triad_regularity,
        cell_type_{1,2,3}, triad_type,
        neighborhood_size,
        neighborhood_mean_area_um2, neighborhood_std_area_um2,
        neighborhood_mean_max_edge_um,
        neighborhood_<triad_type>_count, neighborhood_<triad_type>_prop
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations_with_replacement
from pathlib import Path
from typing import List, Mapping

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import errors
from ..cancel import cancellable_as_completed, critical_section
from ..wsi import _validate_wsi_directory, get_avg_mpp
from ..uri_path import URIPath

from .insight_helpers import compute_cell_center_points
from .graph_cache import get_or_build_delaunay, read_graph_cache, _cache_path
from .simplex_helpers import build_dual_graph, k_hop_adjacency_matrix, triad_geometry

_logger = logging.getLogger(__name__)

_WORKER_STEPS = [
    "load CSV",
    "cell centers",
    "triangulation",
    "triad geometry",
    "dual graph",
    "k-hop neighbors",
    "triad composition",
    "save outputs",
]


def _triad_type_vocab(all_types: list[str]) -> list[str]:
    """Return the sorted list of unordered triad-type labels over *all_types*."""
    sorted_types = sorted(all_types)
    return ["__".join(tri) for tri in combinations_with_replacement(sorted_types, 3)]


def _worker(
    wsi_path: URIPath,
    model_output_csv: URIPath,
    results_dir: URIPath,
    max_edge_um: float,
    tcomp_k: int,
    slide_mpp_lookup: Mapping[str, float] | None = None,
    overwrite: bool = False,
    graph_cache_dir: Path | URIPath | None = None,
    pbar_position: int = 1,
    no_neighborhood: bool = False,
) -> tuple[str, dict | None]:
    """Process a single slide to compute per-triad neighborhood composition."""

    slide_id = wsi_path.stem
    tcomp_csv = results_dir / "tcomp-outputs-csv" / f"{slide_id}.csv"

    if not overwrite and tcomp_csv.exists():
        return slide_id, True

    desc = slide_id if len(slide_id) <= 32 else slide_id[:29] + "..."
    inner = tqdm(
        total=len(_WORKER_STEPS),
        desc=desc,
        position=pbar_position,
        leave=False,
        unit="step",
        dynamic_ncols=True,
    )

    def _step(name: str) -> None:
        inner.set_postfix_str(name)
        inner.update(1)

    # --- MPP resolution -----------------------------------------------------
    mpp = None
    if slide_mpp_lookup:
        mpp = slide_mpp_lookup.get(slide_id) or slide_mpp_lookup.get(str(wsi_path))
    if mpp is None:
        mpp = get_avg_mpp(wsi_path)
    max_edge_px = max_edge_um / mpp

    # --- Load CSV -----------------------------------------------------------
    try:
        with model_output_csv.open("r", encoding="utf-8") as fp:
            nodes_df = pd.read_csv(fp)
    except Exception as exc:
        _logger.warning("Failed to load CSV for %s: %s", slide_id, exc)
        inner.close()
        return slide_id, None
    _step("load CSV")

    prob_columns = [c for c in nodes_df.columns if c.startswith("prob_")]
    if not prob_columns:
        inner.close()
        return slide_id, None

    predicted_labels = nodes_df[prob_columns].idxmax(axis=1)
    nodes_df["cell_type"] = predicted_labels.str.removeprefix("prob_")

    nodes_df = compute_cell_center_points(nodes_df)
    _step("cell centers")

    centers = nodes_df[["center_x", "center_y"]].values
    N = len(nodes_df)

    # --- Delaunay (via shared cache) ----------------------------------------
    if graph_cache_dir is not None:
        _ = get_or_build_delaunay(graph_cache_dir, slide_id, centers, mpp, max_edge_px)
        cache = read_graph_cache(_cache_path(graph_cache_dir, slide_id))
        simplices_all = cache["simplices"].astype(np.int64)
    else:
        from .insight_helpers import _delaunay_full
        simplices_all, _src, _dst, _lengths = _delaunay_full(centers)
        simplices_all = simplices_all.astype(np.int64)
    _step("triangulation")

    # --- Triad geometry -----------------------------------------------------
    if simplices_all.shape[0] == 0:
        inner.close()
        tcomp_csv.parent.mkdir(parents=True, exist_ok=True)
        with tcomp_csv.open("w", encoding="utf-8", newline="") as fp:
            fp.write("triad_id\n")
        return slide_id, True

    geom_all = triad_geometry(centers, simplices_all)

    # Filter out triads whose longest edge exceeds the threshold.
    max_edge_mask = geom_all["max_edge_px"] < max_edge_px
    simplices = simplices_all[max_edge_mask]
    if simplices.shape[0] == 0:
        inner.close()
        tcomp_csv.parent.mkdir(parents=True, exist_ok=True)
        with tcomp_csv.open("w", encoding="utf-8", newline="") as fp:
            fp.write("triad_id\n")
        return slide_id, True

    # Sort each simplex's vertex ids ascending for canonical (v1, v2, v3).
    simplices_sorted = np.sort(simplices, axis=1)

    geom = {k: v[max_edge_mask] for k, v in geom_all.items()}
    T = simplices_sorted.shape[0]
    _step("triad geometry")

    # --- Fast path: skip dual graph + k-hop entirely -----------------------
    if no_neighborhood:
        _step("dual graph")       # advance pbar — skipped
        _step("k-hop neighbors")  # advance pbar — skipped

        cell_type_array = nodes_df["cell_type"].to_numpy()
        ct_triples = cell_type_array[simplices_sorted]
        ct_sorted = np.sort(ct_triples, axis=1)
        triad_type = np.array(
            ["__".join(row) for row in ct_sorted.tolist()], dtype=object
        )
        area_um2 = geom["area_px2"] * (mpp ** 2)
        max_edge_um_arr = geom["max_edge_px"] * mpp
        perimeter_um = geom["perimeter_px"] * mpp

        data: dict[str, np.ndarray] = {
            "triad_id": np.arange(T, dtype=np.int64),
            "vertex_1_id": simplices_sorted[:, 0].astype(np.int64),
            "vertex_2_id": simplices_sorted[:, 1].astype(np.int64),
            "vertex_3_id": simplices_sorted[:, 2].astype(np.int64),
            "centroid_x": geom["centroid_x"],
            "centroid_y": geom["centroid_y"],
            "triad_max_edge_um": max_edge_um_arr,
            "triad_area_um2": area_um2,
            "triad_perimeter_um": perimeter_um,
            "triad_regularity": geom["regularity"],
            "cell_type_1": ct_sorted[:, 0],
            "cell_type_2": ct_sorted[:, 1],
            "cell_type_3": ct_sorted[:, 2],
            "triad_type": triad_type,
        }

        region_cols = [c for c in nodes_df.columns if c.startswith("region_prob_")]
        if region_cols:
            region_labels = [c.removeprefix("region_prob_") for c in region_cols]
            region_probs = nodes_df[region_cols].to_numpy(dtype=np.float32)
            centroid_region_idx = region_probs[simplices_sorted].sum(axis=1).argmax(axis=1)
            data["centroid_region"] = np.array(region_labels, dtype=object)[centroid_region_idx]
        _step("triad composition")

        out_df = pd.DataFrame(data)
        tcomp_csv.parent.mkdir(parents=True, exist_ok=True)
        with critical_section(f"saving tcomp output for {slide_id}"):
            with tcomp_csv.open("w", encoding="utf-8", newline="") as fp:
                out_df.to_csv(fp, index=False)
        _step("save outputs")
        inner.close()
        return slide_id, True

    # --- Dual graph ---------------------------------------------------------
    D = build_dual_graph(simplices_sorted, num_vertices=N)
    _step("dual graph")

    # --- k-hop on dual graph (as sparse matrix for vectorised aggregation) -
    A_k = k_hop_adjacency_matrix(D, tcomp_k)
    _step("k-hop neighbors")

    # --- Per-triad features ------------------------------------------------
    cell_type_array = nodes_df["cell_type"].to_numpy()
    all_types = sorted(c.removeprefix("prob_") for c in prob_columns)
    triad_type_vocab = _triad_type_vocab(all_types)

    # Cell types at each of the 3 vertex positions (already sorted by vertex id).
    # To produce the alphabetized triad_type string we sort the *cell types* per row.
    ct_triples = cell_type_array[simplices_sorted]  # (T, 3)
    ct_sorted = np.sort(ct_triples, axis=1)
    triad_type = np.array(
        ["__".join(row) for row in ct_sorted.tolist()], dtype=object
    )

    # Convert geometric px units to µm.
    area_um2 = geom["area_px2"] * (mpp ** 2)
    max_edge_um_arr = geom["max_edge_px"] * mpp
    perimeter_um = geom["perimeter_px"] * mpp

    # --- Vectorised neighborhood aggregation via sparse matrix products ----
    type_to_col = {t: i for i, t in enumerate(triad_type_vocab)}
    V = len(triad_type_vocab)
    triad_type_idx = np.fromiter(
        (type_to_col[t] for t in triad_type), count=T, dtype=np.int64
    )

    from scipy.sparse import csr_matrix as _csr

    T_onehot = _csr(
        (
            np.ones(T, dtype=np.float32),
            (np.arange(T, dtype=np.int64), triad_type_idx),
        ),
        shape=(T, V),
    )

    counts_mat = (A_k @ T_onehot).toarray().astype(np.float64)
    nhood_size = counts_mat.sum(axis=1).astype(np.int32)

    area_f32 = area_um2.astype(np.float32)
    max_edge_f32 = max_edge_um_arr.astype(np.float32)
    sum_area = np.asarray(A_k @ area_f32, dtype=np.float64)
    sum_area_sq = np.asarray(A_k @ (area_f32 ** 2), dtype=np.float64)
    sum_max_edge = np.asarray(A_k @ max_edge_f32, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        nhood_mean_area = np.where(nhood_size > 0, sum_area / nhood_size, np.nan)
        var_area = np.where(
            nhood_size > 0,
            sum_area_sq / nhood_size - nhood_mean_area ** 2,
            np.nan,
        )
        var_area = np.where(var_area < 0, 0.0, var_area)
        nhood_std_area = np.sqrt(var_area)
        nhood_mean_max_edge = np.where(
            nhood_size > 0, sum_max_edge / nhood_size, np.nan
        )
    _step("triad composition")

    # --- Build output DataFrame --------------------------------------------
    denom = np.where(nhood_size > 0, nhood_size, np.nan)

    data: dict[str, np.ndarray] = {
        "triad_id": np.arange(T, dtype=np.int64),
        "vertex_1_id": simplices_sorted[:, 0].astype(np.int64),
        "vertex_2_id": simplices_sorted[:, 1].astype(np.int64),
        "vertex_3_id": simplices_sorted[:, 2].astype(np.int64),
        "centroid_x": geom["centroid_x"],
        "centroid_y": geom["centroid_y"],
        "triad_max_edge_um": max_edge_um_arr,
        "triad_area_um2": area_um2,
        "triad_perimeter_um": perimeter_um,
        "triad_regularity": geom["regularity"],
        "cell_type_1": ct_sorted[:, 0],
        "cell_type_2": ct_sorted[:, 1],
        "cell_type_3": ct_sorted[:, 2],
        "triad_type": triad_type,
        "neighborhood_size": nhood_size,
        "neighborhood_mean_area_um2": nhood_mean_area,
        "neighborhood_std_area_um2": nhood_std_area,
        "neighborhood_mean_max_edge_um": nhood_mean_max_edge,
    }
    for j, t in enumerate(triad_type_vocab):
        counts_j = counts_mat[:, j].astype(np.float64)
        data[f"neighborhood_{t}_count"] = counts_j
        data[f"neighborhood_{t}_prop"] = counts_j / denom

    out_df = pd.DataFrame(data)

    tcomp_csv.parent.mkdir(parents=True, exist_ok=True)
    with critical_section(f"saving tcomp output for {slide_id}"):
        with tcomp_csv.open("w", encoding="utf-8", newline="") as fp:
            out_df.to_csv(fp, index=False)
    _step("save outputs")

    inner.close()
    return slide_id, True


def tcomp_generation(
    wsi_dir: str | Path | URIPath | None,
    slide_paths: List[URIPath] | None,
    results_dir: URIPath,
    max_edge_um: float = 25.0,
    tcomp_k: int = 2,
    num_workers: int = 8,
    slide_mpp_lookup: Mapping[str, float] | None = None,
    overwrite: bool = False,
    no_neighborhood: bool = False,
) -> list[str]:
    """Compute triad neighborhood composition for WSInsight outputs.

    Parameters mirror :func:`ncomp_generation`.
    """

    def _as_uri_path(p: str | Path | URIPath | None) -> URIPath | None:
        if p is None:
            return None
        return p if isinstance(p, URIPath) else URIPath(str(p))

    results_dir = _as_uri_path(results_dir)  # type: ignore[assignment]
    if results_dir is None:
        raise ValueError("results_dir must be provided")
    if not results_dir.exists():
        raise errors.ResultsDirectoryNotFound(results_dir)

    wsi_dir_path = _as_uri_path(wsi_dir)
    if wsi_dir_path is not None and not wsi_dir_path.exists():
        raise errors.WholeSlideImageDirectoryNotFound(
            f"directory not found: {wsi_dir_path}"
        )

    if slide_paths is not None:
        normalized = [p if isinstance(p, URIPath) else URIPath(str(p)) for p in slide_paths]
    elif wsi_dir_path is not None:
        normalized = [p for p in wsi_dir_path.iterdir() if p.is_file()]
    else:
        raise ValueError("slide_paths must be provided when wsi_dir is None")

    if not normalized:
        context = wsi_dir_path or "provided slide paths"
        raise errors.WholeSlideImagesNotFound(context)

    if wsi_dir_path is not None:
        _validate_wsi_directory(wsi_dir_path)
    else:
        stems = [p.stem for p in normalized]
        if len(stems) != len(set(stems)):
            raise errors.DuplicateFilePrefixesFound(
                "A slide with the same prefix but different extensions has been found"
            )

    slide_paths = normalized

    model_output_dir = results_dir / "model-outputs-csv"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    tcomp_dir = results_dir / "tcomp-outputs-csv"
    tcomp_dir.mkdir(parents=True, exist_ok=True)

    graph_cache_dir = results_dir / "graphs"
    graph_cache_dir.mkdir(parents=True, exist_ok=True)

    failed_generation: list[str] = []
    jobs = []
    for wsi_path in slide_paths:
        model_output_csv = model_output_dir / wsi_path.with_suffix(".csv").name
        if not model_output_csv.exists():
            failed_generation.append(wsi_path.stem)
            continue
        jobs.append((wsi_path, model_output_csv))

    if not jobs:
        return failed_generation

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(
                _worker,
                wsi_path,
                model_output_csv,
                results_dir,
                max_edge_um,
                tcomp_k,
                slide_mpp_lookup,
                overwrite,
                graph_cache_dir,
                (i % num_workers) + 1,
                no_neighborhood,
            ): wsi_path.stem
            for i, (wsi_path, model_output_csv) in enumerate(jobs)
        }
        outer = tqdm(
            total=len(futures),
            desc="Slides",
            position=0,
            leave=True,
            unit="slide",
            dynamic_ncols=True,
        )
        for f in cancellable_as_completed(futures, ex):
            slide_id, ok = f.result()
            if not ok:
                failed_generation.append(slide_id)
            outer.update(1)
        outer.close()

    return failed_generation
