"""Edge-level composition (ecomp) generation for WSInsight.

For each Delaunay edge, compute the edge-type composition of its k-hop
neighborhood in the *line graph* of the Delaunay triangulation (edges sharing
a common vertex are neighbors).  This is the 1-simplex counterpart of
``ncomp`` (0-simplex / cells) and ``tcomp`` (2-simplex / triads).

Outputs
-------
Per slide
    ``<results_dir>/ecomp-outputs-csv/<slide_id>.csv``
    One row per Delaunay edge with columns:
        edge_id, vertex_1_id, vertex_2_id,
        center_x, center_y, edge_length_um,
        cell_type_1, cell_type_2, edge_type,
        neighborhood_size,
        neighborhood_mean_edge_length_um, neighborhood_std_edge_length_um,
        neighborhood_<edge_type>_count, neighborhood_<edge_type>_prop
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations_with_replacement
from pathlib import Path
from typing import List
from typing import Mapping

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import errors
from ..cancel import cancellable_as_completed
from ..cancel import critical_section
from ..uri_path import URIPath
from ..wsi import _validate_wsi_directory
from ..wsi import get_avg_mpp
from .graph_cache import _cache_path
from .graph_cache import get_or_build_delaunay
from .graph_cache import read_graph_cache
from .insight_helpers import compute_cell_center_points
from .insight_helpers import make_short_ids
from .simplex_helpers import build_line_graph
from .simplex_helpers import k_hop_adjacency_matrix

_logger = logging.getLogger(__name__)


def _gpu_available() -> bool:
    """Return True iff cupy is importable and at least one CUDA device exists."""
    try:
        import cupy as cp  # noqa: F401

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


_WORKER_STEPS = [
    "load CSV",
    "cell centers",
    "triangulate",
    "line graph",
    "k-hop nbrs",
    "edge comp.",
    "save outputs",
]
_STEP_LABEL_W = 12  # pad postfix so tqdm bar geometry stays stable across steps


def _edge_type_vocab(all_types: list[str]) -> list[str]:
    """Return the sorted list of unordered edge-type labels over *all_types*."""
    sorted_types = sorted(all_types)
    return ["__".join(pair) for pair in combinations_with_replacement(sorted_types, 2)]


def _worker(
    wsi_path: URIPath,
    model_output_csv: URIPath,
    results_dir: URIPath,
    max_edge_um: float,
    ecomp_k: int,
    slide_mpp_lookup: Mapping[str, float] | None = None,
    overwrite: bool = False,
    graph_cache_dir: Path | URIPath | None = None,
    pbar_position: int = 1,
    display_id: str | None = None,
    device: str = "auto",
    no_neighborhood: bool = False,
) -> tuple[str, dict | None]:
    """Process a single slide to compute per-edge neighborhood composition."""

    slide_id = wsi_path.stem
    ecomp_csv = results_dir / "ecomp-outputs-csv" / f"{slide_id}.csv"

    if not overwrite and ecomp_csv.exists():
        return slide_id, True

    desc = display_id or slide_id
    inner = tqdm(
        total=len(_WORKER_STEPS),
        desc=desc,
        position=pbar_position,
        leave=False,
        dynamic_ncols=True,
        # Steps are heterogeneous (load CSV is ms; k-hop neighbors is
        # seconds-to-minutes), so the default rate/ETA fields would lie.
        # Show only what we actually know: percent, fraction, elapsed,
        # current step name (set via set_postfix_str).
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed} {postfix}]",
    )

    def _step(name: str) -> None:
        inner.set_postfix_str(f"{name:<{_STEP_LABEL_W}}")
        inner.update(1)

    # --- MPP resolution (µm → px) -------------------------------------------
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
    else:
        # Without a cache, fall back to inline Delaunay (rare path).
        from .insight_helpers import _delaunay_full

        simplices, src, dst, lengths = _delaunay_full(centers)
        cache = {
            "simplices": simplices,
            "edges_source": src,
            "edges_target": dst,
            "edges_length": lengths,
        }

    # Prune edges by max_edge_px.
    length_mask = cache["edges_length"] < max_edge_px
    edge_src = cache["edges_source"][length_mask].astype(np.int64)
    edge_dst = cache["edges_target"][length_mask].astype(np.int64)
    edge_len_px = cache["edges_length"][length_mask].astype(np.float64)
    edges = np.stack([edge_src, edge_dst], axis=1)
    # `edges` already has source < target from _delaunay_full.

    E = edges.shape[0]
    if E == 0:
        inner.close()
        # Write an empty CSV with expected header so downstream tools don't fail.
        ecomp_csv.parent.mkdir(parents=True, exist_ok=True)
        with ecomp_csv.open("w", encoding="utf-8", newline="") as fp:
            fp.write("edge_id\n")
        return slide_id, True
    _step("triangulate")

    # --- Fast path: skip line graph + k-hop entirely -----------------------
    if no_neighborhood:
        _step("line graph")  # advance pbar — skipped
        _step("k-hop nbrs")  # advance pbar — skipped

        cell_type_array = nodes_df["cell_type"].to_numpy()
        all_types = sorted(c.removeprefix("prob_") for c in prob_columns)
        K = len(all_types)
        ct_to_idx = {t: i for i, t in enumerate(all_types)}
        ct_to_idx_get = np.vectorize(ct_to_idx.__getitem__, otypes=[np.int64])
        ct_a_idx = ct_to_idx_get(cell_type_array[edge_src])
        ct_b_idx = ct_to_idx_get(cell_type_array[edge_dst])
        lo = np.minimum(ct_a_idx, ct_b_idx)
        hi = np.maximum(ct_a_idx, ct_b_idx)
        swap = ct_a_idx > ct_b_idx
        v1 = np.where(swap, edge_dst, edge_src)
        v2 = np.where(swap, edge_src, edge_dst)
        edge_type_idx = (lo * (2 * K - lo + 1) // 2 + (hi - lo)).astype(np.int64)
        vocab_arr = np.asarray(_edge_type_vocab(all_types), dtype=object)
        all_types_arr = np.asarray(all_types, dtype=object)
        ct1 = all_types_arr[lo]
        ct2 = all_types_arr[hi]
        edge_type = vocab_arr[edge_type_idx]
        edge_len_um = edge_len_px * mpp
        mid_x = ((centers[edge_src, 0] + centers[edge_dst, 0]) / 2).astype(np.int32)
        mid_y = ((centers[edge_src, 1] + centers[edge_dst, 1]) / 2).astype(np.int32)

        region_cols = [c for c in nodes_df.columns if c.startswith("region_prob_")]
        data: dict[str, np.ndarray] = {
            "edge_id": np.arange(E, dtype=np.int64),
            "vertex_1_id": v1.astype(np.int64),
            "vertex_2_id": v2.astype(np.int64),
            "center_x": mid_x,
            "center_y": mid_y,
            "edge_length_um": edge_len_um,
            "cell_type_1": ct1,
            "cell_type_2": ct2,
            "edge_type": edge_type,
        }
        if region_cols:
            region_labels = [c.removeprefix("region_prob_") for c in region_cols]
            region_probs = nodes_df[region_cols].to_numpy(dtype=np.float32)
            edge_region_idx = (region_probs[edge_src] + region_probs[edge_dst]).argmax(
                axis=1
            )
            data["center_region"] = np.array(region_labels, dtype=object)[
                edge_region_idx
            ]
        _step("edge comp.")

        out_df = pd.DataFrame(data)
        ecomp_csv.parent.mkdir(parents=True, exist_ok=True)
        with critical_section(f"saving ecomp output for {slide_id}"):
            with ecomp_csv.open("w", encoding="utf-8", newline="") as fp:
                out_df.to_csv(fp, index=False)
        _step("save outputs")
        inner.close()
        return slide_id, True

    # --- Line graph ---------------------------------------------------------
    use_gpu = str(device).lower() == "cuda"
    if use_gpu:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        from .simplex_helpers_gpu import build_line_graph_gpu
        from .simplex_helpers_gpu import k_hop_adjacency_matrix_gpu

        L_gpu = build_line_graph_gpu(edges, num_vertices=N)
        _step("line graph")

        A_k_gpu = k_hop_adjacency_matrix_gpu(L_gpu, ecomp_k)
        _step("k-hop nbrs")
    else:
        L = build_line_graph(edges, num_vertices=N)
        _step("line graph")

        A_k = k_hop_adjacency_matrix(L, ecomp_k)
        _step("k-hop nbrs")

    # --- Per-edge features --------------------------------------------------
    cell_type_array = nodes_df["cell_type"].to_numpy()
    all_types = sorted(c.removeprefix("prob_") for c in prob_columns)
    edge_type_vocab = _edge_type_vocab(all_types)

    # Map each cell type to its integer index in ``all_types``.  Building the
    # edge-type index from these integers is O(E) numpy, avoiding per-string
    # hashing or ``np.char.add`` concatenation (both are Python-level and
    # were measured to dominate ecomp runtime before this refactor).
    K = len(all_types)
    ct_to_idx = {t: i for i, t in enumerate(all_types)}
    ct_to_idx_get = np.vectorize(ct_to_idx.__getitem__, otypes=[np.int64])
    ct_a_idx = ct_to_idx_get(cell_type_array[edge_src])
    ct_b_idx = ct_to_idx_get(cell_type_array[edge_dst])

    # Canonicalize (lo <= hi) so ct1 always points to the smaller index.
    lo = np.minimum(ct_a_idx, ct_b_idx)
    hi = np.maximum(ct_a_idx, ct_b_idx)
    swap = ct_a_idx > ct_b_idx

    # Also reorder vertex ids correspondingly so vertex_1 corresponds to cell_type_1.
    v1 = np.where(swap, edge_dst, edge_src)
    v2 = np.where(swap, edge_src, edge_dst)

    # combinations_with_replacement(all_types, 2) flat index:
    #   index(a, b) = a*(2K - a + 1) // 2 + (b - a), for 0 <= a <= b < K.
    # This matches the ordering used by _edge_type_vocab().
    edge_type_idx = (lo * (2 * K - lo + 1) // 2 + (hi - lo)).astype(np.int64)

    # Derive string columns by a single O(E) gather (only for CSV output).
    all_types_arr = np.asarray(all_types, dtype=object)
    vocab_arr = np.asarray(edge_type_vocab, dtype=object)
    ct1 = all_types_arr[lo]
    ct2 = all_types_arr[hi]
    edge_type = vocab_arr[edge_type_idx]

    edge_len_um = edge_len_px * mpp
    mid_x = ((centers[edge_src, 0] + centers[edge_dst, 0]) / 2).astype(np.int32)
    mid_y = ((centers[edge_src, 1] + centers[edge_dst, 1]) / 2).astype(np.int32)

    # --- Vectorised neighborhood aggregation via sparse matrix products ----
    V = len(edge_type_vocab)

    if use_gpu:
        T_onehot = cpsp.csr_matrix(
            (
                cp.ones(E, dtype=cp.float32),
                (cp.arange(E, dtype=cp.int64), cp.asarray(edge_type_idx)),
            ),
            shape=(E, V),
        )

        counts_mat = cp.asnumpy((A_k_gpu @ T_onehot).toarray()).astype(np.float64)
        nhood_size = counts_mat.sum(axis=1).astype(np.int32)

        edge_len_um_f32 = edge_len_um.astype(np.float32)
        edge_len_um_gpu = cp.asarray(edge_len_um_f32)
        sum_len = cp.asnumpy(A_k_gpu @ edge_len_um_gpu).astype(np.float64)
        sum_len_sq = cp.asnumpy(A_k_gpu @ (edge_len_um_gpu**2)).astype(np.float64)
    else:
        from scipy.sparse import csr_matrix as _csr

        T_onehot = _csr(
            (
                np.ones(E, dtype=np.float32),
                (np.arange(E, dtype=np.int64), edge_type_idx),
            ),
            shape=(E, V),
        )

        counts_mat = (A_k @ T_onehot).toarray().astype(np.float64)
        nhood_size = counts_mat.sum(axis=1).astype(np.int32)

        edge_len_um_f32 = edge_len_um.astype(np.float32)
        sum_len = np.asarray(A_k @ edge_len_um_f32, dtype=np.float64)
        sum_len_sq = np.asarray(A_k @ (edge_len_um_f32**2), dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_len = np.where(nhood_size > 0, sum_len / nhood_size, np.nan)
        var_len = np.where(
            nhood_size > 0, sum_len_sq / nhood_size - mean_len**2, np.nan
        )
        # Clip tiny negatives from float round-off before sqrt.
        var_len = np.where(var_len < 0, 0.0, var_len)
        std_len = np.sqrt(var_len)

    nhood_mean_len_um = mean_len
    nhood_std_len_um = std_len
    _step("edge comp.")

    # --- Build output DataFrame --------------------------------------------
    denom = np.where(nhood_size > 0, nhood_size, np.nan)

    data: dict[str, np.ndarray] = {
        "edge_id": np.arange(E, dtype=np.int64),
        "vertex_1_id": v1.astype(np.int64),
        "vertex_2_id": v2.astype(np.int64),
        "center_x": mid_x,
        "center_y": mid_y,
        "edge_length_um": edge_len_um,
        "cell_type_1": ct1,
        "cell_type_2": ct2,
        "edge_type": edge_type,
        "neighborhood_size": nhood_size,
        "neighborhood_mean_edge_length_um": nhood_mean_len_um,
        "neighborhood_std_edge_length_um": nhood_std_len_um,
    }
    for j, t in enumerate(edge_type_vocab):
        counts_j = counts_mat[:, j].astype(np.float64)
        data[f"neighborhood_{t}_count"] = counts_j
        data[f"neighborhood_{t}_prop"] = counts_j / denom

    out_df = pd.DataFrame(data)

    ecomp_csv.parent.mkdir(parents=True, exist_ok=True)
    with critical_section(f"saving ecomp output for {slide_id}"):
        with ecomp_csv.open("w", encoding="utf-8", newline="") as fp:
            out_df.to_csv(fp, index=False)
    _step("save outputs")

    inner.close()
    return slide_id, True


def ecomp_generation(
    wsi_dir: str | Path | URIPath | None,
    slide_paths: List[URIPath] | None,
    results_dir: URIPath,
    max_edge_um: float = 25.0,
    ecomp_k: int = 2,
    num_workers: int = 8,
    slide_mpp_lookup: Mapping[str, float] | None = None,
    overwrite: bool = False,
    device: str = "auto",
    no_neighborhood: bool = False,
) -> list[str]:
    """Compute edge neighborhood composition for WSInsight outputs.

    Parameters mirror :func:`ncomp_generation`.

    Returns
    -------
    list[str]
        Slide IDs that failed to process.
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
        normalized = [
            p if isinstance(p, URIPath) else URIPath(str(p)) for p in slide_paths
        ]
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

    ecomp_dir = results_dir / "ecomp-outputs-csv"
    ecomp_dir.mkdir(parents=True, exist_ok=True)

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

    # Resolve device once up-front so we log the backend a single time.
    if str(device).lower() == "auto":
        resolved_device = "cuda" if _gpu_available() else "cpu"
    else:
        resolved_device = str(device).lower()
    _logger.info("ecomp backend: %s", resolved_device)

    short_ids = make_short_ids([wsi_path.stem for wsi_path, _ in jobs])

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(
                _worker,
                wsi_path,
                model_output_csv,
                results_dir,
                max_edge_um,
                ecomp_k,
                slide_mpp_lookup,
                overwrite,
                graph_cache_dir,
                (i % num_workers) + 1,
                resolved_device,
                no_neighborhood,
                display_id=short_ids.get(wsi_path.stem, wsi_path.stem),
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
