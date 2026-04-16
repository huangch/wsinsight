"""Neighborhood composition (ncomp) generation for WSInsight.

For each cell of a specified target type (or all cells if no target is given),
compute the cell-type composition of its k-hop graph neighborhood built via
Delaunay triangulation — the same graph construction used by H-Plot.

Outputs
-------
Per slide
    ``<results_dir>/ncomp-outputs-csv/<slide_id>.csv``
    One row per target cell with columns:
        center_x, center_y, cell_type,
        neighborhood_size, ncomp_count_<type…>, ncomp_prop_<type…>
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Mapping, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import errors
from ..wsi import _validate_wsi_directory, get_avg_mpp
from ..uri_path import URIPath

from .insight_helpers import (
    compute_cell_center_points,
    delaunay_triangulation,
    k_hop_neighbors,
)

_logger = logging.getLogger(__name__)

_WORKER_STEPS = [
    "load CSV",
    "cell centers",
    "triangulation",
    "k-hop neighbors",
    "nhood composition",
    "save outputs",
]


def _worker(
    wsi_path: URIPath,
    model_output_csv: URIPath,
    results_dir: URIPath,
    max_neighbor_distance_um: float,
    target_type_list: Sequence[str],
    ncomp_k: int,
    slide_mpp_lookup: Mapping[str, float] | None = None,
    overwrite: bool = False,
    pbar_position: int = 1,
) -> tuple[str, dict | None]:
    """Process a single slide to compute per-cell neighborhood composition."""

    slide_id = wsi_path.stem
    ncomp_csv = results_dir / "ncomp-outputs-csv" / f"{slide_id}.csv"

    if not overwrite and ncomp_csv.exists():
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

    # --- MPP resolution (µm → px) -------------------------------------------
    mpp = None
    if slide_mpp_lookup:
        mpp = slide_mpp_lookup.get(slide_id) or slide_mpp_lookup.get(str(wsi_path))
    if mpp is None:
        mpp = get_avg_mpp(wsi_path)
    max_neighbor_distance_px = max_neighbor_distance_um / mpp

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

    # Assign a cell type label to each cell via argmax over prob columns
    predicted_labels = nodes_df[prob_columns].idxmax(axis=1)
    # Strip "prob_" prefix to get plain type names (e.g. "tumor", "lymphocyte")
    nodes_df["cell_type"] = predicted_labels.str.removeprefix("prob_")

    nodes_df = compute_cell_center_points(nodes_df)
    _step("cell centers")

    edges_df = delaunay_triangulation(
        nodes_df[["center_x", "center_y"]].values, max_neighbor_distance_px
    )
    _step("triangulation")

    if "source" not in edges_df.columns or "target" not in edges_df.columns:
        inner.close()
        return slide_id, None

    neighbor_lists, _A, _Mk = k_hop_neighbors(len(nodes_df), edges_df, ncomp_k)
    _step("k-hop neighbors")

    # --- Determine target cells ----------------------------------------------
    all_types = [c.removeprefix("prob_") for c in prob_columns]
    if target_type_list:
        target_mask = nodes_df["cell_type"].isin(target_type_list)
    else:
        target_mask = pd.Series(True, index=nodes_df.index)

    target_indices = nodes_df.index[target_mask].tolist()

    # --- Compute per-cell neighborhood composition --------------------------
    type_counts: dict[str, list] = {t: [] for t in all_types}
    ncomp_sizes: list[int] = []
    cell_types: list[str] = []
    center_xs: list[int] = []
    center_ys: list[int] = []

    cell_type_array = nodes_df["cell_type"].to_numpy()

    for i in target_indices:
        nbrs = [n for n in neighbor_lists[i] if n != i]  # exclude self
        ncomp_size = len(nbrs)
        ncomp_sizes.append(ncomp_size)
        cell_types.append(cell_type_array[i])
        center_xs.append(int(nodes_df.at[i, "center_x"]))
        center_ys.append(int(nodes_df.at[i, "center_y"]))

        if ncomp_size > 0:
            nbr_types = cell_type_array[nbrs]
            for t in all_types:
                type_counts[t].append(int(np.sum(nbr_types == t)))
        else:
            for t in all_types:
                type_counts[t].append(0)
    _step("nhood composition")

    # --- Build per-cell DataFrame -------------------------------------------
    n = len(target_indices)
    ncomp_sizes_arr = np.array(ncomp_sizes, dtype=np.float64)
    denom = np.where(ncomp_sizes_arr > 0, ncomp_sizes_arr, np.nan)

    per_cell_data: dict[str, list | np.ndarray] = {
        "center_x": center_xs,
        "center_y": center_ys,
        "cell_type": cell_types,
        "neighborhood_size": ncomp_sizes,
    }
    for t in all_types:
        counts = np.array(type_counts[t], dtype=np.float64)
        per_cell_data[f"neighborhood_{t}_count"] = counts
        per_cell_data[f"neighborhood_{t}_prop"] = counts / denom

    per_cell_df = pd.DataFrame(per_cell_data)

    # --- Save ---------------------------------------------------------------
    ncomp_csv.parent.mkdir(parents=True, exist_ok=True)
    with ncomp_csv.open("w", encoding="utf-8", newline="") as fp:
        per_cell_df.to_csv(fp, index=False)
    _step("save outputs")

    inner.close()
    return slide_id, True


def ncomp_generation(
    wsi_dir: str | Path | URIPath | None,
    slide_paths: List[URIPath] | None,
    results_dir: URIPath,
    target_type_list: Sequence[str] | None = None,
    max_neighbor_distance_um: float = 25.0,
    ncomp_k: int = 2,
    num_workers: int = 8,
    slide_mpp_lookup: Mapping[str, float] | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Compute neighborhood composition for WSInsight outputs and persist CSVs.

    Parameters
    ----------
    wsi_dir:
        Directory of whole slide images (used for MPP lookup and slide enumeration).
    slide_paths:
        Explicit list of slide URIPaths.  Required when *wsi_dir* is ``None``.
    results_dir:
        Root results directory; must contain a ``model-outputs-csv/`` subdirectory.
    target_type_list:
        Cell types to compute ncomp for.  ``None`` or empty → all cells.
    max_neighbor_distance_um:
        Maximum edge length in µm for the Delaunay graph.
    ncomp_k:
        Number of hops that define the neighborhood radius.
    num_workers:
        Number of slides to process concurrently.
    slide_mpp_lookup:
        Optional pre-computed slide-id → µm/px mapping (avoids reading WSIs).
    overwrite:
        When ``False``, skip slides whose per-slide CSV already exists.

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

    ncomp_dir = results_dir / "ncomp-outputs-csv"
    ncomp_dir.mkdir(parents=True, exist_ok=True)

    target_types = list(target_type_list or [])

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
                max_neighbor_distance_um,
                target_types,
                ncomp_k,
                slide_mpp_lookup,
                overwrite,
                (i % num_workers) + 1,
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
        for f in as_completed(futures):
            slide_id, ok = f.result()
            if not ok:
                failed_generation.append(slide_id)
            outer.update(1)
        outer.close()

    return failed_generation
