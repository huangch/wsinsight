"""Generate H-Plot layers and summary metrics from WSInsight detection outputs."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence, List, Mapping

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import errors
from ..wsi import _validate_wsi_directory, get_avg_mpp
from ..uri_path import URIPath

from .insight_helpers import (compute_cell_center_points,
                              delaunay_triangulation,
                              k_hop_neighbors,
                              compute_enrichment_index,
                              identify_region_by_cell_function_enrichment,
                              calculate_distance_to_border,
                              identify_border_cells,
                              compute_hplot,
                              compute_hmetrics,
                              )

_WORKER_STEPS = [
    "load CSV",
    "cell centers",
    "triangulation",
    "k-hop neighbors",
    "enrichment index",
    "tumor regions",
    "border cells",
    "layer distances",
    "hplot curve",
    "hmetrics",
    "save outputs",
]


def _worker(
    wsi_path: URIPath,
    model_output_csv: URIPath,
    insight_dir: URIPath,
    max_neighbor_distance_um: float,
    base_type_list: Sequence[str],
    target_type_list: Sequence[str],
    hplot_k: int,
    hplot_N: int,
    hplot_R: float,
    range_min: int | None,
    range_max: int | None,
    samples_with_valid_range_only: bool,
    slide_mpp_lookup: Mapping[str, float] | None = None,
    pbar_position: int = 1,
):
    """Process a single slide to build cell layers, save intermediates, and compute metrics."""

    slide_id = wsi_path.stem
    hplot_csv_name = wsi_path.with_name(slide_id).with_suffix(".csv").name
    hmetric_json_name = wsi_path.with_name(slide_id).with_suffix(".json").name
    hplot_csv = insight_dir / "hplots" / hplot_csv_name
    hmetric_json = insight_dir / "hmetrics" / hmetric_json_name
    cells_csv = insight_dir / "cells" / hplot_csv_name

    if hplot_csv.exists() and hmetric_json.exists() and cells_csv.exists():
        with hplot_csv.open("r", encoding="utf-8") as fp:
            hplot_df = pd.read_csv(fp)
        with hmetric_json.open("r", encoding="utf-8") as fp:
            hmetric_dict = json.load(fp)
        return slide_id, hplot_df, hmetric_dict

    # desc = slide_id if len(slide_id) <= 32 else slide_id[:29] + "..."
    # inner = tqdm(
    #     total=len(_WORKER_STEPS),
    #     desc=desc,
    #     position=pbar_position,
    #     leave=False,
    #     unit="step",
    #     dynamic_ncols=True,
    # )

    def _step(name: str) -> None:
        pass
        # inner.set_postfix_str(name)
        # inner.update(1)

    mpp = None
    if slide_mpp_lookup:
        # Prefer cached spacing derived during patch extraction (avoids re-reading remote WSIs).
        mpp = slide_mpp_lookup.get(slide_id) or slide_mpp_lookup.get(str(wsi_path))
    if mpp is None:
        mpp = get_avg_mpp(wsi_path)
    max_neighbor_distance_px = max_neighbor_distance_um / mpp

    try:
        with model_output_csv.open("r", encoding="utf-8") as fp:
            nodes_df = pd.read_csv(fp)
    except Exception:
        # inner.close()
        return slide_id, None, None
    _step("load CSV")

    prob_columns = [c for c in nodes_df.columns.to_list() if c.startswith("prob_")]
    if not prob_columns:
        # inner.close()
        return slide_id, None, None

    predicted_labels = nodes_df[prob_columns].idxmax(axis=1)
    prob_prefix = "prob_"
    base_targets = {f"{prob_prefix}{bt}" for bt in base_type_list}
    target_targets = {f"{prob_prefix}{tt}" for tt in target_type_list}
    nodes_df["is_base_type"] = predicted_labels.isin(base_targets)
    nodes_df["is_target_type"] = predicted_labels.isin(target_targets)
    nodes_df = compute_cell_center_points(nodes_df)
    _step("cell centers")

    edges_df = delaunay_triangulation(nodes_df[["center_x", "center_y"]].values, max_neighbor_distance_px)
    _step("triangulation")

    if "source" not in edges_df.columns or "target" not in edges_df.columns:
        # inner.close()
        return slide_id, None, None

    k_neighbors_results, A_sparse, Mk_sparse = k_hop_neighbors(len(nodes_df), edges_df, hplot_k)
    _step("k-hop neighbors")

    nodes_df = compute_enrichment_index(nodes_df, k_neighbors_results, Mk_sparse=Mk_sparse)
    _step("enrichment index")

    nodes_df = identify_region_by_cell_function_enrichment(
        k_neighbors_results, nodes_df, hplot_N, hplot_R, Mk_sparse=Mk_sparse
    )
    _step("tumor regions")

    nodes_df = identify_border_cells(nodes_df, {}, A_sparse=A_sparse)
    _step("border cells")

    nodes_df = calculate_distance_to_border(nodes_df, {}, A_sparse=A_sparse)
    _step("layer distances")

    with cells_csv.open("w", encoding="utf-8", newline="") as fp:
        nodes_df.to_csv(fp, index=False)

    hplot_df = compute_hplot(nodes_df, edges_df)
    _step("hplot curve")

    with hplot_csv.open("w", encoding="utf-8", newline="") as fp:
        hplot_df.to_csv(fp, index=False)

    hmetric_dict = compute_hmetrics(
        hplot_df=hplot_df,
        range_min=range_min,
        range_max=range_max,
        hplot_samples_with_valid_range_only=samples_with_valid_range_only,
    )
    _step("hmetrics")

    with hmetric_json.open("w", encoding="utf-8") as fp:
        json.dump(hmetric_dict, fp, indent=2)
    _step("save outputs")

    # inner.close()
    return slide_id, hplot_df, hmetric_dict


def upsert_by_key(df_old: pd.DataFrame, df_new: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Update/insert rows from df_new into df_old using a unique key.
    - New wins on key clashes (entire row overwrite, including NaNs).
    - Rows in df_new with duplicate keys -> keep the last occurrence.
    - Columns are aligned to df_old's columns (extra cols in df_new are ignored).
    - Preserves original df_old row order; brand-new keys are appended in the
      order they appear (last occurrence) in df_new.
    Returns a NEW DataFrame.
    """

    if key not in df_old.columns or key not in df_new.columns:
        raise KeyError(f"Key column '{key}' must exist in both DataFrames.")

    # 1) Align columns to df_old's schema (safe even if already identical)
    cols = list(df_old.columns)
    new_aligned = df_new.reindex(columns=cols)

    # 2) Ensure df_new is unique on key: keep the last (newest) occurrence
    new_dedup = new_aligned.drop_duplicates(subset=[key], keep="last")

    # 3) Set indices by key for clean overwrite semantics
    old_idx = df_old.set_index(key).copy()
    new_idx = new_dedup.set_index(key)

    # 4) Overwrite existing keys (including NaNs; full-row replace)
    common = old_idx.index.intersection(new_idx.index)
    if len(common):
        old_idx.loc[common] = new_idx.loc[common]

    # 5) Append brand-new keys at the end (preserve df_new order for new keys)
    new_only = new_idx.index.difference(old_idx.index)
    if len(new_only):
        old_idx = pd.concat([old_idx, new_idx.loc[new_only]], axis=0)

    # 6) Restore key as a column; df_old order preserved, new keys appended
    out = old_idx.reset_index()
    return out


def hplot_generation(
    wsi_dir: str | Path | URIPath | None,
    slide_paths: List[URIPath] | None,
    results_dir: URIPath,
    base_type_list: Sequence[str] | None = None,
    target_type_list: Sequence[str] | None = None,
    max_neighbor_distance_um: float = 25.0,
    hplot_k: int = 2,
    hplot_N: int = 8,
    hplot_R: float = 0.5,
    hplot_range_max: int | None = None,
    hplot_range_min: int | None = None,
    hplot_samples_with_valid_range_only: bool = False,
    num_workers: int = 8,
    slide_mpp_lookup: Mapping[str, float] | None = None,
) -> list[str]:
    """Compute H-Plot layers/metrics for WSInsight outputs and persist aggregated CSVs."""

    def _as_uri_path(path_like: str | Path | URIPath | None) -> URIPath | None:
        if path_like is None:
            return None
        if isinstance(path_like, URIPath):
            return path_like
        return URIPath(str(path_like))

    results_dir = _as_uri_path(results_dir)  # type: ignore[assignment]
    if results_dir is None:  # pragma: no cover - signature requires a value
        raise ValueError("results_dir must be provided")
    if not results_dir.exists():
        raise errors.ResultsDirectoryNotFound(results_dir)

    wsi_dir_path = _as_uri_path(wsi_dir)
    if wsi_dir_path is not None and not wsi_dir_path.exists():
        raise errors.WholeSlideImageDirectoryNotFound(
            f"directory not found: {wsi_dir_path}"
        )

    if slide_paths is not None:
        normalized_slide_paths = [
            p if isinstance(p, URIPath) else URIPath(str(p)) for p in slide_paths
        ]
        # Slide path inputs can originate from earlier patching runs and may not exist locally
        # during inference; only their filenames are required to align CSV outputs.
    elif wsi_dir_path is not None:
        normalized_slide_paths = [p for p in wsi_dir_path.iterdir() if p.is_file()]
    else:
        raise ValueError("slide_paths must be provided when wsi_dir is None")

    if not normalized_slide_paths:
        context = wsi_dir_path or "provided slide paths"
        raise errors.WholeSlideImagesNotFound(context)

    if wsi_dir_path is not None:
        _validate_wsi_directory(wsi_dir_path)
    else:
        stems = [p.stem for p in normalized_slide_paths]
        if len(stems) != len(set(stems)):
            raise errors.DuplicateFilePrefixesFound(
                "A slide with the same prefix but different extensions has been found"
            )

    slide_paths = normalized_slide_paths

    model_output_dir = results_dir / "model-outputs-csv"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model_output_paths = [model_output_dir / p.with_suffix(".csv").name for p in slide_paths]
    if len(model_output_paths) != len(slide_paths):
        raise errors.ResultsDirectoryNotFound(
            "The 'model-outputs-csv' and image directory were mismatched."
        )

    hplot_dir = results_dir / "hplot-outputs-csv"
    hplot_dir.mkdir(parents=True, exist_ok=True)
    hplot_hplots_dir = hplot_dir / "hplots"
    hplot_hplots_dir.mkdir(parents=True, exist_ok=True)
    hplot_hmetrics_dir = hplot_dir / "hmetrics"
    hplot_hmetrics_dir.mkdir(parents=True, exist_ok=True)
    hplot_cells_dir = hplot_dir / "cells"
    hplot_cells_dir.mkdir(parents=True, exist_ok=True)

    hplot_hplots_csv = results_dir / "hplot-outputs.csv"
    hplot_hmetrics_csv = results_dir / "hmetrics-outputs.csv"

    failed_generation: list[str] = []
    base_types = list(base_type_list or [])
    target_types = list(target_type_list or [])
    if not base_types or not target_types:
        raise ValueError("base_type_list and target_type_list must be provided")

    hplot_df = pd.DataFrame(
        {"id": [], "layer": [], "value": [], "target_count": [], "all_count": [], "distance": []}
    )
    hmetrics_df = pd.DataFrame(
        {
            "id": [],
            "valid": [],
            "convergence_distance (intra)": [],
            "abundance_score (intra)": [],
            "penetration_score (intra)": [],
            "layerwise_enrichment_index (intra)": [],
            "global_enrichment_index (intra)": [],
            "weighted_global_enrichment_index (intra)": [],
            "convergence_distance (peri)": [],
            "abundance_score (peri)": [],
            "proximity_score (peri)": [],
            "layerwise_enrichment_index (peri)": [],
            "global_enrichment_index (peri)": [],
            "weighted_global_enrichment_index (peri)": [],
            "exclusion_index": [],
            "desert_index": [],
            "inflammation_index": [],
            "layerwise_enrichment_index": [],
            "global_enrichment_index": [],
            "weighted_global_enrichment_index": [],
        }
    )

    jobs = []
    for wsi_path, model_output_csv in zip(slide_paths, model_output_paths):
        if not model_output_csv.exists():
            failed_generation.append(wsi_path.stem)
            continue
        jobs.append(
            (
                wsi_path,
                model_output_csv,
                hplot_dir,
                max_neighbor_distance_um,
                base_types,
                target_types,
                hplot_k,
                hplot_N,
                hplot_R,
                hplot_range_min,
                hplot_range_max,
                hplot_samples_with_valid_range_only,
                slide_mpp_lookup,
            )
        )

    if not jobs:
        return failed_generation

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(_worker, *args, (i % num_workers) + 1)
            for i, args in enumerate(jobs)
        ]
        outer = tqdm(
            total=len(futures),
            desc="Slides",
            position=0,
            leave=True,
            unit="slide",
            dynamic_ncols=True,
        )
        for f in as_completed(futures):
            image_id, df, hm = f.result()

            if df is None or hm is None:
                failed_generation.append(image_id)
                outer.update(1)
                continue

            clean_df = df.copy()
            clean_df["layer"] = pd.to_numeric(clean_df["layer"], errors="coerce")
            required_cols = [
                "layer",
                "target_type_prop",
                "target_type_count",
                "all_type_count",
                "distance",
            ]
            clean_df = clean_df[np.isfinite(clean_df["layer"])][required_cols]
            if clean_df.empty:
                failed_generation.append(image_id)
                outer.update(1)
                continue

            clean_df["layer"] = clean_df["layer"].astype(int)
            mn = int(clean_df["layer"].min())
            mx = int(clean_df["layer"].max())
            layer_lookup = {
                int(layer): (
                    row.get("target_type_prop", np.nan),
                    row.get("target_type_count", np.nan),
                    row.get("all_type_count", np.nan),
                    row.get("distance", np.nan),
                )
                for layer, row in clean_df.set_index("layer")[
                    ["target_type_prop", "target_type_count", "all_type_count", "distance"]
                ].iterrows()
            }

            for layer in range(mn, mx + 1):
                value, target_count, all_count, distance = layer_lookup.get(
                    layer, (np.nan, np.nan, np.nan, np.nan)
                )
                hplot_df.loc[len(hplot_df)] = [image_id, layer, value, target_count, all_count, distance]

            hmetrics_df.loc[len(hmetrics_df)] = [
                image_id,
                hm["valid"],
                hm["intra"]["convergence_distance"],
                hm["intra"]["abundance_score"],
                hm["intra"]["penetration_score"],
                hm["intra"]["layerwise_enrichment_index"],
                hm["intra"]["global_enrichment_index"],
                hm["intra"]["weighted_global_enrichment_index"],
                hm["peri"]["convergence_distance"],
                hm["peri"]["abundance_score"],
                hm["peri"]["proximity_score"],
                hm["peri"]["layerwise_enrichment_index"],
                hm["peri"]["global_enrichment_index"],
                hm["peri"]["weighted_global_enrichment_index"],
                hm["peri"]["abundance_score"]
                / (1e-6 + hm["peri"]["abundance_score"] + hm["intra"]["abundance_score"]),
                1
                - 0.5 * (hm["intra"]["abundance_score"] + hm["peri"]["abundance_score"]),
                0.5 * (hm["intra"]["abundance_score"] + hm["peri"]["abundance_score"]),
                0.5
                * (
                    hm["peri"]["layerwise_enrichment_index"]
                    + hm["intra"]["layerwise_enrichment_index"]
                ),
                0.5
                * (
                    hm["intra"]["global_enrichment_index"]
                    + hm["peri"]["global_enrichment_index"]
                ),
                0.5
                * (
                    hm["intra"]["weighted_global_enrichment_index"]
                    + hm["peri"]["weighted_global_enrichment_index"]
                ),
            ]

            outer.update(1)
        outer.close()

    if hplot_hplots_csv.exists():
        with hplot_hplots_csv.open("r", encoding="utf-8") as fp:
            hplot_df = upsert_by_key(pd.read_csv(fp), hplot_df, key="id")

    with hplot_hplots_csv.open("w", encoding="utf-8", newline="") as fp:
        hplot_df.to_csv(fp, index=False)

    if hplot_hmetrics_csv.exists():
        with hplot_hmetrics_csv.open("r", encoding="utf-8") as fp:
            hmetrics_df = upsert_by_key(pd.read_csv(fp), hmetrics_df, key="id")

    with hplot_hmetrics_csv.open("w", encoding="utf-8", newline="") as fp:
        hmetrics_df.to_csv(fp, index=False)

    return failed_generation