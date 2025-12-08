"""Generate H-Plot layers and summary metrics from WSInsight detection outputs."""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json 
from .. import errors
from ..wsi import _validate_wsi_directory, get_avg_mpp
from ..uri_path import URIPath

from .insight_helpers import (compute_cell_center_points, 
                              delaunay_triangulation,
                              create_adjacency_list_fast,
                              k_hop_neighbors,
                              compute_enrichment_index,
                              identify_region_by_cell_function_enrichment,
                              calculate_distance_to_border,
                              identify_border_cells,
                              compute_hplot,
                              compute_hmetrics,
                              )


def _worker(wsi_path, model_output_csv, insight_dir, max_neighbor_distance_um,
            wsi_dir, base_type_list, target_type_list, hplot_k, hplot_N, hplot_R,
            range_min, range_max, samples_with_valid_range_only):
    """Process a single slide to build cell layers, save intermediates, and compute metrics."""
    
    hplot_csv_name = wsi_path.with_name(wsi_path.stem).with_suffix(".csv").name
    hmetric_json_name = wsi_path.with_name(wsi_path.stem).with_suffix(".json").name
    hplot_csv = insight_dir / "hplots" / hplot_csv_name
    hmetric_json = insight_dir / "hmetrics" / hmetric_json_name
    cells_csv = insight_dir / "cells" / hplot_csv_name
    
    if Path(cells_csv).exists() and Path(hplot_csv).exists() and Path(hmetric_json).exists():
        hplot_df = pd.read_csv(hplot_csv)
        
        with Path(hmetric_json).open("r", encoding="utf-8") as f:
            hmetric_dict = json.load(f)
            
        return wsi_path.stem, hplot_df, hmetric_dict 
    
    mpp = get_avg_mpp(wsi_path)
    max_neighbor_distance_px = max_neighbor_distance_um / mpp
    
    try:
        nodes_df = pd.read_csv(model_output_csv)
    except Exception:
        return wsi_dir.stem, None, None
    
    # Identify base Cells and target Cells (modifies df_with_centers in place)
    prob_columns = [c for c in nodes_df.columns.to_list() if c.startswith('prob_')]
    for bt in base_type_list: nodes_df['is_base_type'] = nodes_df[prob_columns].idxmax(axis=1) == f'prob_{bt}' 
    for tt in target_type_list: nodes_df['is_target_type'] = nodes_df[prob_columns].idxmax(axis=1) == f'prob_{tt}'
           
    nodes_df = compute_cell_center_points(nodes_df)
    edges_df = delaunay_triangulation(nodes_df[['center_x', 'center_y']].values, max_neighbor_distance_px)
    
    if "source" not in edges_df.columns or "target" not in edges_df.columns:
        return wsi_dir.stem, None, None
         
    # K-hop Neighbors Analysis
    adj_list = create_adjacency_list_fast(edges_df)  # Create adjacency list for border identification
    k_neighbors_results = k_hop_neighbors(nodes_df, adj_list, hplot_k)

    # Compute infiltration index per cell.
    nodes_df = compute_enrichment_index(nodes_df, k_neighbors_results)

    # Cell Function Enrichment Analysis (Identify base Regions)
    nodes_df = identify_region_by_cell_function_enrichment(k_neighbors_results, nodes_df, hplot_N, hplot_R)
    
    # Identify base Region Cells and Border Cells (modifies df_with_cell_flags in place)
    nodes_df = identify_border_cells(nodes_df, adj_list)
    
    # Calculate Distance to Border and Assign Signed Distance (modifies df_with_borders in place)
    nodes_df = calculate_distance_to_border(nodes_df, adj_list)

    nodes_df.to_csv(cells_csv, index=False)
    
    # Compute H-Plot    
    hplot_df = compute_hplot(nodes_df, edges_df)
    
    hplot_df.to_csv(hplot_csv, index=False)

    hmetric_dict = compute_hmetrics(hplot_df=hplot_df,
                                    range_min=range_min,
                                    range_max=range_max,
                                    hplot_samples_with_valid_range_only = samples_with_valid_range_only)
        
    with open(hmetric_json, "w") as f:
        json.dump(hmetric_dict, f, indent=2)
    
    return wsi_path.stem, hplot_df, hmetric_dict


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
    wsi_dir: str | URIPath | None,
    wsi_paths: list[Path] | None,
    results_dir: str | Path,
    base_type_list: List | None=None,
    target_type_list: List | None=None,
    max_neighbor_distance_um: float=25.0,
    hplot_k: int=2,
    hplot_N: int=8,
    hplot_R: float=0.5,
    hplot_range_max: int | None=None,
    hplot_range_min: int | None=None,
    hplot_samples_with_valid_range_only=False,
    num_workers: int = 8,
) -> list[str]:
    """Compute H-Plot layers/metrics for WSInsight outputs and persist aggregated CSVs.

    Slides are enumerated from `wsi_dir`, their corresponding `model-outputs-csv/*.csv`
    files are loaded, and per-slide worker processes build adjacency graphs,
    neighborhood statistics, layer assignments, and immune abundance metrics. The
    resulting per-slide artifacts are stored under `results_dir/hplot-outputs-csv`
    while combined summaries are written to `hplot-outputs.csv` and
    `hmetrics-outputs.csv`.
    """
    # Make sure required directories exist.
    wsi_dir = URIPath(wsi_dir)
    if not wsi_dir.exists():
        raise errors.WholeSlideImageDirectoryNotFound(f"directory not found: {wsi_dir}")
    wsi_paths = [p for p in wsi_dir.iterdir() if p.is_file()]
    if not wsi_paths:
        raise errors.WholeSlideImagesNotFound(wsi_dir)
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise errors.ResultsDirectoryNotFound(results_dir)

    _validate_wsi_directory(wsi_dir)

    # Check patches directory.
    model_output_dir = results_dir / "model-outputs-csv"
    if not model_output_dir.exists():
        raise errors.ResultsDirectoryNotFound(
            "The 'model-outputs-csv' directory was not found in results directory."
        )
    # Create the patch paths based on the whole slide image paths. In effect, only
    # create patch paths if the whole slide image patch exists.
    model_output_paths = [model_output_dir / p.with_suffix(".csv").name for p in wsi_paths]
    
    if len(model_output_paths) != len(wsi_paths):
        raise errors.ResultsDirectoryNotFound(
            "The 'model-outputs-csv' and image directory were mismatched."
        )
    
    hplot_dir = results_dir / "hplot-outputs-csv"
    hplot_dir.mkdir(exist_ok=True)
    
    hplot_hplots_dir = results_dir / "hplot-outputs-csv" / "hplots"
    hplot_hplots_dir.mkdir(exist_ok=True)
    
    hplot_hmetrics_dir = results_dir / "hplot-outputs-csv" / "hmetrics"
    hplot_hmetrics_dir.mkdir(exist_ok=True)
    
    hplot_cells_dir = results_dir / "hplot-outputs-csv" / "cells"
    hplot_cells_dir.mkdir(exist_ok=True)

    hplot_hplots_csv = results_dir / "hplot-outputs.csv"
    hplot_hmetrics_csv = results_dir / "hmetrics-outputs.csv"
    
    failed_generation: list[str] = []

    hplot_df = pd.DataFrame({'id':[], 
                             'layer':[], 
                             'value':[], 
                             'distance':[]})
    
    hmetrics_df = pd.DataFrame({'id': [],
                                'valid': [],
                                'convergence_distance (intra)': [],
                                'abundance_score (intra)': [],
                                'penetration_score (intra)': [],
                                'layerwise_enrichment_index (intra)': [],
                                'global_enrichment_index (intra)': [],
                                'weighted_global_enrichment_index (intra)': [],
                                'convergence_distance (peri)': [],
                                'abundance_score (peri)': [],
                                'proximity_score (peri)': [],
                                'layerwise_enrichment_index (peri)': [],
                                'global_enrichment_index (peri)': [],
                                'weighted_global_enrichment_index (peri)': [],
                                'exclusion_index': [],
                                'desert_index': [],
                                'inflammation_index': [],
                                'layerwise_enrichment_index': [],
                                'global_enrichment_index': [],
                                'weighted_global_enrichment_index': [],
                               })
    
    jobs = []

    for _, (wsi_path, model_output_csv) in enumerate(zip(wsi_paths, model_output_paths)):
        # hplot_csv_name = wsi_path.with_name(wsi_path.stem).with_suffix(".csv").name
        # hplot_csv = hplot_hplots_dir / hplot_csv_name
        
        # if hplot_csv.exists():
        #     print("H-Plot CSV exists... skipping.")
        #     print(hplot_csv)
        #     continue
    
        if not model_output_csv.exists():
            print(f"Skipping because model output not found: {model_output_csv}")
            continue
    
        jobs.append((wsi_path, model_output_csv, hplot_dir, max_neighbor_distance_um,
                    wsi_dir, base_type_list, target_type_list, hplot_k,
                    hplot_N, hplot_R, hplot_range_min, hplot_range_max,
                    hplot_samples_with_valid_range_only))
        
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_worker, *args) for args in jobs]
        pbar = tqdm(total=len(futures))
        for f in as_completed(futures):
            image_id, df, hm = f.result()
            
            if df is not None and hm is not None:
                # Clean and validate 'layer' first
                layers = pd.to_numeric(df['layer'], errors='coerce')          # non-numerics -> NaN
                layers = layers[np.isfinite(layers)]                          # drop NaN/±inf
                if not layers.empty:
                    # decide a policy; here we skip this slide
                    # (or set mn=mx=0 if you prefer)
                 
                    # If 'layer' should be integer-ish, round AFTER cleaning
                    mn = int(np.floor(layers.min()))
                    mx = int(np.ceil(layers.max()))
                    # or if you truly want nearest-integer bounds:
                    # mn = int(np.nanmin(np.rint(layers.values)))
                    # mx = int(np.nanmax(np.rint(layers.values)))
        
                    #
                    # Generating h-plot
                    #
                    
                    for layer in range(mn, mx+1):
                        if layer in df['layer'].tolist():
                            value = df[df['layer']==layer]['target_type_prop'].values[0]
                            distance = df[df['layer']==layer]['distance'].values[0]
                        else:
                            value = distance = np.nan
                    
                        hplot_df.loc[len(hplot_df)] = [image_id, layer, value, distance]
        
                    #
                    # Generating h-metrics
                    #
                    
                    hmetrics_df.loc[len(hmetrics_df)] = [
                        image_id,
                        hm['valid'],
                        hm['intra']['convergence_distance'],
                        hm['intra']['abundance_score'],
                        hm['intra']['penetration_score'],
                        hm['intra']['layerwise_enrichment_index'],
                        hm['intra']['global_enrichment_index'],
                        hm['intra']['weighted_global_enrichment_index'],
                        hm['peri']['convergence_distance'],
                        hm['peri']['abundance_score'],
                        hm['peri']['proximity_score'],
                        hm['peri']['layerwise_enrichment_index'],
                        hm['peri']['global_enrichment_index'],
                        hm['peri']['weighted_global_enrichment_index'],
                        hm['peri']['abundance_score']/(1e-6+hm['peri']['abundance_score']+hm['intra']['abundance_score']), # exclusion_index
                        1-0.5*(hm['intra']['abundance_score']+hm['peri']['abundance_score']), # desert_index
                        0.5*(hm['intra']['abundance_score']+hm['peri']['abundance_score']), # inflammation_index
                        0.5*(hm['peri']['layerwise_enrichment_index']+hm['intra']['layerwise_enrichment_index']), # enrichment_v1_index
                        0.5*(hm['intra']['global_enrichment_index']+hm['peri']['global_enrichment_index']), # infiltration_v1_index
                        0.5*(hm['intra']['weighted_global_enrichment_index']+hm['peri']['weighted_global_enrichment_index']), # infiltration_v1_index
                    ]
            
            pbar.update(1)
    
    
    if hplot_hplots_csv.exists():
        hplot_df = upsert_by_key(pd.read_csv(hplot_hplots_csv), hplot_df, key="id")
            
    hplot_df.to_csv(hplot_hplots_csv, index=False)
    
    if hplot_hmetrics_csv.exists():
        hmetrics_df = upsert_by_key(pd.read_csv(hplot_hmetrics_csv), hmetrics_df, key="id")
    
    hmetrics_df.to_csv(hplot_hmetrics_csv, index=False)
    
    return failed_generation 