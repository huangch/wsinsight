"""Generate H-Plot layers and summary metrics from WSInsight detection outputs."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
from typing import Mapping
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import errors
from ..cancel import cancellable_as_completed
from ..cancel import critical_section
from ..uri_path import URIPath
from ..wsi import _validate_wsi_directory
from ..wsi import get_avg_mpp
from .graph_cache import get_or_build_delaunay
from .insight_helpers import calculate_distance_to_border
from .insight_helpers import compute_cell_center_points
from .insight_helpers import compute_hplot
from .insight_helpers import delaunay_triangulation
from .insight_helpers import identify_border_cells
from .insight_helpers import identify_region_by_cell_function_enrichment
from .insight_helpers import k_hop_neighbors
from .insight_helpers import make_short_ids

_logger = logging.getLogger(__name__)

_WORKER_STEPS = [
    "load CSV",
    "cell centers",
    "triangulate",
    "k-hop nbrs",
    "tumor regs",
    "border cells",
    "layer dists",
    "hplot curve",
    "save outputs",
]
_STEP_LABEL_W = 12  # pad postfix so tqdm bar geometry stays stable across steps


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
    overwrite: bool = False,
    graph_cache_dir: Path | URIPath | None = None,
    base_by: str = "celltype",
    target_by: str = "celltype",
    pbar_position: int = 1,
    display_id: str | None = None,
):
    """Process a single slide to build cell layers, save intermediates, and compute metrics."""

    slide_id = wsi_path.stem
    hplot_csv_name = model_output_csv.name
    hplot_csv = insight_dir / "hplots" / hplot_csv_name
    cells_csv = insight_dir / "cells" / hplot_csv_name

    if not overwrite and hplot_csv.exists() and cells_csv.exists():
        with hplot_csv.open("r", encoding="utf-8") as fp:
            hplot_df = pd.read_csv(fp)
        return slide_id, hplot_df

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
    except Exception as exc:
        _logger.warning("Failed to load CSV for %s: %s", slide_id, exc)
        inner.close()
        return slide_id, None, None
    _step("load CSV")

    prob_columns = [c for c in nodes_df.columns.to_list() if c.startswith("prob_")]
    if not prob_columns and (base_by == "celltype" or target_by == "celltype"):
        inner.close()
        return slide_id, None, None

    predicted_labels = nodes_df[prob_columns].idxmax(axis=1) if prob_columns else None
    prob_prefix = "prob_"

    # Build a case-insensitive lookup: lowered column name -> actual column name
    _col_lower_to_actual = {c.lower(): c for c in prob_columns}

    def _resolve_types(type_list: Sequence[str]) -> set[str]:
        """Map user-supplied type names to actual column names, case-insensitively."""
        resolved = set()
        for t in type_list:
            candidate = f"{prob_prefix}{t}".lower()
            actual = _col_lower_to_actual.get(candidate)
            if actual is not None:
                resolved.add(actual)
        return resolved

    # niche one-hot columns (present only in niche-outputs-csv/cells/<id>.csv).
    niche_columns = [c for c in nodes_df.columns.to_list() if c.startswith("niche_")]
    _niche_lower_to_actual = {c.lower(): c for c in niche_columns}

    def _resolve_niches(type_list: Sequence[str]) -> list[str]:
        """Map niche ids ("2" or "niche_2") to actual niche_ columns, case-insensitively."""
        resolved = []
        for t in type_list:
            key = str(t).strip().lower()
            if not key.startswith("niche_"):
                key = f"niche_{key}"
            actual = _niche_lower_to_actual.get(key)
            if actual is not None and actual not in resolved:
                resolved.append(actual)
        return resolved

    # Aggregate membership columns written by `wsinsight agg`:
    # object_<name>_prob_<name> (1.0 = member, 0.0 = not).
    _obj_lower_to_actual = {
        c.lower(): c
        for c in nodes_df.columns.to_list()
        if c.lower().startswith("object_")
    }

    def _resolve_aggregates(type_list: Sequence[str]) -> list[str]:
        """Map aggregate names to their object_<name>_prob_<name> columns."""
        resolved = []
        for t in type_list:
            name = str(t).strip().lower()
            actual = _obj_lower_to_actual.get(f"object_{name}_prob_{name}")
            if actual is not None and actual not in resolved:
                resolved.append(actual)
        return resolved

    # ---- base membership: cell type (prob idxmax) OR niche one-hot ----
    if base_by == "niche":
        base_niche_cols = _resolve_niches(base_type_list)
        if not base_niche_cols:
            _logger.warning(
                "[%s] None of the base niches %s matched niche_ columns %s. Skipping slide.",
                slide_id,
                sorted(base_type_list),
                sorted(niche_columns),
            )
            inner.close()
            return slide_id, None, None
        nodes_df["is_base_type"] = nodes_df[base_niche_cols].fillna(0).max(axis=1) > 0
    elif base_by == "aggregate":
        base_agg_cols = _resolve_aggregates(base_type_list)
        if not base_agg_cols:
            _logger.warning(
                "[%s] None of the base aggregates %s matched object_<name>_prob_<name> "
                "columns %s. Run `wsinsight agg` first. Skipping slide.",
                slide_id,
                sorted(base_type_list),
                sorted(_obj_lower_to_actual.values()),
            )
            inner.close()
            return slide_id, None, None
        nodes_df["is_base_type"] = nodes_df[base_agg_cols].fillna(0).max(axis=1) > 0
    else:
        base_targets = _resolve_types(base_type_list)
        if not base_targets:
            _logger.warning(
                "[%s] None of the base types %s matched available columns %s (case-insensitive). Skipping slide.",
                slide_id,
                sorted(base_type_list),
                sorted(set(prob_columns)),
            )
            inner.close()
            return slide_id, None, None
        nodes_df["is_base_type"] = predicted_labels.isin(base_targets)

    # ---- target membership: cell type (prob idxmax) OR niche one-hot ----
    if target_by == "niche":
        target_niche_cols = _resolve_niches(target_type_list)
        if not target_niche_cols:
            _logger.warning(
                "[%s] None of the target niches %s matched niche_ columns %s. Skipping slide.",
                slide_id,
                sorted(target_type_list),
                sorted(niche_columns),
            )
            inner.close()
            return slide_id, None, None
        # One-hot membership: max over requested niche_ columns is 0/1 per cell, so
        # its per-layer mean is exactly the niche fraction.
        nodes_df["is_target_type"] = nodes_df[target_niche_cols].fillna(0).max(axis=1) > 0
    elif target_by == "aggregate":
        target_agg_cols = _resolve_aggregates(target_type_list)
        if not target_agg_cols:
            _logger.warning(
                "[%s] None of the target aggregates %s matched object_<name>_prob_<name> "
                "columns %s. Run `wsinsight agg` first. Skipping slide.",
                slide_id,
                sorted(target_type_list),
                sorted(_obj_lower_to_actual.values()),
            )
            inner.close()
            return slide_id, None, None
        nodes_df["is_target_type"] = nodes_df[target_agg_cols].fillna(0).max(axis=1) > 0
    else:
        target_targets = _resolve_types(target_type_list)
        if not target_targets:
            _logger.warning(
                "[%s] None of the target types %s matched available columns %s (case-insensitive). Skipping slide.",
                slide_id,
                sorted(target_type_list),
                sorted(set(prob_columns)),
            )
            inner.close()
            return slide_id, None, None
        nodes_df["is_target_type"] = predicted_labels.isin(target_targets)
    nodes_df = compute_cell_center_points(nodes_df)
    _step("cell centers")

    centers = nodes_df[["center_x", "center_y"]].values
    if graph_cache_dir is not None:
        edges_df = get_or_build_delaunay(
            graph_cache_dir, slide_id, centers, mpp, max_neighbor_distance_px
        )
    else:
        edges_df = delaunay_triangulation(centers, max_neighbor_distance_px)
    _step("triangulate")

    if "source" not in edges_df.columns or "target" not in edges_df.columns:
        inner.close()
        return slide_id, None, None

    k_neighbors_results, A_sparse, Mk_sparse = k_hop_neighbors(
        len(nodes_df), edges_df, hplot_k
    )
    _step("k-hop nbrs")

    nodes_df = identify_region_by_cell_function_enrichment(
        k_neighbors_results, nodes_df, hplot_N, hplot_R, Mk_sparse=Mk_sparse
    )
    _step("tumor regs")

    nodes_df = identify_border_cells(nodes_df, {}, A_sparse=A_sparse)
    _step("border cells")

    nodes_df = calculate_distance_to_border(nodes_df, {}, A_sparse=A_sparse)
    _step("layer dists")

    # ``distance_to_border`` is retained so ``wsinsight import --include hplot``
    # can carry it per cell (as ``hplot_distance_to_border``). The two boolean
    # base-region flags remain internal scaffolding and are still dropped.
    _drop_cols = ["is_base_region", "is_base_border"]
    with critical_section(f"saving hplot outputs for {slide_id}"):
        with cells_csv.open("w", encoding="utf-8", newline="") as fp:
            nodes_df.drop(
                columns=[c for c in _drop_cols if c in nodes_df.columns]
            ).to_csv(fp, index=False)

        hplot_df = compute_hplot(nodes_df, edges_df, mpp=mpp)
        _step("hplot curve")

        with hplot_csv.open("w", encoding="utf-8", newline="") as fp:
            hplot_df.to_csv(fp, index=False)

        _step("save outputs")

    inner.close()
    return slide_id, hplot_df


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


def hplot_finalize(output_dir: URIPath, overwrite: bool = False) -> None:
    """Rebuild hplot-outputs.csv from per-slide intermediates.

    Reads all per-slide CSV files written by hplot_generation into
    ``output_dir/hplot-outputs-csv/hplots/`` and assembles the aggregated
    summary file at the top level of ``output_dir``.

    When *overwrite* is False and the summary file already exists, the
    function returns without modifying anything.
    """

    hplot_hplots_csv = output_dir / "hplot-outputs.csv"

    if not overwrite and hplot_hplots_csv.exists():
        print("hplot-outputs.csv already exists. Use --overwrite to regenerate.")
        return

    hplot_outputs_csv_dir = output_dir / "hplot-outputs-csv"
    hplots_dir = hplot_outputs_csv_dir / "hplots"

    hplot_files = sorted(hplots_dir.iterdir()) if hplots_dir.exists() else []

    hplot_files = [f for f in hplot_files if f.name.endswith(".csv")]

    if not hplot_files:
        raise ValueError(
            f"No per-slide hplot CSV files found under {hplot_outputs_csv_dir}."
        )

    print(
        f"Found {len(hplot_files)} per-slide hplot CSV(s) in {hplots_dir}. "
        "Note: finalize aggregates whatever is present — make sure every parallel "
        "subset job has finished before relying on this total."
    )

    _COL_RENAME = {
        "target_type_prop": "target_prop",
        "target_type_count": "target_count",
        "base_type_prop": "base_prop",
        "base_type_count": "base_count",
        "all_type_count": "all_count",
    }
    _COL_ORDER = [
        "id",
        "layer",
        "target_prop",
        "target_count",
        "base_prop",
        "base_count",
        "all_count",
        "distance_um",
    ]

    hplot_frames: list[pd.DataFrame] = []
    for csv_file in tqdm(hplot_files, desc="Assembling hplot CSVs", unit="slide"):
        slide_id = csv_file.stem
        with csv_file.open("r", encoding="utf-8") as fp:
            df = pd.read_csv(fp)
        df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
        df = df[np.isfinite(df["layer"])].copy()
        if df.empty:
            continue
        df["layer"] = df["layer"].astype(int)
        src_cols = [
            c
            for c in [
                "layer",
                "target_type_prop",
                "target_type_count",
                "base_type_prop",
                "base_type_count",
                "all_type_count",
                "distance_um",
            ]
            if c in df.columns
        ]
        df = df[src_cols].copy()
        df.rename(columns=_COL_RENAME, inplace=True)

        # Gap-fill: ensure every integer layer in [min, max] has a row
        mn, mx = int(df["layer"].min()), int(df["layer"].max())
        layer_lookup = df.set_index("layer").to_dict("index")
        rows = []
        for layer in range(mn, mx + 1):
            entry = layer_lookup.get(layer, {})
            rows.append(
                {
                    "id": slide_id,
                    "layer": layer,
                    "target_prop": entry.get("target_prop", np.nan),
                    "target_count": entry.get("target_count", np.nan),
                    "base_prop": entry.get("base_prop", np.nan),
                    "base_count": entry.get("base_count", np.nan),
                    "all_count": entry.get("all_count", np.nan),
                    "distance_um": entry.get("distance_um", np.nan),
                }
            )
        hplot_frames.append(pd.DataFrame(rows, columns=_COL_ORDER))

    if hplot_frames:
        merged_hplot = pd.concat(hplot_frames, ignore_index=True)
        merged_hplot.drop_duplicates(subset=["id", "layer"], keep="last", inplace=True)
        merged_hplot.sort_values(["id", "layer"], inplace=True, ignore_index=True)
        with critical_section("saving aggregated hplot-outputs.csv"):
            with hplot_hplots_csv.open("w", encoding="utf-8", newline="") as fp:
                merged_hplot.to_csv(fp, index=False)
        print(
            f"Wrote {hplot_hplots_csv} aggregating "
            f"{merged_hplot['id'].nunique()} slide(s)."
        )
    else:
        print("No non-empty per-slide hplot CSVs to aggregate; nothing written.")


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
    overwrite: bool = False,
    base_by: str = "celltype",
    target_by: str = "celltype",
    model_output_subdir: str = "model-outputs-csv",
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

    model_output_dir = results_dir / model_output_subdir
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model_output_paths = [
        model_output_dir / p.with_suffix(".csv").name for p in slide_paths
    ]
    if len(model_output_paths) != len(slide_paths):
        raise errors.ResultsDirectoryNotFound(
            "The 'model-outputs-csv' and image directory were mismatched."
        )

    hplot_dir = results_dir / "hplot-outputs-csv"
    hplot_dir.mkdir(parents=True, exist_ok=True)
    hplot_hplots_dir = hplot_dir / "hplots"
    hplot_hplots_dir.mkdir(parents=True, exist_ok=True)
    hplot_cells_dir = hplot_dir / "cells"
    hplot_cells_dir.mkdir(parents=True, exist_ok=True)

    hplot_hplots_csv = results_dir / "hplot-outputs.csv"

    failed_generation: list[str] = []
    base_types = list(base_type_list or [])
    target_types = list(target_type_list or [])
    if not base_types or not target_types:
        raise ValueError("base_type_list and target_type_list must be provided")

    hplot_df = pd.DataFrame(
        {
            "id": [],
            "layer": [],
            "target_prop": [],
            "target_count": [],
            "base_prop": [],
            "base_count": [],
            "all_count": [],
            "distance_um": [],
        }
    )

    graph_cache_dir = results_dir / "graphs"
    graph_cache_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for wsi_path, model_output_csv in zip(
        slide_paths, model_output_paths, strict=False
    ):
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
                overwrite,
                graph_cache_dir,
                base_by,
                target_by,
            )
        )

    if not jobs:
        return failed_generation

    short_ids = make_short_ids([args[0].stem for args in jobs])

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        future_to_id: dict = {}
        futures = []
        for i, args in enumerate(jobs):
            fut = ex.submit(_worker, *args, (i % num_workers) + 1,
                            display_id=short_ids.get(args[0].stem, args[0].stem))
            future_to_id[fut] = args[0].stem
            futures.append(fut)
        outer = tqdm(
            total=len(futures),
            desc="Slides",
            position=0,
            leave=True,
            unit="slide",
            dynamic_ncols=True,
        )
        for f in cancellable_as_completed(futures, ex):
            try:
                image_id, df = f.result()
            except Exception as exc:
                # One slide's unhandled error must not abort the whole run;
                # record it and keep processing the remaining slides.
                sid = future_to_id.get(f, "<unknown>")
                _logger.warning("H-Plot generation failed for %s: %s", sid, exc)
                failed_generation.append(sid)
                outer.update(1)
                continue

            if df is None:
                failed_generation.append(image_id)
                outer.update(1)
                continue

            clean_df = df.copy()
            clean_df["layer"] = pd.to_numeric(clean_df["layer"], errors="coerce")
            required_cols = [
                "layer",
                "target_type_prop",
                "target_type_count",
                "base_type_prop",
                "base_type_count",
                "all_type_count",
                "distance_um",
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
                    row.get("base_type_prop", np.nan),
                    row.get("base_type_count", np.nan),
                    row.get("all_type_count", np.nan),
                    row.get("distance_um", np.nan),
                )
                for layer, row in clean_df.set_index("layer")[
                    [
                        "target_type_prop",
                        "target_type_count",
                        "base_type_prop",
                        "base_type_count",
                        "all_type_count",
                        "distance_um",
                    ]
                ].iterrows()
            }

            for layer in range(mn, mx + 1):
                (
                    target_prop,
                    target_count,
                    base_prop,
                    base_count,
                    all_count,
                    distance_um,
                ) = layer_lookup.get(
                    layer, (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                )
                hplot_df.loc[len(hplot_df)] = [
                    image_id,
                    layer,
                    target_prop,
                    target_count,
                    base_prop,
                    base_count,
                    all_count,
                    distance_um,
                ]

            outer.update(1)
        outer.close()

    if hplot_hplots_csv.exists():
        with hplot_hplots_csv.open("r", encoding="utf-8") as fp:
            hplot_df = upsert_by_key(pd.read_csv(fp), hplot_df, key="id")

    with critical_section("saving hplot-outputs.csv"):
        with hplot_hplots_csv.open("w", encoding="utf-8", newline="") as fp:
            hplot_df.to_csv(fp, index=False)

    return failed_generation
