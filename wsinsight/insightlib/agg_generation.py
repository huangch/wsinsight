"""Aggregate (``agg``) generation for WSInsight.

For each slide, ``wsinsight agg``
builds the cached Delaunay graph, detects connected, density-gated aggregates
of a chosen set of cell types (``--agg-types``), contracts them into a
quotient graph, and writes three framework-native artifacts, all namespaced by
the product label ``--agg-name``:

1. a per-cell membership group ``object_<name>_prob_<name>`` upserted into each
   ``model-outputs-csv/<slide_id>.csv`` (siblings preserved);
2. a per-aggregate sidecar ``agg-<name>-outputs-csv/<slide_id>.csv`` shaped like
   a model-output CSV (``center_x`` / ``center_y`` / ``prob_<name>`` + features)
   so H-Plot can consume it directly;
3. an ``agg/<name>/`` subgroup inside ``graphs/<slide_id>.h5`` holding the
   quotient graph.

The command carries no distance responsibility: distance-to-tumor-border is
always H-Plot's runtime pass.
"""

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
from ..io.schema import make_object_prefix
from ..uri_path import URIPath
from ..wsi import _validate_wsi_directory
from ..wsi import get_avg_mpp
from .aggregate import aggregate_features
from .aggregate import contract_to_quotient
from .aggregate import identify_aggregates
from .graph_cache import get_or_build_delaunay
from .graph_cache import make_aggregate_params_key
from .graph_cache import write_aggregate_cache
from .insight_helpers import compute_cell_center_points

_logger = logging.getLogger(__name__)


def membership_column(name: str) -> str:
    """Return the per-cell membership column name for an aggregate *name*.

    e.g. ``"tls"`` -> ``"object_tls_prob_tls"`` (a 1-class pseudo-probability
    that rides the existing ``object_*_prob_*`` export-discovery path).
    """
    return f"{make_object_prefix(name)}prob_{name}"


def _slide_mpp(
    wsi_path: URIPath,
    slide_id: str,
    slide_mpp_lookup: Mapping[str, float] | None,
) -> float:
    mpp = None
    if slide_mpp_lookup:
        mpp = slide_mpp_lookup.get(slide_id) or slide_mpp_lookup.get(str(wsi_path))
    if mpp is None:
        mpp = get_avg_mpp(wsi_path)
    return float(mpp)


def _worker(
    wsi_path: URIPath,
    model_output_csv: URIPath,
    results_dir: URIPath,
    name: str,
    agg_types: Sequence[str],
    max_neighbor_distance_um: float,
    k: int,
    N: int,
    R: float,
    min_size: int,
    slide_mpp_lookup: Mapping[str, float] | None = None,
    overwrite: bool = False,
    graph_cache_dir: Path | URIPath | None = None,
) -> tuple[str, bool | None]:
    """Detect, contract, and persist aggregates for a single slide."""
    slide_id = wsi_path.stem
    sidecar_csv = results_dir / f"agg-{name}-outputs-csv" / f"{slide_id}.csv"

    if not overwrite and sidecar_csv.exists():
        return slide_id, True

    mpp = _slide_mpp(wsi_path, slide_id, slide_mpp_lookup)
    max_neighbor_distance_px = max_neighbor_distance_um / mpp

    try:
        with model_output_csv.open("r", encoding="utf-8") as fp:
            df = pd.read_csv(fp)
    except Exception as exc:
        _logger.warning("Failed to load CSV for %s: %s", slide_id, exc)
        return slide_id, None

    prob_columns = [c for c in df.columns if c.startswith("prob_")]
    if not prob_columns:
        _logger.warning("No 'prob_*' columns in %s; skipping.", slide_id)
        return slide_id, None

    df = compute_cell_center_points(df)
    predicted_labels = df[prob_columns].idxmax(axis=1)
    centers = df[["center_x", "center_y"]].to_numpy()

    if graph_cache_dir is not None:
        edges_df = get_or_build_delaunay(
            graph_cache_dir,
            slide_id,
            np.asarray(centers, dtype=np.int32),
            mpp,
            max_neighbor_distance_px,
        )
    else:
        from .insight_helpers import delaunay_triangulation

        edges_df = delaunay_triangulation(centers, max_neighbor_distance_px)

    aggregate_id = identify_aggregates(
        df,
        list(agg_types),
        edges_df,
        k=k,
        N=N,
        R=R,
        min_size=min_size,
        predicted_labels=predicted_labels,
    )

    quotient = contract_to_quotient(aggregate_id, edges_df, centers)
    features_df = aggregate_features(
        df,
        aggregate_id,
        slide_id=slide_id,
        mpp=mpp,
        predicted_labels=predicted_labels.str.removeprefix("prob_"),
    )

    # --- 1. upsert per-cell membership group into model-outputs-csv ----------
    col = membership_column(name)
    df[col] = (aggregate_id >= 0).astype(np.float64)
    with critical_section(f"saving agg membership for {slide_id}"):
        with model_output_csv.open("w", encoding="utf-8", newline="") as fp:
            df.to_csv(fp, index=False)

    # --- 2. sidecar: aggregates as meta-cells (model-output-shaped) ----------
    sidecar_df = features_df.copy()
    sidecar_df[f"prob_{name}"] = 1.0
    sidecar_csv.parent.mkdir(parents=True, exist_ok=True)
    with critical_section(f"saving agg sidecar for {slide_id}"):
        with sidecar_csv.open("w", encoding="utf-8", newline="") as fp:
            sidecar_df.to_csv(fp, index=False)

    # --- 3. quotient graph -> agg/<name>/ subgroup in graphs/<slide>.h5 ------
    if graph_cache_dir is not None:
        params_key = make_aggregate_params_key(
            agg_types=list(agg_types),
            k=k,
            N=N,
            R=R,
            min_size=min_size,
            max_edge_length_px=max_neighbor_distance_px,
        )
        h5path = Path(str(graph_cache_dir)) / f"{slide_id}.h5"
        try:
            write_aggregate_cache(
                h5path,
                name,
                params_key=params_key,
                num_cells=len(df),
                aggregate_centers=quotient["aggregate_centers"],
                aggregate_sizes=quotient["aggregate_sizes"],
                cell_to_aggregate=quotient["cell_to_aggregate"],
                quotient_edges_source=quotient["quotient_edges_source"],
                quotient_edges_target=quotient["quotient_edges_target"],
            )
        except Exception as exc:
            _logger.warning("Failed to write agg cache for %s: %s", slide_id, exc)

    return slide_id, True


def _as_uri_path(p: str | Path | URIPath | None) -> URIPath | None:
    if p is None:
        return None
    return p if isinstance(p, URIPath) else URIPath(str(p))


def agg_generation(
    wsi_dir: str | Path | URIPath | None,
    slide_paths: List[URIPath] | None,
    results_dir: URIPath,
    *,
    name: str,
    agg_types: Sequence[str],
    max_neighbor_distance_um: float = 25.0,
    k: int = 2,
    N: int = 8,
    R: float = 0.5,
    min_size: int = 10,
    num_workers: int = 8,
    slide_mpp_lookup: Mapping[str, float] | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Detect aggregates for every slide and persist the three artifacts.

    Returns the list of slide ids that failed to process.
    """
    results_dir = _as_uri_path(results_dir)  # type: ignore[assignment]
    if results_dir is None:
        raise ValueError("results_dir must be provided")
    if not results_dir.exists():
        raise errors.ResultsDirectoryNotFound(results_dir)

    if not name:
        raise ValueError("an aggregate name (--agg-name) is required")
    if not agg_types:
        raise ValueError("at least one ingredient cell type (--agg-types) is required")

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
    if not model_output_dir.exists():
        raise errors.ResultsDirectoryNotFound(
            f"'model-outputs-csv' not found under {results_dir}"
        )

    sidecar_dir = results_dir / f"agg-{name}-outputs-csv"
    sidecar_dir.mkdir(parents=True, exist_ok=True)

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
                name,
                list(agg_types),
                max_neighbor_distance_um,
                k,
                N,
                R,
                min_size,
                slide_mpp_lookup,
                overwrite,
                graph_cache_dir,
            ): wsi_path.stem
            for (wsi_path, model_output_csv) in jobs
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
