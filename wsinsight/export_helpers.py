"""Helpers for the export command: merge per-cell results and write combined CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .uri_path import URIPath

PathLike = Union[Path, URIPath]

# Columns that define patch geometry / identity — never treated as measurements.
_GEOM_COLS = frozenset({
    "minx", "miny", "width", "height",
    "center_x", "center_y",
    "polygon_wkt",
})


def _read_csv(path: PathLike) -> pd.DataFrame:
    if isinstance(path, URIPath):
        with path.open("r", encoding="utf-8") as fp:
            return pd.read_csv(fp)
    return pd.read_csv(path)


def _write_csv(df: pd.DataFrame, path: PathLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(path, URIPath):
        with path.open("w", encoding="utf-8", newline="") as fp:
            df.to_csv(fp, index=False)
    else:
        df.to_csv(path, index=False)


def _ensure_center(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer center_x / center_y if absent (same formula as insight_helpers)."""
    if "center_x" not in df.columns or "center_y" not in df.columns:
        df["center_x"] = np.rint(df["minx"] + df["width"] / 2).astype(np.int32)
        df["center_y"] = np.rint(df["miny"] + df["height"] / 2).astype(np.int32)
    return df


# Per-cell analysis sources that can be merged into export CSVs.
CELL_SOURCES = frozenset({"hplot", "ncomp", "cme", "xenium"})

# Simplex-level sources (edges, triads, aggregates) — exported separately.
SIMPLEX_SOURCES = frozenset({"ecomp", "tcomp"})  # agg:<name> handled dynamically

# Backward compatibility alias
AVAILABLE_SOURCES = CELL_SOURCES


def parse_include_sources(
    include_sources: tuple[str, ...],
    results_dir,  # URIPath
) -> tuple[set[str], set[str]]:
    """Parse and validate --include sources.

    Returns (cell_sources, simplex_sources) where:
    - cell_sources: set of per-cell source names (hplot, ncomp, cme, xenium)
    - simplex_sources: set of simplex source names (ecomp, tcomp, agg:<name>)

    Special values:
    - "all": include all available sources (cell + simplex)
    - "all-cells": include all per-cell sources only
    """
    import re

    if not include_sources:
        # Default: all per-cell sources, no simplex
        return set(), set()

    # Flatten comma-separated values
    raw_sources: list[str] = []
    for item in include_sources:
        raw_sources.extend(s.strip().lower() for s in item.split(",") if s.strip())

    cell_sources: set[str] = set()
    simplex_sources: set[str] = set()

    for src in raw_sources:
        if src == "all":
            # All cell sources + all available simplex sources
            cell_sources.update(CELL_SOURCES)
            simplex_sources.update(SIMPLEX_SOURCES)
            # Also discover any agg-*-outputs-csv directories
            for child in results_dir.iterdir():
                match = re.match(r"agg-([a-z0-9_]+)-outputs-csv", child.name)
                if match and child.is_dir():
                    simplex_sources.add(f"agg:{match.group(1)}")
        elif src == "all-cells":
            cell_sources.update(CELL_SOURCES)
        elif src in CELL_SOURCES:
            cell_sources.add(src)
        elif src in SIMPLEX_SOURCES:
            simplex_sources.add(src)
        elif src.startswith("agg:"):
            # Validate agg name format
            agg_name = src.split(":", 1)[1]
            if re.match(r"^[a-z0-9_]+$", agg_name):
                simplex_sources.add(src)
            else:
                raise ValueError(
                    f"Invalid aggregate name '{agg_name}' — must be lowercase alphanumeric with underscores."
                )
        else:
            valid = sorted(CELL_SOURCES | SIMPLEX_SOURCES) + ["all", "all-cells", "agg:<name>"]
            raise ValueError(
                f"Unknown source '{src}'. Valid sources: {', '.join(valid)}"
            )

    return cell_sources, simplex_sources


def _read_xenium_summaries(h5ad_path: PathLike, slide_id: str) -> pd.DataFrame:
    """Extract per-cell summaries from an imported-xenium h5ad file.

    Returns a DataFrame indexed by model-output row index (0-based) with columns:
    * xenium_barcode — original Xenium cell barcode
    * xenium_total_counts — total UMI counts per cell
    * xenium_n_genes — number of genes detected (count > 0)
    * xenium_matched — always True for matched cells

    Only cells with a valid model_cell_id (matched to a WSInsight detection) are
    included.
    """
    import anndata

    # Handle URIPath by materializing to local path
    if isinstance(h5ad_path, URIPath):
        local_path = Path(h5ad_path.materialize())
    else:
        local_path = Path(h5ad_path)

    adata = anndata.read_h5ad(local_path)

    # Filter to matched cells only
    if "model_cell_id" not in adata.obs.columns:
        return pd.DataFrame()

    matched_mask = adata.obs["model_cell_id"].notna()
    if not matched_mask.any():
        return pd.DataFrame()

    adata = adata[matched_mask].copy()

    # Parse row index from model_cell_id (format: "{sample_id}-{row_index}")
    def parse_row_idx(cell_id: str) -> int:
        # cell_id is like "SAMPLE-123" where 123 is the row index
        parts = cell_id.rsplit("-", 1)
        return int(parts[-1]) if len(parts) == 2 else -1

    row_indices = adata.obs["model_cell_id"].apply(parse_row_idx)

    # Compute summaries
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    summaries = pd.DataFrame({
        "_row_idx": row_indices.values,
        "xenium_barcode": adata.obs.index.values,
        "xenium_total_counts": np.asarray(X.sum(axis=1)).flatten(),
        "xenium_n_genes": np.asarray((X > 0).sum(axis=1)).flatten(),
        "xenium_matched": True,
    })

    # Filter out invalid row indices and set as index
    summaries = summaries[summaries["_row_idx"] >= 0].copy()
    summaries = summaries.set_index("_row_idx")

    return summaries


def build_export_csvs(
    results_dir: PathLike,
    *,
    overwrite: bool = True,
    include: Optional[frozenset[str]] = None,
) -> List[PathLike]:
    """Left-join all per-cell analysis outputs into ``export-csv/``.

    Parameters
    ----------
    results_dir
        Root results directory containing model-outputs-csv/ and other outputs.
    overwrite
        Re-build even if export-csv/ already contains up-to-date files.
    include
        Set of source names to include: {"hplot", "ncomp", "cme", "xenium"}.
        If None or empty, all available sources are included.

    Sources (all optional — skipped when absent or not in ``include``):
    * ``model-outputs-csv/<slide>.csv``  — base inference + reg columns (always)
    * ``hplot-outputs-csv/cells/<slide>.csv`` — H-Plot per-cell features
    * ``ncomp-outputs-csv/<slide>.csv`` — neighbourhood composition
    * ``cme-outputs-csv/cells/<slide>.csv`` — cell morphology embeddings
    * ``imported-xenium/<slide>.h5ad`` — Xenium per-cell summaries

    Returns the list of combined CSV paths that were written.
    """
    # If include is None or empty, include all sources
    if not include:
        include = AVAILABLE_SOURCES
    base_dir = results_dir / "model-outputs-csv"
    if not base_dir.exists():
        return []

    # Source directories
    hplot_cells_dir = results_dir / "hplot-outputs-csv" / "cells"
    ncomp_dir = results_dir / "ncomp-outputs-csv"
    cme_cells_dir = results_dir / "cme-outputs-csv" / "cells"
    xenium_dir = results_dir / "imported-xenium"

    export_dir = results_dir / "export-csv"
    export_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(base_dir, URIPath):
        base_csvs = [p for p in base_dir.iterdir(files_only=True) if p.suffix == ".csv"]
    else:
        base_csvs = sorted(p for p in base_dir.iterdir() if p.is_file() and p.suffix == ".csv")

    written: list[PathLike] = []

    for base_csv in tqdm(base_csvs, desc="Building export CSVs", unit="slide"):
        slide_id = base_csv.stem
        out_csv = export_dir / f"{slide_id}.csv"

        if not overwrite and out_csv.exists():
            written.append(out_csv)
            continue

        df = _read_csv(base_csv)
        df = _ensure_center(df)

        # --- H-Plot cells (join on minx, miny) ------------------------------
        if "hplot" in include:
            hplot_csv = hplot_cells_dir / f"{slide_id}.csv"
            if hplot_csv.exists():
                hdf = _read_csv(hplot_csv)
                new_cols = [c for c in hdf.columns if c not in df.columns]
                if new_cols and "minx" in hdf.columns and "miny" in hdf.columns:
                    df = df.merge(
                        hdf[["minx", "miny"] + new_cols],
                        on=["minx", "miny"],
                        how="left",
                    )

        # --- ncomp (join on center_x, center_y) -----------------------------
        if "ncomp" in include:
            ncomp_csv = ncomp_dir / f"{slide_id}.csv"
            if ncomp_csv.exists():
                ndf = _read_csv(ncomp_csv)
                new_cols = [c for c in ndf.columns if c not in df.columns]
                if new_cols and "center_x" in ndf.columns and "center_y" in ndf.columns:
                    df = df.merge(
                        ndf[["center_x", "center_y"] + new_cols],
                        on=["center_x", "center_y"],
                        how="left",
                    )

        # --- cme (join on minx, miny) ----------------------------------------
        if "cme" in include:
            cme_csv = cme_cells_dir / f"{slide_id}.csv"
            if cme_csv.exists():
                cdf = _read_csv(cme_csv)
                new_cols = [c for c in cdf.columns if c not in df.columns]
                if new_cols and "minx" in cdf.columns and "miny" in cdf.columns:
                    df = df.merge(
                        cdf[["minx", "miny"] + new_cols],
                        on=["minx", "miny"],
                        how="left",
                    )

        # --- Xenium (join by row index from model_cell_id) --------------------
        if "xenium" in include:
            xenium_h5ad = xenium_dir / f"{slide_id}.h5ad"
            if xenium_h5ad.exists():
                try:
                    xdf = _read_xenium_summaries(xenium_h5ad, slide_id)
                    if not xdf.empty:
                        # xdf is indexed by row index; merge by position
                        df = df.reset_index(drop=True)
                        for col in xdf.columns:
                            if col not in df.columns:
                                df[col] = xdf.reindex(df.index).get(col, np.nan)
                except Exception:
                    pass  # Skip silently if h5ad read fails

        _write_csv(df, out_csv)
        written.append(out_csv)

    return written


def measure_columns(df: pd.DataFrame) -> List[str]:
    """Return all numeric columns suitable for inclusion in GeoJSON measurements.

    Excludes geometry / identity columns.  Includes ``prob_*``, ``region_prob_*``,
    ``neighborhood_*``, boolean flags like ``is_base_type``, etc.
    """
    return [
        c for c in df.columns
        if c not in _GEOM_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
