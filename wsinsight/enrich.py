"""Merge per-cell results from all available analyses into a single enriched CSV per slide."""

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


def build_enriched_csvs(
    results_dir: PathLike,
    *,
    overwrite: bool = True,
) -> List[PathLike]:
    """Left-join all per-cell analysis outputs into ``enriched-outputs-csv/``.

    Sources (all optional — skipped when absent):
    * ``model-outputs-csv/<slide>.csv``  — base inference + reg columns
    * ``hplot-outputs-csv/cells/<slide>.csv`` — H-Plot per-cell features
    * ``ncomp-outputs-csv/<slide>.csv`` — neighbourhood composition

    Returns the list of enriched CSV paths that were written.
    """
    base_dir = results_dir / "model-outputs-csv"
    hplot_cells_dir = results_dir / "hplot-outputs-csv" / "cells"
    ncomp_dir = results_dir / "ncomp-outputs-csv"
    enriched_dir = results_dir / "enriched-outputs-csv"
    enriched_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(base_dir, URIPath):
        base_csvs = [p for p in base_dir.iterdir(files_only=True) if p.suffix == ".csv"]
    else:
        base_csvs = sorted(p for p in base_dir.iterdir() if p.is_file() and p.suffix == ".csv")

    written: list[PathLike] = []

    for base_csv in tqdm(base_csvs, desc="Enriching CSVs", unit="slide"):
        slide_id = base_csv.stem
        out_csv = enriched_dir / f"{slide_id}.csv"

        if not overwrite and out_csv.exists():
            written.append(out_csv)
            continue

        df = _read_csv(base_csv)
        df = _ensure_center(df)

        # --- H-Plot cells (join on minx, miny) ------------------------------
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
