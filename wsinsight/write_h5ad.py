"""Conversion of merged per-cell export CSVs into AnnData ``.h5ad`` files.

Each slide's merged ``export-csv/<slide>.csv`` table becomes one AnnData object:

* ``X``            — per-class probability matrix (columns ``<prefix>_*``),
                     or, when no probability columns are present, all numeric
                     non-geometry measurement columns.
* ``var_names``    — class / feature names (the ``<prefix>_`` stripped off).
* ``obs``          — ``slide_id``, ``object_type``, argmax ``classification``,
                     ``niche_id`` (Categorical integer from the ``niche_id`` column
                     in the niche CSV), plus every geometry and extra column.
* ``obsm["spatial"]`` — cell centroid coordinates ``[center_x, center_y]``.
* ``uns["wsinsight"]`` — provenance metadata.

This mirrors the GeoJSON / OME-CSV exporters but produces the AnnData format
consumed by scanpy, squidpy, and the wider single-cell ecosystem.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List
from typing import Union

import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm.auto import tqdm

from .uri_path import URIPath

PathLike = Union[Path, URIPath]

# Columns that carry geometry/identity — never treated as measurements (the X
# matrix); kept in ``obs`` instead.  Matches ``write_geojson._GEOM_COLS``.
_GEOM_COLS = frozenset(
    {
        "minx",
        "miny",
        "width",
        "height",
        "center_x",
        "center_y",
        "polygon_wkt",
    }
)


def _read_csv(path: PathLike) -> pd.DataFrame:
    if isinstance(path, URIPath):
        with path.open("r", encoding="utf-8") as fp:
            return pd.read_csv(fp)
    return pd.read_csv(path)


def _iter_h5ad_stems(root: PathLike) -> set[str]:
    if not root.exists():
        return set()
    if isinstance(root, URIPath):
        entries = root.iterdir(files_only=True)
    else:
        entries = root.iterdir()
    return {p.stem for p in entries if p.suffix == ".h5ad"}


def build_anndata_from_df(
    df: pd.DataFrame,
    *,
    prefix: str = "prob",
    slide_id: str = "",
    object_type: str = "detection",
) -> AnnData:
    """Build an :class:`anndata.AnnData` from one merged per-cell DataFrame."""
    df = df.reset_index(drop=True)
    n = len(df)

    prob_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]

    if prob_cols:
        x_cols = prob_cols
        var_names = [c[len(prefix) + 1 :] for c in prob_cols]
    else:
        # Fall back to every numeric non-geometry column as the feature matrix.
        x_cols = [
            c
            for c in df.columns
            if c not in _GEOM_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]
        var_names = list(x_cols)

    if x_cols:
        X = df[x_cols].to_numpy(dtype=np.float32, copy=True)
    else:
        X = np.zeros((n, 0), dtype=np.float32)

    var = pd.DataFrame(index=pd.Index(var_names, name="feature"))

    obs_index = pd.Index(
        [f"{slide_id}-{i}" if slide_id else str(i) for i in range(n)],
        name="cell_id",
    )
    obs = pd.DataFrame(index=obs_index)
    obs["slide_id"] = slide_id
    obs["object_type"] = object_type

    # Argmax classification label (only meaningful with probability columns).
    if prob_cols and X.shape[1]:
        arg = X.argmax(axis=1)
        obs["classification"] = pd.Categorical([var_names[a] for a in arg])

    # Carry every column that is not part of X into obs (geometry + extras).
    # niche_id (integer niche cluster label) is kept as its own obs column
    # so it does not collide with the argmax 'classification' obs column above.
    x_set = set(x_cols)
    for c in df.columns:
        if c in x_set or c in obs.columns:
            continue
        if c == "niche_id":
            obs["niche_id"] = pd.Categorical(df[c].to_numpy())
        else:
            obs[c] = df[c].to_numpy()

    # Spatial coordinates for squidpy / scanpy spatial tooling.
    if {"center_x", "center_y"}.issubset(df.columns):
        spatial = df[["center_x", "center_y"]].to_numpy(dtype=np.float32)
    elif {"minx", "miny", "width", "height"}.issubset(df.columns):
        cx = np.rint(df["minx"] + df["width"] / 2)
        cy = np.rint(df["miny"] + df["height"] / 2)
        spatial = np.column_stack([cx, cy]).astype(np.float32)
    else:
        spatial = None

    adata = AnnData(X=X, obs=obs, var=var)
    if spatial is not None:
        adata.obsm["spatial"] = spatial
    adata.uns["wsinsight"] = {
        "prefix": prefix,
        "object_type": object_type,
        "slide_id": slide_id,
        "n_cells": int(n),
        "classes": list(var_names) if prob_cols else [],
    }
    return adata


def _write_one(adata: AnnData, out_path: PathLike) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(out_path, URIPath):
        # anndata can only write to a local filesystem path; stage then upload.
        tmp = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
        tmp.close()
        try:
            adata.write_h5ad(Path(tmp.name))
            with open(tmp.name, "rb") as src, out_path.open("wb") as dst:
                dst.write(src.read())
        finally:
            os.unlink(tmp.name)
    else:
        adata.write_h5ad(Path(out_path))


def write_h5ads(
    csvs: List[PathLike],
    *,
    results_dir: PathLike,
    output_dir: Union[str, Path] = "export-h5ad",
    prefix: str = "prob",
    object_type: str = "detection",
    overwrite: bool = False,
    show_progress: bool = True,
) -> List[PathLike]:
    """Convert merged per-cell export CSVs to ``.h5ad`` files.

    Returns the list of ``.h5ad`` paths written (or already present when
    ``overwrite`` is False).
    """
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")

    out_root = results_dir / output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    already = set() if overwrite else _iter_h5ad_stems(out_root)
    pending = [p for p in csvs if p.stem not in already]

    written: list[PathLike] = []
    iterator = (
        tqdm(pending, desc="Writing h5ad", unit="slide") if show_progress else pending
    )
    for csv_path in iterator:
        slide_id = csv_path.stem
        out_path = out_root / f"{slide_id}.h5ad"
        df = _read_csv(csv_path)
        adata = build_anndata_from_df(
            df, prefix=prefix, slide_id=slide_id, object_type=object_type
        )
        _write_one(adata, out_path)
        written.append(out_path)

    return written
