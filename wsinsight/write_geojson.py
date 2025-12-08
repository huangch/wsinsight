"""High-throughput conversion of model-output CSVs into GeoJSON overlays."""

# geojson_vectorized_serial.py
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from colorsys import hsv_to_rgb
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import from_wkt  # shapely >= 2.0
# from shapely.geometry import Polygon, MultiPolygon  # (imported only for typing)
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import multiprocessing

from .uri_path import URIPath

PathLike = Union[Path, URIPath]

Lock = multiprocessing.Lock()

# from .num_worker_optimizer import pick_workers_safe, throttle_when_busy

# ---- fast JSON encoder ----
import orjson
def _dumps(obj: dict) -> bytes:
    """Serialize a GeoJSON dictionary using the fast `orjson` backend."""
    return orjson.dumps(obj)

# --------------------------
# Helpers
# --------------------------
def _make_distinct_colors(
    n: int,
    s: float = 0.70,
    v: float = 0.90,
    shuffle: bool = True,
    seed: Optional[int] = None,
):
    """Return `n` well-spaced HSV-derived colors for class visualization."""
    if n <= 0:
        raise ValueError("n must be > 0")
    hues = [i / n for i in range(n)]
    if shuffle and n > 2:
        order, L, R = [], 0, n - 1
        while L <= R:
            order.append(L)
            if L != R:
                order.append(R)
            L += 1; R -= 1
        hues = [hues[i] for i in order]
    if seed is not None:
        import random; random.seed(seed)

    out = []
    for h in hues:
        r, g, b = hsv_to_rgb(h, s, v)
        R, G, B = (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
        out.append({"hex": f"#{R:02X}{G:02X}{B:02X}", "rgb": (R, G, B), "hsv": (h, s, v)})
    return out

# --------------------------
# BOX path (vectorized)
# --------------------------
def _dataframe_to_geojson_box_fast(
    df: pd.DataFrame,
    prob_cols: List[str],
    overlap: float,
    *,
    prefix: str = "prob",
    object_type: str = "tile",
    set_classification: bool = False,
    color_list: Optional[List[dict]] = None,
) -> dict:
    """Vectorize CSV tile boxes into a GeoJSON FeatureCollection."""
    # Keep only valid rows for probs
    df = df.dropna(subset=[c for c in prob_cols if c.startswith(prefix)])

    # Pull numeric columns to arrays (no per-row pandas calls)
    xywh = df[["minx", "miny", "width", "height"]].to_numpy(dtype=np.int64, copy=False)
    minx, miny, w, h = xywh.T

    # Vectorized patch math
    pw = np.rint(w * (1.0 - overlap)).astype(np.int64)
    ph = np.rint(h * (1.0 - overlap)).astype(np.int64)
    pmx = np.rint((w - pw) * 0.5).astype(np.int64)
    pmy = np.rint((h - ph) * 0.5).astype(np.int64)

    minx2 = minx + pmx
    miny2 = miny + pmy
    maxx2 = minx2 + pw
    maxy2 = miny2 + ph

    # Closed ring coordinates (N, 5, 2)
    coords = np.stack([
        np.stack([maxx2, miny2], axis=1),
        np.stack([maxx2, maxy2], axis=1),
        np.stack([minx2, maxy2], axis=1),
        np.stack([minx2, miny2], axis=1),
        np.stack([maxx2, miny2], axis=1),
    ], axis=1)

    # Probs as matrix, argmax vectorized
    probs = df[prob_cols].to_numpy(dtype=np.float32, copy=False)
    arg = probs.argmax(axis=1)

    if color_list is None:
        color_list = _make_distinct_colors(len(prob_cols))

    class_names = [
        f"{prefix}_{c[len(prefix) + 1:]}" if c.startswith(f"{prefix}_") else f"{prefix}_{c}"
        for c in prob_cols
    ]

    # Build features in one tight Python loop (no pandas ops inside)
    features = []
    for i in range(len(df)):
        measurements = {prob_cols[j]: float(probs[i, j]) for j in range(len(prob_cols))}
        feat = {
            "type": "Feature",
            "id": str(uuid.uuid4()),
            # "geometry": {"type": "Polygon", "coordinates": [list(map(tuple, coords[i]))]},
            "geometry": {"type": "Polygon", "coordinates": [coords[i].tolist()]},
            "properties": {
                "isLocked": True,
                "measurements": measurements,
                "objectType": object_type,
            },
        }
        if set_classification:
            ci = int(arg[i])
            feat["properties"]["classification"] = {
                "name": class_names[ci],
                "color": list(color_list[ci]["rgb"]),
            }
        features.append(feat)

    return {"type": "FeatureCollection", "features": features}

# --------------------------
# POLYGON/WKT path (GeoPandas + Shapely vectorized)
# --------------------------
def _dataframe_to_geojson_polygon_fast(
    df: pd.DataFrame,
    prob_cols: List[str],
    *,
    prefix: str = "prob",
    object_type: str = "tile",
    set_classification: bool = False,
    color_list: Optional[List[dict]] = None,
    crs: Optional[str] = None,
) -> dict:
    """Convert Shapely-backed polygon annotations into GeoJSON quickly."""
    # Vectorized WKT → geometry (Shapely 2 ufunc)
    geom = from_wkt(df["polygon_wkt"])

    # Properties (copy needed bits; avoid huge extra columns)
    props = df.drop(columns=["polygon_wkt"]).copy()

    # Argmax over probs (vectorized)
    probs = props[prob_cols].to_numpy(dtype=np.float32, copy=False)
    idx = probs.argmax(axis=1)

    names = [
        f"{prefix}_{c[len(prefix) + 1:]}" if c.startswith(f"{prefix}_") else f"{prefix}_{c}"
        for c in prob_cols
    ]

    if color_list is None:
        color_list = _make_distinct_colors(len(prob_cols))

    # Pack your custom fields
    props["objectType"] = object_type
    if set_classification:
        props["classification"] = [
            {"name": names[i], "color": list(color_list[i]["rgb"])} for i in idx
        ]
    props["measurements"] = [dict(zip(prob_cols, map(float, row))) for row in probs]
    props["isLocked"] = True  # match box path

    # GeoDataFrame with final columns
    gdf = gpd.GeoDataFrame(props, geometry=geom, crs=crs)

    # gdf.to_json keeps dict-like columns (classification/measurements) as JSON objects
    return json.loads(gdf.to_json(drop_id=True))

# --------------------------
# Unified builder (vectorized)
# --------------------------
def _build_geojson_dict_from_csv(
    csv: PathLike,
    *,
    overlap: float,
    results_dir: PathLike,
    output_dir: PathLike,
    prefix: str = "prob",
    object_type: str = "tile",
    set_classification: bool = False,
    annotation_shape: str = "box",  # "box" or "polygon"
    # CSV read tuning
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict] = None,
) -> Tuple[PathLike, dict]:
    """Load a model-output CSV and build the GeoJSON dict plus destination."""
    # Read only what we need; memory-map large files
    df = pd.read_csv(
        csv,
        usecols=usecols,
        dtype=dtype,
        engine="c",
        memory_map=True,
        low_memory=False,
    )

    prob_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
    if not prob_cols:
        raise KeyError(f"No {prefix}_* columns in {csv}")

    color_list = _make_distinct_colors(len(prob_cols))

    if annotation_shape == "box":
        geojson = _dataframe_to_geojson_box_fast(
            df, prob_cols, overlap,
            prefix=prefix,
            object_type=object_type, set_classification=set_classification,
            color_list=color_list,
        )
    else:
        # requires a 'polygon_wkt' column
        if "polygon_wkt" not in df.columns:
            raise KeyError("polygon_wkt column is required for annotation_shape='polygon'")
        geojson = _dataframe_to_geojson_polygon_fast(
            df, prob_cols,
            prefix=prefix,
            object_type=object_type, set_classification=set_classification,
            color_list=color_list,
        )

    out_path = results_dir / output_dir / f"{csv.stem}.geojson"
    return out_path, geojson

# NOTE: The following function created GeoJSON directly from HDF5 polygon data to keep
#       precise cell contours, but that workflow is no longer in use. The implementation
#       is retained for reference only and is intentionally commented out.
"""
def _build_geojson_dict_from_h5(
    h5_path: Path,
    *,
    overlap: float,
    results_dir: Path,
    output_dir: Path,
    prefix: str = "prob",
    object_type: str = "tile",
    set_classification: bool = False,
    annotation_shape: str = "polygon",   # HDF5 reader supports polygons (recommended)
    # no usecols/dtype knobs here; HDF5 datasets are already typed
) -> Tuple[Path, dict]:
    if annotation_shape != "polygon":
        raise NotImplementedError("HDF5 builder currently supports annotation_shape='polygon' only.")

    # ---- read geometry + per-polygon attributes ----
    with Lock:
        with h5py.File(h5_path, "r") as f:
            if "/polygons" not in f:
                raise KeyError("Group '/polygons' not found in HDF5.")

            g = f["/polygons"]

            if "coords" not in g or "offsets" not in g:
                raise KeyError("Expected datasets '/polygons/coords' and '/polygons/offsets'.")

            coords = np.asarray(g["coords"])         # (K, 2) float32
            offsets = np.asarray(g["offsets"])       # (M+1,) int64
            if offsets.ndim != 1 or coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError("Invalid shapes for coords/offsets in HDF5.")

            num_polys = int(len(offsets) - 1)

            # discover per-polygon probability datasets: {name: array(M,)}
            prob_cols: List[str] = []
            per_poly_data: Dict[str, np.ndarray] = {}
            for name, dset in g.items():
                if not isinstance(dset, h5py.Dataset):
                    continue
                if not name.startswith(f"{prefix}_"):
                    continue
                arr = np.asarray(dset)
                if arr.ndim == 1 and len(arr) == num_polys:
                    per_poly_data[name] = arr
                    prob_cols.append(name)

            if not prob_cols:
                raise KeyError(f"No per-polygon datasets starting with '{prefix}_' in {h5_path}")

            # ---- reconstruct polygons into WKT ----
            def poly_wkt(i: int) -> str:
                s, e = int(offsets[i]), int(offsets[i + 1])
                xy = coords[s:e]  # (N_i, 2)
                if xy.size == 0:
                    # represent an empty polygon as an empty ring (rare)
                    return "POLYGON EMPTY"
                # Ensure closed ring (repeat first point if needed)
                if not np.array_equal(xy[0], xy[-1]):
                    xy = np.vstack([xy, xy[0]])
                # WKT uses "x y" pairs
                ring = ", ".join(f"{float(x)} {float(y)}" for x, y in xy)
                return f"POLYGON (({ring}))"

            polygon_wkts = [poly_wkt(i) for i in range(num_polys)]

    # ---- build a DataFrame that matches your *_to_geojson_polygon_fast() input ----
    data = {"polygon_wkt": polygon_wkts}
    for k, v in per_poly_data.items():
        data[k] = v
    df = pd.DataFrame(data)

    # colors based on discovered probability columns
    color_list = _make_distinct_colors(len(prob_cols))

    # ---- delegate to your existing fast converter ----
    geojson = _dataframe_to_geojson_polygon_fast(
        df,
        prob_cols,
        prefix=prefix,
        object_type=object_type,
        set_classification=set_classification,
        color_list=color_list,
    )

    out_path = results_dir / output_dir / f"{h5_path.stem}.geojson"
    return out_path, geojson
"""



# --------------------------
# Writer (atomic rename)
# --------------------------
def _is_uri_path(obj: object) -> bool:
    """Return True when `obj` is already a `URIPath` instance."""
    return isinstance(obj, URIPath)


def _iter_files(path: PathLike, *, suffix: Optional[str] = None):
    """Iterate over files below `path`, honoring both local and URI paths."""
    if isinstance(path, URIPath):
        iterator = path.iterdir(files_only=True)
    else:
        iterator = (child for child in path.iterdir() if child.is_file())
    for child in iterator:
        if suffix is None or child.suffix == suffix:
            yield child


def _write_geojson_bytes(out_path: PathLike, payload: bytes, atomic: bool = True) -> None:
    """Write GeoJSON bytes, optionally via atomic rename for local paths."""
    parent = out_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    if isinstance(out_path, URIPath):
        # Remote paths rely on URIPath to sync caches back to the backend.
        with out_path.open("wb") as f:
            f.write(payload)
        return

    if atomic:
        tmp = out_path.with_suffix(out_path.suffix + ".PART")
        with open(tmp, "wb", buffering=1 << 20) as f:
            f.write(payload)
        tmp.replace(out_path)
    else:
        with open(out_path, "wb", buffering=1 << 20) as f:
            f.write(payload)


def _worker(csv, 
            overlap, 
            results_dir, 
            output_dir, 
            prefix, 
            object_type, 
            set_classification, 
            annotation_shape,
            usecols,
            dtype,
            atomic_writes,
            ):
    """Process a single CSV in a worker process and persist the GeoJSON."""
    # Build + encode (serial)
    out_path, geojson = _build_geojson_dict_from_csv(
        csv,
        overlap=overlap,
        results_dir=results_dir,
        output_dir=output_dir,
        prefix=prefix,
        object_type=object_type,
        set_classification=set_classification,
        annotation_shape=annotation_shape,
        usecols=usecols,
        dtype=dtype,
    )

    payload = _dumps(geojson)

    # Write (atomic)
    _write_geojson_bytes(out_path, payload, atomic=atomic_writes)

def write_geojsons(
    csvs: List[PathLike],
    *,
    results_dir: PathLike,
    overlap: float,
    # input_dir: Path = Path("."),
    output_dir: Path = Path("."),
    prefix: str = "prob",
    num_workers=8,
    object_type: str = "tile",
    set_classification: bool = False,
    annotation_shape: str = "box",  # "box" or "polygon"
    atomic_writes: bool = True,
    usecols: Optional[List[str]] = None,  # e.g., ["minx","miny","width","height", "polygon_wkt", *prob_cols]
    dtype: Optional[Dict] = None,        # e.g., {"minx":"int32", ...}
    show_progress: bool = True,
    print_timings: bool = False,
) -> None:
    """Convert CSV outputs to GeoJSON concurrently and store results."""
    # Basic validations
    if not results_dir.exists():
        raise FileExistsError(f"results_dir does not exist: {results_dir}")

    missing_dirs = sorted({p.parent for p in csvs if not p.parent.exists()}, key=lambda x: str(x))
    if missing_dirs:
        missing_str = ", ".join(str(d) for d in missing_dirs)
        raise FileExistsError(
            "GeoJSON input CSV directory not found: "
            f"{missing_str}"
        )

    out_root = results_dir / output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    # Skip those already done
    already = {p.stem for p in _iter_files(out_root, suffix=".geojson")}
    csvs = [p for p in csvs if p.stem not in already]
    total = len(csvs)
    if total == 0:
        if print_timings:
            print("No new CSVs to process.")
        return

    pbar = tqdm(total=total, desc="Files completed", dynamic_ncols=True) if show_progress else None
    
    # Run with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_worker, 
                             args,
                             overlap, 
                             results_dir, 
                             output_dir, 
                             prefix, 
                             object_type, 
                             set_classification, 
                             annotation_shape,
                             usecols,
                             dtype,
                             atomic_writes,
                             ) for args in csvs]
        
        # pbar = tqdm(total=len(futures)) if pbar is None else pbar
        for f in as_completed(futures):
            # throttle_when_busy(target_util=0.9)
            f.result()
            if pbar:
                pbar.update(1)

    if pbar:
        pbar.close()