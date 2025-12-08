"""
Convert CSVs of model outputs to compressed OME-CSV files.

OME-CSV files can be loaded into whole slide image viewers like QuPath.
"""

from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional, Union

import gzip
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .uri_path import URIPath

PathLike = Union[Path, URIPath]


def _dataframe_to_omecsv(
    df: pd.DataFrame,
    prob_cols: List[str],
    minx2: np.ndarray,
    miny2: np.ndarray,
    maxx2: np.ndarray,
    maxy2: np.ndarray,
    *,
    class_prefix: str = "prob_",
) -> str:
    """
    Convert a dataframe of tiles to OME-CSV format (no df.apply).

    Parameters
    ----------
    df : DataFrame
        Must contain probability columns in `prob_cols`.
    prob_cols : list[str]
        Columns like ["prob_classA", "prob_classB", ...].
    minx2, miny2, maxx2, maxy2 : np.ndarray
        Arrays of shape (N,) with integer coordinates for the shrunken box.
    class_prefix : str
        Prefix to strip from prob column names when deriving class labels.

    Returns
    -------
    str
        Full OME-CSV content as a single string.
    """
    num_rows = df.shape[0]
    assert (
        len(minx2) == len(miny2) == len(maxx2) == len(maxy2) == num_rows
    ), "Coordinate arrays must match DataFrame length"

    # 1. Header line
    head_str = ",".join(
        ["object", "secondary_object", "polygon", "objectType", "classification", *prob_cols]
    )

    # 2. Probabilities & class labels (vectorized)
    prob_arr = df[prob_cols].to_numpy(copy=False)  # shape (N, C)
    class_names = np.array([c[len(class_prefix):] for c in prob_cols])
    best_idx = prob_arr.argmax(axis=1)             # shape (N,)
    cls_arr = class_names[best_idx]               # shape (N,)

    # 3. Build all lines
    lines = [head_str]
    for i in range(num_rows):
        # Shrunken box vertices (x, y)
        x1, y1 = int(maxx2[i]), int(miny2[i])
        x2, y2 = int(maxx2[i]), int(maxy2[i])
        x3, y3 = int(minx2[i]), int(maxy2[i])
        x4, y4 = int(minx2[i]), int(miny2[i])

        coords = [
            f"{x1} {y1}",
            f"{x2} {y2}",
            f"{x3} {y3}",
            f"{x4} {y4}",
            f"{x1} {y1}",  # close ring
        ]
        poly_str = '"POLYGON ((' + ",".join(coords) + '))"'

        probs = prob_arr[i]
        mvals = ",".join(map(str, probs))  # "p1,p2,..."
        cls = cls_arr[i]

        # object,secondary_object,polygon,objectType,classification,<prob_cols...>
        line = f"{i},{i},{poly_str},tile,{cls},{mvals}"
        lines.append(line)

    return "\n".join(lines)


def make_omecsv(
    csv: PathLike,
    results_dir: PathLike,
    output_dir: PathLike,
    overlap: float,
    prefix: str,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict] = None,
) -> None:
    """
    Read a CSV of model outputs and write a compressed OME-CSV file (.ome.csv.gz).

    Parameters
    ----------
    csv : Path
        Input CSV path (model outputs).
    results_dir : Path
        Base results directory.
    output_dir : Path
        Relative output directory under results_dir.
    overlap : float
        Overlap fraction used to shrink the tile box (0.0–1.0).
    prefix : str
        Prefix for probability columns, e.g. "prob".
    usecols : list[str], optional
        Columns to read from CSV.
    dtype : dict, optional
        Dtype mapping for pd.read_csv.
    """
    filename = csv.stem

    df = pd.read_csv(
        csv,
        usecols=usecols,
        dtype=dtype,
        engine="c",
        memory_map=True,
        low_memory=False,
    )

    # Probability columns, e.g. "prob_*" if prefix="prob"
    full_prefix = f"{prefix}_"
    prob_cols = [c for c in df.columns if c.startswith(full_prefix)]
    if not prob_cols:
        raise KeyError(f"Did not find any columns with '{full_prefix}' prefix.")

    # Drop rows with NaNs in any prob column
    df = df.dropna(subset=prob_cols)

    # Geometry columns to arrays
    xywh = df[["minx", "miny", "width", "height"]].to_numpy(dtype=np.int64, copy=False)
    minx, miny, w, h = xywh.T

    # Vectorized patch box math (shrink box using overlap)
    pw = np.rint(w * (1.0 - overlap)).astype(np.int64)
    ph = np.rint(h * (1.0 - overlap)).astype(np.int64)
    pmx = np.rint((w - pw) * 0.5).astype(np.int64)
    pmy = np.rint((h - ph) * 0.5).astype(np.int64)

    minx2 = minx + pmx
    miny2 = miny + pmy
    maxx2 = minx2 + pw
    maxy2 = miny2 + ph

    # Build full OME-CSV content
    omecsv = _dataframe_to_omecsv(
        df,
        prob_cols,
        minx2,
        miny2,
        maxx2,
        maxy2,
        class_prefix=full_prefix,
    )

    # Write compressed .ome.csv.gz
    out_path = results_dir / output_dir / f"{filename}.ome.csv.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_gzip_bytes(out_path, omecsv.encode("utf-8"))


def _iter_files(path: PathLike, *, suffix: Optional[str] = None):
    """Yield files under ``path`` while supporting ``URIPath`` sources."""
    if isinstance(path, URIPath):
        iterator = path.iterdir(files_only=True)
    else:
        iterator = (child for child in path.iterdir() if child.is_file())
    for child in iterator:
        if suffix is None or child.suffix == suffix:
            yield child


def _write_gzip_bytes(out_path: PathLike, payload: bytes) -> None:
    """Persist compressed OME-CSV bytes to either local or remote storage."""
    parent = out_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    if isinstance(out_path, URIPath):
        with out_path.open("wb") as fh:
            with gzip.GzipFile(fileobj=fh, mode="wb") as gz:
                gz.write(payload)
        return

    with gzip.open(out_path, "wb") as gz:
        gz.write(payload)


def write_omecsvs(
    csvs: List[PathLike],
    h5s: List[PathLike],  # kept for API compatibility; not used
    overlap: float,
    results_dir: PathLike,
    # input_dir: Path,
    output_dir: PathLike,
    prefix: str,
    num_workers: int,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict] = None,
) -> None:
    """
    Convert multiple model-output CSV files into compressed OME-CSV files
    in parallel using a process pool.

    Parameters
    ----------
    csvs : list[Path]
        List of input CSV paths.
    h5s : list[Path]
        Unused, kept only for external API compatibility.
    overlap : float
        Overlap fraction used to shrink the tile box (0.0–1.0).
    results_dir : Path
        Base results directory.
    output_dir : Path
        Relative output directory under results_dir.
    prefix : str
        Prefix for probability columns in the CSVs.
    num_workers : int
        Number of worker processes to use.
    usecols : list[str], optional
        Columns to read from each CSV.
    dtype : dict, optional
        Dtype mapping for pd.read_csv.
    """
    output = results_dir / output_dir

    if not results_dir.exists():
        raise FileExistsError(f"results_dir does not exist: {results_dir}")

    missing_dirs = sorted({p.parent for p in csvs if not p.parent.exists()}, key=lambda d: str(d))
    if missing_dirs:
        if (results_dir / "patches").exists():
            raise FileExistsError(
                "Model outputs have not been generated yet. Please run model inference."
            )
        missing_str = ", ".join(str(d) for d in missing_dirs)
        raise FileExistsError(
            "Expected the following directories to contain model-output CSVs but they do not exist: "
            f"{missing_str}"
        )

    missing_files = [p for p in csvs if not p.exists()]
    if missing_files:
        missing_str = ", ".join(str(p) for p in missing_files)
        raise FileNotFoundError(
            "The following CSV files were not found: "
            f"{missing_str}"
        )

    # Skip CSVs that already have corresponding OME-CSV outputs
    if output.exists():
        omecsvs = list(_iter_files(output, suffix=".ome.csv.gz"))
        omecsv_filenames = []
        for p in omecsvs:
            name = p.name
            suffix = ".ome.csv.gz"
            if name.endswith(suffix):
                # Strip ".ome.csv.gz" to get the base slide name
                omecsv_filenames.append(name[: -len(suffix)])
            else:
                omecsv_filenames.append(p.stem)

        csv_filenames = [p.stem for p in csvs]
        csvs_new = [csv for csv in csv_filenames if csv not in omecsv_filenames]
        csvs = [path for path in csvs if path.stem in csvs_new]
    else:
        output.mkdir(parents=True, exist_ok=True)

    total = len(csvs)
    if total == 0:
        return

    pbar = tqdm(total=total, desc="Files completed", dynamic_ncols=True)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(
                make_omecsv,
                csv,
                results_dir=results_dir,
                output_dir=output_dir,
                overlap=overlap,
                prefix=prefix,
                usecols=usecols,
                dtype=dtype,
            )
            for csv in csvs
        ]
        for fut in as_completed(futures):
            fut.result()
            pbar.update(1)

    pbar.close()




# """Convert CSVs of model outputs to OMECSV files.
#
# OMECSV files can be loaded into whole slide image viewers like QuPath.
# """
#
# from __future__ import annotations
#
# # from functools import partial
# from pathlib import Path
# import numpy as np
# import gzip
# from tqdm.auto import tqdm
# # import numpy.typing as npt
# import pandas as pd
# # from tqdm.contrib.concurrent import process_map
# from concurrent.futures import ProcessPoolExecutor, as_completed
# import multiprocessing
# from typing import List, Dict, Optional
#
# Lock = multiprocessing.Lock()
#
# def _box_to_polygon(
#     *, minx: int, miny: int, width: int, height: int, overlap: float
# ) -> list[tuple[int, int]]:
#     """Get coordinates of a box polygon."""
#
#     patch_width = int(np.ceil(width * (1.0-overlap)))
#     patch_height = int(np.ceil(height * (1.0-overlap)))
#
#     patch_minx = int(np.floor(0.5*(width-patch_width)))
#     patch_miny = int(np.floor(0.5*(height-patch_height)))
#
#     minx = minx + patch_minx
#     miny = miny + patch_miny
#
#     maxx = minx + patch_width
#     maxy = miny + patch_height
#
#     return [(maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny), (maxx, miny)]
#
#
# # def _row_to_omecsv(row: pd.Series, prob_cols: list[str], overlap: float) -> dict:
# #     """Convert information about one tile to a single OMECSV feature."""
# #     minx, miny, width, height = row["minx"], row["miny"], row["width"], row["height"]
# #     coords = _box_to_polygon(minx=minx, miny=miny, width=width, height=height, overlap=overlap)
# #     prob_dict = row[prob_cols].to_dict()
# #
# #     measurements = {}
# #     for k, v in prob_dict.items():
# #         measurements[k] = v
# #
# #     return {
# #         "type": "Feature",
# #         "id": str(uuid.uuid4()),
# #         "geometry": {
# #             "type": "Polygon",
# #             "coordinates": [coords],
# #         },
# #         "properties": {
# #             "isLocked": True,
# #             # measurements is a list of {"name": str, "value": float} dicts.
# #             # https://qupath.github.io/javadoc/docs/qupath/lib/measurements/MeasurementList.html
# #             "measurements": measurements,
# #             "objectType": "tile",
# #             # classification is a dict of "name": str and optionally "color": int.
# #             # https://qupath.github.io/javadoc/docs/qupath/lib/objects/classes/PathClass.html
# #             # We do not include classification because we do not enforce a single class
# #             # per tile.
# #             # "classification": {"name": class_name},
# #         },
# #     }
#
# # "object", "secondary_object", "polygon", "objectType", "classification"
# def _row_to_omecsv(row: pd.Series, prob_cols: list[str]) -> dict:
#     """Convert information about one tile to a single OMECSV feature."""
#     # minx, miny, width, height = int(np.round(row["minx"])), int(np.round(row["miny"])), int(np.round(row["width"])), int(np.round(row["height"]))
#     # maxx, maxy = minx+width, miny+height
#     # coords = _box_to_polygon(minx=minx, miny=miny, width=width, height=height, overlap=overlap)
#     # polygon_str = f'"POLYGON (({minx} {miny},{maxx} {miny},{maxx} {maxy},{minx} {maxy},{minx} {miny}))"'
#
#     prob_dict = row[prob_cols].to_dict()
#
#     measurements = {}
#     for k, v in prob_dict.items():
#         measurements[k] = v
#
#
#     measurement_val_str = ','.join([str(v) for _, v in prob_dict.items()])
#     measurement_keys = [k[5:] for k, _ in prob_dict.items()]
#
#     idx = np.array([v for _, v in prob_dict.items()]).argmax()
#     cls = measurement_keys[idx]
#
#     # obj_id = str(uuid.uuid4())
#
#     return 'tile,'+cls+','+measurement_val_str
#
#
# # def _dataframe_to_omecsv(df: pd.DataFrame, prob_cols: list[str], overlap: float) -> dict:
# #     """Convert a dataframe of tiles to OMECSV format."""
# #     features = df.apply(_row_to_omecsv, axis=1, prob_cols=prob_cols, overlap=overlap)
# #     return {
# #         "type": "FeatureCollection",
# #         "features": features.tolist(),
# #     }
#
# #
# # def _dataframe_to_omecsv(df: pd.DataFrame, prob_cols: list[str], polygon_cols: np.array) -> dict:
# #     """Convert a dataframe of tiles to OMECSV format."""
# #
# #     head_str = ','.join(["object,secondary_object,polygon,objectType,classification", ','.join(prob_cols)])
# #
# #     polygon_heads = polygon_cols[..., :1]
# #     closed_polygon_cols = np.concatenate((polygon_cols, polygon_heads), axis=2)
# #
# #     polygons = ['"POLYGON (('+','.join([f"{int(0.5+p2)} {int(0.5+p1)}" for (p1, p2) in zip(p[0], p[1])])+'))"' for p in closed_polygon_cols]
# #     features = df.apply(_row_to_omecsv, axis=1, prob_cols=prob_cols)
# #
# #     return head_str+'\n'+'\n'.join([','.join([str(i),str(i),p,f]) for i,(p, f) in enumerate(zip(polygons, features))])
#
# def _dataframe_to_omecsv(
#     df: pd.DataFrame,
#     prob_cols: list[str],
#     polygon_cols: np.ndarray,
#     *,
#     prefix: str = "prob_",
# ) -> str:
#     """
#     Convert a dataframe of tiles to OMECSV format.
#
#     Parameters
#     ----------
#     df : DataFrame
#         Must contain the probability columns in `prob_cols`.
#     prob_cols : list[str]
#         Columns like ["prob_classA", "prob_classB", ...]
#     polygon_cols : np.ndarray
#         Array of shape (N, 4, 2) with the 4 corners for each polygon:
#         [ [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ], ... ]
#     prefix : str
#         Prefix used for probability columns, e.g. "prob_".
#
#     Returns
#     -------
#     str
#         Full OME-CSV as a single string.
#     """
#
#     # -----------------------------
#     # 1. Header line
#     # -----------------------------
#     head_str = ",".join(
#         ["object", "secondary_object", "polygon", "objectType", "classification", *prob_cols]
#     )
#
#     # -----------------------------
#     # 2. Build polygon WKT strings
#     #    polygon_cols: (N, 4, 2)  -> close ring -> (N, 5, 2)
#     # -----------------------------
#     # close the ring by repeating the first vertex at the end
#     first_vertex = polygon_cols[:, :1, :]               # (N, 1, 2)
#     closed = np.concatenate([polygon_cols, first_vertex], axis=1)  # (N, 5, 2)
#
#     polygons = []
#     # closed[i] is shape (5, 2): [[x,y], [x,y], ...]
#     for coords in closed:
#         # note: original code did int(0.5 + p) – keep the same rounding behavior
#         coord_str = ",".join(
#             f"{int(0.5 + y)} {int(0.5 + x)}"   # (y,x) order, following your original code
#             for x, y in coords
#         )
#         polygons.append(f'"POLYGON (({coord_str}))"')
#
#     # -----------------------------
#     # 3. Probabilities & classification (vectorized)
#     # -----------------------------
#     # shape: (N, C)
#     prob_arr = df[prob_cols].to_numpy(copy=False)
#
#     # class names from column names, e.g. "prob_classA" -> "classA"
#     class_names = np.array([c[len(prefix):] for c in prob_cols])
#
#     # argmax along columns -> (N,)
#     best_idx = prob_arr.argmax(axis=1)
#     cls_arr = class_names[best_idx]
#
#     # measurement values as strings "p1,p2,p3..."
#     # We still need a Python loop here because it's string work,
#     # but we avoid per-row Series construction.
#     measurement_strs = [
#         ",".join(map(str, row)) for row in prob_arr
#     ]
#
#     # -----------------------------
#     # 4. Assemble all lines
#     # -----------------------------
#     lines = [head_str]
#     # i = object id / secondary_object id
#     for i, (poly, cls, mvals) in enumerate(zip(polygons, cls_arr, measurement_strs)):
#         # object, secondary_object, polygon, objectType, classification, <prob_cols...>
#         # original row shape: i,i,p,"tile,cls,vals..."  -> we expand in-place here
#         line = f"{i},{i},{poly},tile,{cls},{mvals}"
#         lines.append(line)
#
#     return "\n".join(lines)
#
#
#
# # def make_omecsv(csv: Path, results_dir: Path, overlap: float) -> None:
# #     filename = csv.stem
# #     df = pd.read_csv(csv)
# #     prob_cols = [col for col in df.columns.tolist() if col.startswith("prob_")]
# #     if not prob_cols:
# #         raise KeyError("Did not find any columns with prob_ prefix.")
# #     omecsv = _dataframe_to_omecsv(df, prob_cols, overlap)
# #     with open(results_dir / "model-outputs-omecsv" / f"{filename}.ome.csv", "w") as f:
# #         json.dump(omecsv, f)
#
#
# def make_omecsv(csv: Path, 
#                 results_dir: Path, 
#                 output_dir: Path, 
#                 overlap: float,
#                 prefix: str,
#                 usecols: Optional[List[str]] = None,
#                 dtype: Optional[Dict] = None,
#                 ) -> None:
#     filename = csv.stem
#
#
#     df = pd.read_csv(
#         csv,
#         usecols=usecols,
#         dtype=dtype,
#         engine="c",
#         memory_map=True,
#         low_memory=False,
#     )
#
#     prob_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
#     df = df.dropna(subset=[c for c in prob_cols if c.startswith(prefix)])
#
#     # Pull numeric columns to arrays (no per-row pandas calls)
#     xywh = df[["minx", "miny", "width", "height"]].to_numpy(dtype=np.int64, copy=False)
#     minx, miny, w, h = xywh.T
#
#     # Vectorized patch math
#     pw = np.rint(w * (1.0 - overlap)).astype(np.int64)
#     ph = np.rint(h * (1.0 - overlap)).astype(np.int64)
#     pmx = np.rint((w - pw) * 0.5).astype(np.int64)
#     pmy = np.rint((h - ph) * 0.5).astype(np.int64)
#
#     minx2 = minx + pmx
#     miny2 = miny + pmy
#     maxx2 = minx2 + pw
#     maxy2 = miny2 + ph
#
#     # Closed ring coordinates (N, 5, 2)
#     polygon_cols = np.stack([
#         np.stack([maxx2, miny2], axis=1),
#         np.stack([maxx2, maxy2], axis=1),
#         np.stack([minx2, maxy2], axis=1),
#         np.stack([minx2, miny2], axis=1),
#         # np.stack([maxx2, miny2], axis=1),
#     ], axis=1)
#
#
#     prob_cols = [col for col in df.columns.tolist() if col.startswith(f"{prefix}_")]
#     if not prob_cols:
#         raise KeyError("Did not find any columns with prob_ prefix.")
#
#     # with Lock:
#     #     h5_path = results_dir / "patches" / f"{filename}.h5"
#     #     with h5py.File(h5_path, mode="r") as f:
#     #         if "/polygons" not in f:
#     #             raise KeyError("Group '/polygons' not found in HDF5.")
#     #
#     #         g = f["/polygons"]
#     #
#     #         if "coords" not in g or "offsets" not in g:
#     #             raise KeyError("Expected datasets '/polygons/coords' and '/polygons/offsets'.")
#     #
#     #         coords = np.asarray(g["coords"])         # (K, 2) float32
#     #         offsets = np.asarray(g["offsets"])       # (M+1,) int64
#     #         if offsets.ndim != 1 or coords.ndim != 2 or coords.shape[1] != 2:
#     #             raise ValueError("Invalid shapes for coords/offsets in HDF5.")
#     #
#     #         num_polys = int(len(offsets) - 1)
#     #
#     #         # discover per-polygon probability datasets: {name: array(M,)}
#     #         prob_cols: List[str] = []
#     #         per_poly_data: Dict[str, np.ndarray] = {}
#     #         for name, dset in g.items():
#     #             if not isinstance(dset, h5py.Dataset):
#     #                 continue
#     #             if not name.startswith(f"{prefix}_"):
#     #                 continue
#     #             arr = np.asarray(dset)
#     #             if arr.ndim == 1 and len(arr) == num_polys:
#     #                 per_poly_data[name] = arr
#     #                 prob_cols.append(name)
#     #
#     #         # if not prob_cols:
#     #         #     raise KeyError(f"No per-polygon datasets starting with '{prefix}_' in {h5_path}")
#     #
#     #         # ---- reconstruct polygons into WKT ----
#     #         def poly_wkt(i: int) -> str:
#     #             s, e = int(offsets[i]), int(offsets[i + 1])
#     #             xy = coords[s:e]  # (N_i, 2)
#     #             if xy.size == 0:
#     #                 # represent an empty polygon as an empty ring (rare)
#     #                 return "POLYGON EMPTY"
#     #             # Ensure closed ring (repeat first point if needed)
#     #             if not np.array_equal(xy[0], xy[-1]):
#     #                 xy = np.vstack([xy, xy[0]])
#     #             # WKT uses "x y" pairs
#     #             ring = ", ".join(f"{float(x)} {float(y)}" for x, y in xy)
#     #             return f"POLYGON (({ring}))"
#     #
#     #         polygon_wkts = [poly_wkt(i) for i in range(num_polys)]
#     #
#     #     # ---- build a DataFrame that matches your *_to_geojson_polygon_fast() input ----
#     # data = {"polygon_wkt": polygon_wkts}
#     # for k, v in per_poly_data.items():
#     #     data[k] = v
#     # df = pd.DataFrame(data)
#
#     # omecsv = _dataframe_to_omecsv(df, prob_cols, polygon_cols)
#     omecsv = _dataframe_to_omecsv(df, prob_cols, polygon_cols, prefix=prefix)
#
#     with gzip.open(results_dir / output_dir / f"{filename}.ome.csv.gz", "wb") as f:
#         f.write(omecsv.encode('utf-8'))
#
#
# def write_omecsvs(csvs: list[Path], 
#                   h5s: list[Path], 
#                   overlap: float, 
#                   results_dir: Path, 
#                   input_dir: Path,
#                   output_dir: Path,
#                   prefix: str,
#                   num_workers: int,
#                   usecols: Optional[List[str]] = None,  # e.g., ["minx","miny","width","height", "polygon_wkt", *prob_cols]
#                   dtype: Optional[Dict] = None,        # e.g., {"minx":"int32", ...}
#                   ) -> None:
#     output = results_dir / output_dir
#
#     if not results_dir.exists():
#         raise FileExistsError(f"results_dir does not exist: {results_dir}")
#     if (
#         not (results_dir / input_dir).exists()
#         and (results_dir / "patches").exists()
#     ):
#         raise FileExistsError(
#             "Model outputs have not been generated yet. Please run model inference."
#         )
#     if not (results_dir / input_dir).exists():
#         raise FileExistsError(
#             "Expected results_dir to contain a 'model-outputs-csv' "
#             "directory but it does not."
#             "Please provide the path to the directory"
#             "that contains model-outputs, masks, and patches."
#         )
#     if output.exists():
#         omecsvs = list((results_dir / output_dir).glob("*.ome.csv.gz"))
#
#         # Makes a list of filenames for both geojsons and csvs
#         omecsv_filenames = [filename.stem[:-4] for filename in omecsvs]
#         csv_filenames = [filename.stem for filename in csvs]
#
#         # Makes a list of new csvs that need to be converted to geojson
#         csvs_new = [csv for csv in csv_filenames if csv not in omecsv_filenames]
#         csvs = [path for path in csvs if path.stem in csvs_new]
#     else:
#         # If output directory doesn't exist, make one and set csvs_final to csvs
#         output.mkdir(parents=True, exist_ok=True)
#
#     # make_omecsv(csvs[0], 
#     #             results_dir=results_dir,
#     #             output_dir=output_dir,
#     #             overlap=overlap, 
#     #             prefix=prefix,
#     #             usecols=usecols,
#     #             dtype=dtype,
#     #             )
#
#
#     total = len(csvs)
#     if total == 0:
#         return    
#
#     pbar = tqdm(total=total, desc="Files completed", dynamic_ncols=True) 
#
#     # Run with progress bar
#     with ProcessPoolExecutor(max_workers=num_workers) as ex:
#         futures = [ex.submit(make_omecsv, 
#                              csv,
#                              results_dir=results_dir,
#                              output_dir=output_dir,
#                              overlap=overlap, 
#                              prefix=prefix,
#                              usecols=usecols,
#                              dtype=dtype,
#                              ) for csv in csvs]
#
#         # pbar = tqdm(total=len(futures)) if pbar is None else pbar
#         for f in as_completed(futures):
#             # throttle_when_busy(target_util=0.9)
#             f.result()
#             pbar.update(1)
#
#
#
#     # func = partial(make_omecsv, 
#     #                results_dir=results_dir,
#     #                overlap=overlap, 
#     #                prefix=prefix,
#     #                usecols=usecols,
#     #                dtype=dtype,
#     #                )
#     # process_map(func, csvs, max_workers=num_workers, chunksize=1)
#
