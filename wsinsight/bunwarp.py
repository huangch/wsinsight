"""Apply ST2WSI (bUnwarpJ + SIFT) registration to Xenium cell coordinates.

ST2WSI writes per pair:
  registration_params.json  — affine matrix, DAPI pixel size, flip/rotate, scales
  direct_transf.txt         — bUnwarpJ elastic B-spline grid (source->target)

Pipeline mapping a Xenium centroid (microns) to full-resolution H&E pixels
(faithful to qupath-extension-qust XeniumAnnotation.java):

  µm
   ÷ DapiImgPxlSize            -> full-res DAPI (source) px
   flipV / flipH / rotate      -> full-res DAPI px, re-oriented (full-res dims)
   ÷ SourceScale               -> DAPI px at the level the SIFT affine was fit at
   SIFT affine                 -> H&E (target) px at the TargetScale level
   [bUnwarpJ elastic B-spline] -> H&E px at the TargetScale level (optional)
   × TargetScale               -> full-resolution H&E px

`SourceScale` / `TargetScale` are the WSI pyramid *downsample factors* at which the
source (DAPI) and target (H&E) images were opened when the affine/B-spline were
estimated (e.g. 16). The affine therefore expects source coords divided by
SourceScale, and its output lives in target/TargetScale space, scaled back up by
TargetScale at the end.

Modes:
  'affine'          -> SIFT affine only (no elastic warp; target dims not needed).
  'affine+bspline'  -> affine + bUnwarpJ elastic warp (default). Requires the H&E
                       (target) full-resolution dimensions, because the bUnwarpJ
                       lattice is defined over the target image at TargetScale
                       resolution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_params(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    m = [float(v) for v in d["xnumAnnotImgRegParamSiftMatrix"]]  # a,c,e,b,d,f
    return {
        "affine": np.array([[m[0], m[1], m[2]], [m[3], m[4], m[5]]], float),
        "pxl": float(d["xnumAnnotImgRegParamDapiImgPxlSize"]),
        "flip_h": bool(d["xnumAnnotImgRegParamFlipHori"]),
        # FlipVert is present in the ST2WSI params; qust applies it before FlipHori.
        # In practice one flip axis suffices (H+V == 180deg rotation), but we honour
        # it for fidelity when it is set.
        "flip_v": bool(d.get("xnumAnnotImgRegParamFlipVert", False)),
        "rot": str(d["xnumAnnotImgRegParamRotation"]),
        "src_w": int(d["xnumAnnotImgRegParamSrcImgWidth"]),
        "src_h": int(d["xnumAnnotImgRegParamSrcImgHeight"]),
        "src_scale": int(d["xnumAnnotImgRegParamSourceScale"]),
        "tgt_scale": int(d["xnumAnnotImgRegParamTargetScale"]),
    }


def load_elastic(path: Path):
    """Read bUnwarpJ saveElasticTransformation: intervals + (I+3)x(I+3) cx,cy."""
    toks = Path(path).read_text().split()
    intervals = None
    if "Intervals=" in toks:                          # spaced: "Intervals= 8"
        intervals = int(toks[toks.index("Intervals=") + 1])
    else:
        for t in toks:
            if t.startswith("Intervals="):           # concatenated: "Intervals=8"
                intervals = int(t.split("=", 1)[1])
                break
    if intervals is None:
        raise ValueError(f"Cannot find Intervals value in {path}")
    nums = [float(t) for t in toks if _is_float(t)]
    n = (intervals + 3) ** 2
    cx = np.array(nums[:n]).reshape(intervals + 3, intervals + 3)
    cy = np.array(nums[n:2 * n]).reshape(intervals + 3, intervals + 3)
    return intervals, cx, cy


def _is_float(t: str) -> bool:
    try:
        float(t); return True
    except ValueError:
        return False


def _apply_source_transform(xy_px: np.ndarray, p: dict) -> np.ndarray:
    """Full-res DAPI px -> flip/rotate (full-res dims) -> ÷SourceScale -> SIFT affine.

    Returns the affine output (ax, ay) in the TargetScale-downsampled target frame,
    mirroring qupath-extension-qust XeniumAnnotation (flipV, then flipH, then
    rotation, then ÷SourceScale, then affine).
    """
    x = xy_px[:, 0].copy()
    y = xy_px[:, 1].copy()
    w, h = p["src_w"], p["src_h"]

    # ---- re-orient in full-resolution source (DAPI) pixel space ----
    if p["flip_v"]:
        y = h - y
    if p["flip_h"]:
        x = w - x
    r = p["rot"]
    if r in ("-90", "270"):
        x, y = y, w - x
    elif r in ("-180", "180"):
        x, y = w - x, h - y
    elif r in ("-270", "90"):
        x, y = h - y, x

    # ---- down to the level the SIFT affine was estimated at ----
    x = x / p["src_scale"]
    y = y / p["src_scale"]

    # ---- SIFT affine -> target coords at the TargetScale level ----
    a = p["affine"]
    ax = a[0, 0] * x + a[0, 1] * y + a[0, 2]
    ay = a[1, 0] * x + a[1, 1] * y + a[1, 2]
    return np.stack([ax, ay], 1)


def _bspline(axy: np.ndarray, intervals: int, cx: np.ndarray, cy: np.ndarray,
             denom_w: float, denom_h: float) -> np.ndarray:
    """bUnwarpJ elastic warp in the TargetScale-downsampled target frame.

    axy      : affine output (target px at the TargetScale level).
    denom_w/h: (round(W_target / TargetScale) - 1) / (round(H_target / TargetScale) - 1),
               i.e. the target lattice extent at the TargetScale level (per qust).
    """
    out = np.empty_like(axy)
    n3 = intervals + 3
    for i in range(axy.shape[0]):
        # qust rounds the affine output to integer pixels before indexing the lattice
        bu = round(float(axy[i, 0]))
        bv = round(float(axy[i, 1]))
        u = bu * intervals / denom_w + 1.0
        v = bv * intervals / denom_h + 1.0
        iu, iv = int(np.floor(u)), int(np.floor(v))
        sx = sy = 0.0
        for l in range(iv - 1, iv + 3):
            if 0 <= l < n3:
                wv = _b3(v - l)
                for k in range(iu - 1, iu + 3):
                    if 0 <= k < n3:
                        b = _b3(u - k) * wv
                        sx += cx[l, k] * b
                        sy += cy[l, k] * b
        out[i] = (sx, sy)
    return out


def _b3(t):
    t = abs(t)
    if t < 1: return 2/3 - t*t + 0.5*t**3
    if t < 2: return ((2 - t) ** 3) / 6
    return 0.0


def map_cells(xy_um, params, elastic=None, mode="affine+bspline", target_wh=None):
    """Map Xenium centroids (microns) to full-resolution H&E (target) pixels.

    Parameters
    ----------
    xy_um : (N, 2) array of Xenium centroids in microns.
    params : path to registration_params.json.
    elastic : path to direct_transf.txt (required for 'affine+bspline').
    mode : 'affine' | 'affine+bspline' (default).
    target_wh : (W, H) full-resolution H&E (target) pixel dimensions. Required for
        'affine+bspline' because the bUnwarpJ lattice is defined over the target
        image at TargetScale resolution.

    Returns
    -------
    (N, 2) array of full-resolution H&E pixel coordinates.
    """
    p = load_params(params)
    xy_px = np.asarray(xy_um, float) / p["pxl"]          # µm -> full-res DAPI px
    axy = _apply_source_transform(xy_px, p)              # -> target px @ TargetScale level

    if mode == "affine+bspline":
        if elastic is None:
            raise ValueError("mode 'affine+bspline' requires an elastic transform file")
        if target_wh is None:
            raise ValueError(
                "mode 'affine+bspline' requires target_wh=(W, H), the full-resolution "
                "H&E (target) image dimensions"
            )
        intervals, cx, cy = load_elastic(elastic)
        tgt = p["tgt_scale"]
        denom_w = int(target_wh[0] / tgt + 0.5) - 1
        denom_h = int(target_wh[1] / tgt + 0.5) - 1
        axy = _bspline(axy, intervals, cx, cy, denom_w, denom_h)
    elif mode != "affine":
        raise ValueError(f"unknown mode {mode!r}; expected 'affine' or 'affine+bspline'")

    return axy * p["tgt_scale"]                          # -> full-res H&E px
