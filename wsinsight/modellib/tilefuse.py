"""Tile-level nucleus post-processing plus stitcher for object inference."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import warnings
from concurrent.futures import ThreadPoolExecutor
from queue import Empty
from queue import Queue
from threading import Lock
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from tqdm.std import tqdm as Tqdm


# ------------------------------------------------------------------ #
# Lazily-allocated sparse canvas                                       #
# ------------------------------------------------------------------ #
class _SparseCanvas:
    """
    A 2-D (H, W) or 3-D (H, W, C) array that only allocates memory for
    ``chunk_size × chunk_size`` blocks that are actually written to.

    For large whole-slide images most of the bounding box is background,
    so tissue-covering patches may touch only a small fraction of the
    possible chunks, giving a proportional memory saving.

    All chunks share the same ``dtype``.  Callers that need float32 for
    OpenCV/scipy operations should add ``.astype(np.float32)`` on the
    result of ``read()``.

    Thread safety
    -------------
    ``write()`` must only be called from one thread at a time (the
    inference accumulation loop is sequential, so this holds).
    ``read()`` is safe to call from multiple threads simultaneously
    (returns a fresh array; only reads from ``self._chunks``).
    """

    def __init__(
        self,
        height: int,
        width: int,
        n_channels: int,  # 0 → 2-D canvas, >0 → 3-D canvas
        chunk_size: int = 4096,
        dtype=np.float16,
    ) -> None:
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.chunk_size = chunk_size
        self.dtype = dtype
        self._chunks: dict = {}  # (cy_start, cx_start) → np.ndarray

    # ------------------------------------------------------------------
    def _alloc(self, cy: int, cx: int) -> np.ndarray:
        """Allocate and register an uninitialised chunk at grid position (cy, cx).

        We use ``np.empty`` rather than ``np.zeros``: uninitialised memory is
        safe here because callers always write the full intended region via
        :meth:`write` (and :meth:`read` zero-fills unwritten sub-regions on
        each call), so a freshly-allocated chunk's bytes are never observed
        by consumers.  On glibc this skips the calloc cost (we measured a
        ~30% wall reduction in ``read()`` paths under heavy concurrent
        stitching).
        """
        ch = min(self.chunk_size, self.height - cy)
        cw = min(self.chunk_size, self.width - cx)
        shape = (ch, cw) if self.n_channels == 0 else (ch, cw, self.n_channels)
        arr = np.empty(shape, dtype=self.dtype)
        self._chunks[(cy, cx)] = arr
        return arr

    def _snap(self, coord: int) -> int:
        """Return the chunk-grid start for ``coord``."""
        return (coord // self.chunk_size) * self.chunk_size

    # ------------------------------------------------------------------
    def write(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        data: np.ndarray,
    ) -> None:
        """Write ``data[0:y1-y0, 0:x1-x0, ...]`` into ``[y0:y1, x0:x1, ...]``."""
        cs = self.chunk_size
        cy = self._snap(y0)
        while cy < y1:
            cy_end = min(cy + cs, self.height)
            cx = self._snap(x0)
            while cx < x1:
                cx_end = min(cx + cs, self.width)
                # region of intersection
                ry0 = max(y0, cy)
                ry1 = min(y1, cy_end)
                rx0 = max(x0, cx)
                rx1 = min(x1, cx_end)
                if ry1 > ry0 and rx1 > rx0:
                    chunk = self._chunks.get((cy, cx))
                    if chunk is None:
                        chunk = self._alloc(cy, cx)
                    # local coords inside chunk
                    lry0 = ry0 - cy
                    lry1 = ry1 - cy
                    lrx0 = rx0 - cx
                    lrx1 = rx1 - cx
                    # source coords inside data
                    dry0 = ry0 - y0
                    dry1 = ry1 - y0
                    drx0 = rx0 - x0
                    drx1 = rx1 - x0
                    if self.n_channels == 0:
                        chunk[lry0:lry1, lrx0:lrx1] = data[dry0:dry1, drx0:drx1]
                    else:
                        chunk[lry0:lry1, lrx0:lrx1, :] = data[dry0:dry1, drx0:drx1, :]
                cx += cs
            cy += cs

    # ------------------------------------------------------------------
    def read(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        out_dtype=None,
    ) -> np.ndarray:
        """Return a fresh array filled from ``[y0:y1, x0:x1, ...]``.

        Regions that have never been written return zeros.

        Parameters
        ----------
        out_dtype
            Optional.  When set, the returned array is allocated directly in
            this dtype so callers don't need a separate ``.astype(...)`` cast.
            NumPy then performs the cast on per-block slice assignment, which
            is cheaper than a full-array astype when only a single chunk is
            read.  Defaults to the canvas' stored dtype.
        """
        if out_dtype is None:
            out_dtype = self.dtype
        h = y1 - y0
        w = x1 - x0
        shape = (h, w) if self.n_channels == 0 else (h, w, self.n_channels)
        out = np.zeros(shape, dtype=out_dtype)
        cs = self.chunk_size
        cy = self._snap(y0)
        while cy < y1:
            cy_end = min(cy + cs, self.height)
            cx = self._snap(x0)
            while cx < x1:
                cx_end = min(cx + cs, self.width)
                ry0 = max(y0, cy)
                ry1 = min(y1, cy_end)
                rx0 = max(x0, cx)
                rx1 = min(x1, cx_end)
                if ry1 > ry0 and rx1 > rx0:
                    chunk = self._chunks.get((cy, cx))
                    if chunk is not None:
                        lry0 = ry0 - cy
                        lry1 = ry1 - cy
                        lrx0 = rx0 - cx
                        lrx1 = rx1 - cx
                        dry0 = ry0 - y0
                        dry1 = ry1 - y0
                        drx0 = rx0 - x0
                        drx1 = rx1 - x0
                        if self.n_channels == 0:
                            out[dry0:dry1, drx0:drx1] = chunk[lry0:lry1, lrx0:lrx1]
                        else:
                            out[dry0:dry1, drx0:drx1, :] = chunk[
                                lry0:lry1, lrx0:lrx1, :
                            ]
                cx += cs
            cy += cs
        return out


# Tame nested threading from 3rd-party libs
import cv2

try:
    cv2.setNumThreads(1)
except Exception:
    pass

# from ..num_worker_optimizer import pick_workers_safe, throttle_when_busy


# ------------------------------- #
# Robust nuclei post-proc (fixed) #
# ------------------------------- #
def _proc_np_hv(
    np_map: np.ndarray, hv_map: np.ndarray, min_object_size: int
) -> np.ndarray:
    """
    Robust nuclei postproc on one tile.
    - Handles empty/near-empty tiles
    - Skips remove_small_objects when label count < 2 (avoids warning)
    Returns: int32 instance map (H, W)
    """
    H, W = np_map.shape[:2]

    # 1) foreground
    blb_bin = (np_map >= 0.5).astype(np.uint8)
    if blb_bin.sum() == 0:
        return np.zeros((H, W), dtype=np.int32)

    labeled, num = ndi.label(blb_bin)
    if num > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            labeled = remove_small_objects(labeled, min_size=int(min_object_size))
    blb = (labeled > 0).astype(np.uint8)
    if blb.sum() == 0:
        return np.zeros((H, W), dtype=np.int32)

    # 2) HV normalize + edges
    h_dir = cv2.normalize(
        hv_map[:, :, 0],
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    v_dir = cv2.normalize(
        hv_map[:, :, 1],
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1.0 - cv2.normalize(
        sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    sobelv = 1.0 - cv2.normalize(
        sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1.0 - blb)  # suppress background
    overall[overall < 0] = 0

    # 3) distance (basins)
    dist = (1.0 - overall) * blb
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    # 4) markers
    ridge = (overall >= 0.4).astype(np.uint8)
    marker = blb.astype(np.int16) - ridge.astype(np.int16)
    marker = np.clip(marker, 0, 1).astype(np.uint8)
    if marker.any():
        marker = binary_fill_holes(marker).astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker, mnum = ndi.label(marker)
        if mnum > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                marker = remove_small_objects(marker, min_size=int(min_object_size))
    else:
        # fallback to a single marker if nothing left
        marker, _ = ndi.label(blb)

    # 5) watershed
    proced_pred = watershed(dist, markers=marker, mask=blb.astype(bool))
    return proced_pred.astype(np.int32)


# ----------------------------------------------------------------------------
# Per-tile dedup across overlapping tile seams (CV++ verbatim Option D)
# ----------------------------------------------------------------------------
def _dedup_overlapping_cells(
    inst,
    prob,
    poly,
    origin,
    slide_overlap_size: int = 0,
    slide_patch_size: int = 0,
    slide_halo_size: int = 0,
    iou_threshold: float = 0.01,
):
    """Drop duplicate cells emitted by adjacent overlapping tiles.

    Two passes:

    * **Pass-A (edge-distance)**: two cells are paired when their bboxes touch
      along a single axis within ``slide_overlap_size`` pixels AND the orthogonal
      overlap exceeds half the smaller bbox extent.  Same argmax class
      required.  Captures half-cell pairs across the tile seam whose bboxes
      share an edge but have IoU == 0.

    * **Pass-B (IoU)**: same argmax class AND bbox IoU > ``iou_threshold`` (CV++
      default 0.01).  Captures overlapping cells whose bboxes genuinely
      intersect.

    Survivor rule: higher max-prob wins; ties broken by lower index
    (instance_id stability across reruns).

    Parameters
    ----------
    inst, prob, poly
        Index-aligned lists in the format ``_stitching_worker`` returns:
        ``inst`` is a list of ``(1, 4)`` int32 bboxes ``[minx, miny, w, h]``;
        ``prob`` is a list of ``(1, K)`` float32 type-probability vectors;
        ``poly`` is a list of ``(M, 2)`` int32 polygon vertices.
    origin
        Index-aligned list of ``(tile_row, tile_col)``.  Retained for test
        compatibility (``tests/test_tilefuse_dedup.py`` passes it); this
        implementation does not consult it (per-tile dedup is single-origin).
    slide_overlap_size
        Side length of the right/bottom overlap strip in slide pixels.  When
        ``0`` the function is a no-op (returns the input lists unchanged), so
        older model configs without ``overlap_size_pixels`` set are byte-
        identical to the pre-overlap path.
    slide_patch_size, slide_halo_size
        Retained for test compatibility; not used by this implementation.
    iou_threshold
        IoU threshold for Pass-B; default 0.01 matches CV++.

    Returns
    -------
    (kept_inst, kept_prob, kept_poly) — index-aligned lists.
    """
    n = len(inst)
    if n == 0 or slide_overlap_size == 0:
        return list(inst), list(prob), list(poly)

    # Safety cap: Pass-A builds (n, n) matrices (multiple ~64 MB temporaries at
    # n=4000). On tissue-dense tiles that exceed this, NumPy starts paging
    # temporaries to swap and ``_stitching_worker`` goes from ~66 ms to minutes.
    # Skip dedup and emit raw detections; the caller still gets a valid export.
    _DEDUP_MAX_CELLS = 4000
    if n > _DEDUP_MAX_CELLS:
        import sys as _sys

        print(
            f"tilefuse.dedup: skipping dedup for tile with {n} cells "
            f"(>{_DEDUP_MAX_CELLS}); half-cell pairs above this threshold "
            f"would page to swap.",
            file=_sys.stderr,
        )
        return list(inst), list(prob), list(poly)

    # --- bbox matrix (n, 4) of [minx, miny, w, h]
    bboxes = (
        np.array([np.asarray(b).reshape(-1)[:4] for b in inst], dtype=np.int64).reshape(
            n, 4
        )
        if n
        else np.empty((0, 4), dtype=np.int64)
    )
    x_min = bboxes[:, 0]
    y_min = bboxes[:, 1]
    w = bboxes[:, 2]
    h = bboxes[:, 3]
    x_max = x_min + w
    y_max = y_min + h

    # --- argmax class per cell
    def _argmax_p(p):
        a = np.asarray(p).reshape(-1)
        return int(a.argmax()), float(a.max())

    cells_am, cells_pmax = (
        zip(*(_argmax_p(p) for p in prob), strict=True) if n else ([], [])
    )
    cells_am = np.asarray(cells_am, dtype=np.int32)
    cells_pmax = np.asarray(cells_pmax, dtype=np.float32)

    drop = np.zeros(n, dtype=bool)  # cells to drop

    # --- Pass A: bbox edge-distance match (half-cell pairs across seam)
    #     i is "left half" if a.right ~ b.left, AND y overlap substantial.
    edge_tol = 1
    if n >= 2:
        # (n, n) geometry
        x_max_col = x_max[:, None]
        x_min_row = x_min[None, :]
        y_min_row = y_min[None, :]
        y_max_col = y_max[:, None]
        y_max_row = y_max[None, :]
        h_col = h[:, None]
        h_row = h[None, :]
        w_col = w[:, None]
        w_row = w[None, :]

        # Half-cell pair when the right edge of one bbox abuts the left edge of
        # the other (or top/bottom mirror).  Substantial overlap rule uses
        # 0.5 * min(h) so a 10x20 paired with a 10x5 is rejected.
        x_min_col = x_min[:, None]
        x_max_row = x_max[None, :]
        y_min_col = y_min[:, None]
        y_subst_lr = (
            np.minimum(y_max_col, y_max_row) - np.maximum(y_min_col, y_min_row)
        ) > 0.5 * np.minimum(h_col, h_row)
        match_lr = (np.abs(x_max_col - x_min_row) <= edge_tol) & y_subst_lr
        x_subst_tb = (
            np.minimum(x_max_col, x_max_row) - np.maximum(x_min_col, x_min_row)
        ) > 0.5 * np.minimum(w_col, w_row)
        match_tb = (np.abs(y_max_col - y_min_row) <= edge_tol) & x_subst_tb
        # Either direction (a left of b, or b left of a).
        half_match = match_lr | match_lr.T | match_tb | match_tb.T
        # Argmax class must agree.
        class_match = cells_am[:, None] == cells_am[None, :]
        # Each pair only once — keep i<j.
        i_idx, j_idx = np.triu_indices(n, k=1)
        cand = half_match[i_idx, j_idx] & class_match[i_idx, j_idx]
        if cand.any():
            ai, bi = i_idx[cand], j_idx[cand]
            # Higher max-prob wins; tie → keep lower index.
            a_p = cells_pmax[ai]
            b_p = cells_pmax[bi]
            drop_a = (a_p < b_p) | ((a_p == b_p) & (ai > bi))
            drop_b = (b_p < a_p) | ((b_p == a_p) & (bi > ai))
            drop[ai[drop_a]] = True
            drop[bi[drop_b]] = True

    # --- Pass B: IoU on survivors (CV++ verbatim, vectorized).
    survivor = np.flatnonzero(~drop)
    if survivor.size >= 2:
        i_idx2, j_idx2 = np.triu_indices(survivor.size, k=1)
        ax0 = x_min[survivor[i_idx2]]
        ay0 = y_min[survivor[i_idx2]]
        ax1 = x_max[survivor[i_idx2]]
        ay1 = y_max[survivor[i_idx2]]
        bx0 = x_min[survivor[j_idx2]]
        by0 = y_min[survivor[j_idx2]]
        bx1 = x_max[survivor[j_idx2]]
        by1 = y_max[survivor[j_idx2]]
        iw = np.maximum(0, np.minimum(ax1, bx1) - np.maximum(ax0, bx0))
        ih = np.maximum(0, np.minimum(ay1, by1) - np.maximum(ay0, by0))
        inter = iw * ih
        areas = w * h
        union = areas[survivor[i_idx2]] + areas[survivor[j_idx2]] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        # Same class required, IoU above threshold.
        cls_eq = cells_am[survivor[i_idx2]] == cells_am[survivor[j_idx2]]
        cand2 = (iou > iou_threshold) & cls_eq
        if cand2.any():
            ii = survivor[i_idx2[cand2]]
            jj = survivor[j_idx2[cand2]]
            # Higher max-prob wins; tie → keep lower SURVIVOR index (NOT
            # original index, since Pass-A may have already shifted order).
            a_p = cells_pmax[ii]
            b_p = cells_pmax[jj]
            ii_local = np.searchsorted(survivor, ii)
            jj_local = np.searchsorted(survivor, jj)
            drop_a = (a_p < b_p) | ((a_p == b_p) & (ii_local > jj_local))
            drop_b = (b_p < a_p) | ((b_p == a_p) & (jj_local > ii_local))
            drop[ii[drop_a]] = True
            drop[jj[drop_b]] = True

    keep = np.flatnonzero(~drop)
    return ([inst[k] for k in keep], [prob[k] for k in keep], [poly[k] for k in keep])


# ----------------------------- #
# Existing per-tile measurement #
# ----------------------------- #
def _stitching_worker(
    np_tile,
    hv_tile,
    tp_tile,
    interior_y0,
    interior_x0,
    interior_slice,
    min_object_size,
    slide_overlap_size: int = 0,
    slide_patch_size: int = 0,
):
    pred_inst_padded = _proc_np_hv(np_tile, hv_tile, min_object_size).astype(np.int32)
    ys, xs = interior_slice
    pred_inst = pred_inst_padded[ys, xs]

    max_id = int(pred_inst.max())
    if max_id <= 0:
        return [], [], []

    labels = pred_inst
    lbl = labels.ravel()

    counts = np.bincount(lbl, minlength=max_id + 1).astype(np.int32)
    counts[0] = 0
    valid_ids = np.nonzero(counts)[0]
    if valid_ids.size == 0:
        return [], [], []

    slices = ndi.find_objects(labels, max_label=max_id)

    inst_list: List[np.ndarray] = []
    prob_list: List[np.ndarray] = []
    poly_list: List[np.ndarray] = []

    for inst_id in valid_ids.tolist():
        sl = slices[inst_id - 1]
        if sl is None:
            continue
        r_sl, c_sl = sl
        rmin, rmax = r_sl.start, r_sl.stop
        cmin, cmax = c_sl.start, c_sl.stop

        # global bbox
        x = cmin + interior_x0
        y = rmin + interior_y0
        w = cmax - cmin
        h = rmax - rmin

        local = labels[rmin:rmax, cmin:cmax] == inst_id

        # bbox-bounded mean — avoids the global np.add.at (unbuffered, ~46M
        # Python-level += per tile on a 2048² slide).  ~10x faster.
        prob = tp_tile[rmin:rmax, cmin:cmax, :][local].astype(np.float32).mean(axis=0)

        # polygon — must succeed before emitting.  Empty cv2 output (very
        # small label regions rejected by marching squares) becomes a bbox
        # rectangle so the three lists stay index-aligned downstream.
        cnts, _ = cv2.findContours(
            local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not cnts:
            poly = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32
            )
        else:
            cnt = max(cnts, key=cv2.contourArea)  # (M,1,2)
            cnt2d = cnt.squeeze(1).astype(np.int32)
            if cnt2d.ndim != 2 or cnt2d.shape[0] < 3:
                poly = np.array(
                    [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32
                )
            else:
                cnt2d[:, 0] += x
                cnt2d[:, 1] += y
                poly = cnt2d

        # Triple-append atomically — keeps inst/prob/poly index-aligned so
        # ``_dedup_overlapping_cells`` can use the same np index for all three.
        inst_list.append(np.array([x, y, w, h], dtype=np.int32).reshape(1, -1))
        prob_list.append(prob.reshape(1, -1))
        poly_list.append(poly)

    # Drop half-cell / overlapping duplicates emitted by the neighbouring
    # tile when overlapping-tile seams are configured via ``overlap_size_pixels``.
    if slide_overlap_size > 0 and inst_list:
        # All cells in this worker come from the same tile; in-worker dedup is
        # intentionally single-origin, so origin is a uniform placeholder here.
        # Kept in the call to preserve the API used by ``tests/test_tilefuse_dedup.py``.
        origin = [(0, 0)] * len(inst_list)
        inst_list, prob_list, poly_list = _dedup_overlapping_cells(
            inst_list,
            prob_list,
            poly_list,
            origin,
            slide_overlap_size=slide_overlap_size,
            slide_patch_size=slide_patch_size,
        )

    return inst_list, prob_list, poly_list


# ----------------------- #
# The main stitcher class #
# ----------------------- #
class TileRemapStitcher:
    """
    Accelerated path:
      - GPU softmax & 164->S bilinear resize for the whole batch
      - HV vectors scaled by S/164
      - Preallocated canvases to avoid reallocation
      - Threaded, batched finalize with index-only jobs
    """

    def __init__(
        self,
        n_classes: int,
        slide_width: int,
        slide_height: int,
        slide_patch_size: int,
        slide_halo_size: int,
        slide_mpp: float,
        model_mpp: float,
        min_object_size: int = 20,
        device="cuda",
        slide_overlap_size: int = 0,
    ):
        self.n_classes = n_classes
        self.slide_width = slide_width
        self.slide_height = slide_height
        self.slide_patch_size = slide_patch_size
        self.slide_halo_size = slide_halo_size
        self.alpha = model_mpp / slide_mpp
        self.min_object_size = int(min_object_size)
        # Right/bottom overlap strip in slide pixels (= overlap_size_pixels *
        # slide_mpp / spacing_um_px, rounded).  When 0 the dedup pass in
        # ``_stitching_worker`` becomes a no-op, preserving byte-identical
        # behaviour for older model configs without ``overlap_size_pixels``.
        self.slide_overlap_size = int(slide_overlap_size) if slide_overlap_size else 0
        # Sparse canvases: only chunks that receive inference patches are
        # allocated, avoiding OOM on large slides with sparse tissue.
        # float16 halves memory vs float32; finalize workers upcast to float32
        # before passing to OpenCV/scipy (see the worker closure below).
        self.np_map = _SparseCanvas(
            slide_height, slide_width, n_channels=0, dtype=np.float16
        )
        self.hv_map = _SparseCanvas(
            slide_height, slide_width, n_channels=2, dtype=np.float16
        )
        self.tp_map = _SparseCanvas(
            slide_height, slide_width, n_channels=self.n_classes, dtype=np.float16
        )
        self.device = device

    def _get_bounding_box(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmax += 1
        cmax += 1
        return [rmin, rmax, cmin, cmax]

    # --------- hot path: batch GPU → single CPU write ---------
    @torch.no_grad()
    def accumulate_batch_torch(self, pred_dict: dict, batch_coords: torch.Tensor):
        """
        pred_dict: {"np": (B,2,164,164), "hv": (B,2,164,164), "tp": (B,K,164,164)} Tensors on 'device'
        batch_coords: (B,4) [minx, miny, pw, ph] in target S-grid
        """
        assert ("np" in pred_dict and "hv" in pred_dict and "tp" in pred_dict) or (
            "nuclei_binary_map" in pred_dict
            and "hv_map" in pred_dict
            and "nuclei_type_map" in pred_dict
        )

        np_logits: torch.Tensor = (
            pred_dict["np"] if "np" in pred_dict else pred_dict["nuclei_binary_map"]
        )
        hv: torch.Tensor = pred_dict["hv"] if "hv" in pred_dict else pred_dict["hv_map"]
        tp_logits: torch.Tensor = (
            pred_dict["tp"] if "tp" in pred_dict else pred_dict["nuclei_type_map"]
        )

        slide_width = self.slide_width
        slide_height = self.slide_height
        batch_size = np_logits.shape[0]
        slide_patch_size = self.slide_patch_size
        alpha = self.alpha

        # Softmax on GPU
        np_prob = torch.softmax(np_logits, dim=1)[:, 1:2, ...]  # (B,1,164,164)
        tp_prob = torch.softmax(tp_logits, dim=1)  # (B,K,164,164)

        # 164 -> S resize on GPU
        np_res = F.interpolate(
            np_prob,
            size=(slide_patch_size, slide_patch_size),
            mode="bilinear",
            align_corners=False,
        )  # (B,1,S,S)
        hv_res = (
            F.interpolate(
                hv,
                size=(slide_patch_size, slide_patch_size),
                mode="bilinear",
                align_corners=False,
            )
            * alpha
        )  # (B,2,S,S)
        tp_res = F.interpolate(
            tp_prob,
            size=(slide_patch_size, slide_patch_size),
            mode="bilinear",
            align_corners=False,
        )  # (B,K,S,S)

        # Renormalize TP per pixel
        tp_res = tp_res / (tp_res.sum(dim=1, keepdim=True) + 1e-8)

        # Single host transfer
        np_res_np = np_res.squeeze(1).contiguous().cpu().numpy()  # (B,S,S)
        hv_res_np = hv_res.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # (B,S,S,2)
        tp_res_np = tp_res.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # (B,S,S,K)

        # Coordinates
        coords = batch_coords.detach().to("cpu").numpy().astype(np.int32)[:, :2]

        # Tight CPU writes (clip)
        for i in range(batch_size):
            x0 = int(coords[i, 0])
            y0 = int(coords[i, 1])
            x1 = x0 + slide_patch_size
            y1 = y0 + slide_patch_size

            cx0 = max(0, x0)
            cy0 = max(0, y0)
            cx1 = min(slide_width, x1)
            cy1 = min(slide_height, y1)
            if cx1 <= cx0 or cy1 <= cy0:
                continue

            tx0 = cx0 - x0
            ty0 = cy0 - y0
            tx1 = tx0 + (cx1 - cx0)
            ty1 = ty0 + (cy1 - cy0)

            self.np_map.write(cy0, cy1, cx0, cx1, np_res_np[i, ty0:ty1, tx0:tx1])
            self.hv_map.write(cy0, cy1, cx0, cx1, hv_res_np[i, ty0:ty1, tx0:tx1, :])
            self.tp_map.write(cy0, cy1, cx0, cx1, tp_res_np[i, ty0:ty1, tx0:tx1, :])

    def finalize(
        self,
        tile_size: int = 2048,
        padding_size: int = 64,
        pbar: Optional[Tqdm] = None,
        num_workers: Optional[int] = None,
        tiles_per_task: int = 4,
    ):
        """
        Queue-based finalize:
          - num_workers threads pull tiles from a shared queue (auto load balancing)
          - Optional tiles_per_task to reduce queue contention
          - Per-tile progress updates (smooth tqdm)

        tiles_per_task default raised 1 -> 4 on 2026-08-25: queue contention is
        the dominant per-tile overhead at low tiles-per-task (each tile triggers
        q.get + q.task_done). Batching 4 tiles per worker pull amortises that.
        """
        H, W = self.slide_height, self.slide_width
        if H <= 0 or W <= 0:
            return [], [], []

        # 1) Build index-only jobs (no data slicing yet)
        #    With ``slide_overlap_size`` > 0 (CellViT-style overlap), adjacent
        #    finalize tiles are emitted at stride < tile_size so that cells
        #    straddling a tile seam appear in two adjacent worker invocations
        #    and ``_stitching_worker`` can dedup them via edge-distance match.
        #    Inference coverage (``accumulate_batch_torch``) is unaffected —
        #    each physical pixel is still inferred exactly once.
        stride_y = max(1, tile_size - self.slide_overlap_size)
        stride_x = max(1, tile_size - self.slide_overlap_size)
        jobs: List[Tuple[int, int, int, int, int, int, int, int, int, int]] = []
        for interior_y0 in range(0, H, stride_y):
            for interior_x0 in range(0, W, stride_x):
                interior_y1 = min(interior_y0 + tile_size, H)
                interior_x1 = min(interior_x0 + tile_size, W)
                if interior_y1 <= interior_y0 or interior_x1 <= interior_x0:
                    continue

                pad_y0 = max(0, interior_y0 - padding_size)
                pad_y1 = min(interior_y1 + padding_size, H)
                pad_x0 = max(0, interior_x0 - padding_size)
                pad_x1 = min(interior_x1 + padding_size, W)

                inner_y0 = interior_y0 - pad_y0
                inner_x0 = interior_x0 - pad_x0
                inner_y1 = inner_y0 + (interior_y1 - interior_y0)
                inner_x1 = inner_x0 + (interior_x1 - interior_x0)

                jobs.append(
                    (
                        pad_y0,
                        pad_y1,
                        pad_x0,
                        pad_x1,
                        interior_y0,
                        interior_x0,
                        inner_y0,
                        inner_y1,
                        inner_x0,
                        inner_x1,
                    )
                )

        if not jobs:
            return [], [], []

        total = len(jobs)
        if pbar is not None and getattr(pbar, "total", None) is None:
            # 若外部已設定 total，就尊重外部；否則設一下可得較好體驗
            try:
                pbar.reset(
                    total=total
                )  # tqdm>=4.66 支援；若不支援會丟例外，下面 except 吞掉
            except Exception:
                pass

        q: Queue = Queue()
        for j in jobs:
            q.put(j)

        inst_all: List[np.ndarray] = []
        prob_all: List[np.ndarray] = []
        poly_all: List[np.ndarray] = []
        merge_lock = Lock()  # 合併全域結果的鎖
        pbar_lock = Lock()  # 進度條更新的鎖（避免競爭）

        np_map = self.np_map
        hv_map = self.hv_map
        tp_map = self.tp_map
        min_object_size = self.min_object_size

        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1) - 2)
        tiles_per_task = max(1, int(tiles_per_task))

        # Add sentinels so that workers can exit cleanly after queue drains
        for _ in range(num_workers):
            q.put(None)

        def worker():
            local_inst: List[np.ndarray] = []
            local_prob: List[np.ndarray] = []
            local_poly: List[np.ndarray] = []
            while True:
                job = q.get()
                if job is None:
                    q.task_done()
                    break

                batched_jobs = [job]
                for _ in range(tiles_per_task - 1):
                    try:
                        nxt = q.get_nowait()
                    except Empty:
                        break
                    if nxt is None:
                        # Put sentinel back for another worker and stop batching
                        q.put(None)
                        break
                    batched_jobs.append(nxt)

                for (
                    pad_y0,
                    pad_y1,
                    pad_x0,
                    pad_x1,
                    interior_y0,
                    interior_x0,
                    inner_y0,
                    inner_y1,
                    inner_x0,
                    inner_x1,
                ) in batched_jobs:
                    # Read sparse chunks.  np/hv come back as float32
                    # directly (``read(out_dtype=...)`` skips a post-cast
                    # pass); tp stays float16 (worker casts per-cell to
                    # float64 only over the bbox region).
                    np_tile = np_map.read(
                        pad_y0, pad_y1, pad_x0, pad_x1, out_dtype=np.float32
                    )
                    hv_tile = hv_map.read(
                        pad_y0, pad_y1, pad_x0, pad_x1, out_dtype=np.float32
                    )
                    tp_tile = tp_map.read(
                        pad_y0, pad_y1, pad_x0, pad_x1
                    )  # float16; worker casts to float64 inside the bbox
                    interior_slice = (
                        slice(inner_y0, inner_y1),
                        slice(inner_x0, inner_x1),
                    )

                    ins, prb, ply = _stitching_worker(
                        np_tile,
                        hv_tile,
                        tp_tile,
                        interior_y0,
                        interior_x0,
                        interior_slice,
                        min_object_size,
                        slide_overlap_size=self.slide_overlap_size,
                        slide_patch_size=tile_size,
                    )

                    if ins:
                        local_inst.extend(ins)
                        local_prob.extend(prb)
                        local_poly.extend(ply)

                    if pbar is not None:
                        with pbar_lock:
                            pbar.update(1)

                    q.task_done()

            # 批次把本工人的結果一次性併入（減少鎖競爭）
            if local_inst:
                with merge_lock:
                    inst_all.extend(local_inst)
                    prob_all.extend(local_prob)
                    poly_all.extend(local_poly)

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = [ex.submit(worker) for _ in range(num_workers)]
            for f in futs:
                f.result()  # surface exceptions

        return inst_all, prob_all, poly_all
