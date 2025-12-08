"""Tile-level nucleus post-processing plus stitcher for object inference."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.std import tqdm as Tqdm
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed

from queue import Queue, Empty
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

import warnings

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
def _proc_np_hv(np_map: np.ndarray, hv_map: np.ndarray, min_object_size: int) -> np.ndarray:
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
    h_dir = cv2.normalize(hv_map[:, :, 0], None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v_dir = cv2.normalize(hv_map[:, :, 1], None, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1.0 - cv2.normalize(sobelh, None, alpha=0, beta=1,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobelv = 1.0 - cv2.normalize(sobelv, None, alpha=0, beta=1,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

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


# ----------------------------- #
# Existing per-tile measurement #
# ----------------------------- #
def _stitching_worker(np_tile, hv_tile, tp_tile, interior_y0, interior_x0, interior_slice, min_object_size):
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

    n_classes = int(tp_tile.shape[2])
    tp_interior = tp_tile[ys, xs, :]          # (h,w,K)
    tp_flat = tp_interior.reshape(-1, n_classes).astype(np.float64)

    cls_sums = np.zeros((max_id + 1, n_classes), dtype=np.float64)
    np.add.at(cls_sums, lbl, tp_flat)
    cls_sums[0, :] = 0
    denom = counts.astype(np.float64)
    denom[denom == 0] = 1.0
    cls_means = (cls_sums.T / denom).T.astype(np.float32)  # (max_id+1, K)

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
        w = (cmax - cmin)
        h = (rmax - rmin)

        inst_list.append(np.array([x, y, w, h], dtype=np.int32).reshape(1, -1))
        prob_list.append(cls_means[inst_id].copy().reshape(1, -1))

        # polygon
        local = (labels[rmin:rmax, cmin:cmax] == inst_id).astype(np.uint8)
        cnts, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)  # (M,1,2)
        poly = cnt.squeeze(1).astype(np.int32)
        if poly.ndim != 2 or poly.shape[0] < 3:
            continue
        poly[:, 0] += x
        poly[:, 1] += y
        poly_list.append(poly)

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

    def __init__(self, n_classes: int,
                 slide_width: int,
                 slide_height: int,
                 slide_patch_size: int,
                 slide_halo_size: int,
                 slide_mpp: float,
                 model_mpp: float,
                 min_object_size: int = 20,
                 device="cuda"):
        self.n_classes = n_classes
        self.slide_width = slide_width
        self.slide_height = slide_height
        self.slide_patch_size = slide_patch_size
        self.slide_halo_size = slide_halo_size
        self.alpha = model_mpp / slide_mpp
        self.min_object_size = int(min_object_size)
        self.np_map = np.zeros((slide_height, slide_width), dtype=np.float32)
        self.hv_map = np.zeros((slide_height, slide_width, 2), dtype=np.float32)
        self.tp_map = np.zeros((slide_height, slide_width, self.n_classes), dtype=np.float32)
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
        assert ('np' in pred_dict and 'hv' in pred_dict and 'tp' in pred_dict) or \
               ('nuclei_binary_map' in pred_dict and 'hv_map' in pred_dict and 'nuclei_type_map' in pred_dict)

        np_logits: torch.Tensor = pred_dict['np'] if 'np' in pred_dict else pred_dict['nuclei_binary_map']
        hv:        torch.Tensor = pred_dict['hv'] if 'hv' in pred_dict else pred_dict['hv_map']
        tp_logits: torch.Tensor = pred_dict['tp'] if 'tp' in pred_dict else pred_dict['nuclei_type_map']

        slide_width = self.slide_width
        slide_height = self.slide_height
        batch_size = np_logits.shape[0]
        slide_patch_size = self.slide_patch_size
        slide_halo_size = self.slide_halo_size
        alpha = self.alpha

        # Softmax on GPU
        np_prob = torch.softmax(np_logits, dim=1)[:, 1:2, ...]     # (B,1,164,164)
        tp_prob = torch.softmax(tp_logits, dim=1)                   # (B,K,164,164)

        # 164 -> S resize on GPU
        np_res = F.interpolate(np_prob, size=(slide_patch_size, slide_patch_size),
                               mode='bilinear', align_corners=False)                 # (B,1,S,S)
        hv_res = F.interpolate(hv, size=(slide_patch_size, slide_patch_size),
                               mode='bilinear', align_corners=False) * alpha        # (B,2,S,S)
        tp_res = F.interpolate(tp_prob, size=(slide_patch_size, slide_patch_size),
                               mode='bilinear', align_corners=False)                # (B,K,S,S)

        # Renormalize TP per pixel
        tp_res = tp_res / (tp_res.sum(dim=1, keepdim=True) + 1e-8)

        # Single host transfer
        np_res_np = np_res.squeeze(1).contiguous().cpu().numpy()                 # (B,S,S)
        hv_res_np = hv_res.permute(0, 2, 3, 1).contiguous().cpu().numpy()        # (B,S,S,2)
        tp_res_np = tp_res.permute(0, 2, 3, 1).contiguous().cpu().numpy()        # (B,S,S,K)

        # Coordinates
        coords = batch_coords.detach().to('cpu').numpy().astype(np.int32)[:, :2]
        coords = coords[:, :2] + slide_halo_size

        # Tight CPU writes (clip)
        for i in range(batch_size):
            x0 = int(coords[i, 0]); y0 = int(coords[i, 1])
            x1 = x0 + slide_patch_size; y1 = y0 + slide_patch_size

            cx0 = max(0, x0); cy0 = max(0, y0)
            cx1 = min(slide_width, x1); cy1 = min(slide_height, y1)
            if cx1 <= cx0 or cy1 <= cy0:
                continue

            tx0 = cx0 - x0; ty0 = cy0 - y0
            tx1 = tx0 + (cx1 - cx0); ty1 = ty0 + (cy1 - cy0)

            self.np_map[cy0:cy1, cx0:cx1]        = np_res_np[i, ty0:ty1, tx0:tx1]
            self.hv_map[cy0:cy1, cx0:cx1, :]     = hv_res_np[i, ty0:ty1, tx0:tx1, :]
            self.tp_map[cy0:cy1, cx0:cx1, :]     = tp_res_np[i, ty0:ty1, tx0:tx1, :]

    def finalize(self,
                 tile_size: int = 2048,
                 padding_size: int = 64,
                 pbar: Optional[Tqdm] = None,
                 num_workers: Optional[int] = None,
                 tiles_per_task: int = 1):
        """
        Queue-based finalize:
          - No tiles_per_task / inflight_factor
          - num_workers threads pull tiles from a shared queue (auto load balancing)
          - Optional tiles_per_task to reduce queue contention
          - Per-tile progress updates (smooth tqdm)
        """
        H, W = self.slide_height, self.slide_width
        if H <= 0 or W <= 0:
            return [], [], []
        
        # 1) Build index-only jobs (no data slicing yet)
        jobs: List[Tuple[int, int, int, int, int, int, int, int, int, int]] = []
        for interior_y0 in range(0, H, tile_size):
            for interior_x0 in range(0, W, tile_size):
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
    
                jobs.append((pad_y0, pad_y1, pad_x0, pad_x1,
                             interior_y0, interior_x0,
                             inner_y0, inner_y1, inner_x0, inner_x1))
    
        if not jobs:
            return [], [], []
    
        total = len(jobs)
        if pbar is not None and getattr(pbar, "total", None) is None:
            # 若外部已設定 total，就尊重外部；否則設一下可得較好體驗
            try:
                pbar.reset(total=total)  # tqdm>=4.66 支援；若不支援會丟例外，下面 except 吞掉
            except Exception:
                pass
    
        q: Queue = Queue()
        for j in jobs:
            q.put(j)
    
        inst_all: List[np.ndarray] = []
        prob_all: List[np.ndarray] = []
        poly_all: List[np.ndarray] = []
        merge_lock = Lock()   # 合併全域結果的鎖
        pbar_lock = Lock()    # 進度條更新的鎖（避免競爭）
    
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

                for (pad_y0, pad_y1, pad_x0, pad_x1,
                     interior_y0, interior_x0,
                     inner_y0, inner_y1, inner_x0, inner_x1) in batched_jobs:

                    np_tile = np.ascontiguousarray(np_map[pad_y0:pad_y1, pad_x0:pad_x1])
                    hv_tile = np.ascontiguousarray(hv_map[pad_y0:pad_y1, pad_x0:pad_x1, :])
                    tp_tile = np.ascontiguousarray(tp_map[pad_y0:pad_y1, pad_x0:pad_x1, :])
                    interior_slice = (slice(inner_y0, inner_y1), slice(inner_x0, inner_x1))

                    ins, prb, ply = _stitching_worker(
                        np_tile, hv_tile, tp_tile,
                        interior_y0, interior_x0,
                        interior_slice,
                        min_object_size
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

