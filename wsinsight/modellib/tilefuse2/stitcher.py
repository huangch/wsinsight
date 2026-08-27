"""Bucket-ownership stitcher (v2).

Same public surface as :class:`~wsinsight.modellib.tilefuse.TileRemapStitcher`:
``accumulate_batch_torch`` fills the canvases during inference and ``finalize``
returns three index-aligned lists ``(inst, prob, poly)``.

The inference half is a verbatim port — a v1/v2 run writes byte-identical
canvases, so any difference in the result is attributable to the finalize
rewrite alone.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty
from queue import Queue
from threading import Lock
from typing import List
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.std import tqdm as Tqdm

from .canvas import SparseCanvas
from .diagnostics import EmitStats
from .diagnostics import StageTimer
from .emit import emit_instances
from .geometry import BucketGeometry


def _verbose() -> bool:
    """Full geometry / diagnostics / stage timing on stderr (opt-in)."""
    return os.environ.get("WSINSIGHT_TILEFUSE2_VERBOSE", "0") != "0"


class TileRemapStitcherV2:
    def __init__(
        self,
        n_classes: int,
        slide_width: int,
        slide_height: int,
        slide_patch_size: int,
        slide_mpp: float,
        model_mpp: float,
        patch_size_pixels: int,
        halo_size_pixels: int,
        overlap_size_pixels: int,
        min_object_size: int = 20,
        device="cuda",
        report: bool = True,
    ):
        self.n_classes = n_classes
        self.slide_width = slide_width
        self.slide_height = slide_height
        self.slide_patch_size = slide_patch_size
        self.alpha = model_mpp / slide_mpp
        self.min_object_size = int(min_object_size)
        self.device = device
        self.report = bool(report)

        self.geometry = BucketGeometry.from_model_config(
            patch_size_pixels=patch_size_pixels,
            halo_size_pixels=halo_size_pixels,
            overlap_size_pixels=overlap_size_pixels,
            model_mpp=model_mpp,
            slide_mpp=slide_mpp,
            slide_height=slide_height,
            slide_width=slide_width,
        )

        self.np_map = SparseCanvas(
            slide_height, slide_width, n_channels=0, dtype=np.float16
        )
        self.hv_map = SparseCanvas(
            slide_height, slide_width, n_channels=2, dtype=np.float16
        )
        self.tp_map = SparseCanvas(
            slide_height, slide_width, n_channels=self.n_classes, dtype=np.float16
        )
        # Feather blending: adjacent patches overlap by ``slide_patch_size - B``
        # and are summed under complementary linear ramps.  The accumulated
        # weight is tracked explicitly rather than assumed to be 1, because the
        # slide border and tissue-mask gaps both leave a patch without the
        # neighbour that would complete its ramp.
        self.feather = max(0, int(slide_patch_size) - self.geometry.bucket_size)
        # patchlib stores the top-left of the *whole* patch, but a haloed model
        # emits only the patch's centre (HoVer-Net: 164 of 256 model px), so the
        # write origin has to skip the halo.  Zero when halo_size_pixels is 0.
        self.write_offset = int(round(int(halo_size_pixels) * model_mpp / slide_mpp))
        self.w_map = SparseCanvas(
            slide_height, slide_width, n_channels=0, dtype=np.float16
        )
        self._window_np: Optional[np.ndarray] = None
        self._window_pt: Optional[torch.Tensor] = None

    def _window(self, size: int):
        """Separable ramp window; adjacent patches' ramps sum to exactly 1."""
        if self._window_np is None or self._window_np.shape[0] != size:
            w = np.ones(size, dtype=np.float32)
            f = min(self.feather, size // 2)
            if f > 0:
                ramp = (np.arange(f, dtype=np.float32) + 0.5) / f
                w[:f] = ramp
                w[size - f :] = ramp[::-1]
            win = np.outer(w, w)
            self._window_np = win.astype(np.float16)
            self._window_pt = torch.from_numpy(win).to(self.device)
        return self._window_np, self._window_pt

    # --------- hot path: batch GPU → single CPU write ---------
    @torch.no_grad()
    def accumulate_batch_torch(self, pred_dict: dict, batch_coords: torch.Tensor):
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

        np_prob = torch.softmax(np_logits, dim=1)[:, 1:2, ...]
        tp_prob = torch.softmax(tp_logits, dim=1)

        np_res = F.interpolate(
            np_prob,
            size=(slide_patch_size, slide_patch_size),
            mode="bilinear",
            align_corners=False,
        )
        hv_res = (
            F.interpolate(
                hv,
                size=(slide_patch_size, slide_patch_size),
                mode="bilinear",
                align_corners=False,
            )
            * alpha
        )
        tp_res = F.interpolate(
            tp_prob,
            size=(slide_patch_size, slide_patch_size),
            mode="bilinear",
            align_corners=False,
        )

        tp_res = tp_res / (tp_res.sum(dim=1, keepdim=True) + 1e-8)

        # Window applied after the per-pixel renormalisation so the stored
        # values stay a weighted average of proper probability vectors.
        win_np, win_pt = self._window(slide_patch_size)
        np_res = np_res * win_pt
        hv_res = hv_res * win_pt
        tp_res = tp_res * win_pt

        np_res_np = np_res.squeeze(1).contiguous().cpu().numpy()
        hv_res_np = hv_res.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        tp_res_np = tp_res.permute(0, 2, 3, 1).contiguous().cpu().numpy()

        coords = batch_coords.detach().to("cpu").numpy().astype(np.int32)[:, :2]
        write_offset = self.write_offset

        for i in range(batch_size):
            x0 = int(coords[i, 0]) + write_offset
            y0 = int(coords[i, 1]) + write_offset
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

            self.np_map.accumulate(cy0, cy1, cx0, cx1, np_res_np[i, ty0:ty1, tx0:tx1])
            self.hv_map.accumulate(
                cy0, cy1, cx0, cx1, hv_res_np[i, ty0:ty1, tx0:tx1, :]
            )
            self.tp_map.accumulate(
                cy0, cy1, cx0, cx1, tp_res_np[i, ty0:ty1, tx0:tx1, :]
            )
            self.w_map.accumulate(cy0, cy1, cx0, cx1, win_np[ty0:ty1, tx0:tx1])

    # ------------------------------ finalize ------------------------------
    def finalize(
        self,
        pbar: Optional[Tqdm] = None,
        num_workers: Optional[int] = None,
        tiles_per_task: int = 4,
        timing: Optional[bool] = None,
    ):
        if self.slide_height <= 0 or self.slide_width <= 0:
            return [], [], []

        geom = self.geometry
        jobs = geom.jobs()
        if not jobs:
            return [], [], []

        if timing is None:
            timing = _verbose()
        timer = StageTimer(enabled=bool(timing))
        stats = EmitStats()

        if pbar is not None and getattr(pbar, "total", None) is None:
            try:
                pbar.reset(total=len(jobs))
            except Exception:
                pass

        q: Queue = Queue()
        for j in jobs:
            q.put(j)

        inst_all: List[np.ndarray] = []
        prob_all: List[np.ndarray] = []
        poly_all: List[np.ndarray] = []
        merge_lock = Lock()
        pbar_lock = Lock()

        np_map = self.np_map
        hv_map = self.hv_map
        tp_map = self.tp_map
        w_map = self.w_map
        min_object_size = self.min_object_size

        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 1) - 2)
        tiles_per_task = max(1, int(tiles_per_task))

        for _ in range(num_workers):
            q.put(None)

        t_start = time.perf_counter()

        def worker():
            local_timer = timer.child()
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
                        q.put(None)
                        break
                    batched_jobs.append(nxt)

                for jb in batched_jobs:
                    with local_timer.stage("canvas_read"):
                        np_tile = np_map.read(
                            jb.tile_y0,
                            jb.tile_y1,
                            jb.tile_x0,
                            jb.tile_x1,
                            out_dtype=np.float32,
                        )
                        hv_tile = hv_map.read(
                            jb.tile_y0,
                            jb.tile_y1,
                            jb.tile_x0,
                            jb.tile_x1,
                            out_dtype=np.float32,
                        )
                        tp_tile = tp_map.read(
                            jb.tile_y0,
                            jb.tile_y1,
                            jb.tile_x0,
                            jb.tile_x1,
                            out_dtype=np.float32,
                        )
                        # Un-weight the feather blend.  Uncovered pixels have
                        # weight 0 and numerator 0, so the clamp leaves them 0.
                        wt = w_map.read(
                            jb.tile_y0,
                            jb.tile_y1,
                            jb.tile_x0,
                            jb.tile_x1,
                            out_dtype=np.float32,
                        )
                        np.maximum(wt, 1e-4, out=wt)
                        np_tile /= wt
                        hv_tile /= wt[:, :, None]
                        tp_tile /= wt[:, :, None]

                    ins, prb, ply = emit_instances(
                        np_tile,
                        hv_tile,
                        tp_tile,
                        jb,
                        geom,
                        min_object_size,
                        stats,
                        local_timer,
                    )

                    if ins:
                        local_inst.extend(ins)
                        local_prob.extend(prb)
                        local_poly.extend(ply)

                    if pbar is not None:
                        with pbar_lock:
                            pbar.update(1)

                    q.task_done()

            timer.absorb(local_timer)
            if local_inst:
                with merge_lock:
                    inst_all.extend(local_inst)
                    prob_all.extend(local_prob)
                    poly_all.extend(local_poly)

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futs = [ex.submit(worker) for _ in range(num_workers)]
            for f in futs:
                f.result()
        wall = time.perf_counter() - t_start

        if self.report:
            if _verbose():
                print(geom.describe(), file=sys.stderr)
                print(
                    f"  feather={self.feather} px, "
                    f"write_offset={self.write_offset} px",
                    file=sys.stderr,
                )
                print(stats.report(geom.bucket_padding), file=sys.stderr)
                timing_report = timer.report(wall=wall, n_workers=num_workers)
                if timing_report:
                    print(timing_report, file=sys.stderr)
            elif stats.n_touch_tile_edge:
                # Only surfaces when the model's own margin was too small for
                # some nucleus, which is the one case a user must know about.
                print(
                    f"[tilefuse2] {stats.n_touch_tile_edge} of "
                    f"{stats.n_emitted} cells exceeded the bucket padding "
                    f"(M={geom.bucket_padding} px, largest cell radius "
                    f"{stats.max_radius} px) and may be truncated; set "
                    f"WSINSIGHT_TILEFUSE2_VERBOSE=1 for details.",
                    file=sys.stderr,
                )

        return inst_all, prob_all, poly_all
