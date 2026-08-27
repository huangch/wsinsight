"""Instance extraction with bucket ownership.

The v1 worker cropped the label map to its interior *before* extracting
instances, so a cell straddling an interior edge was physically truncated and a
downstream heuristic had to pair the fragment with a whole copy from the
neighbouring tile.  Here the label map is never cropped: watershed runs on the
whole read window and each instance is claimed by the single tile whose bucket
contains its centroid.  Buckets partition the slide, so that is a bijection —
no duplicates, no misses, no dedup pass.
"""

from __future__ import annotations

from typing import List
from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi

from .diagnostics import EmitStats
from .diagnostics import StageTimer
from .geometry import BucketGeometry
from .geometry import BucketJob
from .segment import proc_np_hv


def emit_instances(
    np_tile: np.ndarray,
    hv_tile: np.ndarray,
    tp_tile: np.ndarray,
    job: BucketJob,
    geom: BucketGeometry,
    min_object_size: int,
    stats: EmitStats,
    timer: StageTimer,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Return index-aligned ``(inst, prob, poly)`` for the cells this bucket owns."""
    with timer.stage("proc_np_hv"):
        labels = proc_np_hv(np_tile, hv_tile, min_object_size)

    max_id = int(labels.max())
    if max_id <= 0:
        return [], [], []

    counts = np.bincount(labels.ravel(), minlength=max_id + 1).astype(np.int32)
    counts[0] = 0
    valid_ids = np.nonzero(counts)[0]
    if valid_ids.size == 0:
        return [], [], []

    slices = ndi.find_objects(labels, max_label=max_id)

    tile_y0 = job.tile_y0
    tile_x0 = job.tile_x0
    tile_h = job.tile_height
    tile_w = job.tile_width
    # A read-window edge that coincides with the slide border cannot be widened,
    # so a cell reaching it is not evidence that M is too small.
    edge_top = job.tile_y0 > 0
    edge_left = job.tile_x0 > 0
    edge_bottom = job.tile_y1 < geom.slide_height
    edge_right = job.tile_x1 < geom.slide_width

    inst_list: List[np.ndarray] = []
    prob_list: List[np.ndarray] = []
    poly_list: List[np.ndarray] = []
    n_discarded = 0
    n_touch_edge = 0
    max_radius = 0

    for inst_id in valid_ids.tolist():
        sl = slices[inst_id - 1]
        if sl is None:
            continue
        r_sl, c_sl = sl
        rmin, rmax = r_sl.start, r_sl.stop
        cmin, cmax = c_sl.start, c_sl.stop

        # Ownership: integer bbox centre, half-open bucket interval.
        gy = tile_y0 + (rmin + rmax) // 2
        gx = tile_x0 + (cmin + cmax) // 2
        if not (
            job.bucket_y0 <= gy < job.bucket_y1 and job.bucket_x0 <= gx < job.bucket_x1
        ):
            n_discarded += 1
            continue

        x = cmin + tile_x0
        y = rmin + tile_y0
        w = cmax - cmin
        h = rmax - rmin

        if (
            (rmin == 0 and edge_top)
            or (cmin == 0 and edge_left)
            or (rmax == tile_h and edge_bottom)
            or (cmax == tile_w and edge_right)
        ):
            n_touch_edge += 1
        radius = max(w, h) // 2
        if radius > max_radius:
            max_radius = radius

        local = labels[rmin:rmax, cmin:cmax] == inst_id

        with timer.stage("class_prob"):
            prob = tp_tile[rmin:rmax, cmin:cmax, :][local].mean(axis=0)

        with timer.stage("find_contours"):
            cnts, _ = cv2.findContours(
                local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not cnts:
                poly = np.array(
                    [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32
                )
            else:
                cnt = max(cnts, key=cv2.contourArea)
                cnt2d = cnt.squeeze(1).astype(np.int32)
                if cnt2d.ndim != 2 or cnt2d.shape[0] < 3:
                    poly = np.array(
                        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                        dtype=np.int32,
                    )
                else:
                    cnt2d[:, 0] += x
                    cnt2d[:, 1] += y
                    poly = cnt2d

        # Appended together so the three lists stay index-aligned downstream.
        inst_list.append(np.array([x, y, w, h], dtype=np.int32).reshape(1, -1))
        prob_list.append(prob.reshape(1, -1))
        poly_list.append(poly)

    stats.merge(
        emitted=len(inst_list),
        discarded=n_discarded,
        touch_edge=n_touch_edge,
        max_radius=max_radius,
    )
    return inst_list, prob_list, poly_list
