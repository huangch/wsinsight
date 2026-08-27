"""Bucket-ownership geometry and emission contract for ``tilefuse2``."""

import numpy as np
import pytest

from wsinsight.modellib.tilefuse2 import BucketGeometry
from wsinsight.modellib.tilefuse2 import EmitStats
from wsinsight.modellib.tilefuse2 import StageTimer
from wsinsight.modellib.tilefuse2.emit import emit_instances
from wsinsight.modellib.tilefuse2.geometry import BucketJob

SLIDE_MPP = 0.2738

CELLVIT = dict(patch_size_pixels=1024, halo_size_pixels=0, overlap_size_pixels=64)
HOVERNET = dict(patch_size_pixels=256, halo_size_pixels=46, overlap_size_pixels=0)


def _geom(cfg, height=4000, width=3000):
    return BucketGeometry.from_model_config(
        model_mpp=0.25,
        slide_mpp=SLIDE_MPP,
        slide_height=height,
        slide_width=width,
        **cfg,
    )


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cfg,model_pad,model_bucket,exp_m,exp_b",
    [
        (CELLVIT, 32, 960, 29, 877),
        (HOVERNET, 46, 164, 42, 150),
    ],
)
def test_geometry_matches_model_config(cfg, model_pad, model_bucket, exp_m, exp_b):
    g = _geom(cfg)
    assert g.model_bucket_padding == model_pad
    assert g.model_bucket_size == model_bucket
    assert g.bucket_padding == exp_m
    assert g.bucket_size == exp_b


@pytest.mark.parametrize("cfg", [CELLVIT, HOVERNET])
def test_tile_identity_holds_after_rounding(cfg):
    g = _geom(cfg)
    assert g.tile_size == g.bucket_size + 2 * g.bucket_padding


def test_hovernet_bucket_equals_model_output_size():
    # 256 - 2*46 = 164 is exactly what the network emits for a 256 px input.
    assert _geom(HOVERNET).model_bucket_size == 256 - 2 * 46


def test_zero_bucket_size_is_rejected():
    with pytest.raises(ValueError):
        BucketGeometry.from_model_config(
            patch_size_pixels=64,
            halo_size_pixels=32,
            overlap_size_pixels=0,
            model_mpp=0.25,
            slide_mpp=SLIDE_MPP,
            slide_height=100,
            slide_width=100,
        )


# --------------------------------------------------------------------------
# Bucket grid is a partition
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", [CELLVIT, HOVERNET])
def test_buckets_partition_the_slide(cfg):
    g = _geom(cfg, height=2600, width=1900)
    cover = np.zeros((g.slide_height, g.slide_width), dtype=np.int32)
    for j in g.jobs():
        cover[j.bucket_y0 : j.bucket_y1, j.bucket_x0 : j.bucket_x1] += 1
    assert cover.min() == 1, "gap between buckets"
    assert cover.max() == 1, "buckets overlap"


@pytest.mark.parametrize("cfg", [CELLVIT, HOVERNET])
def test_tile_contains_its_bucket_with_full_margin(cfg):
    g = _geom(cfg, height=2600, width=1900)
    m = g.bucket_padding
    for j in g.jobs():
        assert j.tile_y0 <= j.bucket_y0 and j.tile_y1 >= j.bucket_y1
        assert j.tile_x0 <= j.bucket_x0 and j.tile_x1 >= j.bucket_x1
        # Margin is the full M unless clipped by the slide border.
        assert j.bucket_y0 - j.tile_y0 == m or j.tile_y0 == 0
        assert j.tile_y1 - j.bucket_y1 == m or j.tile_y1 == g.slide_height
        assert j.bucket_x0 - j.tile_x0 == m or j.tile_x0 == 0
        assert j.tile_x1 - j.bucket_x1 == m or j.tile_x1 == g.slide_width


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------
def _disc_canvas(height, width, centres, radius, n_classes=3):
    """np/hv/tp tiles whose ``proc_np_hv`` yields one instance per centre.

    ``hv`` is flat, so the ridge covers the whole foreground, the marker pass
    falls back to ``ndi.label(blb)`` and each disc becomes exactly one label.
    """
    yy, xx = np.ogrid[:height, :width]
    np_map = np.zeros((height, width), dtype=np.float32)
    for cy, cx in centres:
        np_map[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2] = 1.0
    hv_map = np.zeros((height, width, 2), dtype=np.float32)
    tp_map = np.zeros((height, width, n_classes), dtype=np.float16)
    tp_map[..., 1] = 1.0
    return np_map, hv_map, tp_map


def _tiny_geometry(bucket_size=40, padding=10, height=120, width=120):
    return BucketGeometry(
        bucket_size=bucket_size,
        bucket_padding=padding,
        slide_height=height,
        slide_width=width,
        model_tile_size=bucket_size + 2 * padding,
        model_bucket_size=bucket_size,
        model_bucket_padding=padding,
    )


def _run(geom, np_map, hv_map, tp_map, min_object_size=20):
    stats = EmitStats()
    timer = StageTimer(enabled=False)
    inst, prob, poly = [], [], []
    for j in geom.jobs():
        i, p, g = emit_instances(
            np_map[j.tile_y0 : j.tile_y1, j.tile_x0 : j.tile_x1],
            hv_map[j.tile_y0 : j.tile_y1, j.tile_x0 : j.tile_x1],
            tp_map[j.tile_y0 : j.tile_y1, j.tile_x0 : j.tile_x1],
            j,
            geom,
            min_object_size,
            stats,
            timer,
        )
        inst.extend(i)
        prob.extend(p)
        poly.extend(g)
    return inst, prob, poly, stats


def test_cell_on_a_bucket_boundary_is_emitted_once_and_whole():
    geom = _tiny_geometry()
    radius = 8
    # Centre sits exactly on the boundary between bucket 0 and bucket 1.
    np_map, hv_map, tp_map = _disc_canvas(120, 120, [(60, 40)], radius)

    inst, prob, poly, stats = _run(geom, np_map, hv_map, tp_map)

    assert len(inst) == 1, "boundary cell duplicated or lost"
    x, y, w, h = inst[0][0]
    assert w == 2 * radius + 1 and h == 2 * radius + 1, "cell was truncated"
    assert x == 40 - radius and y == 60 - radius
    assert len(prob) == len(poly) == 1
    assert stats.n_touch_tile_edge == 0


def test_every_cell_emitted_exactly_once_across_the_grid():
    geom = _tiny_geometry()
    centres = [(20, 20), (60, 40), (40, 80), (80, 80), (100, 39), (39, 100)]
    np_map, hv_map, tp_map = _disc_canvas(120, 120, centres, 8)

    inst, _, _, stats = _run(geom, np_map, hv_map, tp_map)

    assert len(inst) == len(centres)
    got = sorted((int(b[0][1] + 8), int(b[0][0] + 8)) for b in inst)
    assert got == sorted(centres)
    assert stats.n_emitted == len(centres)


def test_oversized_cell_is_reported_as_touching_the_tile_edge():
    # radius 16 > padding 10: no tile can see this cell whole.
    geom = _tiny_geometry()
    np_map, hv_map, tp_map = _disc_canvas(120, 120, [(60, 40)], 16)

    _, _, _, stats = _run(geom, np_map, hv_map, tp_map)

    assert stats.n_touch_tile_edge > 0
    assert stats.max_radius >= geom.bucket_padding


def test_emission_is_independent_of_bucket_visit_order():
    geom = _tiny_geometry()
    centres = [(20, 20), (60, 40), (40, 80)]
    np_map, hv_map, tp_map = _disc_canvas(120, 120, centres, 8)

    forward, _, _, _ = _run(geom, np_map, hv_map, tp_map)
    reverse_jobs = list(reversed(geom.jobs()))
    stats = EmitStats()
    timer = StageTimer(enabled=False)
    backward = []
    for j in reverse_jobs:
        i, _, _ = emit_instances(
            np_map[j.tile_y0 : j.tile_y1, j.tile_x0 : j.tile_x1],
            hv_map[j.tile_y0 : j.tile_y1, j.tile_x0 : j.tile_x1],
            tp_map[j.tile_y0 : j.tile_y1, j.tile_x0 : j.tile_x1],
            j,
            geom,
            20,
            stats,
            timer,
        )
        backward.extend(i)

    assert sorted(tuple(b[0]) for b in forward) == sorted(tuple(b[0]) for b in backward)


def test_polygon_coordinates_are_global():
    geom = _tiny_geometry()
    np_map, hv_map, tp_map = _disc_canvas(120, 120, [(80, 80)], 8)

    inst, _, poly, _ = _run(geom, np_map, hv_map, tp_map)

    x, y, w, h = inst[0][0]
    pts = poly[0]
    assert pts.ndim == 2 and pts.shape[1] == 2
    assert pts[:, 0].min() >= x and pts[:, 0].max() <= x + w
    assert pts[:, 1].min() >= y and pts[:, 1].max() <= y + h


def test_bucket_ownership_uses_half_open_intervals():
    job_lo = BucketJob(0, 40, 0, 40, 0, 50, 0, 50)
    job_hi = BucketJob(40, 80, 0, 40, 30, 90, 0, 50)
    # A centroid landing on 40 belongs to the upper bucket, never both.
    assert job_lo.bucket_y0 <= 39 < job_lo.bucket_y1
    assert not (job_lo.bucket_y0 <= 40 < job_lo.bucket_y1)
    assert job_hi.bucket_y0 <= 40 < job_hi.bucket_y1


# --------------------------------------------------------------------------
# Feather blending
# --------------------------------------------------------------------------
def _stitcher(cfg, slide_patch_size, height=4000, width=4000):
    from wsinsight.modellib.tilefuse2 import TileRemapStitcherV2

    return TileRemapStitcherV2(
        n_classes=3,
        slide_width=width,
        slide_height=height,
        slide_patch_size=slide_patch_size,
        slide_mpp=SLIDE_MPP,
        model_mpp=0.25,
        min_object_size=20,
        device="cpu",
        report=False,
        **cfg,
    )


def test_feather_width_is_patch_minus_bucket():
    assert _stitcher(CELLVIT, 935).feather == 935 - 877
    # HoVer-Net's halo is already cropped by the network: writes abut, no blend.
    assert _stitcher(HOVERNET, 150).feather == 0


def test_write_offset_skips_the_halo():
    """patchlib coords are the whole patch's corner, not the model output's.

    HoVer-Net emits only the central 164 of 256 model px, so writing at the
    patch corner shifts the entire canvas by the halo.  Measured on a real run
    before the fix: 16 um diagonal = 42 px on each axis.
    """
    assert _stitcher(CELLVIT, 935).write_offset == 0
    assert _stitcher(HOVERNET, 150).write_offset == 42


def test_write_offset_places_the_output_at_the_patch_centre():
    import torch

    st = _stitcher(HOVERNET, 150, height=1000, width=1000)
    # Independent cross-check: skipping the halo must land the output exactly
    # centred inside the patch footprint.
    footprint = round(256 * 0.25 / SLIDE_MPP)
    assert 2 * st.write_offset + 150 == footprint, "output not centred in patch"

    x0 = y0 = 300
    n, k = 1, 3
    st.accumulate_batch_torch(
        {
            "np": torch.stack(
                [torch.zeros(164, 164), torch.full((164, 164), 5.0)]
            ).unsqueeze(0),
            "hv": torch.zeros(n, 2, 164, 164),
            "tp": torch.zeros(n, k, 164, 164),
        },
        torch.tensor([[x0, y0, 150, 150]], dtype=torch.int32),
    )

    w = st.w_map.read(0, 1000, 0, 1000, out_dtype=np.float32)
    ys, xs = np.nonzero(w)
    assert xs.min() == x0 + 42 and ys.min() == y0 + 42
    assert xs.max() == x0 + 42 + 149 and ys.max() == y0 + 42 + 149


def test_feather_window_is_a_partition_of_unity():
    st = _stitcher(CELLVIT, 935)
    win, _ = st._window(935)
    b = st.geometry.bucket_size

    n = 4
    span = (n - 1) * b + 935
    acc = np.zeros((span, span), dtype=np.float64)
    for ky in range(n):
        for kx in range(n):
            acc[ky * b : ky * b + 935, kx * b : kx * b + 935] += win

    # Interior = every pixel that has neighbours on all four sides.
    interior = acc[935 : (n - 1) * b, 935 : (n - 1) * b]
    assert interior.size > 0
    assert np.allclose(
        interior, 1.0, atol=2e-3
    ), f"min={interior.min()} max={interior.max()}"


def test_patchlib_step_equals_the_tilefuse2_bucket():
    """Cross-module contract with ``patchlib.pipeline`` (the ``overlap`` line).

    The blend is only a partition of unity when the patch grid advances by
    exactly one bucket; if these two drift apart the feather ramps stop being
    complementary and seams reappear.
    """
    for cfg in (CELLVIT, HOVERNET):
        ratio = 0.25 / SLIDE_MPP
        patch_size = round(cfg["patch_size_pixels"] * ratio)
        overlap = (2 * cfg["halo_size_pixels"] + cfg["overlap_size_pixels"]) / cfg[
            "patch_size_pixels"
        ]
        patchlib_step = round((1 - overlap) * patch_size)
        assert patchlib_step == _geom(cfg).bucket_size


def test_canvas_accumulate_sums_overlapping_writes():
    from wsinsight.modellib.tilefuse2.canvas import SparseCanvas

    c = SparseCanvas(10, 10, n_channels=0, dtype=np.float32)
    c.accumulate(0, 5, 0, 5, np.full((5, 5), 0.25, dtype=np.float32))
    c.accumulate(3, 8, 3, 8, np.full((5, 5), 0.75, dtype=np.float32))

    out = c.read(0, 10, 0, 10, out_dtype=np.float32)
    assert out[0, 0] == pytest.approx(0.25)
    assert out[7, 7] == pytest.approx(0.75)
    assert out[4, 4] == pytest.approx(1.0), "overlap must sum, not overwrite"
    assert out[9, 9] == 0.0


def test_canvas_chunks_start_zeroed():
    from wsinsight.modellib.tilefuse2.canvas import SparseCanvas

    c = SparseCanvas(64, 64, n_channels=2, dtype=np.float32)
    c.accumulate(10, 12, 10, 12, np.ones((2, 2, 2), dtype=np.float32))
    out = c.read(0, 64, 0, 64, out_dtype=np.float32)
    assert out.sum() == pytest.approx(8.0), "uninitialised memory leaked in"
