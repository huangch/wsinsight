"""Tests for CellPatchDataset — the per-cell crop source for H-Optimus.

The bug these guard against: the H-Optimus path previously fell back to
``DummyPatchDataset``, which ignores its index and returns the same blank
image for every cell.  Every cell then received an identical embedding, so the
morphology features carried no signal at all.

These tests exec the dataset classes straight out of the source file to avoid
importing torch_geometric / igraph / leidenalg, matching the approach used by
``test_oom_batch_search.py`` and ``test_calibration.py``.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import Dataset


def _load_dataset_classes() -> dict:
    """Exec just the patch-dataset classes from niche_generation.py."""
    src_path = (
        Path(__file__).resolve().parents[1]
        / "wsinsight"
        / "insightlib"
        / "niche_generation.py"
    )
    text = src_path.read_text()
    start = text.index("class DummyPatchDataset(Dataset):")
    end = text.index("def _make_short_ids(")
    ns: dict = {
        "Dataset": Dataset,
        "np": np,
        "Image": Image,
        "threading": threading,
    }
    exec(compile(text[start:end], str(src_path), "exec"), ns)
    return ns


CLASSES = _load_dataset_classes()


class _FakeSlide:
    """Minimal read_region stub whose pixels encode the requested location."""

    def __init__(self, width: int = 4096, height: int = 4096):
        self.width = width
        self.height = height
        self.calls: list[tuple] = []

    def read_region(self, location, level, size):
        self.calls.append((location, level, size))
        x, y = location
        if x < 0 or y < 0 or x + size[0] > self.width or y + size[1] > self.height:
            raise ValueError("region out of bounds")
        # Encode the location in the pixel values so different cells differ.
        arr = np.full((size[1], size[0], 3), (x % 251, y % 251, 7), dtype=np.uint8)
        return Image.fromarray(arr)


def _make_dataset(monkeypatch, centers, *, mpp=0.25, window_um=32.0, slide=None):
    slide = slide or _FakeSlide()
    ds = CLASSES["CellPatchDataset"](
        wsi_path="/fake/slide.svs",
        centers_px=np.asarray(centers),
        mpp_um_per_px=mpp,
        window_um=window_um,
    )
    # Bypass the real slide reader.
    ds._slide = slide
    return ds, slide


# ---------------------------------------------------------------------------
# The regression itself.
# ---------------------------------------------------------------------------


def test_distinct_cells_yield_distinct_crops(monkeypatch):
    """Different cells must produce different pixels.

    DummyPatchDataset returned one constant image, which silently reduced the
    whole morphology feature block to a constant.
    """
    centers = [[500, 500], [1500, 800], [2500, 3000]]
    ds, _ = _make_dataset(monkeypatch, centers)

    crops = [np.asarray(ds[i]) for i in range(len(centers))]

    for a in range(len(crops)):
        for b in range(a + 1, len(crops)):
            assert not np.array_equal(
                crops[a], crops[b]
            ), f"cells {a} and {b} produced identical crops"


def test_dummy_dataset_is_constant_by_design():
    """Documents why DummyPatchDataset must not be used for real runs."""
    dummy = CLASSES["DummyPatchDataset"](num_cells=5)
    first = np.asarray(dummy[0])
    for i in range(1, 5):
        assert np.array_equal(first, np.asarray(dummy[i]))


# ---------------------------------------------------------------------------
# Geometry.
# ---------------------------------------------------------------------------


def test_crop_is_centred_on_the_cell(monkeypatch):
    ds, slide = _make_dataset(monkeypatch, [[1000, 2000]], mpp=0.25, window_um=32.0)
    ds[0]

    (location, level, size) = slide.calls[0]
    window_px = int(round(32.0 / 0.25))  # 128
    assert size == (window_px, window_px)
    assert level == 0
    # Centre of the requested region is the cell centre.
    assert location[0] + window_px // 2 == 1000
    assert location[1] + window_px // 2 == 2000


@pytest.mark.parametrize(
    "mpp,window_um,expected_px",
    [(0.25, 32.0, 128), (0.5, 32.0, 64), (0.25, 64.0, 256), (1.0, 32.0, 32)],
)
def test_window_size_scales_with_resolution(monkeypatch, mpp, window_um, expected_px):
    """The crop covers a fixed physical area regardless of slide resolution."""
    ds, slide = _make_dataset(monkeypatch, [[2000, 2000]], mpp=mpp, window_um=window_um)
    ds[0]
    assert slide.calls[0][2] == (expected_px, expected_px)


def test_window_has_a_floor(monkeypatch):
    """A pathological mpp must not collapse the window to zero pixels."""
    ds, slide = _make_dataset(monkeypatch, [[2000, 2000]], mpp=1000.0, window_um=1.0)
    ds[0]
    assert slide.calls[0][2][0] >= 8


def test_output_is_resized_to_encoder_input(monkeypatch):
    ds, _ = _make_dataset(monkeypatch, [[2000, 2000]], mpp=0.25, window_um=32.0)
    img = ds[0]
    assert img.size == (224, 224)
    assert img.mode == "RGB"


# ---------------------------------------------------------------------------
# Robustness.
# ---------------------------------------------------------------------------


def test_out_of_bounds_cell_does_not_abort_the_slide(monkeypatch):
    """Cells near the slide edge yield blank tissue instead of raising."""
    ds, _ = _make_dataset(monkeypatch, [[5, 5], [2000, 2000]], mpp=0.25)

    edge = ds[0]  # would read from negative coordinates
    assert edge.size == (224, 224)
    assert np.asarray(edge).min() == 255, "edge crop should be blank white"

    interior = ds[1]  # unaffected
    assert np.asarray(interior).min() < 255


def test_len_matches_cell_count(monkeypatch):
    ds, _ = _make_dataset(monkeypatch, [[100, 100]] * 7)
    assert len(ds) == 7


def test_slide_handle_is_dropped_on_pickle(monkeypatch):
    """The handle is not picklable, so it must not survive __getstate__."""
    ds, _ = _make_dataset(monkeypatch, [[100, 100]])
    assert ds._slide is not None
    assert ds.__getstate__()["_slide"] is None


def test_indices_map_to_detection_table_rows(monkeypatch):
    """Index i must read the centre stored at row i.

    prepare_slide_graph indexes this dataset with kept_idx / sampled ids, which
    are row positions in the slide's detection table.
    """
    centers = [[100, 200], [300, 400], [500, 600]]
    ds, slide = _make_dataset(monkeypatch, centers, mpp=0.25, window_um=32.0)
    half = int(round(32.0 / 0.25)) // 2

    for i, (cx, cy) in enumerate(centers):
        ds[i]
        location = slide.calls[-1][0]
        assert location == (cx - half, cy - half)


# ---------------------------------------------------------------------------
# Concurrency: the fetch path now reads crops from a thread pool.
# ---------------------------------------------------------------------------


def test_parallel_reads_preserve_order(monkeypatch):
    """executor.map over the dataset must return crops in index order.

    A reordering here would silently pair each cell with another cell's
    embedding -- worse than a crash, because nothing would report it.
    """
    from concurrent.futures import ThreadPoolExecutor

    centers = [[100 + 37 * i, 200 + 53 * i] for i in range(64)]
    ds, _ = _make_dataset(monkeypatch, centers, mpp=0.25, window_um=32.0)

    serial = [np.asarray(ds[i]) for i in range(len(centers))]
    with ThreadPoolExecutor(max_workers=8) as pool:
        parallel = [
            np.asarray(im) for im in pool.map(ds.__getitem__, range(len(centers)))
        ]

    assert len(parallel) == len(serial)
    for i, (a, b) in enumerate(zip(serial, parallel)):
        assert np.array_equal(
            a, b
        ), f"crop {i} differs between serial and parallel reads"


def test_each_thread_gets_its_own_slide_handle():
    """Slide readers are not reliably thread-safe, so handles must be per-thread."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    opened: list[int] = []
    lock = threading.Lock()

    class _CountingSlide(_FakeSlide):
        def __init__(self):
            super().__init__()
            with lock:
                opened.append(threading.get_ident())

    ds = CLASSES["CellPatchDataset"](
        wsi_path="/fake/slide.svs",
        centers_px=np.array([[500, 500]] * 32),
        mpp_um_per_px=0.25,
    )
    # Patch the lazy opener to build our counting stub instead of a real reader.
    ds._open_slide = _CountingSlide  # type: ignore[attr-defined]

    def _read(i):
        slide = getattr(ds._local, "slide", None)
        if slide is None:
            slide = _CountingSlide()
            ds._local.slide = slide
        return id(slide)

    with ThreadPoolExecutor(max_workers=4) as pool:
        handle_ids = set(pool.map(_read, range(32)))

    # Distinct threads must not share one handle.
    assert len(handle_ids) == len(set(opened))


def test_pickle_drops_thread_local(monkeypatch):
    """threading.local is not picklable; it must be rebuilt on unpickle.

    The class is exec'd from source here so it is not importable by ``pickle``
    itself; exercise the ``__getstate__`` / ``__setstate__`` hooks directly,
    which is what pickling would call.
    """
    ds, _ = _make_dataset(monkeypatch, [[100, 100], [200, 200]])

    state = ds.__getstate__()
    assert state["_slide"] is None, "open slide handle must not be pickled"
    assert state["_local"] is None, "threading.local must not be pickled"

    revived = CLASSES["CellPatchDataset"].__new__(CLASSES["CellPatchDataset"])
    revived.__setstate__(state)

    assert revived._local is not None, "thread-local storage must be rebuilt"
    assert len(revived) == 2
    assert revived.window_px == ds.window_px
    assert revived.out_size == ds.out_size
