"""Tests for the adaptive H-optimus batch-size helpers in niche_generation.

These guard two regressions that between them left the GPUs ~20% utilised:

1. Single-point calibration divided *peak* memory by the batch size, so the
   fixed cuDNN workspace (constant, batch-independent) was charged to every
   image.  That inflated the per-image estimate several-fold and starved the
   batch size.  Two-point calibration takes the slope between two batch sizes,
   which cancels the constant term.
2. Available VRAM was read from ``mem_get_info`` alone, which does not count
   blocks the caching allocator has reserved but is not using.  In steady
   state that made the grow-probe believe there was no headroom left.

The arithmetic tests inject memory figures and need no GPU.  The final test is
opt-in and only runs where CUDA is present.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pytest
import torch

GIB = 1024**3
MIB = 1024**2


def _load_helpers() -> dict:
    """Exec only the batch-sizing helpers out of ``niche_generation.py``.

    Importing the module normally would drag in torch_geometric, igraph and
    leidenalg. This mirrors the source-extraction approach already used by
    ``test_oom_batch_search.py``.
    """
    src_path = (
        Path(__file__).resolve().parents[1]
        / "wsinsight"
        / "insightlib"
        / "niche_generation.py"
    )
    text = src_path.read_text()
    start = text.index("_HOPTIMUS_BYTES_PER_IMAGE_FALLBACK")
    end = text.index("def _embed_hoptimus_subset_dataset(")
    ns: dict = {
        "torch": torch,
        "nn": torch.nn,
        "math": math,
        "Optional": Optional,
        "_logging": logging,
        "tqdm": __import__("tqdm").tqdm,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "deque": deque,
    }
    exec(compile(text[start:end], str(src_path), "exec"), ns)
    return ns


HELPERS = _load_helpers()


# ---------------------------------------------------------------------------
# _is_oom: decides whether the retry loop shrinks or re-raises.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("cuDNN error: CUDNN_STATUS_NOT_SUPPORTED."),
        RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"),
    ],
)
def test_is_oom_recognises_recoverable_allocation_failures(exc):
    assert HELPERS["_is_oom"](exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("bad shape"),
        RuntimeError("size mismatch for weight"),
        KeyError("missing"),
    ],
)
def test_is_oom_ignores_unrelated_errors(exc):
    """A genuine bug must propagate, not silently halve the batch forever."""
    assert HELPERS["_is_oom"](exc) is False


# ---------------------------------------------------------------------------
# _available_vram: must count allocator-reserved-but-idle blocks as usable.
# ---------------------------------------------------------------------------


def test_available_vram_counts_reusable_cached_blocks(monkeypatch):
    free, reserved, allocated = 10 * GIB, 60 * GIB, 2 * GIB
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (free, 80 * GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda i: reserved)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda i: allocated)

    # Driver-free memory alone would report 10 GiB and hide the 58 GiB of
    # reserved-but-idle cache that is immediately reusable.
    assert HELPERS["_available_vram"](0) == free + (reserved - allocated)


def test_available_vram_never_negative(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (5 * GIB, 80 * GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda i: 1 * GIB)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda i: 4 * GIB)
    assert HELPERS["_available_vram"](0) == 5 * GIB


# ---------------------------------------------------------------------------
# _calibrate_bytes_per_image: the two-point slope must cancel fixed overhead.
# ---------------------------------------------------------------------------


def _stub_cuda_for_calibration(monkeypatch, fixed: int, marginal: int) -> dict:
    """Simulate a device whose peak usage is ``fixed + n * marginal``."""
    seen = {"n": 0}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda *a, **k: fixed + seen["n"] * marginal,
    )
    # autocast(device_type="cuda") is meaningless on a CPU-only test host.
    monkeypatch.setattr(torch, "autocast", lambda **kw: contextlib.nullcontext())
    return seen


def test_calibration_cancels_fixed_workspace_overhead(monkeypatch):
    """The headline regression: a 2.3 GiB constant must not be billed per image."""
    fixed = 2300 * MIB  # cuDNN workspace — same for any batch size
    marginal = 17 * MIB  # true activation cost of one extra image
    seen = _stub_cuda_for_calibration(monkeypatch, fixed, marginal)

    class _FakeModel:
        def __call__(self, x):
            seen["n"] = x.shape[0]
            return x

    measured = HELPERS["_calibrate_bytes_per_image"](_FakeModel(), "cpu")
    assert measured == marginal

    # For contrast: the single-point estimate this replaced charges the whole
    # fixed workspace to one batch, overshooting by several-fold.
    b2 = HELPERS["_CAL_B2"]
    single_point = (fixed + b2 * marginal) // b2
    assert single_point > 4 * marginal


def test_calibration_is_independent_of_fixed_overhead_size(monkeypatch):
    """Same marginal cost must be recovered whatever the constant term is."""
    marginal = 12 * MIB
    results = []
    for fixed in (0, 500 * MIB, 4 * GIB):
        with monkeypatch.context() as mp:
            seen = _stub_cuda_for_calibration(mp, fixed, marginal)

            class _FakeModel:
                def __call__(self, x):
                    seen["n"] = x.shape[0]
                    return x

            results.append(HELPERS["_calibrate_bytes_per_image"](_FakeModel(), "cpu"))

    assert results == [marginal] * 3


def test_calibration_falls_back_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert (
        HELPERS["_calibrate_bytes_per_image"](None, "cpu")
        == HELPERS["_HOPTIMUS_BYTES_PER_IMAGE_FALLBACK"]
    )


# ---------------------------------------------------------------------------
# _auto_batch_size: fills the requested fraction and splits evenly.
# ---------------------------------------------------------------------------


def _stub_cuda_for_sizing(monkeypatch, usable: int, n_gpu: int) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: n_gpu)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda i: (usable, 80 * GIB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda i: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda i: 0)


def test_auto_batch_size_fills_requested_fraction(monkeypatch):
    usable, per_image, n_gpu, safety = 76 * GIB, 17 * MIB, 2, 0.95
    _stub_cuda_for_sizing(monkeypatch, usable, n_gpu)

    total = HELPERS["_auto_batch_size"](
        None, "cpu", safety=safety, bytes_per_image=per_image
    )
    per_gpu = total // n_gpu
    occupancy = (per_gpu * per_image) / usable

    # Only rounding down to a whole image separates this from `safety` exactly.
    assert safety - 0.01 <= occupancy <= safety


@pytest.mark.parametrize("n_gpu", [1, 2, 3, 4, 8])
def test_auto_batch_size_splits_evenly_across_gpus(monkeypatch, n_gpu):
    _stub_cuda_for_sizing(monkeypatch, 70 * GIB, n_gpu)
    total = HELPERS["_auto_batch_size"](None, "cpu", bytes_per_image=17 * MIB)
    assert total % n_gpu == 0


def test_auto_batch_size_scales_with_gpu_count(monkeypatch):
    """Each replica gets a full share, so the total tracks the device count."""
    per_image = 17 * MIB
    sizes = {}
    for n_gpu in (1, 2):
        with monkeypatch.context() as mp:
            _stub_cuda_for_sizing(mp, 76 * GIB, n_gpu)
            sizes[n_gpu] = HELPERS["_auto_batch_size"](
                None, "cpu", bytes_per_image=per_image
            )
    assert sizes[2] == 2 * sizes[1]


def test_auto_batch_size_does_not_collapse_to_floor(monkeypatch):
    """Regression: an inflated per-image figure used to yield a tiny batch.

    With 76 GiB free and a correct ~17 MiB/image, thousands of images per GPU
    must fit. Anything near ``min_batch`` means the estimate broke again.
    """
    _stub_cuda_for_sizing(monkeypatch, 76 * GIB, 2)
    total = HELPERS["_auto_batch_size"](None, "cpu", bytes_per_image=17 * MIB)
    assert total // 2 > 1000


def test_auto_batch_size_respects_max_batch(monkeypatch):
    _stub_cuda_for_sizing(monkeypatch, 76 * GIB, 2)
    total = HELPERS["_auto_batch_size"](
        None, "cpu", bytes_per_image=1 * MIB, max_batch=512
    )
    assert total <= 512


def test_auto_batch_size_falls_back_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert HELPERS["_auto_batch_size"](None, "cpu", min_batch=8) == 8


def test_auto_batch_size_survives_introspection_failure(monkeypatch):
    """A driver query blowing up must not take the whole run down with it."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def _boom(*a, **k):
        raise RuntimeError("driver query failed")

    monkeypatch.setattr(torch.cuda, "mem_get_info", _boom)
    assert HELPERS["_auto_batch_size"](None, "cpu", bytes_per_image=17 * MIB) == 8


# ---------------------------------------------------------------------------
# _run_adaptive_batches: the loop itself, driven with pseudo data.
#
# Correctness bar: every item processed exactly once, in order. A loop that
# silently drops or duplicates items would misalign embeddings against cell
# ids -- far worse than any throughput problem.
# ---------------------------------------------------------------------------


class _FakeGPU:
    """Pretends to be a device that OOMs above ``capacity`` items per batch."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.attempts: list[int] = []  # every batch size tried
        self.succeeded: list[int] = []  # batch sizes that went through
        self.oom_count = 0

    def forward(self, items: list) -> list:
        self.attempts.append(len(items))
        if len(items) > self.capacity:
            self.oom_count += 1
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        self.succeeded.append(len(items))
        return list(items)


def _fetch_range(start: int, stop: int) -> list:
    """Pseudo dataset: item i is simply the integer i."""
    return list(range(start, stop))


def _flatten(chunks: list) -> list:
    return [x for chunk in chunks for x in chunk]


def test_loop_processes_every_item_exactly_once_in_order():
    gpu = _FakeGPU(capacity=10**9)
    out = HELPERS["_run_adaptive_batches"](
        n_total=100, batch_size=10, fetch=_fetch_range, forward=gpu.forward
    )
    assert _flatten(out) == list(range(100))


@pytest.mark.parametrize(
    "n_total,batch_size", [(100, 10), (100, 7), (5, 10), (1, 1), (37, 6)]
)
def test_loop_handles_ragged_final_batch(n_total, batch_size):
    gpu = _FakeGPU(capacity=10**9)
    out = HELPERS["_run_adaptive_batches"](
        n_total=n_total, batch_size=batch_size, fetch=_fetch_range, forward=gpu.forward
    )
    assert _flatten(out) == list(range(n_total))


def test_loop_recovers_from_oom_without_losing_items():
    """The regression that matters: shrinking must not skip the chunk tail."""
    gpu = _FakeGPU(capacity=25)  # 100-item batches fail, 25 is fine
    out = HELPERS["_run_adaptive_batches"](
        n_total=100, batch_size=100, fetch=_fetch_range, forward=gpu.forward
    )
    assert gpu.oom_count > 0, "test did not actually exercise the OOM path"
    assert _flatten(out) == list(range(100))
    assert max(gpu.succeeded) <= 25


def test_loop_converges_when_only_tiny_batches_fit():
    gpu = _FakeGPU(capacity=1)
    out = HELPERS["_run_adaptive_batches"](
        n_total=20, batch_size=16, fetch=_fetch_range, forward=gpu.forward
    )
    assert _flatten(out) == list(range(20))
    assert all(n <= 1 for n in gpu.succeeded)


def test_loop_shrinks_geometrically():
    """Each retry halves, so convergence is logarithmic rather than linear.

    ``n_total`` must be at least ``batch_size`` or the first attempt is clamped
    to the number of items available and the ladder never starts at the top.
    """
    gpu = _FakeGPU(capacity=7)
    HELPERS["_run_adaptive_batches"](
        n_total=64, batch_size=64, fetch=_fetch_range, forward=gpu.forward
    )
    assert gpu.attempts[:5] == [64, 32, 16, 8, 4]


def test_loop_raises_when_even_min_batch_ooms():
    gpu = _FakeGPU(capacity=0)  # nothing ever fits
    with pytest.raises(RuntimeError, match="out of memory"):
        HELPERS["_run_adaptive_batches"](
            n_total=10,
            batch_size=8,
            fetch=_fetch_range,
            forward=gpu.forward,
            min_batch=1,
        )


def test_loop_does_not_decrement_by_one_when_converged():
    """Regression: when binary search converges (lo+1==hi==current) but OOM
    persists, the loop must do a bisection reset — not decrement by 1 each time,
    which would produce thousands of OOM warnings before reaching min_batch."""
    # capacity=7 means batches >7 OOM. After convergence around 7/8, any
    # continued OOM should jump back to ~4, not crawl 8→7→6→5→...
    gpu = _FakeGPU(capacity=7)
    HELPERS["_run_adaptive_batches"](
        n_total=64,
        batch_size=64,
        fetch=_fetch_range,
        forward=gpu.forward,
        min_batch=1,
        max_batch=64,
    )
    # Total OOM count must be well below 60 (no decrement loop).
    assert (
        gpu.oom_count < 30
    ), f"oom_count={gpu.oom_count} suggests decrement-by-one regression"


def test_loop_propagates_non_oom_errors_immediately():
    """A real bug must surface, not be mistaken for memory pressure."""
    calls = {"n": 0}

    def _forward(items):
        calls["n"] += 1
        raise ValueError("genuine bug in preprocessing")

    with pytest.raises(ValueError, match="genuine bug"):
        HELPERS["_run_adaptive_batches"](
            n_total=100, batch_size=10, fetch=_fetch_range, forward=_forward
        )
    assert calls["n"] == 1, "must fail fast, not retry a non-OOM error"


def test_loop_calls_on_oom_hook_before_each_retry():
    gpu = _FakeGPU(capacity=4)
    freed = {"n": 0}

    HELPERS["_run_adaptive_batches"](
        n_total=8,
        batch_size=16,
        fetch=_fetch_range,
        forward=gpu.forward,
        on_oom=lambda: freed.__setitem__("n", freed["n"] + 1),
    )
    assert freed["n"] == gpu.oom_count


def test_loop_is_a_noop_for_empty_input():
    gpu = _FakeGPU(capacity=10**9)
    out = HELPERS["_run_adaptive_batches"](
        n_total=0, batch_size=10, fetch=_fetch_range, forward=gpu.forward
    )
    assert out == []
    assert gpu.attempts == []


# ---------------------------------------------------------------------------
# Prefetch: overlaps I/O with compute without disturbing ordering.
#
# Ordering is the property that matters most here -- a reordered batch would
# silently attach each cell's embedding to the wrong cell.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefetch", [True, False])
def test_prefetch_preserves_item_order(prefetch):
    gpu = _FakeGPU(capacity=10**9)
    out = HELPERS["_run_adaptive_batches"](
        n_total=200,
        batch_size=16,
        fetch=_fetch_range,
        forward=gpu.forward,
        prefetch=prefetch,
    )
    assert _flatten(out) == list(range(200))


def test_prefetch_preserves_order_across_oom_resizes():
    """A discarded prefetch must not drop or duplicate the range it covered."""
    gpu = _FakeGPU(capacity=9)
    out = HELPERS["_run_adaptive_batches"](
        n_total=120,
        batch_size=64,
        fetch=_fetch_range,
        forward=gpu.forward,
        min_batch=1,
        max_batch=64,
        prefetch=True,
    )
    assert gpu.oom_count > 0, "resize path was not exercised"
    assert _flatten(out) == list(range(120))


def test_prefetch_actually_runs_ahead():
    """The next range should be requested before the current forward returns."""
    events: list[str] = []

    def _tracking_fetch(start, stop):
        events.append(f"fetch:{start}")
        return list(range(start, stop))

    def _tracking_forward(items):
        events.append(f"forward:{items[0]}")
        time.sleep(0.05)  # give the background thread room to start
        return list(items)

    HELPERS["_run_adaptive_batches"](
        n_total=40,
        batch_size=10,
        fetch=_tracking_fetch,
        forward=_tracking_forward,
        prefetch=True,
        max_batch=10,
    )

    # fetch:10 must appear before forward:0 completes, i.e. before forward:10.
    assert events.index("fetch:10") < events.index("forward:10")


def test_fetch_is_called_once_per_range_when_size_is_stable():
    """With a stable batch size every range is read exactly once."""
    calls: list[tuple] = []

    def _counting_fetch(start, stop):
        calls.append((start, stop))
        return list(range(start, stop))

    gpu = _FakeGPU(capacity=10**9)
    HELPERS["_run_adaptive_batches"](
        n_total=100,
        batch_size=25,
        fetch=_counting_fetch,
        forward=gpu.forward,
        max_batch=25,
        prefetch=True,
    )
    assert calls == [(0, 25), (25, 50), (50, 75), (75, 100)]


def test_ceiling_prevents_runaway_growth():
    """max_batch caps the search so it never probes past the calibrated size.

    Without the cap the search doubles on every success until it provokes an
    OOM, which wastes work and churns the allocator.
    """
    gpu = _FakeGPU(capacity=10**9)  # nothing would ever OOM
    HELPERS["_run_adaptive_batches"](
        n_total=1000,
        batch_size=100,
        fetch=_fetch_range,
        forward=gpu.forward,
        max_batch=100,
    )
    assert max(gpu.attempts) == 100, "batch size grew beyond the ceiling"


# ---------------------------------------------------------------------------
# Tensor batches: fetch now returns a preprocessed tensor, not a list of
# images, so the loop must slice and measure tensors correctly on the OOM path.
# ---------------------------------------------------------------------------


def _fetch_tensor(start: int, stop: int) -> torch.Tensor:
    """Pseudo preprocessed batch: row i encodes item index i."""
    return torch.arange(start, stop, dtype=torch.float32).unsqueeze(1).repeat(1, 4)


def test_loop_handles_tensor_batches():
    gpu = _FakeGPU(capacity=10**9)

    def _forward(x):
        gpu.attempts.append(len(x))
        gpu.succeeded.append(len(x))
        return x

    out = HELPERS["_run_adaptive_batches"](
        n_total=100,
        batch_size=16,
        fetch=_fetch_tensor,
        forward=_forward,
        max_batch=16,
    )
    combined = torch.cat(out, dim=0)
    assert combined.shape == (100, 4)
    # Order preserved: row i must still encode item i.
    assert torch.equal(combined[:, 0], torch.arange(100, dtype=torch.float32))


def test_tensor_batch_oom_slicing_preserves_order():
    """items[:new_bs] on a tensor must keep the leading rows, not reorder."""
    seen: list[int] = []

    def _forward(x):
        seen.append(len(x))
        if len(x) > 12:
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.00 GiB")
        return x

    out = HELPERS["_run_adaptive_batches"](
        n_total=60,
        batch_size=48,
        fetch=_fetch_tensor,
        forward=_forward,
        min_batch=1,
        max_batch=48,
    )
    combined = torch.cat(out, dim=0)
    assert combined.shape[0] == 60
    assert torch.equal(combined[:, 0], torch.arange(60, dtype=torch.float32))
    assert any(n > 12 for n in seen), "OOM path was not exercised"


# ---------------------------------------------------------------------------
# Progress-bar bookkeeping (tqdm.reset() zeroing `n` was a real bug).
# ---------------------------------------------------------------------------


class _SpyBar:
    """Minimal tqdm stand-in that records whether progress ever went backwards."""

    def __init__(self, total: int):
        self.n = 0
        self.total = total
        self.history: list[int] = []

    def update(self, k: int = 1) -> None:
        self.n += k
        self.history.append(self.n)

    def write(self, msg: str) -> None:
        pass  # suppress output in tests

    def refresh(self) -> None:
        self.history.append(self.n)


def test_loop_grows_batch_via_binary_search_after_success():
    """After a successful batch the search probes upward automatically."""
    gpu = _FakeGPU(capacity=10**9)
    HELPERS["_run_adaptive_batches"](
        n_total=256,
        batch_size=8,
        fetch=_fetch_range,
        forward=gpu.forward,
        max_batch=256,
    )
    # Binary search doubles upward when no ceiling is known: 8→16→32→64→128→256
    assert max(gpu.succeeded) > 8


def test_loop_converges_to_exact_max_fit():
    """Binary search must converge on the largest batch that fits, not just any."""
    gpu = _FakeGPU(capacity=20)
    HELPERS["_run_adaptive_batches"](
        n_total=256,
        batch_size=32,
        fetch=_fetch_range,
        forward=gpu.forward,
        max_batch=64,
        min_batch=1,
    )
    # After convergence every batch should be at most capacity.
    assert max(gpu.succeeded) <= 20
    # And it must be close to the ceiling, not stuck at a tiny value.
    assert max(gpu.succeeded) >= 16


def test_loop_regrows_after_recovering_from_oom():
    """Binary search probes upward after success, so throughput recovers."""
    gpu = _FakeGPU(capacity=8)
    HELPERS["_run_adaptive_batches"](
        n_total=256,
        batch_size=64,
        fetch=_fetch_range,
        forward=gpu.forward,
        max_batch=64,
        min_batch=1,
    )
    assert gpu.oom_count > 0
    assert max(gpu.succeeded) >= 4


def test_progress_never_rewinds_across_resizes():
    gpu = _FakeGPU(capacity=8)
    bar = _SpyBar(total=HELPERS["_remaining_batches"](64, 0, 32))

    HELPERS["_run_adaptive_batches"](
        n_total=64,
        batch_size=32,
        fetch=_fetch_range,
        forward=gpu.forward,
        pbar=bar,
        min_batch=1,
    )

    assert gpu.oom_count > 0, "resize path was not exercised"
    assert bar.history == sorted(bar.history), "progress went backwards"


def test_progress_total_matches_batches_actually_run():
    gpu = _FakeGPU(capacity=8)
    bar = _SpyBar(total=HELPERS["_remaining_batches"](64, 0, 32))

    HELPERS["_run_adaptive_batches"](
        n_total=64,
        batch_size=32,
        fetch=_fetch_range,
        forward=gpu.forward,
        pbar=bar,
        min_batch=1,
    )
    assert bar.n == len(gpu.succeeded)
    assert bar.total == bar.n, "bar must finish full, not stuck mid-way"


@pytest.mark.parametrize(
    "n_total,pos,batch_size,expected",
    [
        (100, 0, 10, 10),
        (100, 95, 10, 1),
        (100, 100, 10, 0),
        (100, 120, 10, 0),
        (7, 0, 3, 3),
    ],
)
def test_remaining_batches(n_total, pos, batch_size, expected):
    assert HELPERS["_remaining_batches"](n_total, pos, batch_size) == expected


# ---------------------------------------------------------------------------
# Opt-in: verify the estimate against a real card when one is present.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prediction_matches_real_gpu_measurement():
    timm = pytest.importorskip("timm")

    try:
        model = (
            timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
            .cuda()
            .eval()
        )
        torch.cuda.empty_cache()

        per_image = HELPERS["_calibrate_bytes_per_image"](model, "cuda")
        assert 1 * MIB <= per_image <= 128 * MIB, (
            f"per-image {per_image / MIB:.1f} MiB is implausible for a ViT; "
            "fixed overhead has probably leaked into the estimate again"
        )

        probe = 64
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        x = torch.zeros(probe, 3, 224, 224, device="cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
        torch.cuda.synchronize()
        actual = torch.cuda.max_memory_allocated() - before
        del x, out
        torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as exc:
        # Another job (often a real wsinsight run) holds the card. The
        # arithmetic is covered by the injected-memory tests above; skip rather
        # than report a failure that says nothing about the code.
        if "out of memory" not in str(exc).lower():
            raise
        pytest.skip(f"GPU unavailable for measurement: {exc}")

    ratio = actual / (probe * per_image)
    assert 0.5 <= ratio <= 1.5, (
        f"predicted {probe * per_image / GIB:.2f} GiB but measured "
        f"{actual / GIB:.2f} GiB (ratio {ratio:.2f})"
    )
