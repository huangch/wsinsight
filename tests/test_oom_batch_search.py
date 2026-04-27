"""Tests for OOM batch-size binary-search helper in run_inference."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_helper():
    """Load just ``_advance_batch_search`` without triggering the heavy module
    imports (torch, h5py, histomicstk, ...) at the top of run_inference.py.

    Parses the source file and execs only the helper definition.
    """
    src_path = (
        Path(__file__).resolve().parents[1]
        / "wsinsight" / "modellib" / "run_inference.py"
    )
    text = src_path.read_text()
    marker_start = "def _advance_batch_search("
    marker_end = "def run_inference("
    i, j = text.index(marker_start), text.index(marker_end)
    helper_src = text[i:j]
    ns: dict = {}
    exec(compile(helper_src, str(src_path), "exec"), ns)
    return ns["_advance_batch_search"]


_advance_batch_search = _load_helper()


# ---------------------------------------------------------------------------
# The deadlock case that motivated this helper.
# ---------------------------------------------------------------------------

def test_oom_after_convergence_does_not_deadlock():
    """After lo=hi=current=20 succeeds for many slides, an OOM at 20 must
    NOT propose 20 again (that's the bug we're fixing)."""
    nxt, lo, hi = _advance_batch_search(
        current=20, lo=20, hi=21, max_bs=32, oom=True
    )
    assert nxt < 20
    assert hi == 20
    assert lo < 20
    # Helper invariant: in (1, max_bs).
    assert 1 <= nxt <= 32


def test_oom_after_convergence_repeated_progresses():
    """Repeated OOMs at the same value must keep stepping down, never stall."""
    current, lo, hi = 20, 20, 21
    seen = [current]
    for _ in range(40):
        current, lo, hi = _advance_batch_search(
            current, lo, hi, max_bs=32, oom=True
        )
        seen.append(current)
        if current == 1:
            break
    assert seen[-1] == 1
    # No infinite loop on identical values.
    assert len(seen) <= 41


# ---------------------------------------------------------------------------
# First-time OOM (cold start: lo=0, hi=max_bs+1).
# ---------------------------------------------------------------------------

def test_first_oom_bisects_to_half():
    nxt, lo, hi = _advance_batch_search(
        current=32, lo=0, hi=33, max_bs=32, oom=True
    )
    assert nxt == 16
    assert lo == 0
    assert hi == 32


def test_oom_cascade_reaches_one():
    current, lo, hi, max_bs = 32, 0, 33, 32
    for _ in range(20):
        current, lo, hi = _advance_batch_search(
            current, lo, hi, max_bs, oom=True
        )
        if current == 1:
            break
    assert current == 1


# ---------------------------------------------------------------------------
# Success path.
# ---------------------------------------------------------------------------

def test_success_with_no_ceiling_doubles():
    nxt, lo, hi = _advance_batch_search(
        current=4, lo=0, hi=33, max_bs=32, oom=False
    )
    assert nxt == 8
    assert lo == 4
    assert hi == 33  # still no ceiling


def test_success_with_no_ceiling_clamps_to_max():
    nxt, _, _ = _advance_batch_search(
        current=24, lo=0, hi=33, max_bs=32, oom=False
    )
    assert nxt == 32


def test_success_with_known_ceiling_bisects_upward():
    nxt, lo, hi = _advance_batch_search(
        current=16, lo=0, hi=20, max_bs=32, oom=False
    )
    assert nxt == 18
    assert lo == 16
    assert hi == 20


def test_success_at_converged_bracket_holds_steady():
    """lo=hi → no room to grow; helper must not propose 0 or oscillate."""
    nxt, lo, hi = _advance_batch_search(
        current=20, lo=20, hi=21, max_bs=32, oom=False
    )
    assert nxt == 20
    assert lo == 20
    assert hi == 21


# ---------------------------------------------------------------------------
# Invariants.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("oom", [True, False])
@pytest.mark.parametrize(
    "current,lo,hi,max_bs",
    [
        (32, 0, 33, 32),
        (16, 0, 33, 32),
        (1, 0, 2, 32),
        (20, 20, 21, 32),
        (8, 4, 16, 16),
        (16, 0, 17, 16),
    ],
)
def test_next_current_in_range(current, lo, hi, max_bs, oom):
    nxt, _, _ = _advance_batch_search(current, lo, hi, max_bs, oom=oom)
    assert 1 <= nxt <= max_bs


def test_oom_strictly_lowers_hi():
    _, _, new_hi = _advance_batch_search(
        current=16, lo=4, hi=33, max_bs=32, oom=True
    )
    assert new_hi == 16


def test_success_strictly_raises_lo():
    _, new_lo, _ = _advance_batch_search(
        current=16, lo=4, hi=33, max_bs=32, oom=False
    )
    assert new_lo == 16
