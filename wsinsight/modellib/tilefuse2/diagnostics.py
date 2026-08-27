"""Instrumentation: per-stage timing and emission diagnostics.

``EmitStats.n_touch_tile_edge`` is the load-bearing number.  Bucket ownership is
exact for any cell the owning tile sees whole, which holds when the bucket
padding ``M`` is at least the cell radius.  A cell whose bbox reaches its tile's
read-window edge is proof that ``M`` was too small for that cell, so a zero
count is a *proof* that the geometry was sufficient for this slide rather than
an assumption.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from threading import Lock


class StageTimer:
    """Wall-clock accumulator keyed by stage name.

    Stages are entered once per *instance* on the hot path, so workers use an
    unlocked :meth:`child` and the parent :meth:`absorb`s it once on exit —
    a shared lock here would serialise the very code it measures.
    """

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._totals: dict = defaultdict(float)
        self._counts: dict = defaultdict(int)
        self._lock = Lock()

    def child(self) -> "StageTimer":
        return StageTimer(enabled=self.enabled)

    def absorb(self, other: "StageTimer") -> None:
        with self._lock:
            for name, secs in other._totals.items():
                self._totals[name] += secs
                self._counts[name] += other._counts[name]

    @contextmanager
    def stage(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._totals[name] += time.perf_counter() - t0
            self._counts[name] += 1

    def report(self, wall: float = 0.0, n_workers: int = 1) -> str:
        if not self.enabled or not self._totals:
            return ""
        with self._lock:
            items = sorted(self._totals.items(), key=lambda kv: -kv[1])
            total = sum(self._totals.values())
            lines = [
                "finalize stage timing (thread-wall SUM; inflated by GIL waiting "
                "-- use it for ranking, not for absolute cost):"
            ]
            for name, secs in items:
                n = self._counts[name]
                share = 100.0 * secs / total if total > 0 else 0.0
                lines.append(
                    f"  {name:<18} {secs:8.2f}s  {share:5.1f}%  "
                    f"n={n:<7} {1e3 * secs / max(1, n):7.2f} ms/call"
                )
        if wall > 0:
            eff = 100.0 * total / (wall * max(1, n_workers))
            lines.append(
                f"  {'REAL WALL':<18} {wall:8.2f}s  with {n_workers} workers, "
                f"parallel efficiency {eff:.0f}%"
            )
            if eff < 40:
                lines.append(
                    "  -> GIL-bound: fewer stitch workers would give the same "
                    "wall time at a fraction of the CPU."
                )
        return "\n".join(lines)


class EmitStats:
    """Thread-safe counters describing what bucket ownership emitted."""

    def __init__(self) -> None:
        self.n_emitted = 0
        self.n_discarded = 0
        self.n_touch_tile_edge = 0
        self.max_radius = 0
        self._lock = Lock()

    def merge(
        self,
        *,
        emitted: int,
        discarded: int,
        touch_edge: int,
        max_radius: int,
    ) -> None:
        with self._lock:
            self.n_emitted += emitted
            self.n_discarded += discarded
            self.n_touch_tile_edge += touch_edge
            self.max_radius = max(self.max_radius, max_radius)

    def report(self, bucket_padding: int) -> str:
        lines = [
            "bucket-ownership diagnostics:",
            f"  emitted            = {self.n_emitted}",
            f"  discarded (not mine) = {self.n_discarded}",
            f"  max cell radius    = {self.max_radius} px "
            f"(bucket padding M = {bucket_padding} px)",
            f"  touching tile edge = {self.n_touch_tile_edge}",
        ]
        if self.n_touch_tile_edge == 0:
            lines.append("  -> M was sufficient for every emitted cell.")
        else:
            need = max(1, self.max_radius)
            lines.append(
                f"  -> {self.n_touch_tile_edge} cell(s) exceeded M and may be "
                f"truncated; M >= {need} px is required, i.e. raise the model "
                f"config's overlap_size_pixels/halo_size_pixels accordingly."
            )
        return "\n".join(lines)
