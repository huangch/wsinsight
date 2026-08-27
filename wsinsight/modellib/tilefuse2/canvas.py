"""Lazily-allocated sparse canvas (ported from ``tilefuse``, made accumulable).

Chunks are zero-filled because :meth:`accumulate` sums overlapping patch
contributions; v1's ``np.empty`` optimisation is incompatible with that.
"""

from __future__ import annotations

import numpy as np


class SparseCanvas:
    """A (H, W) or (H, W, C) array that allocates only the chunks written to."""

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
        self._chunks: dict = {}

    def _alloc(self, cy: int, cx: int) -> np.ndarray:
        ch = min(self.chunk_size, self.height - cy)
        cw = min(self.chunk_size, self.width - cx)
        shape = (ch, cw) if self.n_channels == 0 else (ch, cw, self.n_channels)
        arr = np.zeros(shape, dtype=self.dtype)
        self._chunks[(cy, cx)] = arr
        return arr

    def _snap(self, coord: int) -> int:
        return (coord // self.chunk_size) * self.chunk_size

    def accumulate(self, y0: int, y1: int, x0: int, x1: int, data: np.ndarray) -> None:
        """Add ``data`` into ``[y0:y1, x0:x1, ...]`` (feather-blend path)."""
        self._apply(y0, y1, x0, x1, data, add=True)

    def write(self, y0: int, y1: int, x0: int, x1: int, data: np.ndarray) -> None:
        """Write ``data[0:y1-y0, 0:x1-x0, ...]`` into ``[y0:y1, x0:x1, ...]``."""
        self._apply(y0, y1, x0, x1, data, add=False)

    def _apply(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        data: np.ndarray,
        add: bool,
    ) -> None:
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
                    if chunk is None:
                        chunk = self._alloc(cy, cx)
                    lry0 = ry0 - cy
                    lry1 = ry1 - cy
                    lrx0 = rx0 - cx
                    lrx1 = rx1 - cx
                    dry0 = ry0 - y0
                    dry1 = ry1 - y0
                    drx0 = rx0 - x0
                    drx1 = rx1 - x0
                    if self.n_channels == 0:
                        dst = chunk[lry0:lry1, lrx0:lrx1]
                        src = data[dry0:dry1, drx0:drx1]
                    else:
                        dst = chunk[lry0:lry1, lrx0:lrx1, :]
                        src = data[dry0:dry1, drx0:drx1, :]
                    if add:
                        dst += src
                    else:
                        dst[...] = src
                cx += cs
            cy += cs

    def read(self, y0: int, y1: int, x0: int, x1: int, out_dtype=None) -> np.ndarray:
        """Return a fresh array from ``[y0:y1, x0:x1, ...]``; unwritten → zeros."""
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
