"""Bucket-ownership geometry for the v2 stitcher.

Every number here is derived from the model's ``config.json``; the module
introduces no tunable constant of its own.

Model-pixel domain::

    tile_size      = patch_size_pixels
    bucket_padding = halo_size_pixels + overlap_size_pixels // 2
    bucket_size    = tile_size - 2 * bucket_padding

``halo_size_pixels`` and ``overlap_size_pixels`` are mutually exclusive in
practice (CellViT declares the latter, HoVer-Net the former) but both express
the same thing for our purpose: the margin of a model patch whose predictions
are context-only.  Summing them is therefore safe and needs no default.

Slide-pixel domain: ``bucket_size`` and ``bucket_padding`` are converted
independently and the tile is *derived* as ``T = B + 2M`` so the identity holds
exactly after rounding.

Grid::

    bucket k = [B*k,     B*k + B)        ownership, tiles the slide exactly
    tile   k = [B*k - M, B*k + B + M)    read window, clipped to the slide

A cell is emitted by the tile whose bucket contains its centroid.  Buckets
partition the slide, so every cell is claimed exactly once — no duplicates, no
misses, and no dedup pass.  Correctness of the emitted geometry additionally
requires ``M >= cell radius``; ``EmitStats.n_touch_tile_edge`` measures that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class BucketJob:
    """One ownership region plus the read window that resolves it."""

    bucket_y0: int
    bucket_y1: int
    bucket_x0: int
    bucket_x1: int
    tile_y0: int
    tile_y1: int
    tile_x0: int
    tile_x1: int

    @property
    def tile_height(self) -> int:
        return self.tile_y1 - self.tile_y0

    @property
    def tile_width(self) -> int:
        return self.tile_x1 - self.tile_x0


@dataclass(frozen=True)
class BucketGeometry:
    """Slide-pixel bucket grid derived from a model config."""

    bucket_size: int  # B
    bucket_padding: int  # M
    slide_height: int
    slide_width: int

    # Model-pixel provenance, kept for logging / tests.
    model_tile_size: int
    model_bucket_size: int
    model_bucket_padding: int

    @property
    def tile_size(self) -> int:
        """T = B + 2M (exact by construction, not re-rounded)."""
        return self.bucket_size + 2 * self.bucket_padding

    @classmethod
    def from_model_config(
        cls,
        *,
        patch_size_pixels: int,
        halo_size_pixels: int,
        overlap_size_pixels: int,
        model_mpp: float,
        slide_mpp: float,
        slide_height: int,
        slide_width: int,
    ) -> "BucketGeometry":
        model_tile = int(patch_size_pixels)
        model_pad = int(halo_size_pixels) + int(overlap_size_pixels) // 2
        model_bucket = model_tile - 2 * model_pad
        if model_bucket <= 0:
            raise ValueError(
                f"bucket_size must be positive: patch_size_pixels={model_tile}, "
                f"halo_size_pixels={halo_size_pixels}, "
                f"overlap_size_pixels={overlap_size_pixels}"
            )

        ratio = float(model_mpp) / float(slide_mpp)
        return cls(
            bucket_size=max(1, int(round(model_bucket * ratio))),
            bucket_padding=int(round(model_pad * ratio)),
            slide_height=int(slide_height),
            slide_width=int(slide_width),
            model_tile_size=model_tile,
            model_bucket_size=model_bucket,
            model_bucket_padding=model_pad,
        )

    def jobs(self) -> List[BucketJob]:
        """Enumerate the bucket grid in raster order."""
        b = self.bucket_size
        m = self.bucket_padding
        h = self.slide_height
        w = self.slide_width

        out: List[BucketJob] = []
        for by0 in range(0, h, b):
            by1 = min(by0 + b, h)
            ty0 = max(0, by0 - m)
            ty1 = min(h, by1 + m)
            for bx0 in range(0, w, b):
                bx1 = min(bx0 + b, w)
                out.append(
                    BucketJob(
                        bucket_y0=by0,
                        bucket_y1=by1,
                        bucket_x0=bx0,
                        bucket_x1=bx1,
                        tile_y0=ty0,
                        tile_y1=ty1,
                        tile_x0=max(0, bx0 - m),
                        tile_x1=min(w, bx1 + m),
                    )
                )
        return out

    def describe(self) -> str:
        return (
            f"bucket geometry: model px tile={self.model_tile_size} "
            f"bucket={self.model_bucket_size} pad={self.model_bucket_padding} "
            f"| slide px T={self.tile_size} B={self.bucket_size} "
            f"M={self.bucket_padding} "
            f"| grid {-(-self.slide_height // self.bucket_size)}x"
            f"{-(-self.slide_width // self.bucket_size)}"
        )
