"""Bucket-ownership tile stitcher (v2).

See :mod:`wsinsight.modellib.tilefuse2.geometry` for the grid derivation and
:mod:`wsinsight.modellib.tilefuse2.emit` for the ownership rule that replaces
v1's crop-then-dedup pass.
"""

from .diagnostics import EmitStats
from .diagnostics import StageTimer
from .geometry import BucketGeometry
from .geometry import BucketJob
from .stitcher import TileRemapStitcherV2

__all__ = [
    "BucketGeometry",
    "BucketJob",
    "EmitStats",
    "StageTimer",
    "TileRemapStitcherV2",
]
