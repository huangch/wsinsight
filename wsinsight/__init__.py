"""WSInsight is a toolkit for fast patch-based inference on whole slide images."""

from __future__ import annotations

import os

# Force ASCII progress bars for tmux compatibility.
# Must be set before tqdm is first imported anywhere in the package.
os.environ.setdefault("TQDM_ASCII", " #")

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.unknown"
