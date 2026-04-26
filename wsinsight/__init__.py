"""WSInsight is a toolkit for fast patch-based inference on whole slide images."""

from __future__ import annotations

import os

# Force ASCII progress bars for tmux compatibility.
# Must be set before tqdm is first imported anywhere in the package.
os.environ.setdefault("TQDM_ASCII", " #")

# Silence TensorFlow info / oneDNN notices. TF is pulled in transitively by
# some deps (e.g. stardist, histomicstk); these vars must be set BEFORE the
# very first ``import tensorflow`` happens anywhere in the process. Putting
# them here ensures both the ``wsinsight`` console script and ``python -m
# wsinsight`` honor them, since ``wsinsight/__init__.py`` is imported first
# in either case.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 3 = ERROR only
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.unknown"
