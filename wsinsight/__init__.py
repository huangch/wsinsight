"""WSInsight is a toolkit for fast patch-based inference on whole slide images."""

from __future__ import annotations

import os

# Force ASCII progress bars for tmux compatibility.
# Must be set before tqdm is first imported anywhere in the package.
os.environ.setdefault("TQDM_ASCII", " =")


def _harden_tqdm_against_resize() -> None:
    """Make every tqdm bar survive terminal / tmux resizes.

    Two changes are applied process-wide:

    1. ``dynamic_ncols=True`` becomes the default for every bar, so tqdm
       re-queries the terminal width on each refresh instead of caching the
       width it saw at construction time (which is what leaves garbage on
       screen after a resize).
    2. A ``SIGWINCH`` handler clears and immediately redraws every live bar
       the moment the terminal is resized, instead of waiting for the next
       ``update()`` call. This is the "reflash the whole bar on resize"
       behaviour.

    Both are best-effort: any failure (e.g. tqdm not installed, handler
    installed from a non-main thread) is swallowed so importing wsinsight
    never fails because of progress-bar cosmetics.
    """
    try:
        from tqdm import std as _tqdm_std
    except Exception:
        return

    # 1. Default every bar to dynamic_ncols so width is recomputed each refresh.
    # Sentinels are deliberately NOT package-prefixed: these packages share one
    # env and land in one process, so a per-package sentinel would let each of
    # them wrap __init__ and chain another SIGWINCH handler.
    if not getattr(_tqdm_std.tqdm, "_tqdm_resize_hardened", False):
        _orig_init = _tqdm_std.tqdm.__init__

        def _init(self, *args, **kwargs):  # noqa: ANN001
            kwargs.setdefault("dynamic_ncols", True)
            _orig_init(self, *args, **kwargs)

        _tqdm_std.tqdm.__init__ = _init
        _tqdm_std.tqdm._tqdm_resize_hardened = True

    # 2. Redraw all active bars on terminal resize (SIGWINCH).
    try:
        import signal

        if not hasattr(signal, "SIGWINCH"):
            return  # not POSIX (e.g. Windows); nothing to do
        if getattr(_tqdm_std.tqdm, "_tqdm_winch_installed", False):
            return

        _prev_handler = signal.getsignal(signal.SIGWINCH)

        def _on_winch(signum, frame):  # noqa: ANN001
            try:
                for inst in list(getattr(_tqdm_std.tqdm, "_instances", [])):
                    inst.clear(nolock=True)
                    inst.refresh(nolock=True)
            except Exception:
                pass
            # Chain to whatever handler was installed before us.
            if callable(_prev_handler):
                _prev_handler(signum, frame)

        signal.signal(signal.SIGWINCH, _on_winch)
        _tqdm_std.tqdm._tqdm_winch_installed = True
    except (ValueError, OSError):
        # signal.signal raises ValueError off the main thread; ignore.
        pass


_harden_tqdm_against_resize()

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
