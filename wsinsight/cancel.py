"""Cooperative cancellation primitives for the WSInsight CLI.

This module provides a process-wide, two-press SIGINT (Ctrl-C) handler so
long-running pipelines respond quickly while still protecting in-flight
file writes:

* **First Ctrl-C** — sets a cancellation flag. Loops cooperating with this
  module observe the flag, cancel queued futures, and stop submitting new
  work. Any worker currently inside a :func:`critical_section` (e.g. a CSV
  or JSON save) is allowed to complete.
* **Second Ctrl-C within :data:`_DOUBLE_PRESS_WINDOW_S`** — escalates to a
  hard exit (``os._exit(130)``). If a save is in flight, the message
  "Saving in progress, please wait…" is shown and the hard exit is
  deferred until the save finishes (or until
  :data:`_FORCE_KILL_TIMEOUT_S` elapses, whichever comes first).

The handler is installed once at startup from :mod:`wsinsight.__main__`.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import Iterable
from typing import Iterator

import click

if TYPE_CHECKING:  # pragma: no cover
    from concurrent.futures import Executor
    from concurrent.futures import Future


# ---- public state --------------------------------------------------------

_cancel_event = threading.Event()
_critical_lock = threading.Lock()
_critical_depth = 0
_press_count = 0
_last_press_ts: float = 0.0
_force_kill_armed = False
_handler_installed = False
_last_save_warn_ts: float = 0.0

_DOUBLE_PRESS_WINDOW_S = 3.0
_FORCE_KILL_TIMEOUT_S = 10.0
_SAVE_WARN_THROTTLE_S = 1.0


def is_cancelled() -> bool:
    """Return ``True`` if a cancellation has been requested."""
    return _cancel_event.is_set()


def request_cancel() -> None:
    """Request cancellation programmatically (without raising SIGINT).

    Equivalent to a single Ctrl-C press: cooperating loops observe the
    flag via :func:`is_cancelled` / :func:`raise_if_cancelled` and unwind
    gracefully. Safe to call from any thread. Idempotent.
    """
    _cancel_event.set()


def clear_cancel() -> None:
    """Clear the cancellation flag.

    Intended for long-lived hosts (e.g. the MCP server) that may run
    multiple sequential jobs in the same process and need to reset
    state between them. Not safe to call while a cancellable loop is
    actively unwinding.
    """
    _cancel_event.clear()


def raise_if_cancelled() -> None:
    """Raise :class:`KeyboardInterrupt` if cancellation has been requested.

    Uses ``KeyboardInterrupt`` (a :class:`BaseException`) rather than
    :class:`click.Abort` so it propagates through ``except RuntimeError``
    blocks (e.g. PyTorch OOM-retry handlers) without being silently
    swallowed.
    """
    if _cancel_event.is_set():
        raise KeyboardInterrupt("WSInsight: cancelled by user")


@contextmanager
def critical_section(msg: str = "") -> Iterator[None]:
    """Mark the enclosed block as a non-interruptible save region.

    A second Ctrl-C arriving while any thread is inside a critical section
    is deferred (with a "Saving in progress" notice) until all critical
    sections unwind, then upgraded to a hard exit.

    The optional ``msg`` is currently informational; reserved for future
    use by progress reporters.
    """
    global _critical_depth
    with _critical_lock:
        _critical_depth += 1
    try:
        yield
    finally:
        with _critical_lock:
            _critical_depth -= 1
            depth = _critical_depth
        # If a force-kill was deferred while we were saving, honor it now
        # that we've drained the last critical section.
        if depth == 0 and _force_kill_armed:
            _hard_exit()


def _hard_exit() -> None:
    """Print a final notice and terminate with POSIX SIGINT exit code."""
    try:
        click.secho("\nWSInsight: aborting now (exit 130).", fg="red", err=True)
    except Exception:  # pragma: no cover - stdio may be torn down
        pass
    os._exit(130)


def _sigint_handler(signum, frame) -> None:  # noqa: ARG001 - signal API
    global _press_count, _last_press_ts, _force_kill_armed, _last_save_warn_ts

    now = time.monotonic()
    within_window = (now - _last_press_ts) <= _DOUBLE_PRESS_WINDOW_S
    _last_press_ts = now

    if not within_window:
        _press_count = 1
    else:
        _press_count += 1

    if _press_count == 1:
        _cancel_event.set()
        try:
            click.secho(
                "\nWSInsight: cancellation requested. In-flight saves will "
                "complete before exit. Press Ctrl-C again within "
                f"{int(_DOUBLE_PRESS_WINDOW_S)}s to force kill.",
                fg="yellow",
                err=True,
            )
        except Exception:  # pragma: no cover
            pass
        return

    # press_count >= 2 within the window
    with _critical_lock:
        depth = _critical_depth

    if depth == 0:
        _hard_exit()
        return

    # A save is in progress. Defer hard kill until the section unwinds,
    # but cap the wait so we never hang forever.
    if not _force_kill_armed:
        _force_kill_armed = True

        def _watchdog() -> None:
            time.sleep(_FORCE_KILL_TIMEOUT_S)
            try:
                click.secho(
                    f"\nWSInsight: save did not finish within "
                    f"{int(_FORCE_KILL_TIMEOUT_S)}s; force killing.",
                    fg="red",
                    err=True,
                )
            except Exception:  # pragma: no cover
                pass
            _hard_exit()

        threading.Thread(target=_watchdog, daemon=True).start()

    # Throttle the "save in progress" warning to once per second.
    if (now - _last_save_warn_ts) >= _SAVE_WARN_THROTTLE_S:
        _last_save_warn_ts = now
        try:
            click.secho(
                "\nWSInsight: save in progress, please wait… "
                "(will exit immediately after the current write completes).",
                fg="yellow",
                err=True,
            )
        except Exception:  # pragma: no cover
            pass


def install_sigint_handler() -> None:
    """Install the two-press SIGINT handler. Idempotent."""
    global _handler_installed
    if _handler_installed:
        return
    # Only install in the main thread of the main process; subprocesses
    # spawned via ``multiprocessing`` set their own handlers.
    try:
        signal.signal(signal.SIGINT, _sigint_handler)
    except (ValueError, OSError):
        # Not in main thread, or platform doesn't allow it. Skip.
        return
    _handler_installed = True


def cancellable_as_completed(
    futures: Iterable["Future"],
    executor: "Executor | None" = None,
) -> Iterator["Future"]:
    """Yield futures as they complete; stop early on cancellation.

    On cancellation the wrapped ``executor`` (if provided) is shut down
    with ``cancel_futures=True`` so queued work is dropped while
    currently-running workers may still complete and flush their
    :func:`critical_section` saves.
    """
    from concurrent.futures import as_completed

    fut_list = list(futures)
    cancelled = False
    for fut in as_completed(fut_list):
        yield fut
        if not cancelled and _cancel_event.is_set():
            cancelled = True
            if executor is not None:
                # Python 3.9+: cancel_futures drops queued work.
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:  # pragma: no cover - older Python
                    executor.shutdown(wait=False)
            # Continue draining results from already-running workers so
            # their critical sections can finish, but no new work starts.


__all__ = [
    "install_sigint_handler",
    "is_cancelled",
    "request_cancel",
    "clear_cancel",
    "raise_if_cancelled",
    "critical_section",
    "cancellable_as_completed",
]
