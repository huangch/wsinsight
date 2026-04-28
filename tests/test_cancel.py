"""Tests for the wsinsight.cancel cooperative cancellation module."""

from __future__ import annotations

import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from wsinsight import cancel


@pytest.fixture(autouse=True)
def _reset_cancel_state():
    """Reset module-level state before/after each test."""
    cancel._cancel_event.clear()
    cancel._press_count = 0
    cancel._last_press_ts = 0.0
    cancel._force_kill_armed = False
    cancel._last_save_warn_ts = 0.0
    with cancel._critical_lock:
        cancel._critical_depth = 0
    yield
    cancel._cancel_event.clear()


def test_is_cancelled_default_false():
    assert cancel.is_cancelled() is False


def test_raise_if_cancelled_when_set():
    cancel._cancel_event.set()
    with pytest.raises(KeyboardInterrupt):
        cancel.raise_if_cancelled()


def test_raise_if_cancelled_no_raise_when_clear():
    cancel.raise_if_cancelled()  # should not raise


def test_critical_section_increments_and_decrements():
    assert cancel._critical_depth == 0
    with cancel.critical_section("save A"):
        assert cancel._critical_depth == 1
        with cancel.critical_section("save B"):
            assert cancel._critical_depth == 2
        assert cancel._critical_depth == 1
    assert cancel._critical_depth == 0


def test_first_press_sets_event_and_does_not_exit(monkeypatch):
    exited = {"called": False}

    def fake_exit():
        exited["called"] = True

    monkeypatch.setattr(cancel, "_hard_exit", fake_exit)
    cancel._sigint_handler(signal.SIGINT, None)
    assert cancel.is_cancelled() is True
    assert exited["called"] is False


def test_double_press_outside_critical_section_hard_exits(monkeypatch):
    exited = {"called": False}

    def fake_exit():
        exited["called"] = True

    monkeypatch.setattr(cancel, "_hard_exit", fake_exit)

    cancel._sigint_handler(signal.SIGINT, None)
    cancel._sigint_handler(signal.SIGINT, None)

    assert exited["called"] is True


def test_double_press_inside_critical_section_defers_exit(monkeypatch):
    exited = {"called": False}

    def fake_exit():
        exited["called"] = True

    monkeypatch.setattr(cancel, "_hard_exit", fake_exit)
    # Make the watchdog timeout effectively infinite so we test
    # the deferral path, not the watchdog.
    monkeypatch.setattr(cancel, "_FORCE_KILL_TIMEOUT_S", 60.0)

    with cancel.critical_section("slow save"):
        cancel._sigint_handler(signal.SIGINT, None)  # arms cancel
        cancel._sigint_handler(signal.SIGINT, None)  # would normally hard-exit
        assert exited["called"] is False  # deferred while in critical section
        assert cancel._force_kill_armed is True

    # On unwinding the critical section, the deferred exit should fire.
    assert exited["called"] is True


def test_press_outside_window_resets_count(monkeypatch):
    monkeypatch.setattr(cancel, "_DOUBLE_PRESS_WINDOW_S", 0.05)
    exited = {"called": 0}
    monkeypatch.setattr(cancel, "_hard_exit", lambda: exited.__setitem__("called", exited["called"] + 1))

    cancel._sigint_handler(signal.SIGINT, None)
    time.sleep(0.1)  # window elapses
    cancel._sigint_handler(signal.SIGINT, None)
    assert exited["called"] == 0  # treated as first press again


def test_cancellable_as_completed_yields_all_when_no_cancel():
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(lambda x: x * 2, i) for i in range(5)]
        results = sorted(f.result() for f in cancel.cancellable_as_completed(futs, ex))
    assert results == [0, 2, 4, 6, 8]


def test_cancellable_as_completed_calls_shutdown_on_cancel():
    """When cancel is set, the wrapper invokes executor.shutdown with cancel_futures."""

    class FakeExecutor:
        def __init__(self):
            self.shutdown_calls = []

        def shutdown(self, wait=True, cancel_futures=False):
            self.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})

    fake_ex = FakeExecutor()
    with ThreadPoolExecutor(max_workers=2) as real_ex:
        futs = [real_ex.submit(lambda i=i: i, i) for i in range(3)]
        # Wait for them to complete so as_completed yields immediately.
        for f in futs:
            f.result()
        # First yield: trigger cancel; subsequent iterations should call shutdown.
        gen = cancel.cancellable_as_completed(futs, fake_ex)
        next(gen)
        cancel._cancel_event.set()
        # Drain remaining yields so the wrapper observes the cancel.
        list(gen)

    assert len(fake_ex.shutdown_calls) == 1
    assert fake_ex.shutdown_calls[0]["cancel_futures"] is True


def test_install_sigint_handler_idempotent():
    # Save and restore the original handler so we don't pollute the test runner.
    original = signal.getsignal(signal.SIGINT)
    try:
        cancel._handler_installed = False
        cancel.install_sigint_handler()
        first = signal.getsignal(signal.SIGINT)
        cancel.install_sigint_handler()  # second call should be a no-op
        second = signal.getsignal(signal.SIGINT)
        assert first is second
    finally:
        signal.signal(signal.SIGINT, original)
        cancel._handler_installed = False


def test_console_script_entry_point_runs_main_not_cli_directly():
    """Regression: the ``wsinsight`` console script must dispatch through
    ``wsinsight.__main__:main`` (which installs the SIGINT handler), not
    jump directly into the Click ``cli`` callable. If this points at the
    raw Click group, Ctrl-C falls back to Python's default handler and
    the two-press cancellation flow never runs.
    """
    import importlib.metadata as md

    raw = md.entry_points()
    if hasattr(raw, "select"):  # Python 3.10+
        eps = list(raw.select(group="console_scripts"))
    else:  # Python 3.9
        eps = list(raw.get("console_scripts", []))
    eps = [e for e in eps if e.name == "wsinsight"]
    assert len(eps) == 1, f"expected exactly one wsinsight console script, got {eps}"
    assert eps[0].value == "wsinsight.__main__:main", (
        f"wsinsight console-script must be wsinsight.__main__:main "
        f"so install_sigint_handler() runs; got {eps[0].value!r}"
    )
