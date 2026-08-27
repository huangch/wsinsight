"""Tests for the WSINSIGHT_TRACEBACK error-reporting switch."""

from __future__ import annotations

import pytest

from wsinsight.errors import TRACEBACK_ENV_VAR
from wsinsight.errors import format_cli_error
from wsinsight.errors import traceback_enabled


def _raised(exc: BaseException) -> BaseException:
    """Return the exception with a real __traceback__ attached."""
    try:
        raise exc
    except BaseException as e:  # noqa: BLE001
        return e


def test_disabled_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRACEBACK_ENV_VAR, raising=False)
    assert traceback_enabled() is False


@pytest.mark.parametrize("value", ["1", "0", "", "yes", "anything"])
def test_enabled_by_mere_presence(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(TRACEBACK_ENV_VAR, value)
    assert traceback_enabled() is True


def test_message_only_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRACEBACK_ENV_VAR, raising=False)
    out = format_cli_error(_raised(ValueError("bad spacing")))
    assert "Error message:" in out
    assert "ValueError: bad spacing" in out
    assert TRACEBACK_ENV_VAR in out
    assert "Traceback (most recent call last)" not in out


def test_traceback_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TRACEBACK_ENV_VAR, "1")
    out = format_cli_error(_raised(ValueError("bad spacing")))
    assert "Traceback (most recent call last)" in out
    assert "ValueError: bad spacing" in out


def test_empty_message_falls_back_to_type(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TRACEBACK_ENV_VAR, raising=False)
    out = format_cli_error(_raised(RuntimeError()))
    assert "RuntimeError" in out
    assert "RuntimeError:" not in out
