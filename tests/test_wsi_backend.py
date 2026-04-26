"""Tests for backend selection and AppMag-based MPP fallback in wsinsight.wsi."""

from __future__ import annotations

import importlib
import logging

import pytest

from wsinsight import wsi
from wsinsight.errors import CannotReadSpacing


def test_appmag_lookup_table_values():
    assert wsi._MPP_FROM_APPMAG[40] == pytest.approx(0.25)
    assert wsi._MPP_FROM_APPMAG[20] == pytest.approx(0.50)
    assert wsi._MPP_FROM_APPMAG[10] == pytest.approx(1.00)
    assert wsi._MPP_FROM_APPMAG[4]  == pytest.approx(2.50)


def _force_all_primary_readers_to_fail(monkeypatch):
    def _fail(*_a, **_kw):
        raise CannotReadSpacing()

    monkeypatch.setattr(wsi, "_get_mpp_openslide", _fail)
    monkeypatch.setattr(wsi, "_get_mpp_tiffslide", _fail)
    monkeypatch.setattr(wsi, "_get_mpp_tifffile", _fail)


def test_get_avg_mpp_appmag_fallback_40x(monkeypatch, caplog):
    _force_all_primary_readers_to_fail(monkeypatch)
    monkeypatch.setattr(wsi, "_get_appmag_openslide", lambda p: 40.0)
    monkeypatch.setattr(wsi, "_get_appmag_tiffslide", lambda p: None)

    with caplog.at_level(logging.WARNING, logger=wsi.logger.name):
        mpp = wsi.get_avg_mpp("dummy.svs")

    assert mpp == pytest.approx(0.25)
    assert any("AppMag=40" in rec.message for rec in caplog.records)


def test_get_avg_mpp_appmag_fallback_20x_via_tiffslide(monkeypatch):
    _force_all_primary_readers_to_fail(monkeypatch)
    monkeypatch.setattr(wsi, "_get_appmag_openslide", lambda p: None)
    monkeypatch.setattr(wsi, "_get_appmag_tiffslide", lambda p: 20.0)
    assert wsi.get_avg_mpp("dummy.svs") == pytest.approx(0.50)


def test_get_avg_mpp_appmag_rounded(monkeypatch):
    """Non-integer mags should round to the nearest table key."""
    _force_all_primary_readers_to_fail(monkeypatch)
    monkeypatch.setattr(wsi, "_get_appmag_openslide", lambda p: 40.001)
    monkeypatch.setattr(wsi, "_get_appmag_tiffslide", lambda p: None)
    assert wsi.get_avg_mpp("dummy.svs") == pytest.approx(0.25)


def test_get_avg_mpp_unknown_appmag_raises(monkeypatch, caplog):
    _force_all_primary_readers_to_fail(monkeypatch)
    monkeypatch.setattr(wsi, "_get_appmag_openslide", lambda p: 7.0)
    monkeypatch.setattr(wsi, "_get_appmag_tiffslide", lambda p: None)

    with caplog.at_level(logging.WARNING, logger=wsi.logger.name):
        with pytest.raises(CannotReadSpacing):
            wsi.get_avg_mpp("dummy.svs")

    assert any("not in fallback table" in rec.message for rec in caplog.records)


def test_get_avg_mpp_no_appmag_raises(monkeypatch):
    _force_all_primary_readers_to_fail(monkeypatch)
    monkeypatch.setattr(wsi, "_get_appmag_openslide", lambda p: None)
    monkeypatch.setattr(wsi, "_get_appmag_tiffslide", lambda p: None)
    with pytest.raises(CannotReadSpacing):
        wsi.get_avg_mpp("dummy.svs")


def test_backend_env_override_tiffslide(monkeypatch):
    if not wsi.HAS_TIFFSLIDE:
        pytest.skip("tiffslide not installed")
    monkeypatch.setenv("WSINSIGHT_WSI_BACKEND", "tiffslide")
    reloaded = importlib.reload(wsi)
    try:
        assert reloaded._BACKEND == "tiffslide"
    finally:
        # Restore module state for subsequent tests.
        monkeypatch.delenv("WSINSIGHT_WSI_BACKEND", raising=False)
        importlib.reload(wsi)


def test_backend_env_override_openslide(monkeypatch):
    if not wsi.HAS_OPENSLIDE:
        pytest.skip("openslide not installed")
    monkeypatch.setenv("WSINSIGHT_WSI_BACKEND", "openslide")
    reloaded = importlib.reload(wsi)
    try:
        assert reloaded._BACKEND == "openslide"
    finally:
        monkeypatch.delenv("WSINSIGHT_WSI_BACKEND", raising=False)
        importlib.reload(wsi)


def test_backend_default_prefers_openslide_when_available(monkeypatch):
    monkeypatch.delenv("WSINSIGHT_WSI_BACKEND", raising=False)
    reloaded = importlib.reload(wsi)
    try:
        if reloaded.HAS_OPENSLIDE:
            assert reloaded._BACKEND == "openslide"
        else:
            assert reloaded._BACKEND == "tiffslide"
    finally:
        importlib.reload(wsi)


def test_backend_invalid_env_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("WSINSIGHT_WSI_BACKEND", "not-a-real-backend")
    reloaded = importlib.reload(wsi)
    try:
        # Invalid override is ignored; smart default applies.
        if reloaded.HAS_OPENSLIDE:
            assert reloaded._BACKEND == "openslide"
        else:
            assert reloaded._BACKEND == "tiffslide"
    finally:
        monkeypatch.delenv("WSINSIGHT_WSI_BACKEND", raising=False)
        importlib.reload(wsi)
