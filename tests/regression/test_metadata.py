"""Tier 1 (fast): per-slide metadata regression.

Reads MPP / AppMag / level dimensions through the wsinsight WSI helpers and
compares against the scalars recorded in ``cases.toml``. No model, no GPU,
no patch extraction — these tests catch backend / fallback drift in seconds.
"""

from __future__ import annotations

import logging

import pytest

from tests.regression.conftest import RegressionCase
from wsinsight import wsi


pytestmark = pytest.mark.regression


def _require_slide(case: RegressionCase) -> None:
    if not case.exists:
        pytest.skip(f"Slide not on disk: {case.path}")


def test_mpp_matches_expected(case: RegressionCase, caplog):
    _require_slide(case)
    if case.expected_mpp is None:
        pytest.skip("expected_mpp not set for this case")

    with caplog.at_level(logging.WARNING, logger=wsi.logger.name):
        mpp = wsi.get_avg_mpp(str(case.path))

    assert mpp == pytest.approx(case.expected_mpp, abs=case.mpp_atol), (
        f"MPP drift for {case.slide_id}: got {mpp}, expected {case.expected_mpp} "
        f"(±{case.mpp_atol})"
    )

    if case.expects_appmag_fallback:
        assert any("AppMag" in r.message for r in caplog.records), (
            f"{case.slide_id}: expected AppMag fallback warning but none was logged"
        )


def test_appmag_matches_expected(case: RegressionCase):
    _require_slide(case)
    if case.expected_appmag is None:
        pytest.skip("expected_appmag not set for this case")

    appmag = (
        wsi._get_appmag_openslide(str(case.path))
        or wsi._get_appmag_tiffslide(str(case.path))
    )
    assert appmag is not None, f"{case.slide_id}: AppMag could not be read"
    assert appmag == pytest.approx(case.expected_appmag, abs=0.5)
