"""Tier 3 (slow): inference-output regression.

Compares per-slide model output CSV against a golden copy. Only numeric
columns are compared with tolerance; structural columns (cell_id, x, y, etc.)
must match exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from tests.regression.conftest import RegressionCase


pytestmark = [pytest.mark.regression, pytest.mark.slow]


GOLDEN_NAME = "inference.csv"
NUMERIC_RTOL = 1e-3
NUMERIC_ATOL = 1e-4


def _find_inference_csv(results_dir: Path, slide_stem: str) -> Path | None:
    # wsinsight writes per-slide CSVs under model-outputs-csv/.
    for sub in ("model-outputs-csv", "model-outputs"):
        cand = results_dir / sub
        if cand.is_dir():
            hits = list(cand.glob(f"{slide_stem}*.csv"))
            if hits:
                return hits[0]
    return None


def test_inference_csv_matches_golden(case: RegressionCase, request):
    if not case.exists:
        pytest.skip(f"Slide not on disk: {case.path}")
    golden = case.fixture_dir / GOLDEN_NAME
    if not golden.is_file():
        pytest.skip(f"No golden inference.csv for {case.slide_id}")

    external = request.config.getoption("--infer-output-dir", default=None)
    if not external:
        pytest.skip("Pass --infer-output-dir=<dir> to compare an existing run.")

    actual_csv = _find_inference_csv(Path(external), case.path.stem)
    if actual_csv is None:
        pytest.fail(f"No inference CSV produced for {case.slide_id} under {external}")

    actual = pd.read_csv(actual_csv)
    expected = pd.read_csv(golden)

    assert list(actual.columns) == list(expected.columns), (
        f"{case.slide_id}: column drift "
        f"(actual={list(actual.columns)}, expected={list(expected.columns)})"
    )
    assert len(actual) == len(expected), (
        f"{case.slide_id}: row count drift "
        f"(actual={len(actual)}, expected={len(expected)})"
    )

    for col in expected.columns:
        a, e = actual[col], expected[col]
        if np.issubdtype(e.dtype, np.number):
            np.testing.assert_allclose(
                a.to_numpy(), e.to_numpy(),
                rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL,
                err_msg=f"{case.slide_id}: numeric drift in column {col!r}",
            )
        else:
            assert (a.fillna("") == e.fillna("")).all(), (
                f"{case.slide_id}: non-numeric drift in column {col!r}"
            )
