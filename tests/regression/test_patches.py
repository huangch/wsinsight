"""Tier 2 (medium): patch-coordinate regression.

For each slide, compares the ``coords`` HDF5 dataset produced by ``wsinsight
patch`` against a golden copy under ``tests/regression/fixtures/<slide_id>/``.

These tests assume the patch stage has already been run by an earlier test or
by ``scripts/regenerate_regression_baselines.py``. To run the patch stage as
part of the test, set ``--run-slow``.

The comparison is deterministic: same backend + same parameters must produce
exactly the same coordinate set. Order-insensitive equality is used because
some segmentation paths emit coordinates in non-deterministic order.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.regression.conftest import RegressionCase

h5py = pytest.importorskip("h5py")

pytestmark = [pytest.mark.regression, pytest.mark.slow]


PATCH_GOLDEN_NAME = "patches.h5"
PATCH_RUN_DIRNAME = "patch_run"


def _golden_path(case: RegressionCase) -> Path:
    return case.fixture_dir / PATCH_GOLDEN_NAME


def _read_coords(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return np.asarray(f["coords"][:])


def _coord_set(arr: np.ndarray) -> set[tuple[int, int]]:
    return {(int(x), int(y)) for x, y in arr}


def _run_patch_stage(case: RegressionCase, tmp_path: Path, zoo_model: str) -> Path:
    """Invoke ``wsinsight patch`` on a single slide; return the patches dir."""
    out_dir = tmp_path / PATCH_RUN_DIRNAME
    cmd = [
        sys.executable,
        "-m",
        "wsinsight",
        "patch",
        "-i",
        str(case.path),
        "-o",
        str(out_dir),
        "-z",
        zoo_model,
        "--overwrite",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        pytest.fail(
            f"`wsinsight patch` failed for {case.slide_id}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return out_dir / "patches"


def test_patch_coords_match_golden(case: RegressionCase, tmp_path, request):
    if not case.exists:
        pytest.skip(f"Slide not on disk: {case.path}")
    golden = _golden_path(case)
    if not golden.is_file():
        pytest.skip(
            f"No golden patches.h5 for {case.slide_id}; "
            "run scripts/regenerate_regression_baselines.py first."
        )

    # If --run-slow was passed, regenerate fresh patches into tmp_path; otherwise
    # look for a previously generated run pointed at by --patch-output-dir.
    if request.config.getoption("--run-slow", default=False):
        zoo_model = request.config.getoption("--zoo-model", default=None)
        if not zoo_model:
            pytest.skip("--zoo-model is required with --run-slow")
        patches_dir = _run_patch_stage(case, tmp_path, zoo_model)
    else:
        external = request.config.getoption("--patch-output-dir", default=None)
        if not external:
            pytest.skip(
                "Pass --patch-output-dir=<dir> to compare an existing run, or "
                "--run-slow --zoo-model=<path> to run the patch stage in-test."
            )
        patches_dir = Path(external) / "patches"

    # Find the slide's HDF5 inside the patches directory.
    candidates = list(patches_dir.glob(f"{case.path.stem}*.h5"))
    if not candidates:
        pytest.fail(f"No .h5 produced for {case.slide_id} in {patches_dir}")
    actual = _read_coords(candidates[0])
    expected = _read_coords(golden)

    assert actual.shape == expected.shape, (
        f"{case.slide_id}: patch count drift "
        f"(got {actual.shape[0]}, expected {expected.shape[0]})"
    )
    assert _coord_set(actual) == _coord_set(
        expected
    ), f"{case.slide_id}: patch coordinates differ from golden"
