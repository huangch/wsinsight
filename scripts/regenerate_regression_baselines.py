#!/usr/bin/env python
"""Regenerate regression baselines for WSInsight.

Workflow:
  1. Check out a known-good revision of wsinsight.
  2. Run this script:

       python scripts/regenerate_regression_baselines.py \\
         --zoo-model /app/zoo/huangch/CellViT-SAM-H-x40/main \\
         --workdir /tmp/wsinsight-baseline

  3. Commit the updated files under tests/regression/fixtures/.
  4. Check out your branch and run:

       pytest tests/regression -m regression
       pytest tests/regression -m regression --run-slow \\
         --zoo-model /app/zoo/huangch/CellViT-SAM-H-x40/main

By default cases are read from ``tests/regression/cases.toml`` (override via
``WSINSIGHT_REGRESSION_CASES`` or ``--cases``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Make ``tests`` importable as a package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.regression.conftest import (  # noqa: E402
    FIXTURES_DIR, RegressionCase, load_cases,
)


def _run_patch(case: RegressionCase, workdir: Path, zoo_model: str) -> Path:
    out = workdir / case.slide_id / "patch"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "wsinsight", "patch",
        "-i", str(case.path),
        "-o", str(out),
        "-z", zoo_model,
        "--overwrite",
    ]
    print(f"\n[+] {case.slide_id}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out


def _run_infer(case: RegressionCase, patch_dir: Path, zoo_model: str) -> Path:
    cmd = [
        sys.executable, "-m", "wsinsight", "infer",
        "-i", str(case.path),
        "-o", str(patch_dir),
        "-z", zoo_model,
    ]
    print(f"\n[+] {case.slide_id}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return patch_dir


def _copy_patch_golden(case: RegressionCase, patch_dir: Path) -> None:
    src_dir = patch_dir / "patches"
    hits = list(src_dir.glob(f"{case.path.stem}*.h5"))
    if not hits:
        print(f"  ! No .h5 produced for {case.slide_id}; skipping patch golden.")
        return
    case.fixture_dir.mkdir(parents=True, exist_ok=True)
    dst = case.fixture_dir / "patches.h5"
    shutil.copy2(hits[0], dst)
    print(f"  -> wrote {dst}")


def _copy_infer_golden(case: RegressionCase, results_dir: Path) -> None:
    for sub in ("model-outputs-csv", "model-outputs"):
        src_dir = results_dir / sub
        if not src_dir.is_dir():
            continue
        hits = list(src_dir.glob(f"{case.path.stem}*.csv"))
        if hits:
            case.fixture_dir.mkdir(parents=True, exist_ok=True)
            dst = case.fixture_dir / "inference.csv"
            shutil.copy2(hits[0], dst)
            print(f"  -> wrote {dst}")
            return
    print(f"  ! No inference CSV for {case.slide_id}; skipping inference golden.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zoo-model", required=True)
    parser.add_argument("--workdir", default="/tmp/wsinsight-baseline",
                        help="Scratch dir for intermediate patch/infer outputs.")
    parser.add_argument("--cases", default=None,
                        help="Override cases.toml path.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated slide_ids to regenerate.")
    parser.add_argument("--skip-infer", action="store_true",
                        help="Only regenerate patch goldens.")
    args = parser.parse_args()

    if args.cases:
        import os
        os.environ["WSINSIGHT_REGRESSION_CASES"] = args.cases

    cases = load_cases()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c.slide_id in wanted]
    if not cases:
        print("No cases to regenerate.", file=sys.stderr)
        return 1

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"Workdir: {workdir}")
    print(f"Fixtures dir: {FIXTURES_DIR}")

    for case in cases:
        if not case.exists:
            print(f"\n[skip] {case.slide_id}: file not found -> {case.path}")
            continue
        patch_dir = _run_patch(case, workdir, args.zoo_model)
        _copy_patch_golden(case, patch_dir)
        if not args.skip_infer:
            results_dir = _run_infer(case, patch_dir, args.zoo_model)
            _copy_infer_golden(case, results_dir)

    print("\nDone. Review the diff under tests/regression/fixtures/ and commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
