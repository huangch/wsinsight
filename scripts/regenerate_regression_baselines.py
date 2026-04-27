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
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[1]
REGRESSION_DIR = REPO_ROOT / "tests" / "regression"
DEFAULT_CASES_FILE = REGRESSION_DIR / "cases.toml"
FIXTURES_DIR = REGRESSION_DIR / "fixtures"


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (REPO_ROOT / pp)


@dataclass(frozen=True)
class RegressionCase:
    slide_id: str
    path: Path

    @property
    def fixture_dir(self) -> Path:
        return FIXTURES_DIR / self.slide_id

    @property
    def exists(self) -> bool:
        return self.path.is_file()


def load_cases() -> list[RegressionCase]:
    cases_file = Path(
        os.environ.get("WSINSIGHT_REGRESSION_CASES", str(DEFAULT_CASES_FILE))
    )
    if not cases_file.is_file():
        return []
    with cases_file.open("rb") as f:
        data = tomllib.load(f)
    return [
        RegressionCase(slide_id=e["slide_id"], path=_resolve_path(e["path"]))
        for e in data.get("case", [])
    ]


def _stage_slides(cases: list[RegressionCase], staging: Path) -> Path:
    """Symlink case slides into a single directory for ``-i, --wsi-dir``."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for c in cases:
        (staging / c.path.name).symlink_to(c.path)
    return staging


def _run_patch(staging: Path, results_dir: Path, zoo_model: str) -> None:
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    cmd = [
        sys.executable, "-m", "wsinsight", "patch",
        "-i", str(staging),
        "-o", str(results_dir),
        "-z", zoo_model,
    ]
    print(f"\n[+] patch: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _run_infer(results_dir: Path, zoo_model: str) -> None:
    cmd = [
        sys.executable, "-m", "wsinsight", "infer",
        "-o", str(results_dir),
        "-z", zoo_model,
    ]
    print(f"\n[+] infer: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _copy_patch_golden(case: RegressionCase, results_dir: Path) -> None:
    src_dir = results_dir / "patches"
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
        os.environ["WSINSIGHT_REGRESSION_CASES"] = args.cases

    cases = load_cases()
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c.slide_id in wanted]
    cases = [c for c in cases if c.exists]
    if not cases:
        print("No cases to regenerate (none exist on disk).", file=sys.stderr)
        return 1

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    staging = workdir / "slides"
    results_dir = workdir / "results"
    print(f"Workdir: {workdir}")
    print(f"Fixtures dir: {FIXTURES_DIR}")
    print(f"Cases: {[c.slide_id for c in cases]}")

    _stage_slides(cases, staging)
    _run_patch(staging, results_dir, args.zoo_model)
    for c in cases:
        _copy_patch_golden(c, results_dir)

    if not args.skip_infer:
        _run_infer(results_dir, args.zoo_model)
        for c in cases:
            _copy_infer_golden(c, results_dir)

    print("\nDone. Review the diff under tests/regression/fixtures/ and commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
