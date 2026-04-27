"""Shared fixtures for the regression suite.

The suite is data-driven: a TOML file (default ``tests/regression/cases.toml``,
overridable via ``WSINSIGHT_REGRESSION_CASES``) lists the slides under test.
Cases whose ``path`` does not exist on disk are skipped, so the suite is safe
to run on machines without the full TCGA dataset.
"""

from __future__ import annotations

import os
try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - dev envs on older Python
    import tomli as tomllib  # type: ignore[no-redef]
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


REGRESSION_DIR = Path(__file__).parent
REPO_ROOT = REGRESSION_DIR.parents[1]
DEFAULT_CASES_FILE = REGRESSION_DIR / "cases.toml"
FIXTURES_DIR = REGRESSION_DIR / "fixtures"


def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


@dataclass(frozen=True)
class RegressionCase:
    slide_id: str
    path: Path
    expected_appmag: float | None = None
    expected_mpp: float | None = None
    mpp_atol: float = 0.01
    expects_appmag_fallback: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def fixture_dir(self) -> Path:
        return FIXTURES_DIR / self.slide_id

    @property
    def exists(self) -> bool:
        return self.path.is_file()


def _cases_file() -> Path:
    override = os.environ.get("WSINSIGHT_REGRESSION_CASES")
    return Path(override) if override else DEFAULT_CASES_FILE


def load_cases() -> list[RegressionCase]:
    cases_file = _cases_file()
    if not cases_file.is_file():
        return []
    with cases_file.open("rb") as f:
        data = tomllib.load(f)
    raw = data.get("case", [])
    cases: list[RegressionCase] = []
    for entry in raw:
        known = {"slide_id", "path", "expected_appmag", "expected_mpp",
                 "mpp_atol", "expects_appmag_fallback"}
        extra = {k: v for k, v in entry.items() if k not in known}
        cases.append(RegressionCase(
            slide_id=entry["slide_id"],
            path=_resolve_path(entry["path"]),
            expected_appmag=entry.get("expected_appmag"),
            expected_mpp=entry.get("expected_mpp"),
            mpp_atol=entry.get("mpp_atol", 0.01),
            expects_appmag_fallback=entry.get("expects_appmag_fallback", False),
            extra=extra,
        ))
    return cases


def _ids(cases: list[RegressionCase]) -> list[str]:
    return [c.slide_id for c in cases]


@pytest.fixture(scope="session")
def all_cases() -> list[RegressionCase]:
    return load_cases()


def pytest_addoption(parser):
    group = parser.getgroup("wsinsight-regression")
    group.addoption(
        "--run-slow", action="store_true", default=False,
        help="Run slow regression tests (full patch / infer pipeline).",
    )
    group.addoption(
        "--zoo-model", default=None,
        help="Path to a wsinfer-zoo model dir, required by --run-slow.",
    )
    group.addoption(
        "--patch-output-dir", default=None,
        help="Existing wsinsight patch output dir to compare against goldens.",
    )
    group.addoption(
        "--infer-output-dir", default=None,
        help="Existing wsinsight infer output dir to compare against goldens.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "regression: WSInsight per-slide regression test.",
    )
    config.addinivalue_line(
        "markers", "slow: requires --run-slow or external pre-computed outputs.",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip regression items when the case file is empty / absent,
    and skip ``slow`` items unless ``--run-slow`` or an external output dir
    is provided."""
    have_cases = bool(load_cases())
    no_cases_skip = pytest.mark.skip(
        reason="No regression cases configured (see tests/regression/cases.toml)."
    )
    run_slow = config.getoption("--run-slow")
    have_external = bool(
        config.getoption("--patch-output-dir")
        or config.getoption("--infer-output-dir")
    )
    slow_skip = pytest.mark.skip(
        reason="Slow test; pass --run-slow or --patch-output-dir/--infer-output-dir."
    )

    for item in items:
        if "regression" in item.keywords and not have_cases:
            item.add_marker(no_cases_skip)
        if "slow" in item.keywords and not (run_slow or have_external):
            item.add_marker(slow_skip)


# Parametrize fixtures shared across test modules. Each test that needs a
# concrete case asks for the ``case`` fixture.
def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = load_cases()
        metafunc.parametrize("case", cases, ids=_ids(cases))
