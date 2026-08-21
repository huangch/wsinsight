"""Guards for the CLI option-naming convention.

A standalone subcommand must not repeat its own name in its option flags:
it is ``wsinsight niche --clusters``, never ``wsinsight niche --niche-clusters``.
Only ``wsinsight run`` prefixes options with a stage name, because it
orchestrates several subcommands and has to keep their namespaces apart.

This convention has silently regressed before (option prefixes were removed and
later reappeared), so it is pinned here rather than left to review.
"""

from __future__ import annotations

import click
import pytest

from wsinsight.cli.cli import cli


def _subcommands() -> list[tuple[str, click.Command]]:
    """Every registered subcommand except the ``run`` orchestrator."""
    return [(name, cmd) for name, cmd in cli.commands.items() if name != "run"]


def _flags(cmd: click.Command) -> list[str]:
    """All long-form option strings declared on a command."""
    return [
        opt
        for param in cmd.params
        for opt in getattr(param, "opts", [])
        if opt.startswith("--")
    ]


@pytest.mark.parametrize(
    "name,cmd", _subcommands(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_standalone_command_does_not_prefix_its_own_options(name, cmd):
    offenders = [flag for flag in _flags(cmd) if flag.startswith(f"--{name}-")]
    assert not offenders, (
        f"`wsinsight {name}` declares self-prefixed option(s) {offenders}. "
        f"Standalone subcommands take unprefixed flags (e.g. --k, --clusters); "
        f"only `wsinsight run` carries the stage prefix (--{name}-...)."
    )


def test_run_forwards_are_prefixed_and_resolvable():
    """Every stage option `run` forwards must exist on the target subcommand.

    ``run`` passes parameters through by name via ``_select_kwargs``, so a
    mismatch between the two signatures is a runtime TypeError rather than a
    startup error -- worth catching here.
    """
    from wsinsight.cli import run as run_mod

    run_cmd = cli.commands["run"]
    run_params = {p.name for p in run_cmd.params}

    forwarded = {
        "hplot": getattr(run_mod, "_HPLOT_PARAM_NAMES", ()),
        "ncomp": getattr(run_mod, "_NCOMP_PARAM_NAMES", ()),
        "ecomp": getattr(run_mod, "_ECOMP_PARAM_NAMES", ()),
        "tcomp": getattr(run_mod, "_TCOMP_PARAM_NAMES", ()),
        "niche": getattr(run_mod, "_NICHE_PARAM_NAMES", ()),
        "agg": getattr(run_mod, "_AGG_PARAM_NAMES", ()),
    }

    for stage, names in forwarded.items():
        if not names:
            continue
        target = cli.commands.get(stage)
        assert target is not None, f"run forwards to unknown subcommand {stage!r}"
        target_params = {p.name for p in target.params}
        for pname in names:
            assert pname in run_params, (
                f"_{stage.upper()}_PARAM_NAMES lists {pname!r}, "
                f"which is not a parameter of `run`."
            )
            assert pname in target_params, (
                f"_{stage.upper()}_PARAM_NAMES lists {pname!r}, "
                f"which is not a parameter of `wsinsight {stage}`."
            )
