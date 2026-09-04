"""`run` must hand each stage the model option that was typed, not its expansion.

`run` expands ``--zoo-model-dir`` into ``--config`` / ``--model-path`` to fail
fast, but the stages have to receive the shorthand so what they report back
matches the invocation. Forwarding both forms trips their mutual-exclusion
check, so the pair has to be blanked when the shorthand is present.
"""

from __future__ import annotations

import inspect

import pytest

from wsinsight.cli import infer as infer_mod
from wsinsight.cli import patch as patch_mod
from wsinsight.cli.run import _INFER_PARAM_NAMES
from wsinsight.cli.run import _PATCH_PARAM_NAMES
from wsinsight.cli.run import _select_kwargs

_ZOO = "/zoo/some-model/main"


@pytest.mark.parametrize(
    "names", [_PATCH_PARAM_NAMES, _INFER_PARAM_NAMES], ids=["patch", "infer"]
)
def test_shorthand_is_forwarded(names):
    assert "zoo_model_dir" in names


@pytest.mark.parametrize(
    "command", [patch_mod.patch, infer_mod.infer], ids=["patch", "infer"]
)
def test_stages_accept_the_shorthand(command):
    callback = command.callback if hasattr(command, "callback") else command
    assert "zoo_model_dir" in inspect.signature(callback).parameters


@pytest.mark.parametrize(
    "names", [_PATCH_PARAM_NAMES, _INFER_PARAM_NAMES], ids=["patch", "infer"]
)
def test_expanded_pair_is_blanked_so_the_stage_can_expand_it_itself(names):
    params = {k: None for k in set(_PATCH_PARAM_NAMES) | set(_INFER_PARAM_NAMES)}
    params.update(
        {
            "config": _ZOO + "/config.json",
            "model_path": _ZOO + "/torchscript_model.pt",
            "zoo_model_dir": _ZOO,
        }
    )
    if params["zoo_model_dir"] is not None:
        params["config"] = None
        params["model_path"] = None

    kwargs = _select_kwargs(params, names)
    assert kwargs["zoo_model_dir"] == _ZOO
    assert kwargs["config"] is None
    assert kwargs["model_path"] is None
