"""Contracts for how a slide's pixel spacing reaches every stage.

``patch`` decides the µm-per-pixel spacing once — from the slide metadata, or
from ``--spacing-um-px`` when that is given — and records it. Every later stage
reads that record instead of re-opening the slide, so a single flag governs the
whole pipeline and the analysis stages need no slide directory at all.

Each rule below has already been broken once during development (a fallback
that silently lost to metadata, a forwarding list that outlived the parameter
it forwarded), so they are pinned here rather than left to review.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import h5py
import numpy as np
import pytest

from wsinsight.cli import run as run_module
from wsinsight.cli.cli import cli
from wsinsight.insightlib.insight_helpers import build_slide_mpp_lookup
from wsinsight.insightlib.insight_helpers import list_slides_from_patches
from wsinsight.insightlib.insight_helpers import read_patch_slide_records
from wsinsight.wsi import CannotReadSpacing
from wsinsight.wsi import get_avg_mpp

# Stages that read slide pixels, and therefore still take a slide directory.
PIXEL_STAGES = {"patch", "run", "import", "infer", "reg"}
# Stages that work from --results-dir alone.
ANALYSIS_STAGES = {"ncomp", "ecomp", "tcomp", "agg", "hplot", "niche"}


def _flags(command: str) -> set[str]:
    cmd = cli.get_command(click.Context(cli), command)
    assert cmd is not None, f"{command!r} is not a registered subcommand"
    return {opt for p in cmd.params for opt in getattr(p, "opts", [])}


def _write_patch_h5(path: Path, slide_path: str, mpp: float | None) -> None:
    """Write the subset of the patch HDF5 layout the downstream lookup reads."""
    with h5py.File(path, "w") as f:
        g = f.create_group("slide")
        g.attrs.create("slide_path", slide_path, dtype=h5py.string_dtype("utf-8"))
        if mpp is not None:
            g.attrs["slide_mpp"] = mpp
        f.create_dataset("/coords", data=np.zeros((1, 2), dtype=np.int32))


@pytest.fixture()
def results_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results"
    (d / "patches").mkdir(parents=True)
    _write_patch_h5(d / "patches" / "slideA.ome.h5", "/slides/slideA.ome.tiff", 0.2738)
    return d


# -- get_avg_mpp: the supplied value overrides, it does not merely fill in ----


def test_supplied_spacing_overrides_slide_metadata(monkeypatch):
    monkeypatch.setattr("wsinsight.wsi._BACKEND", "openslide")
    monkeypatch.setattr("wsinsight.wsi._get_mpp_openslide", lambda p: (0.25, 0.25))
    assert get_avg_mpp("/fake.svs", override_mpp=0.5) == 0.5


def test_zero_and_none_defer_to_slide_metadata(monkeypatch):
    monkeypatch.setattr("wsinsight.wsi._BACKEND", "openslide")
    monkeypatch.setattr("wsinsight.wsi._get_mpp_openslide", lambda p: (0.25, 0.25))
    assert get_avg_mpp("/fake.svs", override_mpp=0.0) == 0.25
    assert get_avg_mpp("/fake.svs") == 0.25


def test_unreadable_spacing_without_override_raises(monkeypatch):
    def boom(path):
        raise CannotReadSpacing(path)

    monkeypatch.setattr("wsinsight.wsi._BACKEND", "openslide")
    for name in ("_get_mpp_openslide", "_get_mpp_tiffslide", "_get_mpp_tifffile"):
        monkeypatch.setattr(f"wsinsight.wsi.{name}", boom)
    for name in ("_get_appmag_openslide", "_get_appmag_tiffslide"):
        monkeypatch.setattr(f"wsinsight.wsi.{name}", lambda p: None)

    with pytest.raises(CannotReadSpacing):
        get_avg_mpp("/fake.svs")
    assert get_avg_mpp("/fake.svs", override_mpp=0.5) == 0.5


# -- the record patch leaves behind -------------------------------------------


def test_lookup_is_keyed_by_both_stem_and_slide_path(results_dir):
    lookup = build_slide_mpp_lookup(results_dir)
    assert lookup["slideA.ome"] == pytest.approx(0.2738)
    assert lookup["/slides/slideA.ome.tiff"] == pytest.approx(0.2738)


def test_recovered_slide_path_round_trips_to_the_csv_name(results_dir):
    """Recovering real paths avoids the ``.ome.tiff`` double-suffix trap."""
    (slide,) = list_slides_from_patches(results_dir)
    assert slide.stem == "slideA.ome"
    assert slide.with_suffix(".csv").name == "slideA.ome.csv"


def test_missing_or_unreadable_patches_degrade_to_empty(tmp_path):
    assert build_slide_mpp_lookup(tmp_path / "absent") == {}
    assert list_slides_from_patches(tmp_path / "absent") == []

    broken = tmp_path / "broken"
    (broken / "patches").mkdir(parents=True)
    (broken / "patches" / "junk.h5").write_bytes(b"not an hdf5 file")
    assert build_slide_mpp_lookup(broken) == {}


def test_patch_h5_without_spacing_is_still_enumerable(tmp_path):
    """An older run has no ``slide_mpp``; the slide must stay discoverable."""
    d = tmp_path / "results"
    (d / "patches").mkdir(parents=True)
    _write_patch_h5(d / "patches" / "old.ome.h5", "/slides/old.ome.tiff", None)

    assert build_slide_mpp_lookup(d) == {}
    assert [p.name for p in list_slides_from_patches(d)] == ["old.ome.tiff"]
    assert read_patch_slide_records(d) == [("old.ome", "/slides/old.ome.tiff", None)]


# -- CLI surface ---------------------------------------------------------------


@pytest.mark.parametrize("command", sorted(ANALYSIS_STAGES))
def test_analysis_stages_reject_a_slide_directory(command):
    assert "--wsi-dir" not in _flags(command)
    assert "-i" not in _flags(command)


@pytest.mark.parametrize("command", sorted(PIXEL_STAGES))
def test_pixel_stages_still_take_a_slide_directory(command):
    assert "--wsi-dir" in _flags(command)


def test_spacing_override_is_offered_where_patches_are_cut():
    assert "--spacing-um-px" in _flags("patch")
    assert "--spacing-um-px" in _flags("run")


@pytest.mark.parametrize("command", sorted(ANALYSIS_STAGES))
def test_analysis_stages_do_not_take_a_spacing_flag(command):
    assert "--spacing-um-px" not in _flags(command)


def test_schema_command_replaced_describe():
    assert cli.get_command(click.Context(cli), "schema") is not None
    assert cli.get_command(click.Context(cli), "describe") is None


def test_schema_output_covers_every_command_except_itself(tmp_path):
    from click.testing import CliRunner

    out = tmp_path / "schema.json"
    result = CliRunner().invoke(cli, ["schema", "--output", str(out)])
    assert result.exit_code == 0, result.output

    emitted = json.loads(out.read_text())["commands"]
    assert "schema" not in emitted
    assert PIXEL_STAGES | ANALYSIS_STAGES <= set(emitted)


# -- run forwards only what each stage accepts --------------------------------


@pytest.mark.parametrize(
    "forward_list,stage",
    [
        ("_PATCH_PARAM_NAMES", "patch"),
        ("_INFER_PARAM_NAMES", "infer"),
        ("_NCOMP_PARAM_NAMES", "ncomp"),
        ("_ECOMP_PARAM_NAMES", "ecomp"),
        ("_TCOMP_PARAM_NAMES", "tcomp"),
        ("_AGG_PARAM_NAMES", "agg"),
        ("_HPLOT_PARAM_NAMES", "hplot"),
        ("_NICHE_PARAM_NAMES", "niche"),
    ],
)
def test_run_forwards_match_the_stage_signature(forward_list, stage):
    """``ctx.invoke`` raises TypeError on a name the stage no longer declares."""
    cmd = cli.get_command(click.Context(cli), stage)
    declared = {p.name for p in cmd.params}
    forwarded = set(getattr(run_module, forward_list))
    assert forwarded <= declared, (
        f"{forward_list} forwards {sorted(forwarded - declared)} to "
        f"`wsinsight {stage}`, which does not accept it"
    )
