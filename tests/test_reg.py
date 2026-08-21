"""Unit tests for the schema helpers and KD-tree object-to-object registration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from wsinsight.cli.reg import reg
from wsinsight.insightlib.region_registration import register_objects_to_objects
from wsinsight.io.schema import discover_prob_prefixes
from wsinsight.io.schema import make_object_prefix
from wsinsight.io.schema import make_region_prefix
from wsinsight.io.schema import resolve_no_tag_prefix

# ---------------------------------------------------------------------------
# schema.py
# ---------------------------------------------------------------------------


class TestMakePrefix:
    def test_region_no_tag(self) -> None:
        assert make_region_prefix("") == "region_"

    def test_region_with_tag(self) -> None:
        assert make_region_prefix("foo") == "region_foo_"
        assert make_region_prefix("a_b_42") == "region_a_b_42_"

    def test_object_no_tag(self) -> None:
        assert make_object_prefix("") == "object_"

    def test_object_with_tag(self) -> None:
        assert make_object_prefix("ki67") == "object_ki67_"

    def test_invalid_tag_uppercase(self) -> None:
        with pytest.raises(ValueError, match=r"--tag must match"):
            make_region_prefix("Foo")

    def test_invalid_tag_dash(self) -> None:
        with pytest.raises(ValueError, match=r"--tag must match"):
            make_object_prefix("a-b")

    def test_invalid_tag_space(self) -> None:
        with pytest.raises(ValueError, match=r"--tag must match"):
            make_region_prefix("a b")


class TestResolveNoTagPrefix:
    def test_empty_returns_bare(self) -> None:
        assert resolve_no_tag_prefix("region", set()) == "region_"
        assert resolve_no_tag_prefix("object", set()) == "object_"

    def test_unrelated_columns_return_bare(self) -> None:
        cols = {"minx", "miny", "prob_tumor", "object_other_thing"}
        assert resolve_no_tag_prefix("region", cols) == "region_"

    def test_bare_present_bumps_to_one(self) -> None:
        cols = {"region_prob_tumor"}
        assert resolve_no_tag_prefix("region", cols) == "region_1_"

    def test_bare_and_one_present_bumps_to_two(self) -> None:
        cols = {"region_prob_a", "region_1_prob_b"}
        assert resolve_no_tag_prefix("region", cols) == "region_2_"

    def test_gap_fill(self) -> None:
        # Bare and 2 present, 1 free.
        cols = {"region_prob_a", "region_2_prob_b"}
        assert resolve_no_tag_prefix("region", cols) == "region_1_"

    def test_gap_fill_higher(self) -> None:
        cols = {"region_prob_a", "region_1_prob_b", "region_3_prob_c"}
        assert resolve_no_tag_prefix("region", cols) == "region_2_"

    def test_object_kind_isolated_from_region(self) -> None:
        cols = {"region_prob_a", "region_1_prob_b"}
        # object namespace untouched -> bare available.
        assert resolve_no_tag_prefix("object", cols) == "object_"

    def test_tagged_columns_ignored(self) -> None:
        # region_foo_prob_X must NOT count as a numeric bump slot.
        cols = {"region_foo_prob_a"}
        assert resolve_no_tag_prefix("region", cols) == "region_"

    def test_invalid_kind(self) -> None:
        with pytest.raises(ValueError):
            resolve_no_tag_prefix("blob", set())


class TestDiscoverProbPrefixes:
    def test_basic(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "minx": [0],
                "miny": [0],
                "width": [1],
                "height": [1],
                "prob_tumor": [0.1],
                "prob_stroma": [0.9],
                "region_prob_tumor": [0.5],
                "region_foo_prob_tumor": [0.4],
                "object_1_prob_pos": [0.7],
                "noise_col": [0],
            }
        )
        p = tmp_path / "slide1.csv"
        df.to_csv(p, index=False)
        result = discover_prob_prefixes([p])
        assert result == sorted(
            ["prob", "region_prob", "region_foo_prob", "object_1_prob"]
        )

    def test_multi_csv_union(self, tmp_path: Path) -> None:
        a = tmp_path / "a.csv"
        b = tmp_path / "b.csv"
        pd.DataFrame({"prob_x": [0.1]}).to_csv(a, index=False)
        pd.DataFrame({"region_prob_x": [0.1]}).to_csv(b, index=False)
        assert discover_prob_prefixes([a, b]) == ["prob", "region_prob"]

    def test_no_prob(self, tmp_path: Path) -> None:
        p = tmp_path / "x.csv"
        pd.DataFrame({"minx": [0], "miny": [0]}).to_csv(p, index=False)
        assert discover_prob_prefixes([p]) == []


# ---------------------------------------------------------------------------
# register_objects_to_objects
# ---------------------------------------------------------------------------


def _bbox_df(
    centers: list[tuple[float, float]], extra: dict | None = None
) -> pd.DataFrame:
    """Build a DataFrame whose objects are 2x2 boxes centred at `centers`."""
    cols: dict[str, list] = {
        "minx": [cx - 1 for cx, _ in centers],
        "miny": [cy - 1 for _, cy in centers],
        "width": [2.0] * len(centers),
        "height": [2.0] * len(centers),
    }
    if extra:
        cols.update(extra)
    return pd.DataFrame(cols)


class TestRegisterObjectsToObjects:
    def test_within_radius_matches(self) -> None:
        primary = _bbox_df([(0, 0), (10, 10)])
        secondary = _bbox_df(
            [(0.5, 0.5), (10.0, 10.5)],
            extra={"prob_pos": [0.7, 0.3], "prob_neg": [0.3, 0.7]},
        )
        # 1 um/px so radius_px == 5.
        out, rate = register_objects_to_objects(
            primary,
            secondary,
            radius_um=5.0,
            spacing_um_px=1.0,
            out_prefix="object_",
        )
        assert rate == pytest.approx(1.0)
        assert "object_prob_pos" in out.columns
        assert out["object_prob_pos"].tolist() == [0.7, 0.3]
        assert out["object_prob_neg"].tolist() == [0.3, 0.7]

    def test_out_of_radius_unmatched(self) -> None:
        primary = _bbox_df([(0, 0), (100, 100)])
        secondary = _bbox_df(
            [(0.0, 0.0)],
            extra={"prob_pos": [0.9]},
        )
        out, rate = register_objects_to_objects(
            primary,
            secondary,
            radius_um=5.0,
            spacing_um_px=1.0,
            out_prefix="object_",
        )
        assert rate == pytest.approx(0.5)
        # First matched, second NaN.
        assert out["object_prob_pos"].iloc[0] == pytest.approx(0.9)
        assert np.isnan(out["object_prob_pos"].iloc[1])

    def test_empty_primary(self) -> None:
        primary = pd.DataFrame({"minx": [], "miny": [], "width": [], "height": []})
        secondary = _bbox_df([(0, 0)], extra={"prob_pos": [1.0]})
        out, rate = register_objects_to_objects(
            primary,
            secondary,
            radius_um=5.0,
            spacing_um_px=1.0,
        )
        assert rate == 0.0
        assert "object_prob_pos" in out.columns
        assert len(out) == 0

    def test_empty_secondary(self) -> None:
        primary = _bbox_df([(0, 0)])
        secondary = pd.DataFrame(
            {"minx": [], "miny": [], "width": [], "height": [], "prob_pos": []}
        )
        out, rate = register_objects_to_objects(
            primary,
            secondary,
            radius_um=5.0,
            spacing_um_px=1.0,
        )
        assert rate == 0.0
        assert np.isnan(out["object_prob_pos"].iloc[0])

    def test_spacing_conversion(self) -> None:
        # 10-px gap, radius 5 um, 0.25 um/px -> radius_px = 20 -> match.
        primary = _bbox_df([(0, 0)])
        secondary = _bbox_df([(10, 0)], extra={"prob_pos": [0.5]})
        _, rate = register_objects_to_objects(
            primary,
            secondary,
            radius_um=5.0,
            spacing_um_px=0.25,
        )
        assert rate == pytest.approx(1.0)
        # Same 10-px gap, but 1 um/px -> radius_px = 5 -> no match.
        primary = _bbox_df([(0, 0)])
        secondary = _bbox_df([(10, 0)], extra={"prob_pos": [0.5]})
        _, rate2 = register_objects_to_objects(
            primary,
            secondary,
            radius_um=5.0,
            spacing_um_px=1.0,
        )
        assert rate2 == pytest.approx(0.0)

    def test_invalid_radius(self) -> None:
        primary = _bbox_df([(0, 0)])
        secondary = _bbox_df([(0, 0)])
        with pytest.raises(ValueError):
            register_objects_to_objects(
                primary,
                secondary,
                radius_um=0.0,
                spacing_um_px=1.0,
            )
        with pytest.raises(ValueError):
            register_objects_to_objects(
                primary,
                secondary,
                radius_um=5.0,
                spacing_um_px=0.0,
            )

    def test_custom_prefix(self) -> None:
        primary = _bbox_df([(0, 0)])
        secondary = _bbox_df([(0.1, 0.1)], extra={"prob_pos": [0.6]})
        out, _ = register_objects_to_objects(
            primary,
            secondary,
            radius_um=5.0,
            spacing_um_px=1.0,
            out_prefix="object_ki67_",
        )
        assert "object_ki67_prob_pos" in out.columns
        assert out["object_ki67_prob_pos"].iloc[0] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# CLI surface — validation
# ---------------------------------------------------------------------------


def _make_results(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    rd = tmp_path / name
    (rd / "model-outputs-csv").mkdir(parents=True)
    df.to_csv(rd / "model-outputs-csv" / "slide1.csv", index=False)
    return rd


class TestRegCLIValidation:
    def test_xor_both_missing(self, tmp_path: Path) -> None:
        rd = _make_results(tmp_path, "obj", _bbox_df([(0, 0)]))
        result = CliRunner().invoke(reg, ["-o", str(rd)])
        assert result.exit_code != 0
        assert "Exactly one of" in result.output

    def test_xor_both_set(self, tmp_path: Path) -> None:
        primary = _make_results(tmp_path, "obj", _bbox_df([(0, 0)]))
        regd = _make_results(
            tmp_path,
            "reg",
            _bbox_df([(0, 0)], extra={"prob_x": [0.5]}),
        )
        objd = _make_results(
            tmp_path,
            "obj2",
            _bbox_df([(0, 0)], extra={"prob_x": [0.5]}),
        )
        result = CliRunner().invoke(
            reg,
            ["-o", str(primary), "-r", str(regd), "-c", str(objd)],
        )
        assert result.exit_code != 0
        assert "Exactly one of" in result.output

    def test_invalid_tag_rejected(self, tmp_path: Path) -> None:
        primary = _make_results(tmp_path, "obj", _bbox_df([(0, 0)]))
        objd = _make_results(
            tmp_path,
            "obj2",
            _bbox_df([(0, 0)], extra={"prob_x": [0.5]}),
        )
        result = CliRunner().invoke(
            reg,
            ["-o", str(primary), "-c", str(objd), "--tag", "Foo"],
        )
        assert result.exit_code != 0
        assert "--tag must match" in result.output


class TestRegCLIEndToEnd:
    def test_object_to_object_run_writes_columns(self, tmp_path: Path) -> None:
        primary = _make_results(tmp_path, "primary", _bbox_df([(0, 0), (5, 5)]))
        secondary = _make_results(
            tmp_path,
            "secondary",
            _bbox_df(
                [(0.1, 0.1), (5.1, 5.1)],
                extra={"prob_pos": [0.8, 0.2]},
            ),
        )
        result = CliRunner().invoke(
            reg,
            [
                "-o",
                str(primary),
                "-c",
                str(secondary),
                "--radius-um",
                "5",
                "--spacing-um-px",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        out = pd.read_csv(primary / "model-outputs-csv" / "slide1.csv")
        assert "object_prob_pos" in out.columns
        assert out["object_prob_pos"].tolist() == [0.8, 0.2]

    def test_auto_bump_no_tag(self, tmp_path: Path) -> None:
        # Pre-seed with an existing object_prob_* column to force bump.
        df = _bbox_df([(0, 0)])
        df["object_prob_existing"] = [0.5]
        primary = _make_results(tmp_path, "primary", df)
        secondary = _make_results(
            tmp_path,
            "secondary",
            _bbox_df([(0.1, 0.1)], extra={"prob_pos": [0.9]}),
        )
        result = CliRunner().invoke(
            reg,
            [
                "-o",
                str(primary),
                "-c",
                str(secondary),
                "--radius-um",
                "5",
                "--spacing-um-px",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        out = pd.read_csv(primary / "model-outputs-csv" / "slide1.csv")
        assert "object_1_prob_pos" in out.columns
        # Existing column preserved.
        assert "object_prob_existing" in out.columns

    def test_tagged_collision_skips(self, tmp_path: Path) -> None:
        df = _bbox_df([(0, 0)])
        df["object_foo_prob_pos"] = [0.5]
        primary = _make_results(tmp_path, "primary", df)
        secondary = _make_results(
            tmp_path,
            "secondary",
            _bbox_df([(0.1, 0.1)], extra={"prob_pos": [0.9]}),
        )
        result = CliRunner().invoke(
            reg,
            [
                "-o",
                str(primary),
                "-c",
                str(secondary),
                "--tag",
                "foo",
                "--radius-um",
                "5",
                "--spacing-um-px",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "skipped: 1" in result.output
        out = pd.read_csv(primary / "model-outputs-csv" / "slide1.csv")
        # Untouched: original 0.5 still there.
        assert out["object_foo_prob_pos"].iloc[0] == 0.5

    def test_overwrite_replaces(self, tmp_path: Path) -> None:
        df = _bbox_df([(0, 0)])
        df["object_foo_prob_pos"] = [0.5]
        primary = _make_results(tmp_path, "primary", df)
        secondary = _make_results(
            tmp_path,
            "secondary",
            _bbox_df([(0.1, 0.1)], extra={"prob_pos": [0.9]}),
        )
        result = CliRunner().invoke(
            reg,
            [
                "-o",
                str(primary),
                "-c",
                str(secondary),
                "--tag",
                "foo",
                "--overwrite",
                "--radius-um",
                "5",
                "--spacing-um-px",
                "1",
            ],
        )
        assert result.exit_code == 0, result.output
        out = pd.read_csv(primary / "model-outputs-csv" / "slide1.csv")
        assert out["object_foo_prob_pos"].iloc[0] == pytest.approx(0.9)
