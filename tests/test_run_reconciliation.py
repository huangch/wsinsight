"""Tests for the run command's completeness reconciliation logic.

These tests verify that `wsinsight run`:
1. Detects missing patches/CSVs from prior incomplete runs
2. Re-processes only the missing slides
3. Reports clear status summaries
"""

from __future__ import annotations

from pathlib import Path

from wsinsight.cli.run import SlideStatus
from wsinsight.cli.run import _log_reconciliation_summary
from wsinsight.cli.run import _scan_existing_artifacts
from wsinsight.uri_path import URIPath


def _touch(path: URIPath) -> None:
    """Create an empty file.

    URIPath implements only the slice of the Path API the package needs, and
    touch() is not part of it; open("wb") is the supported equivalent.
    """
    with path.open("wb"):
        pass


class TestSlideStatus:
    """Tests for SlideStatus dataclass properties."""

    def test_missing_patches(self) -> None:
        status = SlideStatus()
        status.requested = {"a", "b", "c"}
        status.existing_patches = {"a", "b"}
        assert status.missing_patches == {"c"}

    def test_needs_inference(self) -> None:
        status = SlideStatus()
        status.existing_patches = {"a", "b", "c"}
        status.existing_csvs = {"a"}
        assert status.needs_inference == {"b", "c"}

    def test_completed(self) -> None:
        status = SlideStatus()
        status.existing_patches = {"a", "b", "c"}
        status.existing_csvs = {"a", "b", "d"}  # d has no patch
        assert status.completed == {"a", "b"}

    def test_failed_patch(self) -> None:
        status = SlideStatus()
        status.patched_this_run = {"a", "b", "c"}
        status.existing_patches = {"a", "c"}  # b failed
        assert status.failed_patch == {"b"}

    def test_failed_infer(self) -> None:
        status = SlideStatus()
        status.inferred_this_run = {"a", "b", "c"}
        status.existing_csvs = {"a"}  # b, c failed
        assert status.failed_infer == {"b", "c"}


class TestScanExistingArtifacts:
    """Tests for _scan_existing_artifacts."""

    def test_empty_results_dir(self, tmp_path: Path) -> None:
        results_dir = URIPath(str(tmp_path / "results"))
        results_dir.mkdir(parents=True, exist_ok=True)
        slide_paths = [URIPath(f"slide_{i}.svs") for i in range(3)]

        status = _scan_existing_artifacts(slide_paths, results_dir)

        assert status.requested == {"slide_0", "slide_1", "slide_2"}
        assert status.existing_patches == set()
        assert status.existing_csvs == set()

    def test_partial_patches(self, tmp_path: Path) -> None:
        results_dir = URIPath(str(tmp_path / "results"))
        patches_dir = results_dir / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)

        # Create 2 of 3 patches
        _touch(patches_dir / "slide_0.h5")
        _touch(patches_dir / "slide_2.h5")

        slide_paths = [URIPath(f"slide_{i}.svs") for i in range(3)]
        status = _scan_existing_artifacts(slide_paths, results_dir)

        assert status.requested == {"slide_0", "slide_1", "slide_2"}
        assert status.existing_patches == {"slide_0", "slide_2"}
        assert status.missing_patches == {"slide_1"}

    def test_partial_csvs(self, tmp_path: Path) -> None:
        results_dir = URIPath(str(tmp_path / "results"))
        patches_dir = results_dir / "patches"
        csv_dir = results_dir / "model-outputs-csv"
        patches_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)

        # All patches exist
        for i in range(3):
            _touch(patches_dir / f"slide_{i}.h5")

        # Only 1 CSV exists
        _touch(csv_dir / "slide_1.csv")

        slide_paths = [URIPath(f"slide_{i}.svs") for i in range(3)]
        status = _scan_existing_artifacts(slide_paths, results_dir)

        assert status.existing_patches == {"slide_0", "slide_1", "slide_2"}
        assert status.existing_csvs == {"slide_1"}
        assert status.needs_inference == {"slide_0", "slide_2"}
        assert status.completed == {"slide_1"}

    def test_ignores_non_h5_files(self, tmp_path: Path) -> None:
        results_dir = URIPath(str(tmp_path / "results"))
        patches_dir = results_dir / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)

        _touch(patches_dir / "slide_0.h5")
        _touch(patches_dir / "slide_1.txt")  # Not an H5
        _touch(patches_dir / "readme.md")

        slide_paths = [URIPath("slide_0.svs"), URIPath("slide_1.svs")]
        status = _scan_existing_artifacts(slide_paths, results_dir)

        assert status.existing_patches == {"slide_0"}

    def test_ignores_non_csv_files(self, tmp_path: Path) -> None:
        results_dir = URIPath(str(tmp_path / "results"))
        csv_dir = results_dir / "model-outputs-csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        _touch(csv_dir / "slide_0.csv")
        _touch(csv_dir / "slide_1.txt")  # Not a CSV
        _touch(csv_dir / "metadata.json")

        slide_paths = [URIPath("slide_0.svs"), URIPath("slide_1.svs")]
        status = _scan_existing_artifacts(slide_paths, results_dir)

        assert status.existing_csvs == {"slide_0"}


class TestLogReconciliationSummary:
    """Tests for _log_reconciliation_summary (smoke tests)."""

    def test_pre_patch_stage_runs(self, capsys) -> None:
        status = SlideStatus()
        status.requested = {"a", "b", "c"}
        status.existing_patches = {"a"}
        status.existing_csvs = set()

        _log_reconciliation_summary(status, stage="pre-patch")

        captured = capsys.readouterr()
        assert "Requested slides:" in captured.out
        assert "3" in captured.out

    def test_final_stage_all_complete(self, capsys) -> None:
        status = SlideStatus()
        status.requested = {"a", "b"}
        status.existing_patches = {"a", "b"}
        status.existing_csvs = {"a", "b"}

        _log_reconciliation_summary(status, stage="final")

        captured = capsys.readouterr()
        assert "All slides completed successfully!" in captured.out

    def test_final_stage_with_failures(self, capsys) -> None:
        status = SlideStatus()
        status.requested = {"a", "b", "c"}
        status.existing_patches = {"a", "b"}  # c missing patch
        status.existing_csvs = {"a"}  # b missing csv

        _log_reconciliation_summary(status, stage="final")

        captured = capsys.readouterr()
        assert "Still incomplete:" in captured.out


class TestReconciliationScenarios:
    """Integration-style tests for reconciliation scenarios."""

    def test_scenario_12_slides_11_patches_rerun_patches_missing(
        self, tmp_path: Path
    ) -> None:
        """Reproduce the bug: 12 requested slides, 11 patches exist.

        Before fix: rerun processes 0 slides (inference sees 11 patches, all have CSVs).
        After fix: rerun should detect missing patch and attempt to create it.
        """
        results_dir = URIPath(str(tmp_path / "results"))
        patches_dir = results_dir / "patches"
        csv_dir = results_dir / "model-outputs-csv"
        patches_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)

        # Simulate 11 of 12 patches and CSVs
        for i in range(12):
            if i != 5:  # slide_5 is missing
                _touch(patches_dir / f"slide_{i}.h5")
                _touch(csv_dir / f"slide_{i}.csv")

        slide_paths = [URIPath(f"slide_{i}.svs") for i in range(12)]
        status = _scan_existing_artifacts(slide_paths, results_dir)

        # Verify detection
        assert len(status.requested) == 12
        assert len(status.existing_patches) == 11
        assert len(status.existing_csvs) == 11
        assert status.missing_patches == {"slide_5"}
        assert status.needs_inference == set()  # slide_5 has no patch yet
        assert len(status.completed) == 11

    def test_scenario_missing_csv_but_has_patch(self, tmp_path: Path) -> None:
        """Slide has patch but inference failed — needs inference only."""
        results_dir = URIPath(str(tmp_path / "results"))
        patches_dir = results_dir / "patches"
        csv_dir = results_dir / "model-outputs-csv"
        patches_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)

        # All patches exist
        for i in range(5):
            _touch(patches_dir / f"slide_{i}.h5")

        # Only some CSVs
        _touch(csv_dir / "slide_0.csv")
        _touch(csv_dir / "slide_2.csv")

        slide_paths = [URIPath(f"slide_{i}.svs") for i in range(5)]
        status = _scan_existing_artifacts(slide_paths, results_dir)

        assert status.missing_patches == set()  # All patches exist
        assert status.needs_inference == {"slide_1", "slide_3", "slide_4"}
        assert status.completed == {"slide_0", "slide_2"}

    def test_scenario_second_rerun_all_complete(self, tmp_path: Path) -> None:
        """After full completion, second rerun should skip everything."""
        results_dir = URIPath(str(tmp_path / "results"))
        patches_dir = results_dir / "patches"
        csv_dir = results_dir / "model-outputs-csv"
        patches_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)

        for i in range(5):
            _touch(patches_dir / f"slide_{i}.h5")
            _touch(csv_dir / f"slide_{i}.csv")

        slide_paths = [URIPath(f"slide_{i}.svs") for i in range(5)]
        status = _scan_existing_artifacts(slide_paths, results_dir)

        assert status.missing_patches == set()
        assert status.needs_inference == set()
        assert status.completed == {
            "slide_0",
            "slide_1",
            "slide_2",
            "slide_3",
            "slide_4",
        }
