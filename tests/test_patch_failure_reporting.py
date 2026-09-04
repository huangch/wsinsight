"""A slide that fails segmentation must be reported, not silently counted.

Before this contract existed, `segment_and_patch_directory_of_slides` swallowed
every per-slide exception and returned None, so `run` marked the slide as
patched and moved on to inference against whatever stale patches were on disk.
"""

from __future__ import annotations

from unittest import mock

import pytest

from wsinsight.patchlib import pipeline


@pytest.fixture
def slide_dir(tmp_path):
    d = tmp_path / "slides"
    d.mkdir()
    for name in ("a.tiff", "b.tiff", "c.tiff"):
        (d / name).write_bytes(b"not a real slide")
    return d


def _run(slide_dir, tmp_path, one_slide):
    with mock.patch.object(pipeline, "segment_and_patch_one_slide", one_slide):
        return pipeline.segment_and_patch_directory_of_slides(
            wsi_dir=slide_dir,
            slide_paths=[slide_dir / n for n in ("a.tiff", "b.tiff", "c.tiff")],
            save_dir=tmp_path / "out",
            qupath_measurement_detection_dir=None,
            qupath_geojson_detection_dir=None,
            qupath_geojson_annotation_dir=None,
            patch_size_px=256,
            patch_spacing_um_px=0.5,
        )


def test_failed_slides_are_returned(slide_dir, tmp_path):
    def one_slide(slide_path, **kwargs):
        if "b" in str(slide_path):
            raise RuntimeError("segmentation blew up")

    assert _run(slide_dir, tmp_path, one_slide) == ["b"]


def test_a_failure_does_not_abandon_the_rest_of_the_cohort(slide_dir, tmp_path):
    seen = []

    def one_slide(slide_path, **kwargs):
        seen.append(str(slide_path))
        raise RuntimeError("every slide fails")

    failed = _run(slide_dir, tmp_path, one_slide)
    assert len(seen) == 3
    assert failed == ["a", "b", "c"]


def test_success_reports_no_failures(slide_dir, tmp_path):
    assert _run(slide_dir, tmp_path, lambda slide_path, **kwargs: None) == []
