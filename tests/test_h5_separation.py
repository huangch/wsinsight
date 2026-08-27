"""End-to-end test of the Option 3 H5 schema separation (commit J, 2026-08-24).

Uses real h5py to author real patch.h5 files in a tmp dir, then calls
_load_patch_polygons which is the only reader of /polygons.

Verifies:
  1. /infer/polygons is preferred over /polygons when both exist
     with the matching cell count.
  2. /polygons-only legacy H5s still work (backward compat).
  3. count-mismatch in /infer/polygons falls through to /polygons.
  4. When neither matches, returns None.

(No mocking of h5py itself; we author real H5 files in tmp_path.)
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from wsinsight.modellib.run_inference import _load_patch_polygons


@pytest.fixture
def patch_dir(tmp_path):
    """Return a tmp directory for the patch H5 file."""
    return tmp_path


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_patch(root, n_patches, n_cell_polys=None, n_polys_legacy=None):
    """Write a synthetic patch H5 to root / 'slides.h5'."""
    fp = root / "slides.h5"
    with h5py.File(fp, "w") as f:
        # /coords: n_patches entries, 2 cols
        f.create_dataset(
            "/coords",
            data=np.zeros((n_patches, 2), dtype=np.int32),
        )
        f["/coords"].attrs["patch_size"] = 935
        f["/coords"].attrs["patch_spacing_um_px"] = 0.25

        if n_polys_legacy is not None:
            # /polygons: legacy 5-pt tile rings (n_polys_legacy rings)
            offsets = np.arange(n_polys_legacy + 1, dtype=np.int64) * 5
            coords = np.zeros((n_polys_legacy * 5, 2), dtype=np.float32)
            for i in range(n_polys_legacy):
                # Make each tile ring's first vertex diagnostic
                coords[i * 5 + 0] = (1000 + i, 2000 + i)
            g = f.create_group("/polygons")
            g.create_dataset("coords", data=coords, compression="gzip")
            g.create_dataset("offsets", data=offsets, dtype="int64")
            g.attrs["layout"] = "ragged_offsets"

        if n_cell_polys is not None:
            # /infer/polygons: cell contours (n_cell_polys cells, 5 pts each)
            offsets = np.arange(n_cell_polys + 1, dtype=np.int64) * 5
            coords = np.zeros((n_cell_polys * 5, 2), dtype=np.float32)
            for i in range(n_cell_polys):
                # Make each cell poly's first vertex diagnostic
                coords[i * 5 + 0] = (5000 + i, 6000 + i)
            g = f.create_group("/infer").create_group("polygons")
            g.create_dataset("coords", data=coords, compression="gzip")
            g.create_dataset("offsets", data=offsets, dtype="int64")
            g.attrs["layout"] = "ragged_offsets"
    return fp


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_load_prefers_infer_polygons(patch_dir):
    """When both /infer/polygons (count matches) and /polygons (count
    matches) are present, /infer/polygons wins."""
    fp = _write_patch(
        patch_dir,
        n_patches=1003,
        n_cell_polys=42,
        n_polys_legacy=1003,
    )

    out = _load_patch_polygons(fp, expected=42)
    assert out is not None
    assert len(out) == 42
    # First cell poly starts at (5000, 6000)
    assert tuple(int(v) for v in out[0][0]) == (5000, 6000)


def test_legacy_polygons_only_works(patch_dir):
    """Old wsinsight wrote cell contours to /polygons directly. The new
    loader must still load them when /infer/polygons is absent."""
    fp = _write_patch(
        patch_dir,
        n_patches=1003,
        n_cell_polys=None,
        n_polys_legacy=5,  # old wsinsight wrote N cells here
    )

    out = _load_patch_polygons(fp, expected=5)
    assert out is not None
    assert len(out) == 5
    # Legacy first cell starts at (1000, 2000) (tile ring convention)
    assert tuple(int(v) for v in out[0][0]) == (1000, 2000)


def test_count_mismatch_in_infer_falls_through_to_polygons(patch_dir):
    """If /infer/polygons has wrong count (e.g. from a stale run with a
    different model), fall through to /polygons which might match."""
    fp = _write_patch(
        patch_dir,
        n_patches=1003,
        n_cell_polys=999,  # stale run, count does NOT match expected=5
        n_polys_legacy=5,  # legacy still has matching count
    )

    out = _load_patch_polygons(fp, expected=5)
    assert out is not None
    assert len(out) == 5


def test_no_match_returns_none(patch_dir):
    """When neither schema has the right count, return None (which
    triggers _box_wkt_column fallback in the writer)."""
    fp = _write_patch(
        patch_dir,
        n_patches=1003,
        n_cell_polys=999,
        n_polys_legacy=999,
    )

    out = _load_patch_polygons(fp, expected=5)
    assert out is None


def test_empty_h5_returns_none(patch_dir):
    """A patch H5 with /coords only -> no polygons at all -> None."""
    fp = _write_patch(patch_dir, n_patches=1003)
    out = _load_patch_polygons(fp, expected=5)
    assert out is None


def test_both_match_equal_size_picks_infer(patch_dir):
    """When both /infer/polygons and legacy /polygons have matching
    count, /infer/polygons wins by code order."""
    fp = _write_patch(
        patch_dir,
        n_patches=1003,
        n_cell_polys=10,
        n_polys_legacy=10,  # both would match expected=10
    )

    out = _load_patch_polygons(fp, expected=10)
    assert out is not None and len(out) == 10
    # The first cell should be from /infer/polygons (5000+i, 6000+i range)
    # NOT from /polygons (1000+i, 2000+i range).
    assert tuple(int(v) for v in out[0][0])[0] >= 5000
