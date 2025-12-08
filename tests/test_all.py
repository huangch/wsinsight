from __future__ import annotations

import concurrent.futures as cf
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

import wsinsight.write_geojson as write_geojson_module
import wsinsight.write_omecsv as write_omecsv_module
from wsinsight.cli.infer import _coerce_number, _csv_to_list
from wsinsight.write_geojson import (
    _build_geojson_dict_from_csv,
    _dataframe_to_geojson_box_fast,
    _dataframe_to_geojson_polygon_fast,
    _make_distinct_colors,
)
from wsinsight.write_omecsv import make_omecsv, write_omecsvs


class InlineExecutor:
    """Run executor work immediately inside the current process."""

    def __init__(self, *args, **kwargs) -> None:
        self._futures: list[cf.Future] = []

    def __enter__(self) -> InlineExecutor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def submit(self, fn, *args, **kwargs) -> cf.Future:
        fut: cf.Future = cf.Future()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - surfaced when tests fail
            fut.set_exception(exc)
        else:
            fut.set_result(result)
        self._futures.append(fut)
        return fut


def _write_minimal_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "minx": [0, 10],
            "miny": [0, 10],
            "width": [20, 20],
            "height": [20, 20],
            "prob_background": [0.3, 0.6],
            "prob_tumor": [0.7, 0.4],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_make_distinct_colors_unique_hex() -> None:
    colors = _make_distinct_colors(5, seed=123, shuffle=False)
    assert len(colors) == 5
    assert len({c["hex"] for c in colors}) == 5
    for color in colors:
        assert color["hex"].startswith("#") and len(color["hex"]) == 7
        assert all(0 <= channel <= 255 for channel in color["rgb"])


def test_dataframe_to_geojson_box_fast_sets_properties() -> None:
    df = pd.DataFrame(
        {
            "minx": [0, 20],
            "miny": [0, 20],
            "width": [10, 10],
            "height": [10, 10],
            "prob_background": [0.25, 0.75],
            "prob_tumor": [0.75, 0.25],
        }
    )
    result = _dataframe_to_geojson_box_fast(
        df,
        prob_cols=["prob_background", "prob_tumor"],
        overlap=0.0,
        set_classification=True,
    )

    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == len(df)
    for feature in result["features"]:
        props = feature["properties"]
        assert set(props["measurements"].keys()) == {"prob_background", "prob_tumor"}
        assert props["classification"]["name"].startswith("prob_")
        assert feature["geometry"]["type"] == "Polygon"


def test_dataframe_to_geojson_polygon_fast_handles_wkt() -> None:
    df = pd.DataFrame(
        {
            "polygon_wkt": [
                "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                "POLYGON ((0 0, 2 0, 0 2, 0 0))",
            ],
            "prob_background": [0.4, 0.8],
            "prob_tumor": [0.6, 0.2],
        }
    )

    geojson = _dataframe_to_geojson_polygon_fast(
        df,
        prob_cols=["prob_background", "prob_tumor"],
        set_classification=True,
        crs="EPSG:4326",
    )

    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 2
    assert geojson["features"][0]["geometry"]["type"] == "Polygon"


def test_build_geojson_dict_from_csv_box(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    csv_path = results_dir / "model-outputs-csv" / "sample.csv"
    _write_minimal_csv(csv_path)

    out_path, geojson = _build_geojson_dict_from_csv(
        csv_path,
        overlap=0.0,
        results_dir=results_dir,
        output_dir=Path("geojson"),
        prefix="prob",
        object_type="tile",
        set_classification=True,
        annotation_shape="box",
    )

    assert out_path == results_dir / "geojson" / "sample.geojson"
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 2


def test_write_geojsons_creates_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "results"
    csv_dir = results_dir / "model-outputs-csv"
    _write_minimal_csv(csv_dir / "slide_a.csv")
    _write_minimal_csv(csv_dir / "slide_b.csv")

    monkeypatch.setattr(write_geojson_module, "ProcessPoolExecutor", InlineExecutor)

    write_geojson_module.write_geojsons(
        csvs=sorted(csv_dir.glob("*.csv")),
        overlap=0.0,
        results_dir=results_dir,
        output_dir=Path("geojson-out"),
        prefix="prob",
        num_workers=1,
        object_type="tile",
        set_classification=True,
        show_progress=False,
        print_timings=False,
    )

    for slide in ("slide_a", "slide_b"):
        geojson_file = results_dir / "geojson-out" / f"{slide}.geojson"
        assert geojson_file.exists()
        data = json.loads(geojson_file.read_text())
        assert data["features"], "GeoJSON should contain features"


def test_make_omecsv_produces_compressed_file(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    csv_path = results_dir / "model-outputs-csv" / "slide.csv"
    _write_minimal_csv(csv_path)

    make_omecsv(
        csv=csv_path,
        results_dir=results_dir,
        output_dir=Path("omecsv"),
        overlap=0.0,
        prefix="prob",
        usecols=None,
        dtype=None,
    )

    output_file = results_dir / "omecsv" / "slide.ome.csv.gz"
    assert output_file.exists()
    with gzip.open(output_file, "rt", encoding="utf-8") as f:
        header = f.readline().strip()
        assert header.startswith("object,secondary_object,polygon")


def test_write_omecsvs_runs_inline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results_dir = tmp_path / "results"
    csv_dir = results_dir / "model-outputs-csv"
    csv_paths = [csv_dir / "slide_a.csv", csv_dir / "slide_b.csv"]
    for path in csv_paths:
        _write_minimal_csv(path)

    monkeypatch.setattr(write_omecsv_module, "ProcessPoolExecutor", InlineExecutor)

    write_omecsvs(
        csvs=csv_paths,
        h5s=[],
        overlap=0.0,
        results_dir=results_dir,
        output_dir=Path("omecsv-out"),
        prefix="prob",
        num_workers=2,
    )

    outputs = list((results_dir / "omecsv-out").glob("*.ome.csv.gz"))
    assert len(outputs) == 2


def test_coerce_number_handles_int_float_and_text() -> None:
    assert _coerce_number("42") == 42
    assert _coerce_number("  3.14  ") == pytest.approx(3.14)
    assert _coerce_number("NaN") == "nan"
    assert _coerce_number("Tumor") == "tumor"


def test_csv_to_list_parses_values() -> None:
    values = _csv_to_list(None, None, "1, 2, 3.5, tumor")
    assert values == [1, 2, pytest.approx(3.5), "tumor"]

    already_list = _csv_to_list(None, None, ["4", "5", "Other"])
    assert already_list == [4, 5, "other"]


LEGACY_TEST_SUITE = '''
from __future__ import annotations  # updated header

import json
import os
import platform
import sys
import time
from pathlib import Path

import geojson as geojsonlib
import h5py
import numpy as np
import pandas as pd
import pytest
import tifffile
import torch
from click.testing import CliRunner

from wsinsight.cli.cli import cli
from wsinsight.cli.infer import _get_info_for_save
from wsinsight.modellib.models import get_pretrained_torch_module
from wsinsight.modellib.models import get_registered_model
from wsinsight.modellib.run_inference import jit_compile
from wsinsight.wsi import HAS_OPENSLIDE
from wsinsight.wsi import HAS_TIFFSLIDE


@pytest.fixture
def tiff_image(tmp_path: Path) -> Path:
    x = np.empty((4096, 4096, 3), dtype="uint8")
    x[...] = [160, 32, 240]  # rgb for purple
    path = Path(tmp_path / "images" / "purple.tif")
    path.parent.mkdir(exist_ok=True)

    tifffile.imwrite(
        path,
        data=x,
        compression="zlib",
        tile=(256, 256),
        # 0.25 micrometers per pixel.
        resolution=(40_000, 40_000),
        resolutionunit=tifffile.RESUNIT.CENTIMETER,
    )

    return path


# The reference data for this test was made using a patched version of wsinsight 0.3.6.
# The patches fixed an issue when calculating strides and added padding to images.
# Large-image (which was the backend in 0.3.6) did not pad images and would return
# tiles that were not fully the requested width and height.
@pytest.mark.parametrize(
    "model",
    [
        "breast-tumor-resnet34.tcga-brca",
        "lung-tumor-resnet34.tcga-luad",
        "pancancer-lymphocytes-inceptionv4.tcga",
        "pancreas-tumor-preactresnet34.tcga-paad",
        "prostate-tumor-resnet34.tcga-prad",
    ],
)
@pytest.mark.parametrize("speedup", [False, True])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "openslide",
            marks=pytest.mark.skipif(
                not HAS_OPENSLIDE, reason="OpenSlide not available"
            ),
        ),
        pytest.param(
            "tiffslide",
            marks=pytest.mark.skipif(
                not HAS_TIFFSLIDE, reason="TiffSlide not available"
            ),
        ),
    ],
)
def test_cli_run_with_registered_models(
    model: str,
    speedup: bool,
    backend: str,
    tiff_image: Path,
    tmp_path: Path,
) -> None:
    """A regression test of the command 'wsinsight run'."""

    reference_csv = Path(__file__).parent / "reference" / model / "purple.csv"
    if not reference_csv.exists():
        raise FileNotFoundError(f"reference CSV not found: {reference_csv}")

    runner = CliRunner()
    results_dir = tmp_path / "inference"
    result = runner.invoke(
        cli,
        [
            "--backend",
            backend,
            "run",
            "--wsi-dir",
            str(tiff_image.parent),
            "--results-dir",
            str(results_dir),
            "--model",
            model,
            "--speedup" if speedup else "--no-speedup",
        ],
    )
    assert result.exit_code == 0
    assert (results_dir / "model-outputs-csv").exists()
    df = pd.read_csv(results_dir / "model-outputs-csv" / "purple.csv")
    df_ref = pd.read_csv(reference_csv)

    assert set(df.columns) == set(df_ref.columns)
    assert df.shape == df_ref.shape
    assert np.array_equal(df["minx"], df_ref["minx"])
    assert np.array_equal(df["miny"], df_ref["miny"])
    assert np.array_equal(df["width"], df_ref["width"])
    assert np.array_equal(df["height"], df_ref["height"])

    prob_cols = df_ref.filter(like="prob_").columns.tolist()
    for prob_col in prob_cols:
        assert np.allclose(
            df[prob_col], df_ref[prob_col], atol=1e-07
        ), f"Column {prob_col} not allclose at atol=1e-07"

    # Test that metadata path exists.
    metadata_paths = list(results_dir.glob("run_metadata_*.json"))
    assert len(metadata_paths) == 1
    metadata_path = metadata_paths[0]
    assert metadata_path.exists()
    with open(metadata_path) as f:
        meta = json.load(f)
    assert set(meta.keys()) == {"model", "runtime", "timestamp"}
    assert "config" in meta["model"]
    assert "huggingface_location" in meta["model"]
    assert model in meta["model"]["huggingface_location"]["repo_id"]
    assert meta["runtime"]["python_executable"] == sys.executable
    assert meta["runtime"]["python_version"] == platform.python_version()
    assert meta["timestamp"]
    del metadata_path, meta

    # Test conversion to geojson.
    geojson_dir = results_dir / "model-outputs-geojson"
    # result = runner.invoke(cli, ["togeojson", str(results_dir), str(geojson_dir)])
    assert result.exit_code == 0
    with open(geojson_dir / "purple.geojson") as f:
        d: geojsonlib.GeoJSON = geojsonlib.load(f)
    assert d.is_valid, "geojson not valid!"
    assert len(d["features"]) == len(df_ref)

    for geojson_row in d["features"]:
        assert geojson_row["type"] == "Feature"
        isinstance(geojson_row["id"], str)
        assert geojson_row["geometry"]["type"] == "Polygon"
    res = []
    for prob_col in prob_cols:
        res.append(
            np.array(
                [dd["properties"]["measurements"][prob_col] for dd in d["features"]]
            )
        )
    geojson_probs = np.stack(res, axis=0)
    del res
    assert np.allclose(df[prob_cols].T, geojson_probs)

    # Check the coordinate values.
    for df_row, geojson_row in zip(df.itertuples(), d["features"]):
        maxx = df_row.minx + df_row.width  # type: ignore
        maxy = df_row.miny + df_row.height  # type: ignore
        df_coords = [
            [maxx, df_row.miny],
            [maxx, maxy],
            [df_row.minx, maxy],
            [df_row.minx, df_row.miny],
            [maxx, df_row.miny],
        ]
        assert [df_coords] == geojson_row["geometry"]["coordinates"]


def test_cli_run_with_local_model(tmp_path: Path, tiff_image: Path) -> None:
    model = "breast-tumor-resnet34.tcga-brca"
    reference_csv = Path(__file__).parent / "reference" / model / "purple.csv"
    if not reference_csv.exists():
        raise FileNotFoundError(f"reference CSV not found: {reference_csv}")
    w = get_registered_model(model)

    config = {
        "spec_version": "1.0",
        "architecture": "resnet34",
        "num_classes": 2,
        "class_names": ["Other", "Tumor"],
        "patch_size_pixels": 350,
        "spacing_um_px": 0.25,
        "transform": [
            {"name": "Resize", "arguments": {"size": 224}},
            {"name": "ToTensor"},
            {
                "name": "Normalize",
                "arguments": {
                    "mean": [0.7238, 0.5716, 0.6779],
                    "std": [0.112, 0.1459, 0.1089],
                },
            },
        ],
    }

    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    runner = CliRunner()
    results_dir = tmp_path / "inference"
    result = runner.invoke(
        cli,
        [
            "--backend",
            "tiffslide",
            "run",
            "--wsi-dir",
            str(tiff_image.parent),
            "--results-dir",
            str(results_dir),
            "--model-path",
            w.model_path,
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0
    assert (results_dir / "model-outputs-csv").exists()
    df = pd.read_csv(results_dir / "model-outputs-csv" / "purple.csv")
    df_ref = pd.read_csv(reference_csv)

    assert set(df.columns) == set(df_ref.columns)
    assert df.shape == df_ref.shape
    assert np.array_equal(df["minx"], df_ref["minx"])
    assert np.array_equal(df["miny"], df_ref["miny"])
    assert np.array_equal(df["width"], df_ref["width"])
    assert np.array_equal(df["height"], df_ref["height"])

    prob_cols = df_ref.filter(like="prob_").columns.tolist()
    for prob_col in prob_cols:
        assert np.allclose(
            df[prob_col], df_ref[prob_col], atol=1e-07
        ), f"Column {prob_col} not allclose at atol=1e-07"


def test_cli_run_no_model_or_config(tmp_path: Path) -> None:
    """Test that --model or (--config and --model-path) is required."""
    wsi_dir = tmp_path / "slides"
    wsi_dir.mkdir()

    runner = CliRunner()
    args = [
        "run",
        "--wsi-dir",
        str(wsi_dir),
        "--results-dir",
        str(tmp_path / "results"),
    ]
    # No model, weights, or config.
    result = runner.invoke(cli, args)
    assert result.exit_code != 0
    assert "one of --model or (--config and --model-path) is required" in result.output


def test_cli_run_model_and_config(tmp_path: Path) -> None:
    """Test that (model and weights) or config is required."""
    wsi_dir = tmp_path / "slides"
    wsi_dir.mkdir()

    fake_config = tmp_path / "foobar.json"
    fake_config.touch()
    fake_model_path = tmp_path / "foobar.pt"
    fake_model_path.touch()

    runner = CliRunner()
    args = [
        "run",
        "--wsi-dir",
        str(wsi_dir),
        "--results-dir",
        str(tmp_path / "results"),
        "--model",
        "colorectal-tiatoolbox-resnet50.kather100k",
        "--model-path",
        str(fake_model_path),
        "--config",
        str(fake_config),
    ]
    # No model, weights, or config.
    result = runner.invoke(cli, args)
    assert result.exit_code != 0
    assert (
        "--config and --model-path are mutually exclusive with --model" in result.output
    )


@pytest.mark.xfail
def test_convert_to_sbu() -> None:
    # TODO: create a synthetic output and then convert it. Check that it is valid.
    raise AssertionError()


@pytest.mark.parametrize(
    ["patch_size", "patch_spacing"],
    [(256, 0.25), (256, 0.50), (350, 0.25), (100, 0.3), (100, 0.5)],
)
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "openslide",
            marks=pytest.mark.skipif(
                not HAS_OPENSLIDE, reason="OpenSlide not available"
            ),
        ),
        pytest.param(
            "tiffslide",
            marks=pytest.mark.skipif(
                not HAS_TIFFSLIDE, reason="TiffSlide not available"
            ),
        ),
    ],
)
def test_patch_cli(
    patch_size: int,
    patch_spacing: float,
    backend: str,
    tmp_path: Path,
    tiff_image: Path,
) -> None:
    """Test of 'wsinsight patch'."""
    orig_slide_size = 4096
    orig_slide_spacing = 0.25

    runner = CliRunner()
    savedir = tmp_path / "savedir"
    result = runner.invoke(
        cli,
        [
            "--backend",
            backend,
            "patch",
            "--wsi-dir",
            str(tiff_image.parent),
            "--results-dir",
            str(savedir),
            "--patch-size-px",
            str(patch_size),
            "--patch-spacing-um-px",
            str(patch_spacing),
        ],
    )
    assert result.exit_code == 0
    stem = tiff_image.stem
    assert (savedir / "masks" / f"{stem}.jpg").exists()
    assert (savedir / "patches" / f"{stem}.h5").exists()

    expected_patch_size = round(patch_size * patch_spacing / orig_slide_spacing)
    sqrt_expected_num_patches = round(orig_slide_size / expected_patch_size)
    expected_num_patches = sqrt_expected_num_patches**2

    expected_coords = []
    for x in range(0, orig_slide_size, expected_patch_size):
        for y in range(0, orig_slide_size, expected_patch_size):
            # Patch is kept if centroid is inside.
            if (
                x + expected_patch_size // 2 <= orig_slide_size
                and y + expected_patch_size // 2 <= orig_slide_size
            ):
                expected_coords.append([x, y])
    assert len(expected_coords) == expected_num_patches
    with h5py.File(savedir / "patches" / f"{stem}.h5") as f:
        assert f["/coords"].attrs["patch_size"] == expected_patch_size
        coords = f["/coords"][()]
    assert coords.shape == (expected_num_patches, 2)
    assert np.array_equal(expected_coords, coords)


# FIXME: parametrize this test across our models.
def test_jit_compile() -> None:
    w = get_registered_model("breast-tumor-resnet34.tcga-brca")
    model = get_pretrained_torch_module(w)

    x = torch.ones(20, 3, 224, 224, dtype=torch.float32)
    model.eval()
    NUM_SAMPLES = 1
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(NUM_SAMPLES):
            out_nojit = model(x).detach().cpu()
        time_nojit = time.perf_counter() - t0
    model_nojit = model
    model = jit_compile(model)  # type: ignore
    if model is model_nojit:
        pytest.skip("Failed to compile model (would use original model)")
    with torch.no_grad():
        model(x).detach().cpu()  # run it once to compile
        t0 = time.perf_counter()
        for _ in range(NUM_SAMPLES):
            out_jit = model(x).detach().cpu()
        time_yesjit = time.perf_counter() - t0

    assert torch.allclose(out_nojit, out_jit)
    if time_nojit < time_yesjit:
        pytest.skip(
            "JIT-compiled model was SLOWER than original: "
            f"jit={time_yesjit:0.3f} vs nojit={time_nojit:0.3f}"
        )


def test_issue_89() -> None:
    """Do not fail if 'git' is not installed."""
    model_obj = get_registered_model("breast-tumor-resnet34.tcga-brca")
    d = _get_info_for_save(model_obj)
    assert d
    assert "git" in d["runtime"]
    assert d["runtime"]["git"]
    assert d["runtime"]["git"]["git_remote"]
    assert d["runtime"]["git"]["git_branch"]

    # Test that _get_info_for_save does not fail if git is not found.
    orig_path = os.environ["PATH"]
    try:
        os.environ["PATH"] = ""
        d = _get_info_for_save(model_obj)
        assert d
        assert "git" in d["runtime"]
        assert d["runtime"]["git"] is None
    finally:
        os.environ["PATH"] = orig_path  # reset path


def test_issue_94(tmp_path: Path, tiff_image: Path) -> None:
    """Gracefully handle unreadable slides."""

    # We have a valid tiff in 'tiff_image.parent'. We put in an unreadable file too.
    badpath = tiff_image.parent / "bad.svs"
    badpath.touch()

    runner = CliRunner()
    results_dir = tmp_path / "inference"
    result = runner.invoke(
        cli,
        [
            "run",
            "--wsi-dir",
            str(tiff_image.parent),
            "--results-dir",
            str(results_dir),
            "--model",
            "breast-tumor-resnet34.tcga-brca",
        ],
    )
    # Important part is that we run through all of the files, despite the unreadble
    # file.
    assert result.exit_code == 0
    assert results_dir.joinpath("model-outputs-csv").joinpath("purple.csv").exists()
    assert not results_dir.joinpath("model-outputs-csv").joinpath("bad.csv").exists()


def test_issue_97(tmp_path: Path, tiff_image: Path) -> None:
    """Write a run_metadata file per run."""

    runner = CliRunner()
    results_dir = tmp_path / "inference"
    result = runner.invoke(
        cli,
        [
            "run",
            "--wsi-dir",
            str(tiff_image.parent),
            "--results-dir",
            str(results_dir),
            "--model",
            "breast-tumor-resnet34.tcga-brca",
        ],
    )
    assert result.exit_code == 0
    metas = list(results_dir.glob("run_metadata_*.json"))
    assert len(metas) == 1

    time.sleep(2)  # make sure some time has passed so the timestamp is different

    # Run again...
    result = runner.invoke(
        cli,
        [
            "run",
            "--wsi-dir",
            str(tiff_image.parent),
            "--results-dir",
            str(results_dir),
            "--model",
            "breast-tumor-resnet34.tcga-brca",
        ],
    )
    assert result.exit_code == 0
    metas = list(results_dir.glob("run_metadata_*.json"))
    assert len(metas) == 2


def test_issue_125(tmp_path: Path) -> None:
    """Test that path in model config can be saved when a pathlib.Path object."""

    w = get_registered_model("breast-tumor-resnet34.tcga-brca")
    w.model_path = Path(w.model_path)  # type: ignore
    info = _get_info_for_save(w)
    with open(tmp_path / "foo.json", "w") as f:
        json.dump(info, f)


def test_issue_203(tiff_image: Path) -> None:
    """Test that openslide and tiffslide pad an image if an out-of-bounds region
    is requested.
    """
    import openslide
    import tiffslide

    with tiffslide.TiffSlide(tiff_image) as tslide:
        w, h = tslide.dimensions
        img = tslide.read_region((w, h), level=0, size=(256, 256))
        assert img.size == (256, 256)
        assert np.allclose(np.array(img), 0)
    del tslide, img

    with openslide.OpenSlide(tiff_image) as oslide:
        w, h = oslide.dimensions
        img = oslide.read_region((w, h), level=0, size=(256, 256))
        assert img.size == (256, 256)
        assert np.allclose(np.array(img), 0)


def test_issue_214(tmp_path: Path, tiff_image: Path) -> None:
    """Test that symlinked slides don't mess things up."""
    link = tmp_path / "forlinks" / "arbitrary-link-name.tiff"
    link.parent.mkdir(parents=True)
    link.symlink_to(tiff_image)

    runner = CliRunner()
    results_dir = tmp_path / "inference"
    result = runner.invoke(
        cli,
        [
            "run",
            "--wsi-dir",
            str(link.parent),
            "--results-dir",
            str(results_dir),
            "--model",
            "breast-tumor-resnet34.tcga-brca",
        ],
    )
    assert result.exit_code == 0
    assert (results_dir / "patches" / link.with_suffix(".h5").name).exists()
    assert (results_dir / "model-outputs-csv" / link.with_suffix(".csv").name).exists()

'''
