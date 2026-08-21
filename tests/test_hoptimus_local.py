"""Tests for loading H-Optimus from a local directory.

These tests verify that _embed_hoptimus_subset_dataset correctly handles the
--hoptimus-model-dir option (local directory) as well as the default
HuggingFace path.  The tests use a tiny random-weight surrogate model so they
run without GPU, without the real 4 GB checkpoint, and without network access.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyDataset(Dataset):
    """Returns tiny random RGB PIL images — the same interface as production patch datasets."""

    def __init__(self, n: int = 4, size: int = 224):
        self.n = n
        self.size = size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Image.Image:
        arr = np.random.randint(0, 255, (self.size, self.size, 3), dtype=np.uint8)
        return Image.fromarray(arr)


def _make_tiny_vit_model():
    """Return a minimal timm ViT-like model that outputs [B, 8] embeddings."""
    import timm

    # tiny_vit is not a real timm name; use a small registered model instead
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    return model


def _write_fake_checkpoint(path: Path, model: "torch.nn.Module") -> None:
    """Save model state_dict as pytorch_model.bin."""
    torch.save(model.state_dict(), path)


def _write_config_json(directory: Path, architecture: str) -> None:
    cfg = {
        "architecture": architecture,
        "num_classes": 0,
        "num_features": 192,  # vit_tiny output dim
        "global_pool": "token",
        "pretrained_cfg": {
            "custom_load": True,
            "input_size": [3, 224, 224],
            "fixed_input_size": True,
            "mean": [0.707223, 0.578729, 0.703617],
            "std": [0.211883, 0.230117, 0.177517],
            "num_classes": 0,
        },
    }
    (directory / "config.json").write_text(json.dumps(cfg))


# ---------------------------------------------------------------------------
# Tests: local model dir loading
# ---------------------------------------------------------------------------


class TestEmbedHoptimusLocalDir:
    """Tests for _embed_hoptimus_subset_dataset with hoptimus_model_dir set."""

    def test_missing_model_dir_raises(self, tmp_path):
        """Non-existent directory raises FileNotFoundError."""
        from wsinsight.insightlib.niche_generation import _embed_hoptimus_subset_dataset

        ds = _DummyDataset(n=2)
        with pytest.raises(FileNotFoundError, match="hoptimus-model-dir"):
            _embed_hoptimus_subset_dataset(
                ds,
                [0, 1],
                hoptimus_model_dir=tmp_path / "does_not_exist",
            )

    def test_missing_config_json_raises(self, tmp_path):
        """Directory without config.json raises FileNotFoundError."""
        from wsinsight.insightlib.niche_generation import _embed_hoptimus_subset_dataset

        ds = _DummyDataset(n=2)
        with pytest.raises(FileNotFoundError, match="config.json"):
            _embed_hoptimus_subset_dataset(
                ds,
                [0, 1],
                hoptimus_model_dir=tmp_path,  # empty dir, no config.json
            )

    def test_missing_checkpoint_raises(self, tmp_path):
        """Directory with config.json but no weights file raises FileNotFoundError."""
        from wsinsight.insightlib.niche_generation import _embed_hoptimus_subset_dataset

        _write_config_json(tmp_path, "vit_tiny_patch16_224")
        ds = _DummyDataset(n=2)
        with pytest.raises(FileNotFoundError, match="pytorch_model.bin"):
            _embed_hoptimus_subset_dataset(
                ds,
                [0, 1],
                hoptimus_model_dir=tmp_path,
            )

    def test_config_without_architecture_raises(self, tmp_path):
        """config.json missing 'architecture' key raises ValueError."""
        from wsinsight.insightlib.niche_generation import _embed_hoptimus_subset_dataset

        (tmp_path / "config.json").write_text(json.dumps({"num_classes": 0}))
        (tmp_path / "pytorch_model.bin").write_bytes(b"")
        ds = _DummyDataset(n=2)
        with pytest.raises(ValueError, match="architecture"):
            _embed_hoptimus_subset_dataset(
                ds,
                [0, 1],
                hoptimus_model_dir=tmp_path,
            )

    def test_local_dir_produces_float32_embeddings(self, tmp_path):
        """Local dir with valid config + weights returns float32 array of shape [n, D]."""
        from wsinsight.insightlib.niche_generation import _embed_hoptimus_subset_dataset

        # Build surrogate model and save checkpoint
        model = _make_tiny_vit_model()
        _write_config_json(tmp_path, "vit_tiny_patch16_224")
        _write_fake_checkpoint(tmp_path / "pytorch_model.bin", model)

        ds = _DummyDataset(n=3)
        sampled_ids = [0, 1, 2]

        embeddings = _embed_hoptimus_subset_dataset(
            ds,
            sampled_ids,
            device="cpu",
            hoptimus_model_dir=tmp_path,
        )

        assert isinstance(embeddings, np.ndarray), "should return ndarray"
        assert (
            embeddings.dtype == np.float32
        ), f"expected float32, got {embeddings.dtype}"
        assert embeddings.shape[0] == len(
            sampled_ids
        ), f"expected {len(sampled_ids)} rows, got {embeddings.shape[0]}"
        assert embeddings.ndim == 2, "should be 2-D [n, D]"

    def test_local_dir_subset_selection(self, tmp_path):
        """Only the requested sampled_ids are embedded (not the full dataset)."""
        from wsinsight.insightlib.niche_generation import _embed_hoptimus_subset_dataset

        model = _make_tiny_vit_model()
        _write_config_json(tmp_path, "vit_tiny_patch16_224")
        _write_fake_checkpoint(tmp_path / "pytorch_model.bin", model)

        ds = _DummyDataset(n=10)
        sampled_ids = [0, 3, 7]

        embeddings = _embed_hoptimus_subset_dataset(
            ds,
            sampled_ids,
            device="cpu",
            hoptimus_model_dir=tmp_path,
        )

        assert embeddings.shape[0] == 3

    def test_safetensors_checkpoint_accepted(self, tmp_path):
        """model.safetensors is accepted as an alternative checkpoint format."""
        from wsinsight.insightlib.niche_generation import _embed_hoptimus_subset_dataset

        model = _make_tiny_vit_model()
        _write_config_json(tmp_path, "vit_tiny_patch16_224")

        # Write weights as safetensors if available; else skip
        try:
            from safetensors.torch import save_file

            save_file(model.state_dict(), str(tmp_path / "model.safetensors"))
        except ImportError:
            pytest.skip("safetensors not installed")

        ds = _DummyDataset(n=2)
        embeddings = _embed_hoptimus_subset_dataset(
            ds,
            [0, 1],
            device="cpu",
            hoptimus_model_dir=tmp_path,
        )
        assert embeddings.shape[0] == 2


# ---------------------------------------------------------------------------
# Tests: default HuggingFace path (mocked, no network)
# ---------------------------------------------------------------------------


class TestEmbedHoptimusHuggingFacePath:
    """Verify the default (hoptimus_model_dir=None) path calls timm with the
    expected HF hub model string, without touching the network."""

    def test_default_path_calls_hf_hub(self, tmp_path):
        """When hoptimus_model_dir is None, timm.create_model is called with the
        HuggingFace hub model identifier."""
        from wsinsight.insightlib.niche_generation import _embed_hoptimus_subset_dataset

        # Build a surrogate model to return from the mock
        surrogate = _make_tiny_vit_model()
        surrogate.eval()

        with patch("timm.create_model", return_value=surrogate) as mock_create:
            ds = _DummyDataset(n=2)
            # This will fail after create_model because the surrogate isn't the
            # real H-Optimus, but we only care that create_model was called correctly.
            try:
                _embed_hoptimus_subset_dataset(ds, [0, 1], device="cpu")
            except Exception:
                pass

        assert mock_create.called, "timm.create_model should have been called"
        call_args = mock_create.call_args
        model_name = (
            call_args.args[0]
            if call_args.args
            else call_args.kwargs.get("model_name", "")
        )
        assert (
            model_name == "hf-hub:bioptimus/H-optimus-0"
        ), f"Expected HF hub model id, got: {model_name!r}"
        kwargs = call_args.kwargs
        assert kwargs.get("pretrained") is True
        assert kwargs.get("num_classes") == 0
