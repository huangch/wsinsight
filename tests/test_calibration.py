"""Verify the adaptive H-optimus batch sizing against real GPU behaviour.

Imports the *shipped* helpers from ``niche_generation`` so a pass here means the
production code path is correct — not a reimplementation of it.

Run inside the container:
    python tests/test_calibration.py --model-dir /app/zoo/hoptimus/
"""
import argparse
import math
import sys
import torch
import torch.nn as nn

from wsinsight.insightlib.niche_generation import (
    _auto_batch_size,
    _available_vram,
    _calibrate_bytes_per_image,
    _is_oom,
)

GIB = 1024 ** 3
MIB = 1024 ** 2
SAFETY = 0.95


def _load_model(model_dir: str | None, dev: str) -> nn.Module:
    """Load H-optimus from local dir or fall back to a tiny ViT for testing."""
    if model_dir:
        import json
        import timm
        from pathlib import Path
        cfg_path = Path(model_dir) / "config.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        arch = cfg["architecture"]
        ckpt = next(
            p for p in [
                Path(model_dir) / "pytorch_model.bin",
                Path(model_dir) / "model.safetensors",
            ] if p.exists()
        )
        model = timm.create_model(arch, pretrained=False,
                                  num_classes=cfg.get("num_classes", 0),
                                  global_pool=cfg.get("global_pool", "token"),
                                  pretrained_cfg_overlay=cfg.get("pretrained_cfg", {}))
        timm.models.load_checkpoint(model, str(ckpt))
        return model.to(dev).eval()
    else:
        import timm
        print("[INFO] No --model-dir given, using vit_small_patch16_224 as proxy.")
        return timm.create_model("vit_small_patch16_224", pretrained=False,
                                 num_classes=0).to(dev).eval()


def _measure_peak(model: nn.Module, dev: str, n: int) -> int:
    """Peak memory delta for a real forward pass of n images (BF16)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    dummy = torch.zeros(n, 3, 224, 224, device=dev)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(dummy)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del dummy, out
    torch.cuda.empty_cache()
    return peak - baseline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — cannot verify GPU behaviour.")
        return 1

    dev = "cuda:0"
    ngpu = torch.cuda.device_count()
    card = torch.cuda.get_device_properties(0).total_memory

    print(f"\nGPU        : {torch.cuda.get_device_name(0)} x{ngpu}")
    print(f"VRAM/card  : {card / GIB:.1f} GiB")

    model = _load_model(args.model_dir, dev)
    torch.cuda.empty_cache()
    model_mem = torch.cuda.memory_allocated()
    print(f"model VRAM : {model_mem / GIB:.2f} GiB")

    failures: list[str] = []

    # ── 1. shipped calibration + sizing ──────────────────────────────────────
    per_image = _calibrate_bytes_per_image(model, dev)
    usable = _available_vram(torch.cuda.current_device())
    total_bs = _auto_batch_size(model, dev, safety=SAFETY, bytes_per_image=per_image)
    per_gpu = total_bs // ngpu

    print(f"\nper-image  : {per_image / MIB:.1f} MiB")
    print(f"usable VRAM: {usable / GIB:.1f} GiB")
    print(f"batch size : {total_bs} total ({per_gpu}/GPU)")

    # ViT-scale activations land in single-digit-to-tens of MiB. Hundreds of MiB
    # means fixed cuDNN overhead leaked into the estimate (the original bug).
    if not (1 * MIB <= per_image <= 128 * MIB):
        failures.append(
            f"per-image {per_image / MIB:.1f} MiB outside plausible 1-128 MiB band"
        )

    # ── 2. prediction vs measured reality ────────────────────────────────────
    probe_n = max(64, min(per_gpu, 512))
    predicted = probe_n * per_image
    actual = _measure_peak(model, dev, probe_n)
    ratio = actual / predicted if predicted else float("inf")

    print(f"\nprobe batch: {probe_n} images")
    print(f"  predicted: {predicted / GIB:.2f} GiB")
    print(f"  actual   : {actual / GIB:.2f} GiB")
    print(f"  ratio    : {ratio:.2f}")

    if not (0.5 <= ratio <= 1.5):
        failures.append(f"prediction off by {ratio:.2f}x (want 0.5-1.5)")

    # ── 3. projected full-scale utilisation ──────────────────────────────────
    projected = model_mem + per_gpu * per_image
    pct = 100 * projected / card
    print(f"\nprojected at {per_gpu}/GPU: {projected / GIB:.1f} GiB ({pct:.0f}% of card)")
    if pct < 70:
        failures.append(f"projected utilisation only {pct:.0f}% (want >=70%)")

    # ── 4. OOM classification used by the retry loop ─────────────────────────
    if not _is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")):
        failures.append("_is_oom missed a RuntimeError OOM message")
    if _is_oom(ValueError("unrelated")):
        failures.append("_is_oom misclassified a non-OOM error")

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS - calibration, prediction, utilisation and OOM handling all check out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
