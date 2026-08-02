"""
Diagnostic: compare single-point vs two-point GPU memory calibration.

Run inside the container:
    python tests/test_calibration.py --model-dir /app/zoo/hoptimus/

This script proves whether two-point calibration correctly measures the marginal
per-image cost, independent of the fixed cuDNN/allocator overhead that made the
single-point approach produce 5-10× overestimates.
"""
import argparse
import math
import sys
import torch
import torch.nn as nn


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
    """Peak memory delta for a forward pass of n images (BF16)."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    dummy = torch.zeros(n, 3, 224, 224, device=dev)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(dummy)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del dummy
    torch.cuda.empty_cache()
    return peak - baseline


def _single_point(model, dev, cal_batch=16) -> int:
    m = _measure_peak(model, dev, cal_batch)
    return m // cal_batch


def _two_point(model, dev, b1=8, b2=32) -> int:
    m1 = _measure_peak(model, dev, b1)
    m2 = _measure_peak(model, dev, b2)
    if m2 <= m1:
        return m2 // b2
    return (m2 - m1) // (b2 - b1)


def _batch_size_from(bytes_per_image: int, safety: float = 0.95) -> int:
    ngpu = torch.cuda.device_count()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
    per_gpu = int(free * safety) // bytes_per_image
    total_bs = per_gpu * ngpu
    return total_bs, per_gpu, free


def _verify_actual_usage(model: nn.Module, dev: str, batch_size: int) -> float:
    """Run a real batch at batch_size and measure actual GB used per GPU."""
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    dummy = torch.zeros(batch_size, 3, 224, 224, device=dev)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(dummy)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    del dummy, out
    torch.cuda.empty_cache()
    return (after - before) / 1024**3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available — cannot run this test.")
        sys.exit(1)

    dev = "cuda:0"
    ngpu = torch.cuda.device_count()
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"\n{'='*60}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Total VRAM per GPU: {total_mem:.1f} GB  |  GPUs: {ngpu}")
    print(f"{'='*60}\n")

    print("Loading model...")
    model = _load_model(args.model_dir, dev)
    torch.cuda.empty_cache()
    model_mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  Model memory: {model_mem:.2f} GB\n")

    # ── Single-point calibration (old approach) ──────────────────────────────
    sp = _single_point(model, dev, cal_batch=16)
    sp_bs, sp_per_gpu, free = _batch_size_from(sp)
    print(f"[OLD] Single-point (batch=16):")
    print(f"      bytes_per_image = {sp/1024**2:.1f} MB")
    print(f"      → batch_size = {sp_bs}  ({sp_per_gpu}/GPU)")

    # ── Two-point calibration (new approach) ─────────────────────────────────
    tp = _two_point(model, dev, b1=8, b2=32)
    tp_bs, tp_per_gpu, free = _batch_size_from(tp)
    print(f"\n[NEW] Two-point (b1=8, b2=32):")
    print(f"      bytes_per_image = {tp/1024**2:.1f} MB")
    print(f"      → batch_size = {tp_bs}  ({tp_per_gpu}/GPU)")

    print(f"\n  Overestimate factor (old/new): {sp/tp:.1f}×")
    print(f"  Free VRAM (after empty_cache): {free/1024**3:.1f} GB\n")

    # ── Verify actual GPU usage at the NEW batch size ─────────────────────────
    verify_bs = min(tp_per_gpu, 512)   # cap at 512 for the verification run
    print(f"Verifying actual GPU memory at batch_size={verify_bs} (capped for safety)...")
    actual_gb = _verify_actual_usage(model, dev, verify_bs)
    expected_gb = verify_bs * tp / 1024**3
    print(f"  Expected (two-point estimate): {expected_gb:.2f} GB")
    print(f"  Actual measured:               {actual_gb:.2f} GB")
    ratio = actual_gb / expected_gb if expected_gb > 0 else float('inf')
    print(f"  Accuracy ratio (actual/expected): {ratio:.2f}  (1.0 = perfect)\n")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("="*60)
    if 0.7 <= ratio <= 1.4:
        print("  PASS: two-point calibration is accurate.")
        print(f"  At full scale ({tp_per_gpu}/GPU × {ngpu} GPUs = {tp_bs} total):")
        expected_total = model_mem + tp_per_gpu * tp / 1024**3
        print(f"  Estimated GPU usage: {expected_total:.1f} GB / {total_mem:.0f} GB"
              f"  ({100*expected_total/total_mem:.0f}%)")
    else:
        print(f"  WARN: ratio {ratio:.2f} outside expected 0.7–1.4.")
        print("  Calibration may need further investigation.")
    print("="*60)


if __name__ == "__main__":
    main()
