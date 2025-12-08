from __future__ import annotations
import os, psutil, math, time
from typing import Optional, Callable

# ---------- small helpers ----------
def _cpu_count_physical_or_logical() -> int:
    """Return physical core count when available, otherwise logical count."""
    phys = psutil.cpu_count(logical=False)
    return phys if phys and phys > 0 else (os.cpu_count() or 1)

def _ewma(prev: Optional[float], new: float, alpha: float = 0.5) -> float:
    """Compute an exponentially weighted moving average sample."""
    return new if prev is None else (alpha * new + (1 - alpha) * prev)

def _cpu_util_sample(sample_interval: float = 0.30) -> float:
    """Return total CPU utilization as fraction [0..1]. One sample."""
    return psutil.cpu_percent(interval=sample_interval) / 100.0

def _mem_util_sample() -> tuple[float, int, int]:
    """Return (util_frac, available_bytes, total_bytes)."""
    vm = psutil.virtual_memory()
    return vm.percent / 100.0, vm.available, vm.total

def _optional_gpu_util() -> Optional[float]:
    """Return GPU util [0..1] if NVML present; else None (don’t import if missing)."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        if n == 0:
            return 0.0
        u = []
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            u.append(util.gpu / 100.0)
        pynvml.nvmlShutdown()
        return max(u) if u else 0.0
    except Exception:
        return None

# ---------- dynamic footprint probe (optional) ----------
def _probe_memory_per_worker_bytes(
    probe_fn: Optional[Callable[[], None]] = None,
    warmup_seconds: float = 0.1
) -> Optional[int]:
    """
    Run a tiny 'representative' unit once to estimate RSS delta as per-worker memory.
    If probe_fn is None, return None.
    """
    if probe_fn is None:
        return None
    proc = psutil.Process(os.getpid())
    before = proc.memory_info().rss
    t0 = time.time()
    try:
        probe_fn()   # do one tile worth of work if you can
    except Exception:
        pass
    finally:
        # optional tiny warmup wait (allocators settle)
        dt = time.time() - t0
        if dt < warmup_seconds:
            time.sleep(warmup_seconds - dt)
    after = proc.memory_info().rss
    delta = max(0, after - before)
    # add safety overhead (fragmentation, temp arrays)
    return int(delta * 1.5) if delta > 0 else None

# ---------- public API ----------
_cpu_ewma: Optional[float] = None
_mem_ewma: Optional[float] = None

def pick_workers_safe(
    target_cpu_util: float = 0.60,
    target_mem_util: float = 0.75,
    max_workers: int = 32,
    min_workers: int = 2,
    *,
    memory_per_worker_bytes: Optional[int] = None,
    reserve_mem_bytes: int = 512 * 1024 * 1024,
    cpu_core_reserve: int = 1,          # leave cores for OS/GPU feeders
    sample_interval_sec: float = 0.30,
    ewma_alpha: float = 0.5,
    dynamic_probe_fn: Optional[Callable[[], None]] = None,  # set to estimate footprint once
) -> int:
    """
    Choose worker count from both CPU and RAM headroom with smoothing and (optional) dynamic footprint probe.
    """
    global _cpu_ewma, _mem_ewma

    cores_total = _cpu_count_physical_or_logical()
    cores_usable = max(1, cores_total - cpu_core_reserve)

    # samples (EWMA-smoothed)
    cpu_now = _cpu_util_sample(sample_interval=sample_interval_sec)
    mem_now, mem_avail, _ = _mem_util_sample()
    _cpu_ewma = _ewma(_cpu_ewma, cpu_now, ewma_alpha)
    _mem_ewma = _ewma(_mem_ewma, mem_now, ewma_alpha)
    cpu_util = _cpu_ewma
    mem_util = _mem_ewma

    # optional GPU util awareness (don’t upsize if GPU saturated)
    gpu_util = _optional_gpu_util()
    if gpu_util is not None and gpu_util > 0.90:
        # back off a bit to leave a core for GPU driver / dataloader
        cores_usable = max(1, cores_usable - 1)

    # CPU-based guess: how much room to approach target
    cpu_headroom = max(0.0, target_cpu_util - cpu_util)
    cpu_guess = int(cpu_headroom * cores_usable)

    # Memory-based guess
    mpw = memory_per_worker_bytes
    if mpw is None:
        # Try dynamic probe once if provided
        mpw = _probe_memory_per_worker_bytes(dynamic_probe_fn)
    if mpw:
        avail_for_us = max(0, mem_avail - reserve_mem_bytes)
        mem_guess = int(avail_for_us // mpw)
    else:
        # conservative scaling when footprint unknown
        mem_headroom = max(0.0, target_mem_util - mem_util)
        # scale by max_workers rather than cores (your original idea), but clamp to cores_usable
        mem_guess = min(cores_usable, int(math.floor(mem_headroom * max_workers)))

    # Combine & clamp
    guess = min(cpu_guess, mem_guess, cores_usable, max_workers)
    if guess <= 0:
        # fallback to minimum while respecting hard RAM cap if known
        if mpw:
            hard_cap = int(max(0, (mem_avail - reserve_mem_bytes)) // mpw)
            return max(0, min(hard_cap, min_workers))
        return min_workers

    return max(min_workers, guess)

def throttle_when_busy(
    target_cpu_util: float = 0.80,
    target_mem_util: float = 0.90,
    *,
    reserve_mem_bytes: int = 512 * 1024 * 1024,
    min_sleep: float = 0.25,
    max_sleep: float = 2.0,
    backoff_multiplier: float = 1.5,
) -> None:
    """
    Sleep with exponential backoff while machine is 'busy':
      - CPU utilization > target_cpu_util
      - OR memory utilization > target_mem_util
      - OR available RAM < reserve_mem_bytes
    Uses EWMA’d signals; backs off when still hot.
    """
    sleep_dur = min_sleep
    while True:
        cpu_now = _cpu_util_sample(sample_interval=0.20)
        mem_now, mem_avail, _ = _mem_util_sample()

        cpu_ok = cpu_now <= target_cpu_util
        mem_ok = (mem_now <= target_mem_util) and (mem_avail >= reserve_mem_bytes)

        if cpu_ok and mem_ok:
            return
        time.sleep(sleep_dur)
        sleep_dur = min(max_sleep, sleep_dur * backoff_multiplier)
