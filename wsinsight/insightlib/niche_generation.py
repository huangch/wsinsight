"""niche graph construction, embedding, and clustering pipelines for WSInsight."""

# nichegcn_multi_from_your_funcs_h0.py
# pip install torch torch_geometric scikit-learn numpy scipy pandas timm pillow

from __future__ import annotations
import math, os, time, shutil, tempfile, atexit
import multiprocessing as mp
from typing import Any, List, Dict, Iterable, Optional, Sequence, Tuple # , Callable
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.cluster import KMeans
from torch_geometric.data import Data # , DataLoader as GeoDataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader, DataListLoader
from torch_geometric.nn import GCNConv, DataParallel as GeoDataParallel

# PyG's DataParallel emits a UserWarning recommending DistributedDataParallel.
# DP is used here intentionally (bounded DGI embedding pass; a DDP port would be
# a scoped refactor with no correctness gain), so silence just that message.
import warnings as _warnings
_warnings.filterwarnings(
    "ignore",
    message=r".*'DataParallel' is usually much slower than 'DistributedDataParallel'.*",
)
from torch_geometric.nn.models import DeepGraphInfomax
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm # , trange
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
import igraph as ig
import leidenalg as la
from torchvision import transforms
import click
# import pickle, gzip
import joblib

from .. import errors
from .insight_helpers import compute_cell_center_points
from .insight_helpers import delaunay_triangulation
from .insight_helpers import create_adjacency_list_fast  # adjacency builder
from .insight_helpers import make_short_ids
from .graph_cache import get_or_build_delaunay
from ..uri_path import URIPath
from ..wsi import _validate_wsi_directory, get_avg_mpp
from ..cancel import critical_section, raise_if_cancelled
from ..insightlib.vorononi_niche_region_helper import merge_same_label_by_shared_edges_iterative, remap_edges_to_valid_indices
from ..num_worker_optimizer import pick_workers_safe, throttle_when_busy
           
# =============================================================================
# Utilities: probabilities, edges, isolation
# =============================================================================

def probs_from_df(df: pd.DataFrame,
                  class_order: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Extract [N,C] soft probabilities from columns like 'prob_*'."""
    cols = [c for c in df.columns if c.startswith("prob_")]
    if class_order is not None:
        want = [f"prob_{k}" for k in class_order]
        missing = [c for c in want if c not in cols]
        if missing:
            raise ValueError(f"Missing probability columns: {missing}")
        cols = want
        classes = class_order
    else:
        classes = [c[len("prob_"):] for c in cols]

    P = df[cols].to_numpy(dtype=np.float32)  # [N,C]
    s = P.sum(axis=1, keepdims=True) + 1e-8
    P = P / s
    return P, classes


def to_edge_index(edges_df: pd.DataFrame,
                  src_col: str = "source", dst_col: str = "target",
                  undirected: bool = True, drop_self_loops: bool = True) -> np.ndarray:
    """DataFrame -> edge_index [2,E]. Assumes 0-based indices and length already capped by your function."""
    u = edges_df[src_col].to_numpy()
    v = edges_df[dst_col].to_numpy()
    if drop_self_loops:
        keep = (u != v)
        u, v = u[keep], v[keep]
    if undirected:
        ei = np.r_[u, v]; ej = np.r_[v, u]
    else:
        ei, ej = u, v
    return np.vstack([ei, ej]).astype(np.int64)


def drop_isolated(edge_index: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Remove nodes with degree 0. Returns (edge_index_kept, kept_indices)."""
    if edge_index.size == 0:
        return edge_index, np.array([], dtype=np.int64)
    ei, ej = edge_index
    deg = np.bincount(np.r_[ei, ej], minlength=N)
    kept = np.where(deg > 0)[0]
    if len(kept) == N:
        return edge_index, kept

    # remap
    map_old2new = -np.ones(N, dtype=np.int64)
    map_old2new[kept] = np.arange(len(kept), dtype=np.int64)
    ei_m = map_old2new[ei]; ej_m = map_old2new[ej]
    mask = (ei_m >= 0) & (ej_m >= 0)
    edge_index_new = np.vstack([ei_m[mask], ej_m[mask]]).astype(np.int64)
    return edge_index_new, kept


def _cleanup_tmpdir() -> None:
    """Clean up temporary directories to prevent /tmp from filling up during long-running jobs.
    Especially important when PyTorch/timm models create temporary files during import."""
    tmpdir = tempfile.gettempdir()
    try:
        for item in os.listdir(tmpdir):
            item_path = os.path.join(tmpdir, item)
            # Only remove directories starting with 'tmp' (standard tempfile pattern)
            if os.path.isdir(item_path) and item.startswith("tmp"):
                try:
                    shutil.rmtree(item_path, ignore_errors=True)
                except Exception:
                    pass  # silently ignore failures for in-use directories
    except Exception:
        pass  # silently ignore if tmpdir listing fails


# =============================================================================
# k-hop soft-composition (EXACT hop bins) using your adjacency
# =============================================================================

def _exact_hop_bins(adj: Dict[int, List[int]], src: int, k: int) -> List[List[int]]:
    """Return nodes at EXACT hop distances 1..k from src using BFS."""
    # from collections import deque
    seen = {src}
    q = deque([(src, 0)])
    bins = [list() for _ in range(k + 1)]  # 0..k
    bins[0].append(src)
    while q:
        u, d = q.popleft()
        if d == k:
            continue
        for v in adj.get(u, []):
            if v in seen:
                continue
            seen.add(v)
            nh = d + 1
            bins[nh].append(v)
            q.append((v, nh))
    return bins

# def khop_soft_features(P: np.ndarray, edge_index: np.ndarray, N: int,
#                        k: int = 2, alpha: float = 1.0) -> np.ndarray:
#     """
#     X: [N,(k+1)*C]
#       - 0-hop: own P[i]
#       - h=1..k: Laplace-smoothed mean of neighbors at EXACT hop h
#     """
#     # Build minimal edges_df to reuse your create_adjacency_list
#     if edge_index.size == 0:
#         C = P.shape[1]
#         X = np.zeros((N, (k + 1) * C), dtype=np.float32)
#         X[:, :C] = P
#         for h in range(1, k + 1):
#             X[:, h*C:(h+1)*C] = 1.0 / C
#         return X
#
#     ei, ej = edge_index
#     a = np.minimum(ei, ej); b = np.maximum(ei, ej)
#     pairs = np.unique(np.stack([a, b], axis=1), axis=0)
#     edges_df = pd.DataFrame({"source": pairs[:, 0], "target": pairs[:, 1]})
#     adj = create_adjacency_list_parallel(edges_df)
#
#     N_nodes, C = P.shape
#     X = np.zeros((N_nodes, (k + 1) * C), dtype=np.float32)
#     X[:, :C] = P  # 0-hop
#
#     for i in range(N_nodes):
#         bins = _exact_hop_bins(adj, i, k)
#         for h in range(1, k + 1):
#             idx = bins[h]
#             off = h * C
#             if not idx:
#                 X[i, off:off + C] = 1.0 / C
#             else:
#                 mean_prob = P[idx].mean(axis=0)
#                 # light Laplace smoothing to avoid zeros
#                 X[i, off:off + C] = (mean_prob + (alpha / C)) / (1.0 + alpha)
#     return X





# def _khop_rows_worker(start: int, end: int, k: int, alpha: float,
#                       P: np.ndarray, adj: dict) -> np.ndarray:
#     """
#     Compute X rows [start:end) using EXACT-hop BFS on 'adj'.
#     Returns X_block with shape [(end-start), (k+1)*C].
#     """
#     _, C = P.shape
#     H = end - start
#     Xblk = np.zeros((H, (k + 1) * C), dtype=np.float32)
#     # 0-hop (own probabilities)
#     Xblk[:, :C] = P[start:end]
#
#     for row, i in enumerate(range(start, end)):
#         # EXACT-hop BFS bins
#         seen = {i}
#         q = deque([(i, 0)])
#         bins = [list() for _ in range(k + 1)]
#         bins[0].append(i)
#         while q:
#             u, d = q.popleft()
#             if d == k:
#                 continue
#             for v in adj.get(u, []):
#                 if v in seen:
#                     continue
#                 seen.add(v)
#                 nh = d + 1
#                 bins[nh].append(v)
#                 q.append((v, nh))
#
#         # aggregate per hop with Laplace smoothing
#         for h in range(1, k + 1):
#             idx = bins[h]
#             off = h * C
#             if not idx:
#                 Xblk[row, off:off + C] = 1.0 / C
#             else:
#                 mean_prob = P[idx].mean(axis=0)
#                 Xblk[row, off:off + C] = (mean_prob + (alpha / C)) / (1.0 + alpha)
#     return Xblk
#
#
# def khop_features(P: np.ndarray, edge_index: np.ndarray, N: int,
#                        k: int = 2, alpha: float = 1.0) -> np.ndarray:
#     """
#     Parallel version:
#       X: [N,(k+1)*C]
#         - 0-hop: own P[i]
#         - h=1..k: Laplace-smoothed mean of neighbors at EXACT hop h
#     """
#     N_nodes, C = P.shape
#     assert N_nodes == N, "P and N mismatch"
#
#     # No edges → return baseline
#     if edge_index.size == 0:
#         X = np.zeros((N, (k + 1) * C), dtype=np.float32)
#         X[:, :C] = P
#         for h in range(1, k + 1):
#             X[:, h*C:(h+1)*C] = 1.0 / C
#         return X
#
#     # Build undirected unique edge list → adjacency (your parallel builder)
#     ei, ej = edge_index
#     a = np.minimum(ei, ej); b = np.maximum(ei, ej)
#     pairs = np.unique(np.stack([a, b], axis=1), axis=0)
#     edges_df = pd.DataFrame({"source": pairs[:, 0], "target": pairs[:, 1]})
#     adj = create_adjacency_list_parallel(edges_df, dedup_neighbors=True, sort_neighbors=False)
#
#     # Output
#     X = np.zeros((N, (k + 1) * C), dtype=np.float32)
#     X[:, :C] = P  # 0-hop
#
#     # Decide worker count & chunking (simple heuristic)
#     # cpu = os.cpu_count() or 1
#     # MIN_ROWS_PER_PROC = 256
#     # max_workers = max(1, min(cpu, N // max(1, MIN_ROWS_PER_PROC)))
#     max_workers = pick_workers_safe(max_workers=os.cpu_count()-8, min_workers=8)
#
#     if max_workers == 1:
#         # single-process fallback with same worker logic
#         X[:, :] = _khop_rows_worker(0, N, k, alpha, P, adj)
#         return X
#
#     chunk_size = math.ceil(N / max_workers)
#     ranges = [(s, min(s + chunk_size, N)) for s in range(0, N, chunk_size)]
#
#     # Launch workers and stitch results
#     with ThreadPoolExecutor(max_workers=max_workers) as ex:
#         futures = {ex.submit(_khop_rows_worker, s, e, k, alpha, P, adj): (s, e) for (s, e) in ranges}
#         for fut in as_completed(futures):
#             throttle_when_busy()
#             s, e = futures[fut]
#             X[s:e, :] = fut.result()
#
#     return X




def _khop_rows_worker(start: int, end: int, k: int, alpha: float,
                      P: np.ndarray, adj: dict,
                      mode: str = "soft",
                      labels: np.ndarray | None = None) -> np.ndarray:
    """
    Compute X rows [start:end) using EXACT-hop BFS on 'adj'.
    Returns X_block with shape [(end-start), (k+1)*C].

    mode="soft": 
      - 0-hop: P[i]
      - h>=1 : Laplace-smoothed mean of neighbors' P at EXACT hop h
               out = (mean + alpha/C) / (1+alpha)

    mode="hard":
      - 0-hop: one-hot of argmax(P[i])
      - h>=1 : histogram proportions of argmax labels among EXACT hop h nodes,
               Dirichlet-smoothed with alpha (same formula as above applied to proportions)
    """
    _, C = P.shape
    H = end - start
    Xblk = np.zeros((H, (k + 1) * C), dtype=np.float32)

    if mode == "hard":
        # labels[i] already provided by caller; else compute here as fallback
        if labels is None:
            labels = np.asarray(P.argmax(axis=1), dtype=np.int64)

    # 0-hop block
    if mode == "soft":
        Xblk[:, :C] = P[start:end]
    else:  # hard
        oh = np.zeros((H, C), dtype=np.float32)
        oh[np.arange(H), labels[start:end]] = 1.0
        Xblk[:, :C] = oh

    for row, i in enumerate(range(start, end)):
        # EXACT-hop BFS bins
        seen = {i}
        q = deque([(i, 0)])
        bins = [list() for _ in range(k + 1)]
        bins[0].append(i)
        while q:
            u, d = q.popleft()
            if d == k:
                continue
            for v in adj.get(u, []):
                if v in seen:
                    continue
                seen.add(v)
                nh = d + 1
                bins[nh].append(v)
                q.append((v, nh))

        # aggregate per hop
        for h in range(1, k + 1):
            idx = bins[h]
            off = h * C
            if not idx:
                # no nodes at this hop: fall back to uniform
                Xblk[row, off:off + C] = 1.0 / C
                continue

            if mode == "soft":
                mean_prob = P[idx].mean(axis=0)
                Xblk[row, off:off + C] = (mean_prob + (alpha / C)) / (1.0 + alpha)
            else:
                # hard: histogram proportions of predicted classes
                counts = np.bincount(labels[idx], minlength=C).astype(np.float32)
                props  = counts / counts.sum()
                Xblk[row, off:off + C] = (props + (alpha / C)) / (1.0 + alpha)

    return Xblk


def khop_features(P: np.ndarray, edge_index: np.ndarray, N: int,
                  k: int = 2, alpha: float = 1.0,
                  mode: str = "soft") -> np.ndarray:
    """
    Build k-hop feature blocks X of shape [N, (k+1)*C].

    mode="soft":
      0-hop: P[i]
      h>=1 : Laplace-smoothed mean of neighbors' P at EXACT hop h.

    mode="hard":
      0-hop: one-hot of argmax(P[i])
      h>=1 : histogram proportions of argmax labels at EXACT hop h, Dirichlet-smoothed.

    Notes:
      - EXACT-hop rings (not ≤h).
      - When a hop ring is empty, fill with uniform 1/C.
      - Uses ThreadPoolExecutor; safe with large Python objects and avoids pickling.
    """
    N_nodes, C = P.shape
    assert N_nodes == N, "P and N mismatch"

    # No edges → return baseline blocks
    if edge_index.size == 0:
        X = np.zeros((N, (k + 1) * C), dtype=np.float32)
        if mode == "soft":
            X[:, :C] = P
        else:
            labels = P.argmax(axis=1)
            X[np.arange(N), labels] = 1.0  # 0-hop one-hot
        for h in range(1, k + 1):
            X[:, h*C:(h+1)*C] = 1.0 / C
        return X

    # Build undirected unique edge list → adjacency
    ei, ej = edge_index
    a = np.minimum(ei, ej); b = np.maximum(ei, ej)
    pairs = np.unique(np.stack([a, b], axis=1), axis=0)
    edges_df = pd.DataFrame({"source": pairs[:, 0], "target": pairs[:, 1]})
    adj = create_adjacency_list_fast(edges_df, dedup_neighbors=True, sort_neighbors=False)

    # Output buffer and 0-hop block
    X = np.zeros((N, (k + 1) * C), dtype=np.float32)
    if mode == "soft":
        X[:, :C] = P
        labels = None
    else:
        labels = P.argmax(axis=1).astype(np.int64)
        oh = np.zeros((N, C), dtype=np.float32)
        oh[np.arange(N), labels] = 1.0
        X[:, :C] = oh

    # Decide workers and chunking
    max_workers = pick_workers_safe(max_workers=(os.cpu_count()-8 or 1), min_workers=8)
    chunk_size = max(1, math.ceil(N / max_workers))
    ranges = [(s, min(s + chunk_size, N)) for s in range(0, N, chunk_size)]

    if max_workers == 1:
        X[:, :] = _khop_rows_worker(0, N, k, alpha, P, adj, mode=mode, labels=labels)
        return X

    # Parallel threads
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_khop_rows_worker, s, e, k, alpha, P, adj, mode, labels): (s, e)
            for (s, e) in ranges
        }
        for fut in as_completed(futures):
            throttle_when_busy()
            s, e = futures[fut]
            X[s:e, :] = fut.result()

    return X


# =============================================================================
# H-Optimus-0 via torch DataLoader (switchable)
# =============================================================================

class DummyPatchDataset(Dataset):
    """
    Placeholder dataset you can replace.
    __getitem__(cell_id) should return a PIL.Image (224x224 RGB) or a 3xHxW tensor.
    """
    def __init__(self, num_cells: int, size: int = 224):
        self.num_cells = num_cells
        self.size = size
    def __len__(self):
        return self.num_cells
    def __getitem__(self, idx):
        # A blank RGB image placeholder; replace with real crops later.
        from PIL import Image
        return Image.new("RGB", (self.size, self.size), color=(0, 0, 0))

def _make_short_ids(stems: List[str]) -> dict:
    """Alias kept for backward compatibility; see insight_helpers.make_short_ids."""
    return make_short_ids(stems)


# Fallback per-image memory estimate used when GPU calibration is unavailable.
_HOPTIMUS_BYTES_PER_IMAGE_FALLBACK: int = 32 * 1024 ** 2  # 32 MiB
# Calibration batch size: large enough to amortize PyTorch's fixed per-batch
# allocator blocks so the per-image estimate reflects large-batch inference cost.
_CALIBRATION_BATCH: int = 16


def _calibrate_bytes_per_image(model: nn.Module, dev: str, cal_batch: int = _CALIBRATION_BATCH) -> int:
    """Measure amortized peak GPU memory per image over a small calibration batch.

    Running with cal_batch > 1 amortizes PyTorch's fixed per-batch allocator
    blocks so the per-image estimate reflects actual large-batch inference cost,
    not the inflated cost seen when batch_size=1.

    Falls back to ``_HOPTIMUS_BYTES_PER_IMAGE_FALLBACK`` on any error.
    """
    if not torch.cuda.is_available():
        return _HOPTIMUS_BYTES_PER_IMAGE_FALLBACK
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        dummy = torch.zeros(cal_batch, 3, 224, 224, device=dev)
        # Calibrate in BF16 so the measured cost reflects the actual inference
        # dtype; BF16 activations are half the size of FP32.
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(dummy)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del dummy
        torch.cuda.empty_cache()
        per_image = max((peak - baseline) // cal_batch, 1 * 1024 * 1024)  # at least 1 MiB
        return per_image
    except Exception:
        torch.cuda.empty_cache()
        return _HOPTIMUS_BYTES_PER_IMAGE_FALLBACK


def _auto_batch_size(
    model: nn.Module,
    dev: str,
    safety: float = 0.95,
    min_batch: int = 8,
    max_batch: int = 8192,
    bytes_per_image: Optional[int] = None,
) -> int:
    """Return a batch size that fills ~75% of available VRAM across all GPUs.

    Calibrates per-image memory cost by running a single dummy forward pass,
    then queries free VRAM on the primary device, computes per-GPU capacity,
    and multiplies by *ngpu* so DataParallel distributes a full per-GPU load
    to every device.

    Falls back to *min_batch* on CPU or if VRAM introspection is unavailable.
    """
    if not torch.cuda.is_available():
        return min_batch
    try:
        if bytes_per_image is None:
            bytes_per_image = _calibrate_bytes_per_image(model, dev)
        n_gpu = max(1, torch.cuda.device_count())
        # Query free VRAM on the primary device (model already loaded there).
        free_bytes, _total = torch.cuda.mem_get_info(torch.cuda.current_device())
        # Per-GPU capacity: how many images fit in one GPU's free VRAM.
        per_gpu = int(free_bytes * safety) // bytes_per_image
        per_gpu = max(min_batch, per_gpu)
        # Total batch scaled by number of GPUs; round to nearest multiple of
        # n_gpu so DataParallel always distributes evenly.
        total = per_gpu * n_gpu
        total = max(min_batch, min(max_batch, (total // n_gpu) * n_gpu))
        return total
    except Exception:
        return min_batch


def _embed_hoptimus_subset_dataset(
    dataset: Dataset, sampled_ids: List[int],
    batch_size: Optional[int] = None, device: Optional[str] = None,
    hoptimus_model_dir: Optional[Path] = None,
    slide_id: Optional[str] = None,
    display_id: Optional[str] = None,
) -> np.ndarray:
    """
    Embed only a subset of cells using H-Optimus-0.
    'dataset' must support __getitem__(cell_id) -> PIL.Image or Tensor for that cell.
    If hoptimus_model_dir is provided, the model is loaded from that local directory
    instead of being downloaded from HuggingFace.
    """
    import json
    import logging as _logging
    import timm
    from timm.data import create_transform, resolve_data_config
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # Suppress timm's "Loaded from checkpoint" INFO messages — they clutter
    # the nested progress bar display without adding actionable information.
    _timm_logger = _logging.getLogger("timm")
    _timm_prev_level = _timm_logger.level
    _timm_logger.setLevel(_logging.WARNING)
    try:
      if hoptimus_model_dir is not None:
        hoptimus_model_dir = Path(hoptimus_model_dir)
        config_path = hoptimus_model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"--hoptimus-model-dir does not contain config.json: {hoptimus_model_dir}"
            )
        with open(config_path) as _f:
            _cfg = json.load(_f)
        architecture = _cfg.get("architecture")
        if not architecture:
            raise ValueError(
                f"config.json in --hoptimus-model-dir does not contain 'architecture' key: {config_path}"
            )
        checkpoint_candidates = [
            hoptimus_model_dir / "pytorch_model.bin",
            hoptimus_model_dir / "model.safetensors",
            hoptimus_model_dir / "pytorch_model.safetensors",
        ]
        checkpoint_path = next((p for p in checkpoint_candidates if p.exists()), None)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"--hoptimus-model-dir does not contain a recognised checkpoint file "
                f"(pytorch_model.bin or model.safetensors): {hoptimus_model_dir}"
            )
        # Build pretrained_cfg from the values stored in config.json so that
        # create_transform picks up the correct mean/std/input_size.
        pretrained_cfg_overlay = _cfg.get("pretrained_cfg", {})
        model = timm.create_model(
            architecture,
            pretrained=False,
            num_classes=_cfg.get("num_classes", 0),
            global_pool=_cfg.get("global_pool", "token"),
            pretrained_cfg_overlay=pretrained_cfg_overlay,
        )
        timm.models.load_checkpoint(model, str(checkpoint_path))
        model = model.to(dev).eval()
      else:
        model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, num_classes=0).to(dev).eval()
    finally:
        _timm_logger.setLevel(_timm_prev_level)
    pre = create_transform(**resolve_data_config(model=model), is_training=False)

    # Calibrate per-image memory cost on the plain single-GPU model BEFORE
    # replicating to other GPUs.  A dummy batch of 1 on the plain model gives
    # the true per-image activation footprint without replication overhead.
    _bytes_per_image = _calibrate_bytes_per_image(model, dev)

    # Build one persistent model replica per GPU.
    # Unlike nn.DataParallel (which re-replicates weights every forward pass),
    # this copies the model once per slide and dispatches sub-batches in
    # parallel via threads — each GPU runs independently with no gather bottleneck.
    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if ngpu > 1 and next(model.parameters()).is_cuda:
        import copy
        from concurrent.futures import ThreadPoolExecutor as _TPE
        _replicas = [model] + [
            copy.deepcopy(model).to(f"cuda:{i}").eval() for i in range(1, ngpu)
        ]
    else:
        _replicas = [model]
        ngpu = max(ngpu, 1)

    # Determine batch size dynamically from available VRAM when caller passes None.
    if batch_size is None:
        batch_size = _auto_batch_size(model, dev, bytes_per_image=_bytes_per_image)
        _logging.getLogger(__name__).debug(
            "H-optimus auto batch_size=%d (ngpu=%d, per-GPU≈%d, bytes_per_image=%dMB)",
            batch_size, ngpu, batch_size // ngpu, _bytes_per_image // 1024 // 1024,
        )

    # Build a Subset where sample index equals the cell_id we want
    subset = Subset(dataset, sampled_ids)

    def _preprocess_batch_cpu(batch) -> torch.Tensor:
        """Preprocess batch to a CPU tensor; GPU transfer done per-replica."""
        if isinstance(batch, list):
            return torch.stack([pre(im) for im in batch])
        elif isinstance(batch, torch.Tensor):
            if batch.dim() == 4 and batch.shape[1] in (1, 3):
                return torch.stack([pre(transforms.ToPILImage()(t)) for t in batch])
            return torch.stack([pre(b) for b in batch])
        else:
            try:
                return torch.stack([pre(im) for im in batch])
            except Exception:
                return batch if isinstance(batch, torch.Tensor) else torch.stack(list(batch))

    def _run_on_replica(replica_idx: int, x_cpu: torch.Tensor) -> torch.Tensor:
        """Transfer sub-batch to one GPU replica and run BF16 forward pass."""
        x = x_cpu.to(f"cuda:{replica_idx}")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return _replicas[replica_idx](x).float().cpu()  # cast back to FP32 for downstream

    def _forward(x_cpu: torch.Tensor) -> torch.Tensor:
        """Dispatch a preprocessed CPU tensor across all GPU replicas in parallel."""
        if len(_replicas) > 1:
            splits = x_cpu.chunk(len(_replicas), dim=0)
            with _TPE(max_workers=len(_replicas)) as _pool:
                _futs = [_pool.submit(_run_on_replica, ri, s) for ri, s in enumerate(splits)]
                return torch.cat([f.result() for f in _futs], dim=0)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return _replicas[0](x_cpu.to(dev)).float().cpu()

    # ── Adaptive inference loop ───────────────────────────────────────────────
    # Manual index-based batching (no DataLoader rebuild) so we can grow/shrink
    # batch_size freely at any point without restarting iteration:
    #   • OOM → halve batch_size, clear caches, retry immediately
    #   • Every _GROW_EVERY successful batches → probe free VRAM; grow back if
    #     memory has been freed by other processes
    _GROW_EVERY = 10   # how often to probe for growth opportunity
    _n_ok = 0          # consecutive successful batches since last resize
    _label = display_id or slide_id or "H-optimus"
    desc = f"  [{_label}]"
    feats: list = []
    n_total = len(sampled_ids)
    pos = 0            # index into sampled_ids

    pbar = tqdm(total=math.ceil(n_total / batch_size), desc=desc,
                leave=False, position=1, unit="batch")

    while pos < n_total:
        end = min(pos + batch_size, n_total)
        ids_chunk = sampled_ids[pos:end]
        items = [dataset[idx] for idx in ids_chunk]

        while True:   # OOM retry loop for this chunk
            try:
                x_cpu = _preprocess_batch_cpu(items)
                feats.append(_forward(x_cpu))
                pbar.update(1)
                pos += len(items)
                _n_ok += 1

                # Periodically probe free VRAM and grow batch_size if space freed up.
                if _n_ok % _GROW_EVERY == 0:
                    candidate = _auto_batch_size(
                        _replicas[0], dev, bytes_per_image=_bytes_per_image)
                    if candidate > batch_size:
                        batch_size = candidate
                        remaining_batches = math.ceil((n_total - pos) / batch_size)
                        pbar.reset(total=pbar.n + remaining_batches)
                break

            except torch.cuda.OutOfMemoryError:
                for ri in range(len(_replicas)):
                    with torch.cuda.device(ri):
                        torch.cuda.empty_cache()
                new_bs = max(len(_replicas), len(items) // 2)
                if new_bs >= len(items):
                    raise   # already at minimum; propagate
                _logging.getLogger(__name__).warning(
                    "H-optimus OOM at batch_size=%d; retrying with %d",
                    len(items), new_bs,
                )
                batch_size = new_bs
                items = items[:new_bs]   # retry with the smaller chunk; remainder picked up in next iteration
                _n_ok = 0
                remaining_batches = math.ceil((n_total - pos) / batch_size)
                pbar.reset(total=pbar.n + remaining_batches)

    pbar.close()
    return torch.cat(feats, dim=0).numpy().astype(np.float32)

def _impute_knn(coords_um: np.ndarray, sampled_idx: np.ndarray, sampled_feats: np.ndarray,
                k: int = 3, sigma_um: float = 60.0) -> np.ndarray:
    """Distance-weighted KNN imputation in microns: w = exp(-(d/sigma)^2)."""
    from scipy.spatial import cKDTree
    # N = coords_um.shape[0]
    tree = cKDTree(coords_um[sampled_idx])
    d, nn = tree.query(coords_um, k=min(k, len(sampled_idx)))
    if k == 1 or np.ndim(nn) == 1:
        d = d[:, None]; nn = nn[:, None]
    eps = 1e-8
    W = np.exp(- (d / max(sigma_um, eps)) ** 2).astype(np.float32) + eps
    W /= W.sum(axis=1, keepdims=True)
    H = sampled_feats[nn]                      # [N,k,D]
    return (W[..., None] * H).sum(axis=1).astype(np.float32)


# =============================================================================
# PyG: GCN + DGI (shared across slides)
# =============================================================================

# --- unchanged encoder ---
class GCLEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=32, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index):   # exactly as you had
        h = self.drop(self.act(self.conv1(x, edge_index)))
        z = self.conv2(h, edge_index)
        return z


# --- DGI wrapper: DO NOT change your encoder; just read its output dim ---
class DGIModule(nn.Module):
    """Wrap DGI so it can accept a PyG Data object; does NOT change encoder behavior."""
    def __init__(self, encoder: nn.Module):
        super().__init__()

        # read the encoder's true output dimension (prevents 32/64 mismatch)
        if not hasattr(encoder, "conv2") or not hasattr(encoder.conv2, "out_channels"):
            raise ValueError("Encoder must expose conv2.out_channels")
        enc_out_dim = int(encoder.conv2.out_channels)

        def summary(z, *args, **kwargs):
            return torch.sigmoid(z.mean(dim=0))

        # corruption with two args (what DGI expects)
        def corruption(x_in, edge_in):
            perm = torch.randperm(x_in.size(0), device=x_in.device)
            return x_in[perm], edge_in

        # hidden_channels MUST equal encoder output dim
        self.dgi = DeepGraphInfomax(
            hidden_channels=enc_out_dim,
            encoder=encoder,
            summary=summary,
            corruption=corruption,
        )

    def forward(self, data: Data):
        # Call DGI EXACTLY like single-GPU path (no batch passed to your encoder)
        return self.dgi(data.x, data.edge_index)

    def loss(self, pos_z, neg_z, s):
        # return self.dgi.loss(pos_z, neg_z, s)
        # Ensure summary vector is 1D [hidden], regardless of DataParallel gather
        hd = self.dgi.hidden_channels
        if s.ndim != 1 or s.numel() != hd:
            s = s.reshape(-1, hd).mean(dim=0)   # collapse [num_replicas*hidden] or [R, hidden] -> [hidden]
        return self.dgi.loss(pos_z, neg_z, s)



def train_dgi_multi(slides, hidden=64, out_dim=32, epochs=300, lr=1e-3, wd=1e-4, seed=0,
                    amp=False,
                    early_stop_patience=20, early_stop_min_delta=1e-4,
                    early_stop_min_epochs=50):
    """Train a shared DGI encoder across slide graphs and return embeddings.

    The encoder consumes the raw k-hop composition ``s["X"]`` directly: the
    features are already proportions on a common [0, 1] scale, and keeping them
    untransformed means a given neighbourhood maps to the same input regardless
    of which slides shared the run.

    ``amp`` enables CUDA automatic mixed precision (no-op on CPU/MPS).  Early
    stopping is always active: ``epochs`` is the upper bound, and training stops
    once the mean epoch loss fails to improve by more than ``early_stop_min_delta``
    (relative to the best loss) for ``early_stop_patience`` consecutive epochs,
    but never before ``early_stop_min_epochs``.
    """
    # Seed all RNGs so the encoder init, mini-batch shuffling and DGI corruption
    # are reproducible across runs for a given seed.
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _loader_gen = torch.Generator()
    _loader_gen.manual_seed(seed)

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    primary = torch.device('cuda:0' if ngpu > 0 else 'cpu')

    in_dim = slides[0]["X"].shape[1]
    enc = GCLEncoder(in_dim, hidden, out_dim).to(primary)

    # Build DGI using the encoder's own output dim (prevents size mismatch)
    model = DGIModule(enc)

    # graphs
    data_list = [Data(x=torch.from_numpy(s["X"]).float(),
                      edge_index=torch.from_numpy(s["edge_index"]).long())
                 for s in slides]

    if ngpu > 1:
        # per_gpu_graphs = 2
        # batch_size = per_gpu_graphs * ngpu
        
        
        # ngpu = torch.cuda.device_count()
        # max_per = 8  # don’t go crazy; DP overhead grows
        per_gpu_graphs = 1
        
        # simple ramp-up to find what fits
        for cand in range(4, 0, -1):  # try 4,3,2,1
            try:
                test_bs = cand * max(1, ngpu)
                _ = DataListLoader(data_list[:test_bs], batch_size=test_bs).__iter__().__next__()
                per_gpu_graphs = cand
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
        batch_size = per_gpu_graphs * max(1, ngpu)

        
        
        
        loader = DataListLoader(data_list, batch_size=batch_size, shuffle=True, generator=_loader_gen)
        model = GeoDataParallel(model, device_ids=list(range(ngpu))).to(primary)
    else:
        loader = GeoDataLoader(data_list, batch_size=1, shuffle=True, generator=_loader_gen)
        model = model.to(primary)

    # sanity print once; should be equal (e.g., 32)
    enc_out = enc.conv2.out_channels
    if ngpu > 1:
        print(f"[DGI check] encoder_out_dim={enc_out}, dgi_hidden={model.module.dgi.hidden_channels}")
        assert model.module.dgi.hidden_channels == enc_out
    else:
        print(f"[DGI check] encoder_out_dim={enc_out}, dgi_hidden={model.dgi.hidden_channels}")
        assert model.dgi.hidden_channels == enc_out

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Automatic mixed precision (CUDA only). The GradScaler/autocast are gated by
    # ``enabled`` so that with amp=False the math path is identical to plain FP32.
    use_amp = bool(amp) and primary.type == "cuda"
    if primary.type == "cuda":
        torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Early-stopping state (patience on the mean epoch loss; --epochs is the cap).
    best_loss = float("inf")
    epochs_no_improve = 0

    # training (works for single or multi-GPU)
    for epoch in tqdm(range(epochs)):
        epoch_loss_sum = 0.0
        n_batches = 0
        for batch in loader:
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                if ngpu > 1:
                    pos_z, neg_z, s = model(batch)
                    loss = model.module.loss(pos_z, neg_z, s)
                else:
                    batch = batch.to(primary)
                    pos_z, neg_z, s = model(batch)
                    loss = model.loss(pos_z, neg_z, s)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            epoch_loss_sum += float(loss.detach())
            n_batches += 1

        if n_batches > 0:
            epoch_loss = epoch_loss_sum / n_batches
            # Relative improvement threshold so it adapts to the loss scale.
            # First finite epoch always seeds best_loss (comparing against inf
            # yields NaN, so guard it explicitly).
            if not math.isfinite(best_loss) or epoch_loss < best_loss - early_stop_min_delta * max(abs(best_loss), 1.0):
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if (epoch + 1) >= early_stop_min_epochs and epochs_no_improve >= early_stop_patience:
                print(f"[DGI early-stop] no improvement > {early_stop_min_delta} (relative) for "
                      f"{early_stop_patience} epochs; stopping at epoch {epoch + 1}/{epochs} "
                      f"(best mean loss={best_loss:.4f}).")
                break



    # inference (unchanged)
    enc_eval = (model.module.dgi.encoder if ngpu > 1 else model.dgi.encoder).to(primary)
    enc_eval.eval()
    Z_list = []
    with torch.no_grad():
        for s in slides:
            x = torch.from_numpy(s["X"]).float().to(primary)
            ei = torch.from_numpy(s["edge_index"]).long().to(primary)
            Z_list.append(enc_eval(x, ei).cpu().numpy().astype(np.float32))
    return enc_eval, Z_list

# def train_dgi_multi(slides: List[Dict[str, np.ndarray]],
#                     hidden: int = 64, out_dim: int = 32,
#                     epochs: int = 300, lr: float = 1e-3, wd: float = 1e-4,
#                     device: Optional[str] = None) -> Tuple[nn.Module, List[np.ndarray]]:
#     """Train one DGI encoder across slides; return (encoder, [embeddings_per_slide])."""
#     dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
#     in_dim = slides[0]["X"].shape[1]
#     enc = GCLEncoder(in_dim, hidden, out_dim).to(dev)
#
#     def summary(z, *args, **kwargs): return torch.sigmoid(z.mean(dim=0))
#     def corruption(x_in, edge_in):
#         perm = torch.randperm(x_in.size(0), device=x_in.device)
#         return x_in[perm], edge_in
#
#     dgi = DeepGraphInfomax(hidden_channels=out_dim, encoder=enc, summary=summary, corruption=corruption).to(dev)
#     opt = torch.optim.Adam(dgi.parameters(), lr=lr, weight_decay=wd)
#
#     # Build PyG DataLoader
#     data_list = [Data(x=torch.from_numpy(s["X"]).float(),
#                       edge_index=torch.from_numpy(s["edge_index"]).long())
#                  for s in slides]
#     loader = GeoDataLoader(data_list, batch_size=1, shuffle=True)
#
#     dgi.train()
#     for _ in range(epochs):
#         for batch in loader:
#             batch = batch.to(dev)
#             opt.zero_grad()
#             pos_z, neg_z, s = dgi(batch.x, batch.edge_index)
#             loss = dgi.loss(pos_z, neg_z, s)
#             loss.backward()
#             opt.step()
#
#     # Inference per slide
#     enc.eval()
#     Z_list = []
#     with torch.no_grad():
#         for s in slides:
#             x = torch.from_numpy(s["X"]).float().to(dev)
#             ei = torch.from_numpy(s["edge_index"]).long().to(dev)
#             Z = enc(x, ei).cpu().numpy().astype(np.float32)
#             Z_list.append(Z)
#     return enc, Z_list






# =============================================================================
# Slide building (reusing YOUR functions) + optional H-Optimus
# =============================================================================

def prepare_slide_graph(
    niche_detection_df: pd.DataFrame,
    mpp_um_per_px: float,
    max_edge_len_um: float,
    class_order: Optional[List[str]] = None,
    k_hops: int = 2,
    alpha: float = 1.0,
    # H-Optimus
    use_hoptimus: bool = False,
    hoptimus_only: bool = False,
    hoptimus_model_dir: Optional[Path] = None,
    patch_dataset: Optional[Dataset] = None,  # your dataset: __getitem__(cell_id)-> PIL.Image / Tensor
    sample_frac: Optional[float] = 0.2,
    sample_count: Optional[int] = None,
    pca_dim: Optional[int] = 32,
    knn_k: int = 3,
    knn_sigma_um: float = 60.0,
    device: Optional[str] = None,
    graph_cache_dir: Optional[Path] = None,
    slide_id: Optional[str] = None,
    display_id: Optional[str] = None,
    mode: str = "hard",
    # seed: int = 0
) -> Dict[str, np.ndarray]:
    """
    Build one slide graph:
      - centers from bbox (your fn)
      - Delaunay + distance cap (your fn) in *pixels*
      - drop isolated cells
      - features = k-hop soft-composition (always) + optional H-Optimus (sample+KNN impute)
    Returns: {'X','edge_index','kept_idx','classes'}
    """
    # rng = np.random.default_rng(seed)
    rng = np.random.default_rng()

    # centers in px (your function)
    df = compute_cell_center_points(niche_detection_df.copy())
    centers_px = df[["center_x", "center_y"]].to_numpy(dtype=np.float32)
    N = len(df)

    # Delaunay + cap in px — reuse shared graph cache when available
    max_edge_len_px = float(max_edge_len_um) / float(mpp_um_per_px)
    if graph_cache_dir is not None and slide_id is not None:
        centers_int = np.asarray(centers_px, dtype=np.int32)
        edges_df = get_or_build_delaunay(
            graph_cache_dir, slide_id, centers_int, mpp_um_per_px, max_edge_len_px
        )
    else:
        edges_df = delaunay_triangulation(centers_px, max_edge_len_px)

    # edge_index (undirected), drop isolated
    edge_index = to_edge_index(edges_df, src_col="source", dst_col="target", undirected=True, drop_self_loops=True)
    edge_index, kept_idx = drop_isolated(edge_index, N)
    if kept_idx.size == 0:
        raise ValueError("All nodes are isolated after distance cap; nothing to train.")
    N_kept = len(kept_idx)

    # probs (soft) → subset kept
    P_all, classes = probs_from_df(df, class_order=class_order)  # [N,C]
    # P_all, _ = probs_from_df(df, class_order=class_order)  # [N,C]
    P = P_all[kept_idx]  # [N_kept,C]

    if hoptimus_only and not use_hoptimus:
        raise ValueError("hoptimus_only=True requires use_hoptimus=True")

    # k-hop features (default path unless H-Optimus-only mode is requested).
    X_khop = khop_features(P=P, edge_index=edge_index, N=N_kept, k=k_hops, alpha=alpha, mode=mode)  # [N_kept,(k+1)C]
    blocks = [] if hoptimus_only else [X_khop.astype(np.float32)]
    hoptimus_dim = 0

    # Optional: H-Optimus (sample subset via DataLoader, then KNN impute to all)
    if use_hoptimus:
        # coords in microns for KNN weighting
        coords_um = centers_px[kept_idx] * float(mpp_um_per_px)

        # ensure dataset provided; if not, use dummy
        if patch_dataset is None:
            patch_dataset = DummyPatchDataset(num_cells=N)

        # choose sample
        if sample_count is not None:
            m = max(1, min(int(sample_count), N_kept))
        else:
            frac = float(sample_frac or 0.2)
            m = max(1, min(int(round(frac * N_kept)), N_kept))
        sampled_local_idx = np.sort(rng.choice(N_kept, size=m, replace=False))        # indices in kept space
        sampled_global_ids = kept_idx[sampled_local_idx].tolist()                      # map to original IDs for dataset

        # embed sampled
        Hs = _embed_hoptimus_subset_dataset(patch_dataset, sampled_global_ids, device=device, hoptimus_model_dir=hoptimus_model_dir, slide_id=slide_id, display_id=display_id)  # [m,1536]
        # optional PCA
        if pca_dim is not None and Hs.shape[1] > pca_dim:
            from sklearn.decomposition import PCA
            # Hs = PCA(n_components=pca_dim, random_state=seed).fit_transform(Hs).astype(np.float32)
            Hs = PCA(n_components=pca_dim).fit_transform(Hs).astype(np.float32)

        # KNN impute to all kept nodes (micron distances)
        H_full = _impute_knn(coords_um=coords_um, sampled_idx=sampled_local_idx,
                             sampled_feats=Hs, k=knn_k, sigma_um=knn_sigma_um)  # [N_kept,D]
        hoptimus_dim = int(H_full.shape[1])
        blocks.append(H_full.astype(np.float32))

    # concatenate feature blocks (khop [+ H0])
    X = np.hstack(blocks).astype(np.float32)

    return {
        "X": X,
        "edge_index": edge_index.astype(np.int64),
        "kept_idx": kept_idx.astype(np.int64),
        "classes": classes,
        "edges_df": edges_df,
        "khop_dim": int(X_khop.shape[1]),
        "hoptimus_dim": int(hoptimus_dim),
        "hoptimus_only": bool(hoptimus_only),
    }
    
# =============================================================================
# End-to-end multi-image training + clustering
# =============================================================================

def _approx_knn_connectivity(Z: np.ndarray, k_nn: int = 15, seed: int = 0):
    """Approximate symmetric kNN connectivity via pynndescent (then faiss).

    Returns a symmetric CSR connectivity matrix with the same semantics as
    :func:`_knn_graph_connectivity` (unweighted, self-loops removed), or
    ``None`` if no approximate-NN backend is importable.
    """
    import scipy.sparse as sp
    n = Z.shape[0]
    Zf = np.ascontiguousarray(Z, dtype=np.float32)
    idx = None
    try:
        from pynndescent import NNDescent
        index = NNDescent(Zf, n_neighbors=k_nn + 1, metric="euclidean",
                          random_state=seed)
        idx, _ = index.neighbor_graph            # idx: [n, k_nn+1] incl. self
    except Exception:
        try:
            import faiss
            fi = faiss.IndexFlatL2(Zf.shape[1])
            fi.add(Zf)
            _, idx = fi.search(Zf, k_nn + 1)     # idx: [n, k_nn+1] incl. self
        except Exception:
            return None
    idx = np.asarray(idx)
    rows = np.repeat(np.arange(n, dtype=np.int64), idx.shape[1])
    cols = idx.reshape(-1).astype(np.int64)
    keep = rows != cols                          # drop self-matches
    rows, cols = rows[keep], cols[keep]
    data = np.ones(rows.shape[0], dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A = A.maximum(A.T).tocsr()                   # symmetrize
    return A


def _knn_graph_connectivity(Z: np.ndarray, k_nn: int = 15, seed: int = 0):
    """Symmetric kNN connectivity using approximate NN (pynndescent, then faiss).

    Approximate NN is the only intended path; the exact sklearn kNN is kept
    solely as a safety net for environments where neither pynndescent nor faiss
    is importable, so the pipeline never hard-fails.
    """
    A = _approx_knn_connectivity(Z, k_nn=k_nn, seed=seed)
    if A is not None:
        return A
    click.secho(
        "  no approximate-kNN backend (pynndescent/faiss) is available; "
        "falling back to exact sklearn kNN.", fg="yellow")
    A = kneighbors_graph(Z, n_neighbors=k_nn, mode='connectivity', include_self=False)
    A = A.maximum(A.T).tocsr()  # symmetrize
    return A

def _igraph_from_sparse(A) -> ig.Graph:
    """Convert a scipy sparse adjacency matrix to an undirected igraph graph."""
    A = A.tocoo()
    g = ig.Graph(n=A.shape[0], edges=list(zip(A.row.tolist(), A.col.tolist())), directed=False)
    g.simplify(combine_edges="ignore")
    return g

# ---------------- worker ----------------
def _leiden_worker(n_nodes: int, 
                   edges: np.ndarray, 
                   resolution: float, 
                   seed: int,
                   ) -> Tuple[np.ndarray, float, float]:
    """Run a single Leiden clustering pass and return labels plus modularity."""
    g_local = ig.Graph(n=n_nodes, edges=edges.tolist(), directed=False)
    g_local.simplify(combine_edges="ignore")
    part = la.find_partition(
        g_local, la.RBConfigurationVertexPartition,
        resolution_parameter=float(resolution),
        seed=int(seed),
    )
    labels = np.asarray(part.membership, dtype=int)
    return labels, float(part.modularity), float(resolution)

def _reduce_resolution_worker(args):
    """Summarize repeated Leiden runs for one resolution value."""
    r, runs, Z = args
    # choose best modularity run as representative
    best_labels, best_mod = max(runs, key=lambda x: x[1])

    # Stability: average NMI to best (skip degenerate single-cluster cases)
    nmis = []
    if len(np.unique(best_labels)) > 1:
        for lab, _ in runs:
            if len(np.unique(lab)) > 1:
                nmis.append(normalized_mutual_info_score(lab, best_labels))
    stability = float(np.mean(nmis)) if nmis else 0.0

    # Silhouette on Z if ≥2 clusters
    if len(np.unique(best_labels)) > 1:
        sil = float(silhouette_score(Z, best_labels, sample_size=np.min([len(Z), 10000]), metric='euclidean'))
    else:
        sil = -1.0

    counts = np.bincount(best_labels)
    min_frac = float(counts.min() / counts.sum()) if counts.size else 0.0

    log = {
        "resolution": float(r),
        "n_clusters": int(len(np.unique(best_labels))),
        "modularity": float(best_mod),
        "stability": stability,
        "silhouette": sil,
        "min_frac": min_frac,
        "labels": best_labels,
    }
    return log

def _leiden_sweep_on_graph(
    Z: np.ndarray,
    g: ig.Graph,
    niche_clustering_resolutions: Iterable[float] = np.arange(0.2, 2.05, 0.1),
    n_repeats: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Parallel sweep over (resolution, repeat) AND parallel reduction per resolution.
    Returns {"winner": {...}, "all": [ per-resolution dicts ... ]}.
    """
    rng = np.random.default_rng(seed)

    # Convert igraph to (n_nodes, edges) once (edges is picklable)
    n_nodes = g.vcount()
    el = np.array(g.get_edgelist(), dtype=np.int64)
    if el.size == 0:
        labels = np.zeros(n_nodes, dtype=int)
        out = {
            "resolution": float(next(iter(niche_clustering_resolutions), 1.0)),
            "n_clusters": 1, "modularity": 0.0, "stability": 1.0,
            "silhouette": -1.0, "min_frac": 1.0, "labels": labels,
        }
        return {"winner": out, "all": [out]}

    # ---- Phase A: parallel Leiden runs over (resolution, repeat) ----
    tasks = []
    for r in niche_clustering_resolutions:
        for _ in range(n_repeats):
            tasks.append((n_nodes, el, float(r), int(rng.integers(1_000_000_000))))

    n_jobs = pick_workers_safe(max_workers=os.cpu_count()-2, min_workers=2)
    results_by_r: Dict[float, list] = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = [ex.submit(_leiden_worker, *t) for t in tasks]
        for fut in as_completed(futs):
            throttle_when_busy()
            labels, modularity, r = fut.result()
            results_by_r.setdefault(r, []).append((labels, modularity))

    # ---- Phase B: parallel reduction per resolution ----
    logs = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = [ex.submit(_reduce_resolution_worker, (r, results_by_r[r], Z)) for r in results_by_r.keys()]
        for fut in as_completed(futs):
            throttle_when_busy()
            logs.append(fut.result())

    # Keep logs sorted by resolution (optional)
    logs.sort(key=lambda d: d["resolution"])

    # Pick winner (stable + high modularity; avoid tiny clusters)
    filtered = [d for d in logs if d["min_frac"] >= 0.005] or logs
    winner = sorted(filtered, key=lambda d: (d["stability"], d["modularity"], d["silhouette"]), reverse=True)[0]

    return {"winner": winner, "all": logs}


def estimate_niches_from_Z_list(
    Z_list: List[np.ndarray],
    mode: str = "global",           # "global" (recommended) or "per_slide"
    k_nn: int = 15,
    niche_clustering_resolutions = np.arange(0.2, 2.05, 0.1),
    n_repeats: int = 5,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "clusters_k": int,
        "labels_list": List[np.ndarray],    # per slide labels
        "winner": dict,                     # chosen sweep result (resolution, metrics)
        "all_results": List[dict] or List[List[dict]]   # sweep logs
      }
    """
    if mode == "global":
        # concat for a single clustering (consistent niche IDs across slides)
        offsets = np.cumsum([0] + [Z.shape[0] for Z in Z_list[:-1]])
        Z_all = np.vstack(Z_list)
        A = _knn_graph_connectivity(Z_all, k_nn=k_nn, seed=seed)
        g = _igraph_from_sparse(A)
        sweep = _leiden_sweep_on_graph(Z_all, 
                                       g, 
                                       niche_clustering_resolutions=niche_clustering_resolutions, 
                                       n_repeats=n_repeats, 
                                       seed=seed,
                                       )
        w = sweep["winner"]
        labels_all = w["labels"]
        # split back to per slide
        labels_list = []
        for off, Z in zip(offsets, Z_list):
            labels_list.append(labels_all[off:off+len(Z)])
        return {
            "clusters_k": w["n_clusters"],
            "labels_list": labels_list,
            "winner": w,
            "all_results": sweep["all"],
        }

    elif mode == "per_slide":
        labels_list = []
        winners = []
        all_logs = []
        n_clusters_list = []
        for Z in Z_list:
            A = _knn_graph_connectivity(Z, k_nn=k_nn, seed=seed)
            g = _igraph_from_sparse(A)
            sweep = _leiden_sweep_on_graph(Z, 
                                           g, 
                                           niche_clustering_resolutions=niche_clustering_resolutions, 
                                           n_repeats=n_repeats, 
                                           seed=seed,
                                           )
            w = sweep["winner"]
            labels_list.append(w["labels"])
            winners.append(w)
            all_logs.append(sweep["all"])
            n_clusters_list.append(w["n_clusters"])
        # You can keep per-slide cluster counts, or choose a consensus (e.g., median)
        return {
            "clusters_k": int(np.median(n_clusters_list)),
            "labels_list": labels_list,
            "winner": winners,        # list of winners per slide
            "all_results": all_logs,  # list per slide
        }
    else:
        raise ValueError("mode must be 'global' or 'per_slide'")


def _prepare_slide_graph_worker(i, wsi_path, csv_path, ds,
        max_edge_len_um, class_order, k_hops, alpha,
        sample_frac, sample_count, pca_dim, knn_k, knn_sigma_um,
    device, niche_soft_mode, use_hoptimus, hoptimus_only, hoptimus_model_dir, graph_cache_dir):
    """Background worker to build one slide graph and return it with index."""
    df = pd.read_csv(csv_path)
    mpp = get_avg_mpp(wsi_path)
    slide_id = Path(wsi_path).stem
    s = prepare_slide_graph(
        df,
        mpp_um_per_px=mpp,
        max_edge_len_um=max_edge_len_um,
        class_order=class_order,
        k_hops=k_hops, alpha=alpha,
        hoptimus_only=hoptimus_only,
        hoptimus_model_dir=hoptimus_model_dir,
        use_hoptimus=use_hoptimus, patch_dataset=ds,
        sample_frac=sample_frac, sample_count=sample_count,
        pca_dim=pca_dim, knn_k=knn_k, knn_sigma_um=knn_sigma_um,
        device=device,
        graph_cache_dir=graph_cache_dir,
        slide_id=slide_id,
        mode="soft" if niche_soft_mode else "hard",
    )
    return i, s


def _niche_cellular_worker(args):
    """Background worker: write one slide's per-cell niche CSV (Phase 4).

    Each slide is independent, so this is safe to run in a process pool. The CSV
    is written to a temp file and atomically renamed to protect against partial
    writes on interruption (the parent handles cancellation between futures).
    """
    (wsi_path, model_output_csv, kept_idx, X_raw, classes,
     labels, k_hops, niche_clustering_k, cell_csv, overwrite,
     hoptimus_only, khop_dim) = args
    cell_csv = Path(cell_csv)
    if not overwrite and cell_csv.exists():
        return str(cell_csv), "skip"

    niche_detection_df = pd.read_csv(model_output_csv)
    if hoptimus_only:
        feature_cols = [f"hoptimus_feature_{j}" for j in range(X_raw.shape[1])]
        niche_detection_df.loc[kept_idx, feature_cols] = X_raw
    else:
        feature_cols = [f"feature_k{k}_{c.replace('prob_', '')}"
                        for k in range(k_hops + 1) for c in classes]
        # Keep the CSV contract stable: export only the k-hop feature block even
        # when niche training used concatenated k-hop + H-Optimus features.
        niche_detection_df.loc[kept_idx, feature_cols] = X_raw[:, :int(khop_dim)]
    niche_cols = [f"niche_{l}" for l in range(niche_clustering_k)]
    label_one_hot = np.eye(niche_clustering_k, dtype=np.float32)[labels]
    niche_detection_df.loc[kept_idx, niche_cols] = label_one_hot

    tmp = str(cell_csv) + ".tmp"
    niche_detection_df.to_csv(tmp, index=False)
    os.replace(tmp, cell_csv)
    return str(cell_csv), "ok"


def _niche_annotation_worker(args):
    """Background worker: write one slide's annotation-level niche CSV (Phase 5)."""
    (wsi_path, cell_csv, niche_csv, kept_idx, edges_df,
     niche_clustering_k, max_cell_radius_um, overwrite) = args
    niche_csv = Path(niche_csv)
    if not overwrite and niche_csv.exists():
        return str(niche_csv), "skip"

    mpp = get_avg_mpp(wsi_path)
    niche_detection_df = pd.read_csv(cell_csv)
    valid_mask = np.zeros(len(niche_detection_df), dtype=bool)
    valid_mask[np.asarray(kept_idx, dtype=int)] = True
    edges_df = remap_edges_to_valid_indices(edges_df, valid_mask)

    niche_annotation_df = merge_same_label_by_shared_edges_iterative(
        niche_detection_df,
        edges_df,
        niche_clustering_k=niche_clustering_k,
        mpp=mpp,
        max_radius_um=max_cell_radius_um,
    )

    tmp = str(niche_csv) + ".tmp"
    niche_annotation_df.to_csv(tmp, index=False)
    os.replace(tmp, niche_csv)
    return str(niche_csv), "ok"


def niche_generation(
    wsi_dir: str | URIPath | None,
    wsi_paths: Sequence[Path | URIPath] | None,
    results_dir: str | Path,
    max_edge_len_um: float,
    max_cell_radius_um: float,
    class_order: Optional[List[str]] = None,
    k_hops: int = 2,
    alpha: float = 1.0,
    # H-Optimus switch & params
    use_hoptimus: bool = False,
    hoptimus_only: bool = False,
    hoptimus_model_dir: Optional[Path] = None,
    patch_datasets: Optional[List[Dataset]] = None,  # list aligned with slides_inputs; if None, Dummy is used
    sample_frac: Optional[float] = 0.2,
    sample_count: Optional[int] = None,
    pca_dim: Optional[int] = 32,
    knn_k: int = 3,
    knn_sigma_um: float = 60.0,
    # encoder
    hidden: int = 64,
    out_dim: int = 32,
    epochs: int = 300,
    # clustering
    niche_cellular: bool = False,
    niche_annotation: bool = False,
    niche_clustering_k: int | None = 10,
    niche_clustering_resolutions: List[float]=[0.5,1.0,2.0],
    # # device
    # device: Optional[str] = None,
    niche_soft_mode: bool = False,
    overwrite: bool = False,
    seed: int = 0,
    # early stopping (always active)
    early_stop_patience: int = 20,
    early_stop_min_delta: float = 1e-4,
    early_stop_min_epochs: int = 50,
    # performance
    amp: bool = False,
) -> Dict[str, List[np.ndarray]]:
    """
    Prepare graphs for multiple slides, train one DGI on the raw k-hop
    composition features, and cluster per slide.
    """
    if hoptimus_only and not use_hoptimus:
        raise ValueError("hoptimus_only=True requires use_hoptimus=True")
    
    if os.getenv("WSINFER_FORCE_CPU", "0").lower() not in {"0", "f", "false"}:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # DataParallel uses all GPUs from cuda:0
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f'Using device "{device}"')
    
    # Make sure required directories exist.
    wsi_dir_path = URIPath(wsi_dir) if wsi_dir is not None else None
    if wsi_dir_path is not None and not wsi_dir_path.exists():
        raise errors.WholeSlideImageDirectoryNotFound(f"directory not found: {wsi_dir_path}")

    if wsi_paths is not None:
        slide_paths = [p if isinstance(p, URIPath) else URIPath(p) for p in wsi_paths]
    elif wsi_dir_path is not None:
        slide_paths = [p for p in wsi_dir_path.iterdir() if p.is_file()]
    else:
        raise ValueError("wsi_paths must be provided when wsi_dir is None")

    if not slide_paths:
        context = wsi_dir_path or "provided slide paths"
        raise errors.WholeSlideImagesNotFound(context)

    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise errors.ResultsDirectoryNotFound(results_dir)

    if wsi_dir_path is not None:
        _validate_wsi_directory(wsi_dir_path)
    else:
        stems = [p.stem for p in slide_paths]
        if len(stems) != len(set(stems)):
            raise errors.DuplicateFilePrefixesFound(
                "A slide with the same prefix but different extensions has been found"
            )

    # Check patches directory.
    model_output_dir = results_dir / "model-outputs-csv"
    if not model_output_dir.exists():
        raise errors.ResultsDirectoryNotFound(
            "The 'model-outputs-csv' directory was not found in results directory."
        )
    # Create the patch paths based on the whole slide image paths. In effect, only
    # create patch paths if the whole slide image patch exists.
    model_output_paths = [model_output_dir / p.with_suffix(".csv").name for p in slide_paths]
    
    if len(model_output_paths) != len(slide_paths):
        raise errors.ResultsDirectoryNotFound(
            "The 'model-outputs-csv' and image directory were mismatched."
        )
    niche_output_dir = results_dir / "niche-outputs-csv"
    niche_output_dir.mkdir(exist_ok=True)
    niche_cells_output_dir = results_dir / "niche-outputs-csv" / "cells"
    niche_cells_output_dir.mkdir(exist_ok=True)
    niche_niches_output_dir = results_dir / "niche-outputs-csv" / "niches"
    niche_niches_output_dir.mkdir(exist_ok=True)
    niche_slide_graph_file = results_dir / "slide-graphs.joblib"
    niche_dgi_embeddings_file = results_dir / "dgi-embeddings.joblib"
    niche_labels_file = results_dir / "niche-labels.joblib"

    # If overwrite requested, remove cached checkpoints so all phases re-run.
    if overwrite:
        for _ckpt in (niche_slide_graph_file, niche_dgi_embeddings_file, niche_labels_file):
            if _ckpt.exists():
                _ckpt.unlink()
        # Also wipe the per-slide graph cache so Phase 1 recomputes every slide.
        _per_slide_cache_dir = Path(str(results_dir)) / "slide-graphs-cache"
        if _per_slide_cache_dir.exists():
            shutil.rmtree(_per_slide_cache_dir, ignore_errors=True)
    
    # 1) Build slides (reusing your funcs)
    slides = []
    classes = None
    
    if niche_slide_graph_file.exists():
        click.secho("\nPhase 1/5: Build slide graphs for NicheGCN.\n"
                    f"Load existing slide graph file: {niche_slide_graph_file}\n", fg="green")
        # with gzip.open(niche_slide_graph_file, "rb") as f:
        #     slides = pickle.load(f)
        
        slides = joblib.load(niche_slide_graph_file)
            
    else:
        click.secho("\nPhase 1/5: build slide graphs for NicheGCN.\n", fg="green")
    
        # for i, (wsi_path, model_output_csv) in tqdm(enumerate(zip(slide_paths, model_output_paths)), total=len(slide_paths)):
        #     # print(f"Slide {i+1} of {len(slide_paths)}")
        #     # print(f" Slide path: {wsi_path}")
        #     # print(f" Model output path: {model_output_csv}")
        #
        #     df = pd.read_csv(model_output_csv)
        #     mpp = get_avg_mpp(wsi_path)
        #
        #     ds = None
        #     if use_hoptimus:
        #         if patch_datasets is not None and i < len(patch_datasets) and patch_datasets[i] is not None:
        #             ds = patch_datasets[i]
        #         else:
        #             ds = None  # will default to Dummy inside prepare_slide_graph
        #
        #     s = prepare_slide_graph(
        #         df, 
        #         mpp_um_per_px=mpp, 
        #         max_edge_len_um=max_edge_len_um,
        #         class_order=class_order,
        #         k_hops=k_hops, alpha=alpha,
        #         use_hoptimus=use_hoptimus, patch_dataset=ds,
        #         sample_frac=sample_frac, sample_count=sample_count,
        #         pca_dim=pca_dim, knn_k=knn_k, knn_sigma_um=knn_sigma_um,
        #         device=device, 
        #         mode = "soft" if niche_soft_mode else "hard"
        #         # seed=seed
        #     )
        #     slides.append(s)
        #
        #     if classes is None:
        #         classes = s["classes"]
            
            
            
        
        

        
        
        
        slides = [None] * len(slide_paths)
        classes = None

        graph_cache_dir = Path(str(results_dir)) / "graphs"
        graph_cache_dir.mkdir(parents=True, exist_ok=True)

        # Per-slide checkpoint directory: each slide's graph is saved as soon
        # as it completes so that an interrupted run can resume without
        # recomputing already-finished slides.
        slide_graph_cache_dir = Path(str(results_dir)) / "slide-graphs-cache"
        slide_graph_cache_dir.mkdir(parents=True, exist_ok=True)

        # Process slides sequentially so DataParallel can use all GPUs for each
        # slide's H-optimus embedding, with a clean nested progress display.
        slide_stems = [Path(str(p)).stem for p in slide_paths]
        short_ids = _make_short_ids(slide_stems)

        slide_bar = tqdm(total=len(slide_paths), desc="  slides", unit="slide", position=0)
        for i, (wsi_path, csv_path) in enumerate(zip(slide_paths, model_output_paths)):
            slide_id = Path(str(wsi_path)).stem
            display_id = short_ids.get(slide_id, slide_id)
            per_slide_cache = slide_graph_cache_dir / f"{slide_id}.joblib"

            # Resume: skip slides already computed in a previous interrupted run.
            if not overwrite and per_slide_cache.exists():
                slides[i] = joblib.load(per_slide_cache)
                if classes is None:
                    classes = slides[i]["classes"]
                slide_bar.update(1)
                continue

            ds = None
            if use_hoptimus and patch_datasets is not None and i < len(patch_datasets):
                ds = patch_datasets[i]
            df = pd.read_csv(csv_path)
            mpp = get_avg_mpp(wsi_path)
            s = prepare_slide_graph(
                df,
                mpp_um_per_px=mpp,
                max_edge_len_um=max_edge_len_um,
                class_order=class_order,
                k_hops=k_hops, alpha=alpha,
                hoptimus_only=hoptimus_only,
                hoptimus_model_dir=hoptimus_model_dir,
                use_hoptimus=use_hoptimus, patch_dataset=ds,
                sample_frac=sample_frac, sample_count=sample_count,
                pca_dim=pca_dim, knn_k=knn_k, knn_sigma_um=knn_sigma_um,
                device=device,
                graph_cache_dir=graph_cache_dir,
                slide_id=slide_id,
                display_id=display_id,
                mode="soft" if niche_soft_mode else "hard",
            )
            slides[i] = s
            if classes is None:
                classes = s["classes"]
            # Persist this slide immediately so an interrupted run can resume.
            joblib.dump(s, per_slide_cache, compress=("lz4", 3))
            # Clean up temporary files after each slide to prevent /tmp from filling up
            # (especially important when using H-optimus, which may create many temp files)
            _cleanup_tmpdir()
            slide_bar.update(1)
        slide_bar.close()
        
        # with gzip.open(niche_slide_graph_file, "wb") as f:
        #     pickle.dump(slides, f, protocol=pickle.HIGHEST_PROTOCOL)

        joblib.dump(slides, niche_slide_graph_file, compress=("lz4", 3))
        

    if niche_dgi_embeddings_file.exists():
        click.secho("\nPhase 2/5: Train shared DGI encoder and get DGI embeddings per slide.\n"
                    f"Load existing DGI embeddings file: {niche_dgi_embeddings_file}\n", fg="green")
        # with gzip.open(niche_dgi_embeddings_file, "rb") as f:
        #     Z_list = pickle.load(f)
        
        Z_list = joblib.load(niche_dgi_embeddings_file)
            
    else:
        click.secho("\nPhase 2/5: Train shared DGI encoder and get DGI embeddings per slide.\n", fg="green")
        
        # 3) Train shared DGI encoder and get embeddings per slide
        _, Z_list = train_dgi_multi(slides, hidden=hidden, out_dim=out_dim, epochs=epochs, seed=seed,
                                    early_stop_patience=early_stop_patience,
                                    early_stop_min_delta=early_stop_min_delta,
                                    early_stop_min_epochs=early_stop_min_epochs,
                                    amp=amp)
        
        # with gzip.open(niche_dgi_embeddings_file, "wb") as f:
        #     pickle.dump(Z_list, f, protocol=pickle.HIGHEST_PROTOCOL)

        joblib.dump(Z_list, niche_dgi_embeddings_file, compress=("lz4", 3))
        
    _labels_cached = niche_labels_file.exists() and not overwrite
    if _labels_cached:
        click.secho("\nPhase 3/5: Load cached niche labels.\n"
                    f"Load existing niche labels file: {niche_labels_file}\n", fg="green")
        _cached = joblib.load(niche_labels_file)
        labels_list = _cached["labels_list"]
        niche_clustering_k = _cached["niche_clustering_k"]
    elif not niche_clustering_k:
        # Resolution selection. When several resolutions are supplied we sweep
        # them and keep the most stable "winner"; when a SINGLE resolution is
        # given there is nothing to compare, so we skip the repeat/scoring
        # machinery and just run Leiden once at that resolution (direct mode).
        _res_list = list(niche_clustering_resolutions)
        _single = len(_res_list) == 1
        _n_repeats = 1 if _single else 5
        if _single:
            click.secho(
                f"\nPhase 3/5: Direct Leiden clustering at resolution="
                f"{_res_list[0]:g} (single resolution, no sweep).\n", fg="green")
        else:
            click.secho(
                f"\nPhase 3/5: Estimate niche clustering number via Leiden sweep "
                f"over resolutions {[float(r) for r in _res_list]}.\n", fg="green")

        estimate_niches_from_Z_list_res = estimate_niches_from_Z_list(Z_list,
                                                                      mode="global",
                                                                      niche_clustering_resolutions=_res_list,
                                                                      n_repeats=_n_repeats,
                                                                      k_nn=15,
                                                                      seed=seed)

        _w = estimate_niches_from_Z_list_res['winner']
        niche_clustering_k = _w['n_clusters']
        labels_list  = estimate_niches_from_Z_list_res["labels_list"]     # per-slide niche labels
        click.secho(
            f"  selected Leiden resolution={_w['resolution']:g} -> k={niche_clustering_k} "
            f"clusters (modularity={_w['modularity']:.3f}, "
            f"silhouette={_w['silhouette']:.3f})\n", fg="cyan")

    else:
        click.secho(f"\nPhase 3/5: Use predefined niche clustering number: niche_clustering_k={niche_clustering_k}.\n", fg="green")

        # Cluster ONCE over all slides pooled together so the integer niche_N ids
        # are consistent (and deterministic) across slides, then split the
        # labels back per slide. Running KMeans per slide would give each slide
        # its own arbitrary label permutation, so niche_0 in one slide would not
        # correspond to niche_0 in another.
        offsets = np.cumsum([0] + [Z.shape[0] for Z in Z_list[:-1]])
        Z_all = np.vstack(Z_list)
        labels_all = KMeans(n_clusters=niche_clustering_k,
                            n_init='auto',
                            random_state=seed,
                            ).fit_predict(Z_all).astype(np.int32)
        labels_list = [labels_all[off:off + Z.shape[0]]
                       for off, Z in zip(offsets, Z_list)]

    # Cache the per-slide labels so repeated Phase 4/5 runs skip re-clustering.
    if not _labels_cached:
        joblib.dump({"labels_list": labels_list,
                     "niche_clustering_k": int(niche_clustering_k)},
                    niche_labels_file, compress=("lz4", 3))

    click.secho("\nPhase 4/5: Perform cellular-level niche analysis per slide.\n", fg="green")

    if niche_cellular:
        # Each slide's per-cell CSV is independent -> build in a process pool.
        _p4_tasks = []
        for i, wsi_path in enumerate(slide_paths):
            niche_csv_name = Path(wsi_path).with_suffix(".csv").name
            cell_csv = niche_cells_output_dir / niche_csv_name
            _p4_tasks.append((wsi_path, model_output_paths[i], slides[i]["kept_idx"],
                              slides[i]["X"], slides[i]["classes"],
                              labels_list[i], k_hops, niche_clustering_k, cell_csv, overwrite,
                              bool(slides[i].get("hoptimus_only", False)), int(slides[i].get("khop_dim", 0))))
        _p4_workers = pick_workers_safe(max_workers=os.cpu_count() - 2, min_workers=2)
        _p4_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=_p4_workers, mp_context=_p4_ctx) as _ex:
            _futs = [_ex.submit(_niche_cellular_worker, t) for t in _p4_tasks]
            for _f in tqdm(as_completed(_futs), total=len(_futs)):
                raise_if_cancelled()
                _f.result()          # surfaces worker exceptions

    click.secho("\nPhase 5/5: Perform annotation-level niche analysis per slide.\n", fg="green")
   
    if niche_annotation:
        # Annotation-level region merge is per-slide independent -> process pool.
        _p5_tasks = []
        for i, wsi_path in enumerate(slide_paths):
            niche_csv_name = Path(wsi_path).with_suffix(".csv").name
            cell_csv = niche_cells_output_dir / niche_csv_name
            niche_csv = niche_niches_output_dir / niche_csv_name
            _p5_tasks.append((wsi_path, cell_csv, niche_csv, slides[i]["kept_idx"],
                              slides[i]["edges_df"], niche_clustering_k,
                              max_cell_radius_um, overwrite))
        _p5_workers = pick_workers_safe(max_workers=os.cpu_count() - 2, min_workers=2)
        _p5_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=_p5_workers, mp_context=_p5_ctx) as _ex:
            _futs = [_ex.submit(_niche_annotation_worker, t) for t in _p5_tasks]
            for _f in tqdm(as_completed(_futs), total=len(_futs)):
                raise_if_cancelled()
                _f.result()          # surfaces worker exceptions

        # print("-" * 40)
            