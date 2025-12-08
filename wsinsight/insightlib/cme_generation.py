"""CME graph construction, embedding, and clustering pipelines for WSInsight."""

# cmegcn_multi_from_your_funcs_h0.py
# pip install torch torch_geometric scikit-learn numpy scipy pandas timm pillow

from __future__ import annotations
import math, os
import multiprocessing as mp
from typing import Any, List, Dict, Iterable, Optional, Tuple # , Callable
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data # , DataLoader as GeoDataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader, DataListLoader
from torch_geometric.nn import GCNConv, DataParallel as GeoDataParallel
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
from ..uri_path import URIPath
from ..wsi import _validate_wsi_directory, get_avg_mpp
from ..insightlib.vorononi_cme_region_helper import merge_same_label_by_shared_edges_iterative, remap_edges_to_valid_indices
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

def _embed_hoptimus_subset_dataset(
    dataset: Dataset, sampled_ids: List[int],
    batch_size: int = 128, device: Optional[str] = None
) -> np.ndarray:
    """
    Embed only a subset of cells using H-Optimus-0.
    'dataset' must support __getitem__(cell_id) -> PIL.Image or Tensor for that cell.
    """
    import timm
    from timm.data import create_transform
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, num_classes=0).to(dev).eval()
    pre = create_transform(**model.pretrained_cfg, is_training=False)

    # Build a Subset where sample index equals the cell_id we want
    subset = Subset(dataset, sampled_ids)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    feats = []
    with torch.no_grad():
        for batch in loader:
            # batch may be PIL Images (list) or already tensors
            if isinstance(batch, list):
                x = torch.stack([pre(im) for im in batch]).to(dev)      # [B,3,224,224]
            elif isinstance(batch, torch.Tensor):
                # If user returns raw tensors, ensure shape and normalize via pre
                if batch.dim() == 4 and batch.shape[1] in (1, 3):
                    x = torch.stack([pre(transforms.ToPILImage()(t)) for t in batch]).to(dev)
                else:
                    x = torch.stack([pre(b) if not isinstance(b, torch.Tensor) else pre(b) for b in batch]).to(dev)
            else:
                # If DataLoader collate returns arbitrary type, try element-wise preprocessing
                try:
                    x = torch.stack([pre(im) for im in batch]).to(dev)
                except Exception:
                    # Fallback: assume batch is already preprocessed tensor
                    x = batch.to(dev)
            z = model(x)                                               # [B,1536]
            feats.append(z.detach().cpu())
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



def train_dgi_multi(slides, hidden=64, out_dim=32, epochs=300, lr=1e-3, wd=1e-4):
    """Train a shared DGI encoder across slide graphs and return embeddings."""
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

        
        
        
        loader = DataListLoader(data_list, batch_size=batch_size, shuffle=True)
        model = GeoDataParallel(model, device_ids=list(range(ngpu))).to(primary)
    else:
        loader = GeoDataLoader(data_list, batch_size=1, shuffle=True)
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

    # training (works for single or multi-GPU)
    for _ in tqdm(range(epochs)):
        for batch in loader:
            opt.zero_grad()
    
            if ngpu > 1:
                pos_z, neg_z, s = model(batch)
                loss = model.module.loss(pos_z, neg_z, s)
            else:
                batch = batch.to(primary)
                pos_z, neg_z, s = model(batch)
                loss = model.loss(pos_z, neg_z, s)

    
        
    
            loss.backward()
            opt.step()



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
    cme_detection_df: pd.DataFrame,
    mpp_um_per_px: float,
    max_edge_len_um: float,
    class_order: Optional[List[str]] = None,
    k_hops: int = 2,
    alpha: float = 1.0,
    # H-Optimus
    use_hoptimus: bool = False,
    patch_dataset: Optional[Dataset] = None,  # your dataset: __getitem__(cell_id)-> PIL.Image / Tensor
    sample_frac: Optional[float] = 0.2,
    sample_count: Optional[int] = None,
    pca_dim: Optional[int] = 128,
    knn_k: int = 3,
    knn_sigma_um: float = 60.0,
    device: Optional[str] = None,
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
    df = compute_cell_center_points(cme_detection_df.copy())
    centers_px = df[["center_x", "center_y"]].to_numpy(dtype=np.float32)
    N = len(df)

    # Delaunay + cap in px (your function needs px)
    max_edge_len_px = float(max_edge_len_um) / float(mpp_um_per_px)
    edges_df = delaunay_triangulation(centers_px, max_edge_len_px)  # columns: source,target,length

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

    # k-hop features (always on; biologically grounded)
    X_khop = khop_features(P=P, edge_index=edge_index, N=N_kept, k=k_hops, alpha=alpha, mode=mode)  # [N_kept,(k+1)C]
    blocks = [X_khop.astype(np.float32)]

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
        Hs = _embed_hoptimus_subset_dataset(patch_dataset, sampled_global_ids, batch_size=128, device=device)  # [m,1536]
        # optional PCA
        if pca_dim is not None and Hs.shape[1] > pca_dim:
            from sklearn.decomposition import PCA
            # Hs = PCA(n_components=pca_dim, random_state=seed).fit_transform(Hs).astype(np.float32)
            Hs = PCA(n_components=pca_dim).fit_transform(Hs).astype(np.float32)

        # KNN impute to all kept nodes (micron distances)
        H_full = _impute_knn(coords_um=coords_um, sampled_idx=sampled_local_idx,
                             sampled_feats=Hs, k=knn_k, sigma_um=knn_sigma_um)  # [N_kept,D]
        blocks.append(H_full.astype(np.float32))

    # concatenate feature blocks (khop [+ H0])
    X = np.hstack(blocks).astype(np.float32)

    return {
        "X": X,
        "edge_index": edge_index.astype(np.int64),
        "kept_idx": kept_idx.astype(np.int64),
        "classes": classes,
        "edges_df": edges_df,
    }
    
# =============================================================================
# End-to-end multi-image training + clustering
# =============================================================================

def _knn_graph_connectivity(Z: np.ndarray, k_nn: int = 15):
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
                   # seed: int,
                   ) -> Tuple[np.ndarray, float, float]:
    """Run a single Leiden clustering pass and return labels plus modularity."""
    g_local = ig.Graph(n=n_nodes, edges=edges.tolist(), directed=False)
    g_local.simplify(combine_edges="ignore")
    part = la.find_partition(
        g_local, la.RBConfigurationVertexPartition,
        resolution_parameter=float(resolution),
        # seed=int(seed)
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
    cme_clustering_resolutions: Iterable[float] = np.arange(0.2, 2.05, 0.1),
    n_repeats: int = 5,
    # seed: int = 0,
) -> Dict[str, Any]:
    """
    Parallel sweep over (resolution, repeat) AND parallel reduction per resolution.
    Returns {"winner": {...}, "all": [ per-resolution dicts ... ]}.
    """
    # rng = np.random.default_rng(seed)

    # Convert igraph to (n_nodes, edges) once (edges is picklable)
    n_nodes = g.vcount()
    el = np.array(g.get_edgelist(), dtype=np.int64)
    if el.size == 0:
        labels = np.zeros(n_nodes, dtype=int)
        out = {
            "resolution": float(next(iter(cme_clustering_resolutions), 1.0)),
            "n_clusters": 1, "modularity": 0.0, "stability": 1.0,
            "silhouette": -1.0, "min_frac": 1.0, "labels": labels,
        }
        return {"winner": out, "all": [out]}

    # ---- Phase A: parallel Leiden runs over (resolution, repeat) ----
    tasks = []
    for r in cme_clustering_resolutions:
        for _ in range(n_repeats):
            # tasks.append((n_nodes, el, float(r), int(rng.integers(1_000_000_000))))
            tasks.append((n_nodes, el, float(r)))

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


def estimate_cmes_from_Z_list(
    Z_list: List[np.ndarray],
    mode: str = "global",           # "global" (recommended) or "per_slide"
    k_nn: int = 15,
    cme_clustering_resolutions = np.arange(0.2, 2.05, 0.1),
    n_repeats: int = 5,
    # seed: int = 0,
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
        # concat for a single clustering (consistent cme IDs across slides)
        offsets = np.cumsum([0] + [Z.shape[0] for Z in Z_list[:-1]])
        Z_all = np.vstack(Z_list)
        A = _knn_graph_connectivity(Z_all, k_nn=k_nn)
        g = _igraph_from_sparse(A)
        sweep = _leiden_sweep_on_graph(Z_all, 
                                       g, 
                                       cme_clustering_resolutions=cme_clustering_resolutions, 
                                       n_repeats=n_repeats, 
                                       # seed=seed,
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
            A = _knn_graph_connectivity(Z, k_nn=k_nn)
            g = _igraph_from_sparse(A)
            sweep = _leiden_sweep_on_graph(Z, 
                                           g, 
                                           cme_clustering_resolutions=cme_clustering_resolutions, 
                                           n_repeats=n_repeats, 
                                           # seed=seed,
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
        device, cme_soft_mode, use_hoptimus):
    """Background worker to build one slide graph and return it with index."""
    df = pd.read_csv(csv_path)
    mpp = get_avg_mpp(wsi_path)
    s = prepare_slide_graph(
        df,
        mpp_um_per_px=mpp,
        max_edge_len_um=max_edge_len_um,
        class_order=class_order,
        k_hops=k_hops, alpha=alpha,
        use_hoptimus=use_hoptimus, patch_dataset=ds,
        sample_frac=sample_frac, sample_count=sample_count,
        pca_dim=pca_dim, knn_k=knn_k, knn_sigma_um=knn_sigma_um,
        device=device,
        mode="soft" if cme_soft_mode else "hard",
    )
    return i, s
        
def cme_generation(
    wsi_dir: str | URIPath | None,
    wsi_paths: list[Path] | None,
    results_dir: str | Path,
    max_edge_len_um: float,
    max_cell_radius_um: float,
    class_order: Optional[List[str]] = None,
    k_hops: int = 2,
    alpha: float = 1.0,
    # H-Optimus switch & params
    use_hoptimus: bool = False,
    patch_datasets: Optional[List[Dataset]] = None,  # list aligned with slides_inputs; if None, Dummy is used
    sample_frac: Optional[float] = 0.2,
    sample_count: Optional[int] = None,
    pca_dim: Optional[int] = 128,
    knn_k: int = 3,
    knn_sigma_um: float = 60.0,
    # encoder
    hidden: int = 64,
    out_dim: int = 32,
    epochs: int = 300,
    # clustering
    cme_cellular: bool = False,
    cme_annotation: bool = False,
    cme_clustering_k: int | None = 10,
    cme_clustering_resolutions: List[float]=[0.5,1.0,2.0],
    # # device
    # device: Optional[str] = None,
    cme_soft_mode: bool = False,
    # seed: int = 0,
) -> Dict[str, List[np.ndarray]]:
    """
    Prepare graphs for multiple slides, global-standardize features, train one DGI, and cluster per slide.
    """
    
    if os.getenv("WSINFER_FORCE_CPU", "0").lower() not in {"0", "f", "false"}:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f'Using device "{device}"')
    
    # Make sure required directories exist.
    wsi_dir = URIPath(wsi_dir)
    if not wsi_dir.exists():
        raise errors.WholeSlideImageDirectoryNotFound(f"directory not found: {wsi_dir}")
    wsi_paths = [p for p in wsi_dir.iterdir() if p.is_file()]
    if not wsi_paths:
        raise errors.WholeSlideImagesNotFound(wsi_dir)
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise errors.ResultsDirectoryNotFound(results_dir)

    _validate_wsi_directory(wsi_dir)

    # Check patches directory.
    model_output_dir = results_dir / "model-outputs-csv"
    if not model_output_dir.exists():
        raise errors.ResultsDirectoryNotFound(
            "The 'model-outputs-csv' directory was not found in results directory."
        )
    # Create the patch paths based on the whole slide image paths. In effect, only
    # create patch paths if the whole slide image patch exists.
    model_output_paths = [model_output_dir / p.with_suffix(".csv").name for p in wsi_paths]
    
    if len(model_output_paths) != len(wsi_paths):
        raise errors.ResultsDirectoryNotFound(
            "The 'model-outputs-csv' and image directory were mismatched."
        )
    cme_output_dir = results_dir / "cme-outputs-csv"
    cme_output_dir.mkdir(exist_ok=True)
    cme_cells_output_dir = results_dir / "cme-outputs-csv" / "cells"
    cme_cells_output_dir.mkdir(exist_ok=True)
    cme_cmes_output_dir = results_dir / "cme-outputs-csv" / "cmes"
    cme_cmes_output_dir.mkdir(exist_ok=True)
    cme_slide_graph_file = results_dir / "slide-graphs.joblib"
    cme_dgi_embeddings_file = results_dir / "dgi-embeddings.joblib"
    
    # 1) Build slides (reusing your funcs)
    slides = []
    classes = None
    
    if cme_slide_graph_file.exists():
        click.secho("\nPhase 1/5: Build slide graphs for CMEGCN.\n"
                    f"Load existing slide graph file: {cme_slide_graph_file}\n", fg="green")
        # with gzip.open(cme_slide_graph_file, "rb") as f:
        #     slides = pickle.load(f)
        
        slides = joblib.load(cme_slide_graph_file)
            
    else:
        click.secho("\nPhase 1/5: build slide graphs for CMEGCN.\n", fg="green")
    
        # for i, (wsi_path, model_output_csv) in tqdm(enumerate(zip(wsi_paths, model_output_paths)), total=len(wsi_paths)):
        #     # print(f"Slide {i+1} of {len(wsi_paths)}")
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
        #         mode = "soft" if cme_soft_mode else "hard"
        #         # seed=seed
        #     )
        #     slides.append(s)
        #
        #     if classes is None:
        #         classes = s["classes"]
            
            
            
        
        

        
        
        
        slides = [None] * len(wsi_paths)
        classes = None
        
        ctx = mp.get_context("spawn")  # safer with NumPy/pandas
        tasks = []
        for i, (wsi_path, csv_path) in enumerate(zip(wsi_paths, model_output_paths)):
            ds = None
            if use_hoptimus and patch_datasets is not None and i < len(patch_datasets):
                ds = patch_datasets[i]
            tasks.append((i, wsi_path, csv_path, ds,
                          max_edge_len_um, class_order, k_hops, alpha,
                          sample_frac, sample_count, pca_dim, knn_k, knn_sigma_um,
                          device, cme_soft_mode, use_hoptimus))
            
        num_workers = pick_workers_safe(max_workers=os.cpu_count()-8, min_workers=8)
        
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
            futs = [ex.submit(_prepare_slide_graph_worker, *t) for t in tasks]
            for f in tqdm(as_completed(futs), total=len(futs)):
                idx, s = f.result()         # surfaces exceptions
                slides[idx] = s
                if classes is None:
                    classes = s["classes"]

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # 2) Global z-score over concatenated features (keeps scale consistent across slides)
        X_all = np.vstack([s["X"] for s in slides]).astype(np.float32)
        scaler = StandardScaler(with_mean=True, with_std=True).fit(X_all)
        for s in slides:
            s["X_normalized"] = scaler.transform(s["X"]).astype(np.float32)
            
        # with gzip.open(cme_slide_graph_file, "wb") as f:
        #     pickle.dump(slides, f, protocol=pickle.HIGHEST_PROTOCOL)

        joblib.dump(slides, cme_slide_graph_file, compress=("lz4", 3))
        

    if cme_dgi_embeddings_file.exists():
        click.secho("\nPhase 2/5: Train shared DGI encoder and get DGI embeddings per slide.\n"
                    f"Load existing DGI embeddings file: {cme_dgi_embeddings_file}\n", fg="green")
        # with gzip.open(cme_dgi_embeddings_file, "rb") as f:
        #     Z_list = pickle.load(f)
        
        Z_list = joblib.load(cme_dgi_embeddings_file)
            
    else:
        click.secho("\nPhase 2/5: Train shared DGI encoder and get DGI embeddings per slide.\n", fg="green")
        
        # 3) Train shared DGI encoder and get embeddings per slide
        _, Z_list = train_dgi_multi(slides, hidden=hidden, out_dim=out_dim, epochs=epochs)
        
        # with gzip.open(cme_dgi_embeddings_file, "wb") as f:
        #     pickle.dump(Z_list, f, protocol=pickle.HIGHEST_PROTOCOL)

        joblib.dump(Z_list, cme_dgi_embeddings_file, compress=("lz4", 3))
        
    if not cme_clustering_k:
        click.secho("\nPhase 3/5: Estimate cme clustering number.\n", fg="green")
        
        estimate_cmes_from_Z_list_res = estimate_cmes_from_Z_list(Z_list, 
                                                                      mode="global", 
                                                                      cme_clustering_resolutions=cme_clustering_resolutions,
                                                                      k_nn=15)
        
        cme_clustering_k = estimate_cmes_from_Z_list_res['winner']['n_clusters']
        labels_list  = estimate_cmes_from_Z_list_res["labels_list"]     # per-slide cme labels
        
    else:
        click.secho(f"\nPhase 3/5: Use predefined cme clustering number: cme_clustering_k={cme_clustering_k}.\n", fg="green")
        
        labels_list = [KMeans(n_clusters=cme_clustering_k, 
                              n_init='auto', 
                              # random_state=seed
                              ).fit_predict(Z).astype(np.int32)
                       for Z in Z_list]
    
    click.secho("\nPhase 4/5: Perform cellular-level cme analysis per slide.\n", fg="green")

    if cme_cellular:
        for i, (wsi_path, model_output_csv) in tqdm(enumerate(zip(wsi_paths, model_output_paths)), total=len(wsi_paths)):
            cme_csv_name = Path(wsi_path).with_suffix(".csv").name
            cell_csv = cme_cells_output_dir / cme_csv_name
            cme_csv = cme_cmes_output_dir / cme_csv_name
            
            if cell_csv.exists():
                continue
            
            mpp = get_avg_mpp(wsi_path)
            
            model_output_df = pd.read_csv(model_output_csv)
            cme_detection_df = model_output_df
            
            feature_normalized_cols = [f"feature_normalized_k{k}_{c.replace('prob_', '')}" for k in range(k_hops+1) for c in slides[i]["classes"]]
            feature_cols = [f"feature_raw_k{k}_{c.replace('prob_', '')}" for k in range(k_hops+1) for c in slides[i]["classes"]]
            cme_detection_df.loc[slides[i]["kept_idx"], feature_normalized_cols] = slides[i]["X_normalized"]
            cme_detection_df.loc[slides[i]["kept_idx"], feature_cols] = slides[i]["X"]
            cme_cols = [f"cme_{l}" for l in range(cme_clustering_k)]
            label_one_hot = np.eye(cme_clustering_k, dtype=np.float32)[labels_list[i]]
            cme_detection_df.loc[slides[i]["kept_idx"], cme_cols] = label_one_hot
            
            cme_detection_df.to_csv(cell_csv, index=False)
            
            # valid_mask = np.zeros(len(cme_detection_df), dtype=bool)
            # valid_mask[np.asarray(slides[i]["kept_idx"], dtype=int)] = True
            # edges_df = remap_edges_to_valid_indices(slides[i]['edges_df'], valid_mask)
            
            # cme_annotation_df = merge_same_label_by_shared_edges_iterative(cme_detection_df, 
            #                                                                  edges_df,
            #                                                                  cme_clustering_k=cme_clustering_k,
            #                                                                  mpp=mpp,
            #                                                                  max_radius_um=max_cell_radius_um)
            #
            # cme_annotation_df.to_csv(cme_csv, index=False)
        
    
    click.secho("\nPhase 5/5: Perform annotation-level cme analysis per slide.\n", fg="green")
   
    if cme_annotation:
        for i, (wsi_path, model_output_csv) in tqdm(enumerate(zip(wsi_paths, model_output_paths)), total=len(wsi_paths)):
            cme_csv_name = Path(wsi_path).with_suffix(".csv").name
            cell_csv = cme_cells_output_dir / cme_csv_name
            cme_csv = cme_cmes_output_dir / cme_csv_name
            
            if cme_csv.exists():
                continue
            
            cme_detection_df = pd.read_csv(cell_csv)
            valid_mask = np.zeros(len(cme_detection_df), dtype=bool)
            valid_mask[np.asarray(slides[i]["kept_idx"], dtype=int)] = True
            edges_df = remap_edges_to_valid_indices(slides[i]['edges_df'], valid_mask)
            
            cme_annotation_df = merge_same_label_by_shared_edges_iterative(cme_detection_df, 
                                                                             edges_df,
                                                                             cme_clustering_k=cme_clustering_k,
                                                                             mpp=mpp,
                                                                             max_radius_um=max_cell_radius_um)
            
            cme_annotation_df.to_csv(cme_csv, index=False)
            
        
        # print("-" * 40)
            