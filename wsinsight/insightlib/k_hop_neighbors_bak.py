"""Archived adjacency/k-hop helpers kept for reference during refactors.

Created on Sep 8, 2025 by huangc78.
"""

from __future__ import annotations

from collections import deque
# from typing import Dict, List, Optional, Tuple
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# def create_adjacency_list(edges_df):
#     """
#     Creates an adjacency list from a DataFrame of edges.
#
#     Args:
#         edges_df: DataFrame with 'source' and 'target' columns representing edges.
#
#     Returns:
#         A dictionary representing the adjacency list.
#     """
#     adjacency_list = {}
#     for _, row in edges_df.iterrows():
#         # Convert source and target to integers explicitly
#         source = int(row['source'])
#         target = int(row['target'])
#
#         if source not in adjacency_list:
#             adjacency_list[source] = []
#         if target not in adjacency_list:
#             adjacency_list[target] = []
#
#         adjacency_list[source].append(target)
#         adjacency_list[target].append(source) # Assuming undirected graph
#
#     return adjacency_list





def _prep_edges_numpy(
    edges_df: pd.DataFrame,
    src_col: str = "source",
    dst_col: str = "target",
    ensure_undirected: bool = True,
    dedup_edges: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert DF to two aligned arrays src, dst with optional symmetrization & dedup."""
    u = edges_df[src_col].to_numpy(dtype=np.int64, copy=False)
    v = edges_df[dst_col].to_numpy(dtype=np.int64, copy=False)

    if ensure_undirected:
        src = np.concatenate([u, v], axis=0)
        dst = np.concatenate([v, u], axis=0)
    else:
        src, dst = u, v

    if dedup_edges:
        # drop exact duplicate directed edges
        edges = np.stack([src, dst], axis=1)
        edges = np.unique(edges, axis=0)
        src, dst = edges[:, 0], edges[:, 1]

    return src, dst


def create_adjacency_list_fast(
    edges_df: pd.DataFrame,
    *,
    src_col: str = "source",
    dst_col: str = "target",
    ensure_undirected: bool = True,
    dedup_edges: bool = True,
    dedup_neighbors: bool = False,
    sort_neighbors: bool = False,
    # num_nodes: Optional[int] = None,
) -> Dict[int, List[int]]:
    """
    Vectorized adjacency builder (no Python row loop).
    Returns: {node: [neighbors...]}

    - ensure_undirected: add (v,u) for every (u,v)
    - dedup_edges: remove duplicate directed edges
    - dedup_neighbors: np.unique neighbor list per node
    - sort_neighbors: sorted neighbor list per node
    """
    src, dst = _prep_edges_numpy(edges_df, src_col, dst_col, ensure_undirected, dedup_edges)

    if src.size == 0:
        return {}

    # group by source using argsort + split (O(E log E) but all in C/NumPy)
    order = np.argsort(src, kind="mergesort")  # stable
    src_sorted = src[order]
    dst_sorted = dst[order]

    # where src changes -> split points
    split_points = np.flatnonzero(np.diff(src_sorted)) + 1
    dst_groups = np.split(dst_sorted, split_points)
    src_keys = src_sorted[np.r_[0, split_points]]  # unique sources

    adj: Dict[int, List[int]] = {}
    if not (dedup_neighbors or sort_neighbors):
        # fast path: just tolist() without extra work
        for key, grp in zip(src_keys, dst_groups):
            adj[int(key)] = grp.tolist()
        return adj

    # optional: unique / sort per node
    for key, grp in zip(src_keys, dst_groups):
        arr = grp
        if dedup_neighbors:
            arr = np.unique(arr)
        if sort_neighbors:
            # if we already unique() we can set kind='mergesort' above; here normal sort is fine
            arr = np.sort(arr)
        adj[int(key)] = arr.tolist()
    return adj


# -------------------------- Parallel variant -------------------------- #

# def _build_chunk(
#     key_chunk: np.ndarray,
#     groups_chunk: List[np.ndarray],
#     dedup_neighbors: bool,
#     sort_neighbors: bool,
# ) -> Dict[int, List[int]]:
#     out: Dict[int, List[int]] = {}
#     for k, grp in zip(key_chunk, groups_chunk):
#         arr = grp
#         if dedup_neighbors:
#             arr = np.unique(arr)
#         if sort_neighbors:
#             arr = np.sort(arr)
#         out[int(k)] = arr.tolist()
#     return out
#
#
# def create_adjacency_list_parallel(
#     edges_df: pd.DataFrame,
#     *,
#     src_col: str = "source",
#     dst_col: str = "target",
#     ensure_undirected: bool = True,
#     dedup_edges: bool = True,
#     dedup_neighbors: bool = False,
#     sort_neighbors: bool = False,
#     n_procs: Optional[int] = None,
#     # min_groups_per_proc: int = 256,
# ) -> Dict[int, List[int]]:
#     """
#     Parallel adjacency builder. Recommended only for *very large* graphs where the
#     single-threaded fast version becomes a bottleneck.
#
#     - n_procs: number of processes (default = os.cpu_count()).
#     - min_groups_per_proc: groups per worker; prevents oversharding on small graphs.
#     """
#     # src, dst = _prep_edges_numpy(edges_df, src_col, dst_col, ensure_undirected, dedup_edges)
#     src, _ = _prep_edges_numpy(edges_df, src_col, dst_col, ensure_undirected, dedup_edges)
#
#     if src.size == 0:
#         return {}
#
#     order = np.argsort(src, kind="mergesort")
#     src_sorted = src[order]
#     # dst_sorted = dst[order]
#
#     split_points = np.flatnonzero(np.diff(src_sorted)) + 1
#     # dst_groups = np.split(dst_sorted, split_points)
#     src_keys = src_sorted[np.r_[0, split_points]]
#
#     num_groups = len(src_keys)
#     if num_groups == 0:
#         return {}
#
#     if n_procs is None:
#         import os
#         n_procs = os.cpu_count() or 1
#
#     # ensure we don't spawn too many processes for tiny workloads
#     # max_workers = max(1, min(n_procs, num_groups // max(1, min_groups_per_proc)))
#     # max_workers = pick_workers_safe(max_workers=os.cpu_count()-8, min_workers=8)
#     # if max_workers == 1:
#         # fallback to fast single-threaded
#     return create_adjacency_list_fast(
#         edges_df,
#         src_col=src_col,
#         dst_col=dst_col,
#         ensure_undirected=ensure_undirected,
#         dedup_edges=dedup_edges,
#         dedup_neighbors=dedup_neighbors,
#         sort_neighbors=sort_neighbors,
#     )
#
#     # # shard keys/groups evenly
#     # chunk_size = math.ceil(num_groups / max_workers)
#     # futures = []
#     # out: Dict[int, List[int]] = {}
#     # # with ProcessPoolExecutor(max_workers=max_workers) as ex:
#     # # with ThreadPoolExecutor(max_workers=max_workers) as ex:
#     # for i in range(0, num_groups, chunk_size):
#     #     key_chunk = src_keys[i : i + chunk_size]
#     #     groups_chunk = dst_groups[i : i + chunk_size]
#     #     futures.append(
#     #         # ex.submit(_build_chunk, key_chunk, groups_chunk, dedup_neighbors, sort_neighbors)
#     #         (key_chunk, groups_chunk, dedup_neighbors, sort_neighbors)
#     #     )
#     # # for fut in as_completed(futures):
#     # #     throttle_when_busy()
#     # #     out.update(fut.result())
#     # for fut in futures:
#     #     out.update(_build_chunk(*fut))
#     #
#     # return out





def k_hop_neighbors(nodes_df, adjacency_list, k):
    """
    Finds k-hop neighbors for all cells in a DataFrame using a DataFrame of edges.

    Args:
        nodes_df: DataFrame with cell data.
        adjacency_list: A dictionary representing the adjacency list.
        k: The number of hops.

    Returns:
        A list of lists, where each inner list contains the indices of all cells
        reachable from the corresponding cell within k hops.
    """

    # Create adjacency list from the edges DataFrame
    # adjacency_list = create_adjacency_list(edges_df)

    def k_hop_search(start_node, k, adjacency_list):
        """
        Performs a k-hop search starting from a given node using BFS.

        Args:
            start_node: The index of the starting node.
            k: The number of hops.
            adjacency_list: The adjacency list representation of the graph.

        Returns:
            A set of reachable node indices within k hops (including the start node).
        """
        visited = set()
        queue = deque([(start_node, 0)])  # (node, distance)
        reachable_nodes = set()

        while queue:
            current_node, distance = queue.popleft() # Using popleft for BFS

            if current_node not in visited and distance <= k:
                visited.add(current_node)
                reachable_nodes.add(current_node)

                if distance < k and current_node in adjacency_list:
                    for neighbor in adjacency_list[current_node]:
                        if neighbor not in visited:
                            queue.append((neighbor, distance + 1))

        return sorted(list(reachable_nodes))

    # Apply k-hop search to all cells
    all_k_hop_neighbors = []
    for i in nodes_df.index:
        reachable_neighbors = k_hop_search(start_node=i, k=k, adjacency_list=adjacency_list)
        all_k_hop_neighbors.append(reachable_neighbors)

    return all_k_hop_neighbors
