"""Shared geometric, graph, and statistical helpers for WSInsight analytics."""

from __future__ import annotations

from collections import deque
import numpy as np
import pandas as pd
from typing import Dict, Any, Iterable, List, Tuple
from scipy.spatial import Delaunay
from concurrent.futures import ThreadPoolExecutor


def compute_cell_center_points(model_output_df):
    """
    Computes cell center points

    Args:
        model_output_df: DataFrame with 'minx', 'miny', 'width', 'height' columns.
        
    Returns:
        A tuple containing:
        - The DataFrame with 'center_x' and 'center_y' columns added.
    """
    # Calculate cell center points if not already present
    if 'center_x' not in model_output_df.columns or 'center_y' not in model_output_df.columns:
        model_output_df['center_x'] = np.rint(model_output_df['minx'] + (model_output_df['width'] / 2)).astype(np.int32)
        model_output_df['center_y'] = np.rint(model_output_df['miny'] + (model_output_df['height'] / 2)).astype(np.int32)

    return model_output_df


def delaunay_triangulation(point2d_ary, max_edge_length):
    """
    Performs Delaunay triangulation on cell center points and filters edges by length.

    Args:
        point2d_ary: N x 2 numpy array for center_x, center_y of nuclei
        max_edge_length: Maximum length for edges to be included in the analysis.

    Returns:
        A tuple containing:
        - A DataFrame of edges with 'source', 'target', and 'length' columns representing point indices and edge length,
          filtered by max_edge_length.
        - The DataFrame with 'center_x' and 'center_y' columns added.
    """

    # Perform Delaunay triangulation
    tri = Delaunay(point2d_ary)

    # Get the edges of the triangulation
    # The edges are pairs of indices into the points array
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[0], simplex[2]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))

    # Filter edges by length and store in a list of dictionaries
    filtered_edges_list = []
    for i, j in edges:
        point1 = point2d_ary[i]
        point2 = point2d_ary[j]
        distance = np.linalg.norm(point1 - point2)
        if distance < max_edge_length:
            filtered_edges_list.append({'source': i, 'target': j, 'length': distance})

    # Convert the list of dictionaries to a pandas DataFrame
    filtered_edges_df = pd.DataFrame(filtered_edges_list)

    return filtered_edges_df # , model_output_df # Return both filtered edges DataFrame and the updated DataFrame


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


# --------------------------- helpers ---------------------------

def _reindex_nearest(series: pd.Series, target_levels: Iterable[int]) -> pd.Series:
    """
    Reindex a per-layer Series (index = layer) to a desired list of 'target_levels'
    using nearest available index (clamped to ends). Missing -> NaN.
    """
    target_levels = list(target_levels)
    if series.empty:
        return pd.Series([np.nan] * len(target_levels), index=target_levels, dtype=float)
    s = series.sort_index()
    return s.reindex(target_levels, method="nearest").astype(float)

def _depth_weights(levels: Iterable[int],
                   mode: str,
                   s: float,
                   range_min: int,
                   range_max: int,
                   side: str) -> pd.Series:
    """
    Compute depth weights for 'inside' or 'outside' layers.

    - linear:
        inside  (levels <= 0): w = |level| / |range_min|
        outside (levels >= 1): w = 1 - level / range_max
    - sigmoid (1 - sigmoid(s * level)):
        layer=0 -> 0.5; deeper inside (more negative) -> closer to 1; farther outside (more positive) -> closer to 0
    """
    levels = list(levels)
    if not levels:
        return pd.Series(dtype=float)

    mode = (mode or "linear").lower()
    lv = np.asarray(levels, dtype=float)

    if mode == "sigmoid":
        # w = 1 - sigmoid(s*layer); sigmoid(x)=1/(1+e^-x)
        w = 1.0 - 1.0 / (1.0 + np.exp(-s * lv))
        return pd.Series(np.clip(w, 0.0, 1.0), index=levels, dtype=float)

    # default linear
    if side == "inside":
        denom = max(abs(int(range_min)), 1)
        w = np.clip(np.abs(lv) / denom, 0.0, 1.0)
    else:  # outside
        denom = float(max(int(range_max), 1))
        w = np.clip(1.0 - (lv / denom), 0.0, 1.0)

    return pd.Series(w, index=levels, dtype=float)

def _center_of_mass(values: pd.Series, coords: pd.Series) -> float:
    """
    Weighted center-of-mass along 1D coordinates.
    values: weights per level (e.g., immune abundance per layer)
    coords: coordinate per level (e.g., signed distance per layer)
    Returns NaN if no finite weights.
    """
    v = np.asarray(values.values, dtype=float)
    x = np.asarray(coords.values, dtype=float)
    m = np.isfinite(v) & np.isfinite(x) & (v > 0)
    if not np.any(m):
        return np.nan
    v, x = v[m], x[m]
    return float(np.sum(v * x) / np.sum(v))

def _safe_mean(series: pd.Series) -> float:
    """Mean with NaN tolerance -> float in [0,1] or 0.0 when empty."""
    if series is None or len(series) == 0:
        return 0.0
    return float(np.nanmean(series.values))

def _weighted_mean(numer: pd.Series, denom: pd.Series) -> float:
    """
    Compute sum(numer)/sum(denom) with NaN masking.
    Returns 0.0 when effective denominator is zero.
    """
    nv = np.asarray(numer.values, dtype=float)
    dv = np.asarray(denom.values, dtype=float)
    m = np.isfinite(nv) & np.isfinite(dv) & (dv > 0)
    if not np.any(m):
        return 0.0
    return float(np.sum(nv[m]) / np.sum(dv[m]))


# ---- helper for a single cell ----
def _enrichment_for_cell(args) -> float:
    """
    Helper used by ThreadPoolExecutor.

    Args:
        args: tuple (i, neigh_ids, target_s, base_s, eps)

    Returns:
        (i, enrichment_value)  # i is position in nodes_df (0-based row index)
    """
    i, neigh_ids, target_s, base_s, eps = args

    n = len(neigh_ids)
    if n == 0:
        return i, 0.0

    # 将邻居 ID 映射到布林值；不在 index 的 ID 视为 False
    neigh_target = target_s.reindex(neigh_ids).fillna(False)
    neigh_base   = base_s.reindex(neigh_ids).fillna(False)

    t_count = neigh_target.sum()
    b_count = neigh_base.sum()

    T = float(t_count) / n
    B = float(b_count) / n
    value = T * T / (T + B + eps)
    return i, value


# ---- parallel main ----
def compute_enrichment_index(
    nodes_df: pd.DataFrame,
    k_neighbors_results: List[List],
    target_col: str = "is_target_type",
    base_col: str = "is_base_type",
    eps: float = 1e-6,
    max_workers: int = None,
) -> pd.DataFrame:
    """
    为每个 cell 计算 enrichment_index = T * T / (T + B + eps) 的并行版本（ThreadPoolExecutor）。

    其中：
      T = (k 邻居中 target=True 的数量) / (邻居总数)
      B = (k 邻居中 base=True   的数量) / (邻居总数)

    约定：
      - k_neighbors_results[i] 对应 nodes_df.iloc[i] 这个 cell
      - k_neighbors_results[i] 内的元素是该 cell 邻居的 nodes_df.index（cell ID）

    参数:
      nodes_df: 包含 target_col / base_col 布林列的 DataFrame
      k_neighbors_results: 每个 cell 的邻居 ID 列表
      target_col: 目标类型标记列名
      base_col: 基底 / 对照类型标记列名
      eps: 防止除零的微小常数
      max_workers: 线程数（缺省为 None = ThreadPoolExecutor 自动选择）

    返回:
      原始 nodes_df，附加一列 "hplot_enrichment_index"
    """
    # 基本检查
    for col in (target_col, base_col):
        if col not in nodes_df.columns:
            raise KeyError(f"missing required column '{col}' in nodes_df")
    if len(k_neighbors_results) != len(nodes_df):
        raise ValueError("k_neighbors_results length must match len(nodes_df)")

    # 布林 Series，以 index 为键，方便 reindex 到邻居 ID
    target_s = nodes_df[target_col].astype(bool)
    base_s   = nodes_df[base_col].astype(bool)

    # 结果容器
    out = np.empty(len(nodes_df), dtype=float)

    # 准备任务参数：每个 cell 一组 args
    tasks = [
        (i, neigh_ids, target_s, base_s, eps)
        for i, neigh_ids in enumerate(k_neighbors_results)
    ]

    # 并行计算
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, value in ex.map(_enrichment_for_cell, tasks):
            out[i] = value

    # 写入新列
    nodes_df["hplot_enrichment_index"] = out
    return nodes_df



# # --------------------------- main ---------------------------
# def compute_enrichment_index(
#     nodes_df: pd.DataFrame,
#     k_neighbors_results: list[list],
#     target_col: str = "is_target_type",
#     base_col: str = "is_base_type",
#     eps: float = 1e-6,
# ) -> pd.DataFrame:
#     """
#     为每个 cell 计算 enrichment_index = T * T / (T + B + eps)
#     其中：
#       T = (k 邻居中 target=True 的数量) / (邻居总数)
#       B = (k 邻居中 base=True   的数量) / (邻居总数)
#
#     约定：
#       - k_neighbors_results[i] 对应 nodes_df.iloc[i] 这个 cell
#       - k_neighbors_results[i] 内的元素是该 cell 邻居的 nodes_df.index（cell ID）
#     """
#     # 基本检查
#     for col in (target_col, base_col):
#         if col not in nodes_df.columns:
#             raise KeyError(f"missing required column '{col}' in nodes_df")
#     if len(k_neighbors_results) != len(nodes_df):
#         raise ValueError("k_neighbors_results length must match len(nodes_df)")
#
#     # 取布林 Series（以 index 为键，方便 reindex 到邻居 ID）
#     target_s = nodes_df[target_col].astype(bool)
#     base_s   = nodes_df[base_col].astype(bool)
#
#     # 结果容器
#     out = np.empty(len(nodes_df), dtype=float)
#
#     # 逐 cell 计算
#     # 注：如果某 cell 没有邻居，则 T=B=0 → 指数=0/(0+0+eps)=0
#     for i, neigh_ids in enumerate(k_neighbors_results):
#         n = len(neigh_ids)
#         if n == 0:
#             out[i] = 0.0
#             continue
#
#         # 将邻居 ID 映射到布林值；不在 index 的 ID 视为 False
#         t_count = target_s.reindex(neigh_ids).fillna(False).sum()
#         b_count = base_s.reindex(neigh_ids).fillna(False).sum()
#
#         T = float(t_count) / n
#         B = float(b_count) / n
#         out[i] = T * T / (T + B + eps)
#
#     # 写入新列
#     nodes_df["hplot_enrichment_index"] = out
#     return nodes_df




def _check_enrichment_for_cell(
    args
):
    """
    Helper for parallel execution.

    Args:
        args: tuple (i, neighbors, model_output_df, N, R)

    Returns:
        i if cell i is enriched, else None
    """
    i, neighbors, model_output_df, N, R = args

    # size filter
    if len(neighbors) < N:
        return None

    # neighbors are row indices for model_output_df
    neighbor_df = model_output_df.iloc[neighbors]

    # ratio of base-type cells
    base_type_prop = neighbor_df["is_base_type"].sum() / len(neighbors)

    if base_type_prop >= R:
        return i
    return None


def identify_region_by_cell_function_enrichment(
    k_hop_neighbors_list: List[List[int]],
    model_output_df,
    N: int,
    R: float,
    max_workers: int = None,
):
    """
    Parallel version of identify_region_by_cell_function_enrichment
    using ThreadPoolExecutor.

    Args:
        k_hop_neighbors_list: list of neighbor index lists per cell
        model_output_df: DataFrame with column 'is_base_type'
        N: minimal neighborhood size
        R: minimal base-type ratio
        max_workers: passed to ThreadPoolExecutor (default: None = auto)

    Returns:
        model_output_df with boolean column 'is_base_region'
    """
    # prepare argument tuples once
    tasks = [
        (i, neighbors, model_output_df, N, R)
        for i, neighbors in enumerate(k_hop_neighbors_list)
    ]

    enriched_cells = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for result in ex.map(_check_enrichment_for_cell, tasks):
            if result is not None:
                enriched_cells.append(result)

    model_output_df["is_base_region"] = model_output_df.index.isin(enriched_cells)
    return model_output_df



# def identify_region_by_cell_function_enrichment(k_hop_neighbors_list, model_output_df, N, R):
#     """
#     Identifies cells whose k-hop neighborhood meets specified criteria for size and base cell ratio,
#     and adds an 'is_base_region' column to the DataFrame.
#
#     Args:
#         k_hop_neighbors_list: A list of lists, where each inner list contains the indices
#                               of k-hop neighbors for the corresponding cell.
#         model_output_df: A pandas DataFrame containing cell information, including a boolean column
#             'is_base' indicating base cells.
#         N: The minimal number of neighbors required in the k-hop neighborhood.
#         R: The minimal ratio of base cells required in the k-hop neighborhood.
#
#     Returns:
#         The input DataFrame with a new boolean column 'is_base_region' indicating cells
#         that meet the enrichment criteria.
#     """
#     # Create a copy to avoid modifying the original DataFrame outside the function's scope
#     # df = model_output_df.copy()
#     enriched_cells = []
#     for i, neighbors in enumerate(k_hop_neighbors_list):
#         if len(neighbors) >= N:
#             # Get the subset of the DataFrame for the neighbors
#             neighbor_df = model_output_df.iloc[neighbors]
#             # Calculate the ratio of base cells in the neighborhood
#             base_type_prop = neighbor_df['is_base_type'].sum() / len(neighbors)
#             if base_type_prop >= R:
#                 enriched_cells.append(i)
#
#     # Add the 'is_base_region' column to the DataFrame
#     model_output_df['is_base_region'] = model_output_df.index.isin(enriched_cells)
#
#     return model_output_df



def _is_border_for_index(args) -> tuple:
    """
    Helper for ThreadPoolExecutor.
    Args:
        args: (index, adjacency_list, df_index_set, is_base_region_series)
    Returns:
        (index, is_border: bool)
    """
    index, adjacency_list, df_index_set, is_base_region = args

    # 如果这个 cell 没有邻居，直接不是边界
    neighbors = adjacency_list.get(index)
    if not neighbors:
        return index, False

    # 只要有一个邻居存在于 DataFrame 中且不是 base_region，就视为边界 cell
    for neighbor_index in neighbors:
        if neighbor_index in df_index_set:
            # is_base_region 是一个 Series，只读访问在线程里是安全的
            if not bool(is_base_region.get(neighbor_index, False)):
                return index, True

    return index, False


def identify_border_cells(
    model_output_df: pd.DataFrame,
    adjacency_list: Dict[Any, List[Any]],
    max_workers: int = None,
) -> pd.DataFrame:
    """
    Identifies base border cells based on connections to non-base region cells,
    using ThreadPoolExecutor to parallelize over base_region_indices.

    Args:
        model_output_df: DataFrame with a boolean column 'is_base_region'.
        adjacency_list: dict-like adjacency list: node_id -> list of neighbor node_ids.
        max_workers: optional number of threads for ThreadPoolExecutor.

    Returns:
        The DataFrame with a new boolean column 'is_base_border'.
    """
    if "is_base_region" not in model_output_df.columns:
        raise KeyError("model_output_df must contain column 'is_base_region'")

    # 所有 cell index 的集合，用于 O(1) membership 检查
    df_index_set = set(model_output_df.index)

    # 只读 Series，可安全在多线程中访问
    is_base_region = model_output_df["is_base_region"].astype(bool)

    # 只在 base_region = True 的 cell 上做检查
    base_region_indices = is_base_region[is_base_region].index

    # 初始化输出 Series，全 False
    border_series = pd.Series(False, index=model_output_df.index)

    # 准备任务参数
    tasks = [
        (idx, adjacency_list, df_index_set, is_base_region)
        for idx in base_region_indices
    ]

    # 并行执行
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for idx, is_border in ex.map(_is_border_for_index, tasks):
            if is_border:
                border_series.loc[idx] = True

    # 一次性写回 DataFrame
    model_output_df["is_base_border"] = border_series

    return model_output_df


# def identify_border_cells(model_output_df, adjacency_list):
#     """
#     Identifies base border cells based on connections to non-base region cells.
#
#     Args:
#         model_output_df: DataFrame with 'is_base_region' column.
#         adjacency_list: Adjacency list representing connections between cells.
#
#     Returns:
#         The DataFrame with a new boolean column 'is_base_border'.
#     """
#     model_output_df['is_base_border'] = False
#     # Optimize by iterating through base region cells only
#     base_region_indices = model_output_df[model_output_df['is_base_region']].index
#     for index in base_region_indices:
#         if index in adjacency_list:
#             for neighbor_index in adjacency_list[index]:
#                 # Ensure neighbor exists in the DataFrame index and is not in a base region
#                 if neighbor_index in model_output_df.index and not model_output_df.loc[neighbor_index, 'is_base_region']:
#                     model_output_df.loc[index, 'is_base_border'] = True
#                     break # No need to check other neighbors if one non-base neighbor is found
#     return model_output_df


def calculate_distance_to_border(model_output_df, adjacency_list):
    """
    Calculates the shortest edge count from every cell to the nearest base border cell.

    Args:
        model_output_df: DataFrame with 'is_base_border' column.
        adjacency_list: Adjacency list representing connections between cells.

    Returns:
        The DataFrame with a new column 'edge_distance_to_border'.
    """
    # Initialize distances to infinity for all cells
    edge_distance_to_border = {index: float('inf') for index in model_output_df.index}
    queue = deque()

    # Start BFS from all base border cells
    border_cells = model_output_df[model_output_df['is_base_border']].index.tolist()
    for border_index in border_cells:
        # Ensure border cell exists in the graph keys before adding to queue
        if border_index in adjacency_list:
            edge_distance_to_border[border_index] = 0
            queue.append(border_index)

    while queue:
        current_cell = queue.popleft()

        if current_cell in adjacency_list:
            for neighbor in adjacency_list[current_cell]:
                # Ensure neighbor exists in the DataFrame index and hasn't been visited
                if neighbor in model_output_df.index and edge_distance_to_border[neighbor] == float('inf'):
                    edge_distance_to_border[neighbor] = edge_distance_to_border[current_cell] + 1
                    queue.append(neighbor)

    # Add the calculated edge count to the DataFrame
    model_output_df['distance_to_border'] = model_output_df.index.map(edge_distance_to_border)
    model_output_df['hplot_signed_distance_to_border'] = model_output_df['distance_to_border']
    model_output_df.loc[model_output_df['is_base_region'], 'hplot_signed_distance_to_border'] *= -1
    model_output_df['hplot_signed_distance_to_border'] = model_output_df['hplot_signed_distance_to_border'].replace([np.inf, -np.inf], np.nan)
    
    return model_output_df


def compute_hplot(df_with_distances, filtered_edges_df):
    """
    Calculates the target ratio by cumulative average distance to the tumor border.

    Args:
        df_with_distances: DataFrame with 'hplot_signed_distance_to_border' and 'is_target' columns.
        filtered_edges_df: DataFrame with 'source', 'target', and 'length' columns representing filtered edges.

    Returns:
        A pandas DataFrame with 'cumulative_avg_edge_length' and 'target_type_prop' columns,
        sorted by cumulative_avg_edge_length, ready for plotting.
    """
    # Group by hplot_signed_distance_to_border and calculate the ratio of targets
    # Handle potential empty groups or no targets at a distance
    # Exclude NaN distances from grouping
    base_type_prop_by_distance = df_with_distances.dropna(subset=['hplot_signed_distance_to_border']).groupby('hplot_signed_distance_to_border')[f'is_base_type'].apply(lambda x: x.sum() / len(x) if len(x) > 0 else 0)
    target_type_prop_by_distance = df_with_distances.dropna(subset=['hplot_signed_distance_to_border']).groupby('hplot_signed_distance_to_border')[f'is_target_type'].apply(lambda x: x.sum() / len(x) if len(x) > 0 else 0)

    # Step 1: Calculate average edge length between adjacent layers
    average_edge_length_between_layers = {}
    unique_distances = sorted(df_with_distances['hplot_signed_distance_to_border'].dropna().unique())

    for i in range(len(unique_distances) - 1):
        dist1 = unique_distances[i]
        dist2 = unique_distances[i+1]

        # Identify cells in the two adjacent layers
        cells_in_dist1 = df_with_distances[df_with_distances['hplot_signed_distance_to_border'] == dist1].index
        cells_in_dist2 = df_with_distances[df_with_distances['hplot_signed_distance_to_border'] == dist2].index

        # Find edges connecting cells in dist1 to cells in dist2
        connecting_edges = filtered_edges_df[
            ((filtered_edges_df['source'].isin(cells_in_dist1)) & (filtered_edges_df['target'].isin(cells_in_dist2))) |
            ((filtered_edges_df['source'].isin(cells_in_dist2)) & (filtered_edges_df['target'].isin(cells_in_dist1)))
        ]

        # Calculate the average length of these connecting edges
        if not connecting_edges.empty:
            average_length = connecting_edges['length'].mean()
            # Store the average length associated with the lower distance value of the pair
            # This makes the cumulative sum calculation more straightforward
            average_edge_length_between_layers[dist1] = average_length
        else:
            # Assign NaN if no edges connect these layers
            average_edge_length_between_layers[dist1] = np.nan

    # Step 2 & 3: Order average lengths by signed distance and calculate cumulative average edge length
    # Convert the dictionary to a pandas Series for easy sorting and cumulative sum
    avg_lengths_series = pd.Series(average_edge_length_between_layers)

    # Sort by the signed distance (index)
    avg_lengths_series = avg_lengths_series.sort_index()

    # A clearer way for cumulative sum with sign:
    cumulative_avg_lengths_dict = {0.0: 0.0} # Start at the border

    # Cumulative outwards (positive distances)
    current_dist = 0.0
    for signed_dist in sorted(unique_distances):
        if signed_dist > 0:
            prev_dist = unique_distances[unique_distances.index(signed_dist) - 1]
            if prev_dist in average_edge_length_between_layers: # avg length between prev_dist and signed_dist
                current_dist += average_edge_length_between_layers[prev_dist]
                cumulative_avg_lengths_dict[signed_dist] = current_dist
            elif signed_dist-1 in average_edge_length_between_layers: # Check if avg length from n-1 to n is available
                current_dist += average_edge_length_between_layers[signed_dist-1]
                cumulative_avg_lengths_dict[signed_dist] = current_dist
            else:
                cumulative_avg_lengths_dict[signed_dist] = np.nan # If no edge to prev layer, cumulative is NaN

    # Cumulative inwards (negative distances)
    current_dist = 0.0
    for signed_dist in sorted(unique_distances, reverse=True):
        if signed_dist < 0:
            # next_dist = unique_distances[unique_distances.index(signed_dist) + 1]
            if signed_dist in average_edge_length_between_layers: # avg length between signed_dist and next_dist
                current_dist -= average_edge_length_between_layers[signed_dist] # Subtract as we move inwards
                cumulative_avg_lengths_dict[signed_dist] = current_dist
            else:
                cumulative_avg_lengths_dict[signed_dist] = np.nan # If no edge to next layer, cumulative is NaN

    # Convert the dictionary to a Series and align with signed distances in plot_df
    cumulative_avg_lengths_series = pd.Series(cumulative_avg_lengths_dict)

    # Step 4 & 5: Group target ratio by signed distance and align with cumulative average edge lengths
    plot_df = pd.DataFrame({
        'layer': target_type_prop_by_distance.index,
        'base_type_prop': base_type_prop_by_distance.values,
        'target_type_prop': target_type_prop_by_distance.values
    })

    # Map the cumulative average edge lengths to the signed_distance in plot_df
    plot_df['distance'] = plot_df['layer'].map(cumulative_avg_lengths_series)

    # Drop rows where we couldn't calculate the cumulative average edge length
    plot_df = plot_df.dropna(subset=['distance'])

    # Sort by the new x-axis values for a clear line plot
    plot_df = plot_df.sort_values('layer')

    return plot_df


def compute_hmetrics(
    hplot_df: pd.DataFrame,
    range_min: int | None,
    range_max: int | None,
    hplot_samples_with_valid_range_only: bool = False,
    depth_weight_mode: str = "linear",   # "linear" (default) or "sigmoid"
    s: float = 6.0,                      # slope for sigmoid when depth_weight_mode="sigmoid"
) -> Dict[str, Any]:
    """
    Compute spatial interaction metrics between a target cell class (e.g., immune)
    and a base/tumor cell class across concentric layers around the tumor boundary.

    Required df_cells columns (row = cell):
      - layer (int): 0 = boundary; negative = inside tumor; positive = outside
      - target_type_prop (float in [0,1]): target-type proportion at the cell
      - base_type_prop   (float in [0,1]): tumor/base-type proportion at the cell
      - distance (float): signed distance for the cell's layer (mean per layer is used)
      - id (optional):   sample/image identifier (for valid-range filtering)

    Returns:
      {
        "valid": bool,  # whether observed layers cover [range_min, range_max]
        "intra": {
           "convergence_distance": float (negative toward deeper inside),
           "abundance_score":      float in [0,1] (immune abundance mean inside),
           "penetration_score":    float in [0,1] (depth-normalized immune CoM inside),
           "layerwise_enrichment_index": float in [0,1] (immune-weighted & depth-weighted mean of T/(T+B)),
           "global_enrichment_index":    float in [0,1],  # global T_mean/(T_mean+B_mean)
           "weighted_global_enrichment_index": float in [0,1],  # penetration * global_enrichment
        },
        "peri": {
           "convergence_distance": float (positive outward),
           "abundance_score":      float in [0,1] (immune abundance mean outside),
           "proximity_score":      float in [0,1] (distance-normalized immune CoM outside),
           "layerwise_enrichment_index": float in [0,1],
           "global_enrichment_index":    float in [0,1],
           "weighted_global_enrichment_index": float in [0,1],  # proximity * global_enrichment
        }
      }
    """

    # -------- optional: keep only images fully covering [range_min, range_max] --------
    df_work = hplot_df
    if hplot_samples_with_valid_range_only and ("id" in hplot_df.columns):
        layer_clean = pd.to_numeric(hplot_df["layer"], errors="coerce")
        img_layer = (
            pd.concat([hplot_df["id"], layer_clean.rename("layer")], axis=1)
            .dropna(subset=["id", "layer"])
        )
        img_layer["layer"] = img_layer["layer"].astype(int)
        if not img_layer.empty:
            per_img = img_layer.groupby("id")["layer"].agg(["min", "max"]).astype(int)
            valid_mask = (per_img["min"] <= range_min) & (per_img["max"] >= range_max)
            valid_ids = per_img.index[valid_mask].tolist()
            df_work = hplot_df[hplot_df["id"].isin(valid_ids)].copy()
        else:
            df_work = hplot_df.iloc[0:0].copy()  # nothing passes

    # -------- required columns & sanitize --------
    for col in ("layer", "target_type_prop", "base_type_prop", "distance"):
        if col not in df_work.columns:
            raise KeyError(f"missing required column '{col}'")

    layer = pd.to_numeric(df_work["layer"], errors="coerce").astype("Int64")
    target_prop = pd.to_numeric(df_work["target_type_prop"], errors="coerce").clip(0.0, 1.0)
    tumor_prop  = pd.to_numeric(df_work["base_type_prop"],   errors="coerce").clip(0.0, 1.0)
    distance    = pd.to_numeric(df_work["distance"],         errors="coerce")

    m = layer.notna() & target_prop.notna() & tumor_prop.notna() & distance.notna()
    layer, target_prop, tumor_prop, distance = layer[m].astype(int), target_prop[m], tumor_prop[m], distance[m]

    # Early exit if nothing remains
    if len(layer) == 0 or range_max is None or range_min is None:
        return {
            "valid": False,
            "intra": {
                "penetration_score": 0.0,
                "abundance_score": 0.0,
                "convergence_distance": 0.0,
                "layerwise_enrichment_index": 0.0,
                "global_enrichment_index": np.nan,
                "weighted_global_enrichment_index": 0.0,
            },
            "peri": {
                "proximity_score": 0.0,
                "abundance_score": 0.0,
                "convergence_distance": 0.0,
                "layerwise_enrichment_index": 0.0,
                "global_enrichment_index": np.nan,
                "weighted_global_enrichment_index": 0.0,
            },
        }

    # -------- coverage check --------
    obs_min, obs_max = int(layer.min()), int(layer.max())
    valid = (range_min >= obs_min) and (range_max <= obs_max)

    # -------- per-layer aggregates --------
    target_by_layer = target_prop.groupby(layer).mean()  # immune abundance per layer
    tumor_by_layer  = tumor_prop.groupby(layer).mean()   # tumor abundance per layer
    dist_by_layer   = distance.groupby(layer).mean()     # signed distance per layer

    # desired indices on both sides
    inside_levels  = list(range(0, range_min - 1, -1))  #  0, -1, -2, ..., range_min
    outside_levels = list(range(1, range_max + 1))      #  1,  2,  3, ..., range_max

    # align series to the indices we want (nearest)
    target_intra = _reindex_nearest(target_by_layer[target_by_layer.index <= 0], inside_levels).clip(0.0, 1.0)
    target_peri= _reindex_nearest(target_by_layer[target_by_layer.index >= 1], outside_levels).clip(0.0, 1.0)
    tumor_intra  = _reindex_nearest(tumor_by_layer[tumor_by_layer.index   <= 0], inside_levels).clip(0.0, 1.0)
    tumor_peri = _reindex_nearest(tumor_by_layer[tumor_by_layer.index   >= 1], outside_levels).clip(0.0, 1.0)

    dist_intra   = _reindex_nearest(dist_by_layer[dist_by_layer.index     <= 0], inside_levels)
    dist_peri  = _reindex_nearest(dist_by_layer[dist_by_layer.index     >= 1], outside_levels)

    # shorthand: immune abundance per layer (weights)
    p_intra, p_peri = target_intra, target_peri

    # -------- base spatial metrics --------
    # abundance = mean immune fraction per side
    abundance_intra  = _safe_mean(p_intra)
    abundance_peri = _safe_mean(p_peri)

    # outside proximity: immune-weighted CoM outward normalized by max distance
    total_p_out = float(np.nansum(p_peri.values))
    if total_p_out > 0 and len(dist_peri) > 0:
        com_out = _center_of_mass(p_peri, dist_peri)           # >0 outward
        D_out_max = float(np.nanmax(dist_peri.values)) or 0.0
        if D_out_max > 0:
            proximity_peri = float(np.clip(1.0 - (com_out / D_out_max), 0.0, 1.0))
        else:
            proximity_peri = 1.0
        convergence_distance_out = float(com_out)
    else:
        proximity_peri = 0.0
        convergence_distance_out = float(np.nanmax(dist_peri.values)) if len(dist_peri) else 0.0

    # inside penetration: immune-weighted mean depth normalized by max |range_min|
    # also save convergence_distance_intra as negative magnitude inward
    depth_intra_mag = (-dist_intra).clip(lower=0.0)  # positive inward depth
    p_intra_neg = p_intra[p_intra.index < 0]
    depth_intra_neg = depth_intra_mag.reindex(p_intra_neg.index)
    if len(p_intra_neg) and float(np.nansum(p_intra_neg.values)) > 0.0:
        com_in_mag = _center_of_mass(p_intra_neg, depth_intra_neg)  # positive magnitude inward
        convergence_distance_intra = -float(com_in_mag)              # report negative inward
    else:
        com_in_mag = 0.0
        convergence_distance_intra = 0.0

    max_depth = max(abs(int(range_min)), 1)
    if len(p_intra) and float(np.nansum(p_intra.values)) > 0.0:
        all_depths = pd.Series([abs(l) for l in inside_levels], index=inside_levels, dtype=float)
        mean_depth = float(np.nansum((all_depths * p_intra).values) / np.nansum(p_intra.values))
        penetration_intra = float(np.clip(mean_depth / max_depth, 0.0, 1.0))
    else:
        penetration_intra = 0.0

    # -------- layerwise enrichment (immune- & depth-weighted mean of R_i) --------
    eps = 1e-6
    R_intra  = (target_intra  / (target_intra  + tumor_intra  + eps)).clip(0.0, 1.0)
    R_peri = (target_peri / (target_peri + tumor_peri + eps)).clip(0.0, 1.0)
    valid_intra  = (target_intra  + tumor_intra)  > 0
    valid_peri = (target_peri + tumor_peri) > 0

    w_depth_intra  = _depth_weights(inside_levels,  depth_weight_mode, s, range_min, range_max, side="inside")
    w_depth_peri = _depth_weights(outside_levels, depth_weight_mode, s, range_min, range_max, side="outside")

    # final weights = immune abundance * depth weight
    w_intra  = (p_intra  * w_depth_intra ).where(valid_intra,  np.nan)
    w_peri = (p_peri * w_depth_peri).where(valid_peri, np.nan)

    num_intra  = (R_intra  * w_intra ).where(valid_intra,  np.nan)
    num_peri = (R_peri * w_peri).where(valid_peri, np.nan)

    layerwise_enrichment_intra  = _weighted_mean(num_intra,  w_intra)
    layerwise_enrichment_peri = _weighted_mean(num_peri, w_peri)

    # -------- global enrichment (side-wise mean T and B) --------
    target_intra_mean  = _safe_mean(target_intra)
    tumor_intra_mean   = _safe_mean(tumor_intra)
    target_peri_mean = _safe_mean(target_peri)
    tumor_peri_mean  = _safe_mean(tumor_peri)

    global_enrichment_intra  = float(target_intra_mean  / (target_intra_mean  + tumor_intra_mean  + eps))
    global_enrichment_peri = float(target_peri_mean / (target_peri_mean + tumor_peri_mean + eps))

    # -------- combine into outputs --------
    return {
        "valid": valid,
        "intra": {
            "convergence_distance":  convergence_distance_intra,  # negative inward
            "abundance_score":       abundance_intra,
            "penetration_score":     penetration_intra,
            "layerwise_enrichment_index":   layerwise_enrichment_intra,
            "global_enrichment_index":      global_enrichment_intra,
            "weighted_global_enrichment_index": penetration_intra * global_enrichment_intra,
        },
        "peri": {
            "convergence_distance":  convergence_distance_out,
            "abundance_score":       abundance_peri,
            "proximity_score":       proximity_peri,
            "layerwise_enrichment_index":   layerwise_enrichment_peri,
            "global_enrichment_index":      global_enrichment_peri,
            "weighted_global_enrichment_index": proximity_peri * global_enrichment_peri,
        }
    }