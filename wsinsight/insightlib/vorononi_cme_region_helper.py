"""Voronoi-based CME region helpers for merging cell clusters into polygons."""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
import math
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely import set_precision
import geopandas as gpd

# Optional acceleration (only used if available and use_gpd_overlap=True)
# try:
#     import geopandas as gpd
#     _HAS_GPD = True
# except Exception:
#     _HAS_GPD = False


# ============================ STRtree pairs (version-agnostic) ============================

def _strtree_query_pairs(tree: STRtree, items):
    """Return unique geometry index pairs that intersect, abstracting STRtree APIs."""
    if hasattr(tree, "query_bulk"):
        pairs = tree.query_bulk(items)
        out, ia, ib = set(), np.asarray(pairs[0]).tolist(), np.asarray(pairs[1]).tolist()
        for i, j in zip(ia, ib):
            if j > i:
                out.add((int(i), int(j)))
        return sorted(out)

    out = set()
    probe = tree.query(items[0])
    returns_indices = (len(probe) > 0 and isinstance(probe[0], (int, np.integer)))

    if returns_indices:
        for i, gi in enumerate(items):
            for j in tree.query(gi):
                j = int(j)
                if j > i:
                    out.add((i, j))
        return sorted(out)

    def _geom_key(g):
        try:
            gt = g.geom_type
            if gt == "LineString":
                coords = tuple(map(tuple, np.asarray(g.coords, dtype=float)))
                return ("LS", coords)
            if gt == "Polygon":
                coords = tuple(map(tuple, np.asarray(g.exterior.coords, dtype=float)))
                return ("PG", coords)
            return ("WKB", g.wkb)
        except Exception:
            try:
                return ("BA", g.bounds, getattr(g, "area", None))
            except Exception:
                return None

    key2idxs: Dict[Any, List[int]] = {}
    for idx, g in enumerate(items):
        k = _geom_key(g)
        if k is not None:
            key2idxs.setdefault(k, []).append(idx)

    for i, gi in enumerate(items):
        hits = tree.query(gi)
        for h in hits:
            k = _geom_key(h)
            if k is None:
                for j in range(i + 1, len(items)):
                    try:
                        if h.equals_exact(items[j], 0.0):
                            out.add((i, j))
                    except Exception:
                        pass
                continue
            for j in key2idxs.get(k, []):
                if j > i:
                    out.add((i, j))
    return sorted(out)


# ============================ Voronoi helpers ============================

def _finite_voronoi_regions_no_bbox(vor: Voronoi, far_mult: float = 10.0) -> Dict[int, Polygon]:
    """Construct bounded Voronoi regions by extending rays to a distant envelope."""
    out: Dict[int, Polygon] = {}
    center = vor.points.mean(axis=0)
    R = np.linalg.norm(vor.points - center, axis=1).max() * float(far_mult)

    ridges: Dict[int, List[Tuple[int, int, int]]] = {}
    for (p, q), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        ridges.setdefault(p, []).append((q, v1, v2))
        ridges.setdefault(q, []).append((p, v1, v2))

    for p, reg_idx in enumerate(vor.point_region):
        verts = vor.regions[reg_idx]
        if not verts:
            continue
        if all(v >= 0 for v in verts):
            poly = Polygon(vor.vertices[verts]).buffer(0)
            if not poly.is_empty:
                out[p] = poly
            continue

        new_vertices = []
        for q, v1, v2 in ridges.get(p, []):
            if v1 >= 0 and v2 >= 0:
                new_vertices.append(tuple(vor.vertices[v1]))
                new_vertices.append(tuple(vor.vertices[v2]))
                continue
            t = vor.points[q] - vor.points[p]
            n = np.array([-t[1], t[0]], dtype=float)
            n /= (np.linalg.norm(n) + 1e-12)
            midpoint = (vor.points[p] + vor.points[q]) / 2.0
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = midpoint + direction * R
            if v1 >= 0:
                new_vertices.append(tuple(vor.vertices[v1]))
            elif v2 >= 0:
                new_vertices.append(tuple(vor.vertices[v2]))
            else:
                new_vertices.append(tuple(far_point))

        if not new_vertices:
            continue
        pts = np.asarray(list({v for v in new_vertices}), dtype=float)
        if pts.shape[0] < 3:
            continue
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        poly = Polygon(pts[np.argsort(ang)]).buffer(0)
        if not poly.is_empty:
            out[p] = poly
    return out


def build_capped_voronoi_from_df(
    df: pd.DataFrame,
    *,
    mpp: float,
    max_radius_um: float = 15.0,
    cme_prefix: str = "cme_",
    circle_resolution: int = 64,
    min_area: float = 0.0
) -> Tuple[Dict[int, List[Polygon]], np.ndarray]:
    """Generate per-cluster capped Voronoi polygons and return label mapping."""
    cx = df["minx"].to_numpy(float) + df["width"].to_numpy(float)/2.0
    cy = df["miny"].to_numpy(float) + df["height"].to_numpy(float)/2.0
    pts_all = np.column_stack([cx, cy])

    cme_cols = [c for c in df.columns if c.startswith(cme_prefix)]
    if not cme_cols:
        raise ValueError(f"No columns start with '{cme_prefix}'.")
    cme_mat = df[cme_cols].to_numpy(float)
    valid_mask = np.asarray(cme_mat.sum(axis=1) > 0.0, dtype=bool)
    if not np.any(valid_mask):
        return {}, np.array([], dtype=int)

    labels_full = cme_mat.argmax(axis=1)
    pts = pts_all[valid_mask]
    labels = labels_full[valid_mask]

    if len(pts) < 2:
        return {}, labels

    vor = Voronoi(pts)
    regions = _finite_voronoi_regions_no_bbox(vor)
    r_px = float(max_radius_um) / float(mpp)

    label_to_polys: Dict[int, List[Polygon]] = {}
    for i, reg in regions.items():
        if reg.is_empty:
            continue
        x, y = pts[i]
        disk = Point(x, y).buffer(r_px, resolution=circle_resolution)
        p = reg.intersection(disk).buffer(0)
        if p.is_empty:
            continue
        if min_area > 0.0 and p.area < min_area:
            continue
        lab = int(labels[i])
        if p.geom_type == "Polygon":
            label_to_polys.setdefault(lab, []).append(p)
        else:
            label_to_polys.setdefault(lab, []).extend([g for g in p.geoms if not g.is_empty])

    return label_to_polys, labels


# ============================ Index / remap helpers ============================

def _union_find_components(n: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """Group vertex indices into connected components given undirected edges."""
    parent = list(range(n))
    rank = [0] * n
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
    for i, j in edges: union(i, j)
    buckets: Dict[int, List[int]] = {}
    for i in range(n): buckets.setdefault(find(i), []).append(i)
    return list(buckets.values())

def remap_edges_to_valid_indices(edges_df: pd.DataFrame, valid_mask: np.ndarray) -> pd.DataFrame:
    """Shrink edge indices down to the subset of nodes flagged as valid."""
    valid_idx = np.flatnonzero(valid_mask)
    orig2valid = {orig: k for k, orig in enumerate(valid_idx)}
    e = edges_df.copy()
    e["source"] = e["source"].map(orig2valid)
    e["target"] = e["target"].map(orig2valid)
    e = e.dropna().astype({"source": int, "target": int})
    return e


# ============================ Exact shared-edge merge helpers ============================

def _q_coord(x: float, grid: float) -> float:
    return float(np.round(x / grid) * grid) if grid > 0 else float(x)

def _normalize_edge(p1, p2, grid):
    a = (_q_coord(p1[0], grid), _q_coord(p1[1], grid))
    b = (_q_coord(p2[0], grid), _q_coord(p2[1], grid))
    return (a, b) if a <= b else (b, a)

def _polygon_edges(poly: Polygon, grid: float, include_holes: bool = True):
    edges = []
    def ring_edges(coords):
        for i in range(len(coords) - 1):
            p1 = coords[i]; p2 = coords[i+1]
            edges.append(_normalize_edge((p1[0], p1[1]), (p2[0], p2[1]), grid))
    ring_edges(list(poly.exterior.coords))
    if include_holes:
        for hole in poly.interiors:
            ring_edges(list(hole.coords))
    return edges

def _build_shared_edge_adjacency(polys: List[Polygon], edge_grid_px: float) -> List[Tuple[int,int]]:
    edge_to_polys: Dict[Tuple[Tuple[float,float], Tuple[float,float]], List[int]] = {}
    for i, g in enumerate(polys):
        if g.is_empty: continue
        for e in _polygon_edges(g, edge_grid_px, True):
            lst = edge_to_polys.get(e)
            if lst is None:
                edge_to_polys[e] = [i]
            else:
                if lst[-1] != i:
                    lst.append(i)
    adj = []
    for lst in edge_to_polys.values():
        if len(lst) < 2: continue
        uniq = sorted(set(lst))
        for a in range(len(uniq)):
            for b in range(a+1, len(uniq)):
                adj.append((uniq[a], uniq[b]))
    return sorted(set((min(i,j), max(i,j)) for (i,j) in adj)) if adj else []


# ============================ Tolerant overlap (classic) ============================

def _segmentize_polygon_edges(poly: Polygon, grid: float, include_holes: bool = True):
    segs: List[Tuple[LineString, int]] = []
    rid = 0
    def add_ring(coords, rid):
        for i in range(len(coords) - 1):
            x1,y1 = coords[i]; x2,y2 = coords[i+1]
            x1 = _q_coord(x1, grid); y1 = _q_coord(y1, grid)
            x2 = _q_coord(x2, grid); y2 = _q_coord(y2, grid)
            if (x1 == x2) and (y1 == y2): continue
            segs.append((LineString([(x1,y1),(x2,y2)]), rid))
    add_ring(list(poly.exterior.coords), rid); rid += 1
    if include_holes:
        for hole in poly.interiors:
            add_ring(list(hole.coords), rid); rid += 1
    return segs

def _angle_between(a, b) -> float:
    ax, ay = a; bx, by = b
    na = math.hypot(ax, ay); nb = math.hypot(bx, by)
    if na == 0 or nb == 0: return math.pi
    cosv = max(-1.0, min(1.0, (ax*bx + ay*by) / (na*nb)))
    return math.acos(cosv)

def _colinear_close_overlap_len(s1: LineString, s2: LineString, angle_tol_rad: float, dist_tol: float) -> float:
    if not s1.envelope.intersects(s2.envelope):
        return 0.0
    (x1,y1),(x2,y2) = list(s1.coords)
    (u1,v1),(u2,v2) = list(s2.coords)
    v1a = (x2-x1, y2-y1); v2a = (u2-u1, v2-v1)
    ang = _angle_between(v1a, v2a)
    ang = min(ang, abs(math.pi - ang))
    if ang > angle_tol_rad: return 0.0
    if s1.distance(s2) > dist_tol: return 0.0

    def proj(seg):
        (a,b),(c,d) = seg.coords
        return sorted([a,c]) if abs(c-a) >= abs(d-b) else sorted([b,d])
    r1 = proj(s1); r2 = proj(s2)
    lo = max(r1[0], r2[0]); hi = min(r1[1], r2[1])
    return float(max(0.0, hi - lo))

# def _build_overlap_edge_adjacency(polys: List[Polygon],
#                                   edge_grid_px: float,
#                                   angle_tol_deg: float,
#                                   dist_tol_px: float,
#                                   min_overlap_px: float) -> List[Tuple[int,int]]:
#     all_segs: List[LineString] = []
#     owners: List[int] = []
#     for i, g in enumerate(polys):
#         if g.is_empty: continue
#         segs = [ls for (ls, _) in _segmentize_polygon_edges(g, edge_grid_px, True)]
#         all_segs.extend(segs); owners.extend([i] * len(segs))
#     if not all_segs: return []
#
#     tree = STRtree(all_segs)
#     pairs = _strtree_query_pairs(tree, all_segs)
#     angle_tol_rad = math.radians(angle_tol_deg)
#     pairs_set = set()
#
#     for a, b in pairs:
#         oi, oj = owners[a], owners[b]
#         if oi == oj: continue
#         s1, s2 = all_segs[a], all_segs[b]
#         overlap = _colinear_close_overlap_len(s1, s2, angle_tol_rad, dist_tol_px)
#         if overlap >= min_overlap_px:
#             i, j = sorted((oi, oj))
#             pairs_set.add((i, j))
#     return sorted(pairs_set)


# ============================ Tolerant overlap (GeoPandas / sindex) ============================

def _build_overlap_edge_adjacency_gpd(polys: List[Polygon],
                                      edge_grid_px: float,
                                      angle_tol_deg: float,
                                      dist_tol_px: float,
                                      min_overlap_px: float) -> List[Tuple[int,int]]:
    """
    GeoPandas+sindex accelerated version. Same thresholds & logic as classic path.
    """
    # if not _HAS_GPD:
    #     # fall back if geopandas not available
    #     return _build_overlap_edge_adjacency(polys, edge_grid_px, angle_tol_deg, dist_tol_px, min_overlap_px)

    # segmentize & build owner table
    seg_geom, owners = [], []
    for i, g in enumerate(polys):
        if g.is_empty: continue
        for ls, _ in _segmentize_polygon_edges(g, edge_grid_px, True):
            seg_geom.append(ls); owners.append(i)
    if not seg_geom:
        return []

    gdf = gpd.GeoDataFrame({"owner": owners}, geometry=gpd.GeoSeries(seg_geom), crs=None)  # pixel space
    # Expand bboxes by dist_tol_px to get candidate pairs
    # (buffer(0) keeps them as LineStrings; but we only need bbox)
    b = gdf.geometry.bounds
    # Build an R-tree once:
    sidx = gdf.sindex

    cand_pairs = set()
    for i, row in gdf.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        query_rect = (minx - dist_tol_px, miny - dist_tol_px, maxx + dist_tol_px, maxy + dist_tol_px)
        for j in sidx.intersection(query_rect):
            if j <= i: 
                continue
            if gdf.at[i, "owner"] == gdf.at[j, "owner"]:
                continue
            cand_pairs.add((int(i), int(j)))

    angle_tol_rad = math.radians(angle_tol_deg)
    pairs_set = set()
    for i, j in cand_pairs:
        s1 = gdf.geometry.iat[i]; s2 = gdf.geometry.iat[j]
        overlap = _colinear_close_overlap_len(s1, s2, angle_tol_rad, dist_tol_px)
        if overlap >= min_overlap_px:
            oi, oj = int(gdf.at[i, "owner"]), int(gdf.at[j, "owner"])
            a, b = sorted((oi, oj))
            pairs_set.add((a, b))
    return sorted(pairs_set)


# ============================ Self-clean & fallback boundary length ============================

def _clean_polygon_self_overlaps(poly: Polygon, grid_px: float) -> Polygon:
    """Resolve self-overlaps/numerical noise in polygons using snap+buffer tricks."""
    if poly.is_empty: return poly
    if grid_px > 0: poly = set_precision(poly, grid_px)
    p = poly.buffer(0)
    if p.is_empty: return p
    if p.geom_type == "Polygon": return p
    try:
        u = unary_union(p)
        if u.geom_type == "Polygon": return u
        parts = [g for g in u.geoms if not g.is_empty]
        if not parts: return u
        parts.sort(key=lambda g: g.area, reverse=True)
        return parts[0]
    except Exception:
        return p

def _shared_boundary_len(a: Polygon, b: Polygon, tol_len_px: float, boundary_buffer_px: float = 0.0) -> float:
    """Measure shared boundary length between polygons with optional buffering."""
    if boundary_buffer_px > 0:
        a = a.buffer(boundary_buffer_px)
        b = b.buffer(boundary_buffer_px)
    inter = a.boundary.intersection(b.boundary)
    if inter.is_empty: return 0.0
    if inter.geom_type == "LineString": return inter.length
    if inter.geom_type == "MultiLineString": return sum(seg.length for seg in inter.geoms)
    return 0.0


# ============================ Main pipeline ============================

def merge_same_label_by_shared_edges_iterative(
    df: pd.DataFrame,
    edges_df: pd.DataFrame,          # ['source','target'] from your filtered Delaunay
    cme_clustering_k = 10,
    *,
    mpp: float,
    max_radius_um: float = 15.0,
    cme_prefix: str = "cme_",
    circle_resolution: int = 64,
    min_area: float = 0.0,
    # exact-edge (orientation-free) snapping grid
    edge_grid_px: float = 0.25,
    # partial-overlap tolerances
    angle_tol_deg: float = 3.0,
    dist_tol_px: float = 0.5,
    min_overlap_px: float = 1.0,
    # fallback boundary-length test
    tol_len_px: float = 0.5,
    boundary_buffer_px: float = 0.0,
    # iteration
    max_iter: int = 6,
    grow_buffer: float = 1.0,
    # NEW: use GeoPandas+sindex for tolerant overlap (keeps same thresholds)
    # use_gpd_overlap: bool = False
) -> pd.DataFrame:
    """Merge CME Voronoi pieces of the same label using exact and tolerant edges."""
    # Step 0: centers & labels
    cx = df["minx"].to_numpy(float) + df["width"].to_numpy(float)/2.0
    cy = df["miny"].to_numpy(float) + df["height"].to_numpy(float)/2.0
    pts_all = np.column_stack([cx, cy])

    cme_cols = [c for c in df.columns if c.startswith(cme_prefix)]
    if not cme_cols:
        raise ValueError(f"No columns start with '{cme_prefix}'.")
    cme_mat = df[cme_cols].to_numpy(float)
    valid_mask = cme_mat.sum(axis=1) > 0.0
    if not np.any(valid_mask):
        return _pieces_dict_to_dataframe({}, cme_clustering_k=cme_clustering_k)

    labels_full = cme_mat.argmax(axis=1)
    pts = pts_all[valid_mask]
    labels = labels_full[valid_mask]
    N = len(pts)
    if N < 2:
        return _pieces_dict_to_dataframe({}, cme_clustering_k=cme_clustering_k)

    if edges_df["source"].max() >= N or edges_df["target"].max() >= N:
        edges_df = remap_edges_to_valid_indices(edges_df, valid_mask)

    # Step 1: Voronoi + cap
    vor = Voronoi(pts)
    regions = _finite_voronoi_regions_no_bbox(vor)
    r_px = float(max_radius_um) / float(mpp)

    capped: Dict[int, Polygon] = {}
    for i, poly in regions.items():
        if poly.is_empty: continue
        x, y = pts[i]
        disk = Point(x, y).buffer(r_px, resolution=circle_resolution)
        p = poly.intersection(disk).buffer(0)
        if not p.is_empty and (min_area <= 0 or p.area >= min_area):
            capped[i] = p

    # Step 2: initial merge via Delaunay neighbor pairs
    adj_edges: List[Tuple[int, int]] = []
    for _, row in edges_df.iterrows():
        i = int(row["source"]); j = int(row["target"])
        if i not in capped or j not in capped: continue
        if int(labels[i]) != int(labels[j]): continue
        shared = _shared_boundary_len(capped[i], capped[j], tol_len_px, boundary_buffer_px)
        if shared > tol_len_px:
            adj_edges.append((i, j))

    active = sorted(capped.keys())
    idx_map = {i: k for k, i in enumerate(active)}
    comp_edges = [(idx_map[i], idx_map[j]) for (i, j) in adj_edges]
    comps = _union_find_components(len(active), comp_edges)

    out: Dict[int, List[Polygon]] = {}
    rev = {k: i for i, k in idx_map.items()}
    for comp in comps:
        nodes = [rev[k] for k in comp if rev[k] in capped]
        if not nodes: continue
        lab = int(labels[nodes[0]])
        merged = unary_union([capped[u] for u in nodes]).buffer(0)
        parts = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)
        parts = [g for g in parts if (min_area <= 0 or g.area >= min_area)]
        if parts:
            out.setdefault(lab, []).extend(parts)

    used_nodes = set([i for e in adj_edges for i in e])
    for i in active:
        if i in used_nodes: continue
        lab = int(labels[i])
        if (min_area <= 0 or capped[i].area >= min_area):
            out.setdefault(lab, []).append(capped[i])

    # Step 3: iterative polygon-level merging
    buf = float(boundary_buffer_px)
    for _ in range(max_iter):
        changed = False
        new_out: Dict[int, List[Polygon]] = {}

        for lab, polys in out.items():
            if not polys:
                continue
            polys = [_clean_polygon_self_overlaps(g, grid_px=edge_grid_px) for g in polys]
            polys = [g for g in polys if (not g.is_empty and (min_area <= 0 or g.area >= min_area))]
            if len(polys) <= 1:
                new_out.setdefault(lab, []).extend(polys)
                continue

            snap_polys = [set_precision(g, edge_grid_px) if edge_grid_px > 0 else g for g in polys]
            edge_adj = _build_shared_edge_adjacency(snap_polys, edge_grid_px=edge_grid_px)

            work_polys = polys
            if edge_adj:
                comps_edge = _union_find_components(len(polys), edge_adj)
                merged_after_edges: List[Polygon] = []
                for comp in comps_edge:
                    m = unary_union([polys[k] for k in comp]).buffer(0)
                    if m.is_empty: continue
                    cands = [m] if m.geom_type == "Polygon" else list(m.geoms)
                    for g in cands:
                        if (min_area <= 0 or g.area >= min_area):
                            merged_after_edges.append(g)
                if len(merged_after_edges) != len(polys):
                    changed = True
                work_polys = merged_after_edges

            if len(work_polys) > 1:
                # if use_gpd_overlap:
                pairs = _build_overlap_edge_adjacency_gpd(
                    work_polys, edge_grid_px, angle_tol_deg, dist_tol_px, min_overlap_px
                )
                # else:
                #     pairs = _build_overlap_edge_adjacency(
                #         work_polys, edge_grid_px, angle_tol_deg, dist_tol_px, min_overlap_px
                #     )

                if pairs:
                    comps_ol = _union_find_components(len(work_polys), pairs)
                    after_ol: List[Polygon] = []
                    for comp in comps_ol:
                        m = unary_union([work_polys[k] for k in comp]).buffer(0)
                        if m.is_empty: continue
                        cands = [m] if m.geom_type == "Polygon" else list(m.geoms)
                        for g in cands:
                            if (min_area <= 0 or g.area >= min_area):
                                after_ol.append(g)
                    if len(after_ol) != len(work_polys):
                        changed = True
                    work_polys = after_ol

            work_polys = [_clean_polygon_self_overlaps(g, grid_px=edge_grid_px) for g in work_polys]
            work_polys = [g for g in work_polys if (not g.is_empty and (min_area <= 0 or g.area >= min_area))]
            new_out.setdefault(lab, []).extend(work_polys)

        out = new_out
        if not changed:
            break
        if grow_buffer != 1.0 and buf > 0:
            buf = buf * float(grow_buffer)

    return _pieces_dict_to_dataframe(out, cme_clustering_k=cme_clustering_k)


# ============================ Output serialization ============================

def _pieces_dict_to_dataframe(
    pieces_dict: dict,
    *,
    cme_clustering_k: int,
    geom_col: str = "polygon_wkt",
    geom_format: str = "wkt",
) -> pd.DataFrame:
    """Serialize merged polygons back to a DataFrame with CME one-hot columns."""
    import json, binascii
    cme_cols = [f"cme_{i}" for i in range(cme_clustering_k)]
    rows = []

    def serialize_geom(poly: Polygon):
        if geom_format == "wkt":
            return poly.wkt
        elif geom_format == "wkb_hex":
            return binascii.hexlify(poly.wkb).decode("ascii")
        elif geom_format == "coords_json":
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda p: p.area)
            xs, ys = poly.exterior.xy
            return json.dumps([[float(x), float(y)] for x, y in zip(xs, ys)])
        else:
            raise ValueError("geom_format must be 'wkt', 'wkb_hex', or 'coords_json'")

    def label_to_one_hot(lab: Any):
        if isinstance(lab, str):
            if lab.startswith("cme_"):
                lab = int(lab.split("_", 1)[1])
            else:
                raise ValueError(f"String label '{lab}' not in 'cme_i' form.")
        lab = int(lab)
        if not (0 <= lab < cme_clustering_k):
            raise ValueError(f"Label {lab} out of range for K={cme_clustering_k}.")
        vec = np.zeros(cme_clustering_k, dtype=np.float32); vec[lab] = 1.0
        return vec

    for lab, polys in pieces_dict.items():
        for poly in polys:
            if poly.is_empty or not isinstance(poly, (Polygon, MultiPolygon)):
                continue
            geom_val = serialize_geom(poly)
            one_hot = label_to_one_hot(lab)
            row = {n: float(v) for n, v in zip(cme_cols, one_hot)}
            row[geom_col] = geom_val
            row["area"] = float(poly.area)
            rows.append(row)

    return pd.DataFrame(rows, columns=cme_cols + [geom_col, "area"])
