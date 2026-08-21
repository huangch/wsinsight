"""Profile discovered niches so each ``niche_n`` can be named.

Reads ``niche-outputs-csv/cells/<id>.csv`` produced by ``wsinsight niche`` and, for
each niche cluster, summarises:

* the mean cell-type composition (from ``prob_*`` columns), and
* the most enriched marker genes (from ``expr_*`` columns, if present).

For whole-slide (H&E) cohorts there are no ``expr_`` columns, so only the
composition fingerprint is produced; that is exactly what lets a human assign
biological names (tumor nest, immune-stroma, ...) to the otherwise arbitrary
cluster ids.

This module deliberately depends only on numpy/pandas so it can be imported and
run without the deep-learning stack used by ``wsinsight niche``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd


def _niche_columns(columns: List[str]) -> List[str]:
    """Return ``niche_*`` columns ordered by their integer index."""
    cols = [c for c in columns if c.startswith("niche_")]

    def _idx(c: str) -> int:
        try:
            return int(c.split("_", 1)[1])
        except (IndexError, ValueError):
            return 1 << 30

    return sorted(cols, key=_idx)


def niche_profile(
    results_dir: str | Path,
    top_genes: int = 10,
    top_types: int = 5,
    write: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Aggregate per-cell niche labels into per-cluster composition + marker tables.

    Returns ``(composition_df, markers_df)``. ``markers_df`` is ``None`` when the
    cells CSVs carry no ``expr_`` columns. When ``write`` is True the tables are
    saved to ``<results-dir>/niche-profile-composition.csv`` and
    ``<results-dir>/niche-profile-markers.csv``.
    """
    results_dir = Path(str(results_dir))
    cells_dir = results_dir / "niche-outputs-csv" / "cells"
    csvs = sorted(cells_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No niche cell CSVs found under {cells_dir}. Run `wsinsight niche` first."
        )

    head = pd.read_csv(csvs[0], nrows=1)
    niche_cols = _niche_columns(list(head.columns))
    if not niche_cols:
        raise ValueError(
            f"No niche_ columns in {csvs[0]}; was this produced by `wsinsight niche`?"
        )
    prob_cols = [c for c in head.columns if c.startswith("prob_")]
    expr_cols = [c for c in head.columns if c.startswith("expr_")]

    K = len(niche_cols)
    counts = np.zeros(K, dtype=np.float64)
    sum_prob = np.zeros((K, len(prob_cols)), dtype=np.float64)
    sum_expr = np.zeros((K, len(expr_cols)), dtype=np.float64) if expr_cols else None

    usecols = niche_cols + prob_cols + expr_cols
    for csv in csvs:
        df = pd.read_csv(csv, usecols=lambda c: c in usecols)
        oh = df[niche_cols].fillna(0).to_numpy()
        assigned = oh.sum(axis=1) > 0
        if not assigned.any():
            continue
        lab = oh[assigned].argmax(axis=1)
        P = (
            df.loc[assigned, prob_cols].fillna(0).to_numpy(dtype=np.float64)
            if prob_cols
            else None
        )
        E = (
            df.loc[assigned, expr_cols].fillna(0).to_numpy(dtype=np.float64)
            if expr_cols
            else None
        )
        for k in range(K):
            m = lab == k
            nk = int(m.sum())
            if nk == 0:
                continue
            counts[k] += nk
            if P is not None:
                sum_prob[k] += P[m].sum(axis=0)
            if E is not None:
                sum_expr[k] += E[m].sum(axis=0)

    denom = np.maximum(counts, 1.0)[:, None]
    total = max(counts.sum(), 1.0)

    # ---- composition table ----
    comp = pd.DataFrame(
        sum_prob / denom,
        columns=[c[len("prob_") :] for c in prob_cols],
        index=[f"niche_{k}" for k in range(K)],
    )
    comp.insert(0, "n_cells", counts.astype(int))
    comp.insert(1, "frac", counts / total)
    if prob_cols:
        type_names = [c[len("prob_") :] for c in prob_cols]
        comp["top_types"] = [
            ", ".join(
                f"{type_names[j]}={comp.iloc[k][type_names[j]]:.2f}"
                for j in np.argsort(-(sum_prob[k] / denom[k]))[:top_types]
            )
            for k in range(K)
        ]

    # ---- marker table ----
    markers = None
    if expr_cols and sum_expr is not None:
        mean_expr = sum_expr / denom
        global_mean = sum_expr.sum(axis=0) / total
        eps = 1e-6
        enr = np.log2((mean_expr + eps) / (global_mean[None, :] + eps))
        genes = [c[len("expr_") :] for c in expr_cols]
        rows = []
        for k in range(K):
            order = np.argsort(-enr[k])[:top_genes]
            for rank, gi in enumerate(order, start=1):
                rows.append(
                    {
                        "niche": f"niche_{k}",
                        "rank": rank,
                        "gene": genes[gi],
                        "mean_expr": float(mean_expr[k, gi]),
                        "log2_enrichment": float(enr[k, gi]),
                    }
                )
        markers = pd.DataFrame(rows)

    if write:
        comp.to_csv(results_dir / "niche-profile-composition.csv")
        if markers is not None:
            markers.to_csv(results_dir / "niche-profile-markers.csv", index=False)

    return comp, markers
