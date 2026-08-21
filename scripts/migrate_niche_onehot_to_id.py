#!/usr/bin/env python3
"""Convert niche one-hot columns (niche_0 … niche_N) to a single niche_id column.

Handles both CSV files (niche-outputs-csv/) and AnnData h5ad files
(export-h5ad/).  In the old format, niche_0…niche_N were written as separate
float columns.  This script replaces them with a single integer niche_id.

Usage
-----
# Single file, in-place:
    python migrate_niche_onehot_to_id.py slide_A.csv
    python migrate_niche_onehot_to_id.py slide_A.h5ad

# Single file, to a new path:
    python migrate_niche_onehot_to_id.py slide_A.csv --out slide_A_v2.csv

# Multiple files (glob), in-place:
    python migrate_niche_onehot_to_id.py niche-outputs-csv/cells/*.csv
    python migrate_niche_onehot_to_id.py export-h5ad/*.h5ad

# Dry-run (print what would change, write nothing):
    python migrate_niche_onehot_to_id.py slide_A.csv --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _detect_niche_onehot_cols(columns) -> list[str]:
    """Return sorted list of niche_N one-hot columns."""
    cols = [c for c in columns if c.startswith("niche_") and c[6:].isdigit()]
    return sorted(cols, key=lambda c: int(c[6:]))


def _onehot_to_id(matrix: np.ndarray) -> np.ndarray:
    """argmax of one-hot rows; rows that are all-zero get -1 (unassigned)."""
    niche_id = matrix.argmax(axis=1).astype(int)
    niche_id[matrix.sum(axis=1) == 0] = -1
    return niche_id


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def migrate_csv(src: Path, dst: Path) -> str:
    df = pd.read_csv(src)
    onehot_cols = _detect_niche_onehot_cols(df.columns)

    if not onehot_cols:
        return "already_done" if "niche_id" in df.columns else "no_niche"

    niche_id = _onehot_to_id(df[onehot_cols].to_numpy(float))
    last_pos = df.columns.get_loc(onehot_cols[-1])
    df.insert(last_pos + 1, "niche_id", niche_id)
    df = df.drop(columns=onehot_cols)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(dst) + ".tmp"
    df.to_csv(tmp, index=False)
    Path(tmp).replace(dst)
    return "migrated"


# ---------------------------------------------------------------------------
# h5ad
# ---------------------------------------------------------------------------

def migrate_h5ad(src: Path, dst: Path) -> str:
    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata is required for .h5ad migration: pip install anndata")

    adata = ad.read_h5ad(src)
    onehot_cols = _detect_niche_onehot_cols(adata.obs.columns)

    if not onehot_cols:
        return "already_done" if "niche_id" in adata.obs.columns else "no_niche"

    niche_id = _onehot_to_id(adata.obs[onehot_cols].to_numpy(float))
    last_pos = adata.obs.columns.get_loc(onehot_cols[-1])

    # Insert niche_id at the same position and drop the one-hot columns.
    obs = adata.obs.copy()
    obs.insert(last_pos + 1, "niche_id", pd.Categorical(niche_id))
    obs = obs.drop(columns=onehot_cols)
    adata.obs = obs

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(dst) + ".tmp.h5ad"
    adata.write_h5ad(tmp)
    Path(tmp).replace(dst)
    return "migrated"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def migrate_file(src: Path, dst: Path) -> str:
    if src.suffix == ".h5ad":
        return migrate_h5ad(src, dst)
    return migrate_csv(src, dst)


def dry_run_file(src: Path) -> str:
    if src.suffix == ".h5ad":
        try:
            import anndata as ad
            adata = ad.read_h5ad(src)
            cols = _detect_niche_onehot_cols(adata.obs.columns)
            columns = adata.obs.columns
        except Exception as exc:
            return f"ERROR reading {src}: {exc}"
    else:
        df = pd.read_csv(src)
        cols = _detect_niche_onehot_cols(df.columns)
        columns = df.columns

    if cols:
        return f"would migrate: {src}  ({len(cols)} one-hot cols → niche_id)"
    if "niche_id" in columns:
        return f"already done:  {src}"
    return f"no niche cols: {src}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Migrate niche one-hot columns to niche_id (.csv and .h5ad)."
    )
    ap.add_argument(
        "src",
        type=Path,
        nargs="+",
        help="One or more .csv or .h5ad files to migrate.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Only valid with a single input file. Defaults to in-place.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files.",
    )
    args = ap.parse_args(argv)

    if args.out is not None and len(args.src) > 1:
        print("ERROR: --out can only be used with a single input file.", file=sys.stderr)
        return 1

    counts: dict[str, int] = {"migrated": 0, "already_done": 0, "no_niche": 0, "error": 0}
    labels = {
        "migrated":     "migrated",
        "already_done": "skipped (already has niche_id)",
        "no_niche":     "skipped (no niche columns)",
    }

    for src in args.src:
        if args.dry_run:
            print(" ", dry_run_file(src))
            continue

        dst = args.out if args.out is not None else src
        try:
            status = migrate_file(src, dst)
            counts[status] += 1
            suffix = f" → {dst}" if dst != src else ""
            print(f"  {labels[status]}: {src}{suffix}")
        except Exception as exc:
            counts["error"] += 1
            print(f"  ERROR: {src}: {exc}", file=sys.stderr)

    if not args.dry_run and len(args.src) > 1:
        print(
            f"\nDone. migrated={counts['migrated']}  "
            f"already_done={counts['already_done']}  "
            f"no_niche={counts['no_niche']}  "
            f"errors={counts['error']}"
        )
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())


def migrate_csv(src: Path, dst: Path) -> str:
    """Read *src*, replace one-hot niche columns with niche_id, write *dst*.

    Returns a short status string: 'migrated', 'already_done', or 'no_niche'.
    """
    df = pd.read_csv(src)
    onehot_cols = _detect_niche_onehot_cols(df)

    if not onehot_cols:
        if "niche_id" in df.columns:
            return "already_done"
        return "no_niche"

    # argmax of the one-hot block → integer niche_id.
    # Cells with all-zero rows (not assigned) get niche_id = -1.
    matrix = df[onehot_cols].to_numpy(float)
    row_sum = matrix.sum(axis=1)
    niche_id = matrix.argmax(axis=1).astype(int)
    niche_id[row_sum == 0] = -1  # unassigned cells

    # Insert niche_id immediately after the last one-hot column, then drop them.
    last_col_pos = df.columns.get_loc(onehot_cols[-1])
    df.insert(last_col_pos + 1, "niche_id", niche_id)
    df = df.drop(columns=onehot_cols)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(dst) + ".tmp"
    df.to_csv(tmp, index=False)
    Path(tmp).replace(dst)
    return "migrated"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Migrate niche one-hot columns to niche_id in one or more CSV files."
    )
    ap.add_argument(
        "src",
        type=Path,
        nargs="+",
        help="One or more niche CSV files to migrate.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output path. Only valid when a single input file is given. "
            "Defaults to in-place (overwrites the original)."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files.",
    )
    args = ap.parse_args(argv)

    if args.out is not None and len(args.src) > 1:
        print("ERROR: --out can only be used with a single input file.", file=sys.stderr)
        return 1

    counts = {"migrated": 0, "already_done": 0, "no_niche": 0, "error": 0}
    for src in args.src:
        dst = args.out if args.out is not None else src

        if args.dry_run:
            try:
                df = pd.read_csv(src)
            except Exception as exc:
                print(f"  ERROR reading {src.name}: {exc}", file=sys.stderr)
                continue
            cols = _detect_niche_onehot_cols(df)
            if cols:
                print(f"  would migrate: {src}  ({len(cols)} one-hot cols → niche_id)")
            elif "niche_id" in df.columns:
                print(f"  already done:  {src}")
            else:
                print(f"  no niche cols: {src}")
            continue

        try:
            status = migrate_csv(src, dst)
            counts[status] += 1
            label = {
                "migrated":     "migrated",
                "already_done": "skipped (already has niche_id)",
                "no_niche":     "skipped (no niche columns)",
            }[status]
            suffix = f" → {dst}" if dst != src else ""
            print(f"  {label}: {src}{suffix}")
        except Exception as exc:
            counts["error"] += 1
            print(f"  ERROR: {src}: {exc}", file=sys.stderr)

    if not args.dry_run and len(args.src) > 1:
        print(
            f"\nDone. migrated={counts['migrated']}  "
            f"already_done={counts['already_done']}  "
            f"no_niche={counts['no_niche']}  "
            f"errors={counts['error']}"
        )
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
