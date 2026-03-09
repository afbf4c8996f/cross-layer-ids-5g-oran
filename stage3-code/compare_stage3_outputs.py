#!/usr/bin/env python3
"""Compare two Stage-3 output directories.

Goal: a practical reproducibility / regression check.

It compares:
- metrics/metrics_binary.csv
- metrics/metrics_multiclass.csv
- prediction parquets under predictions/

It reports max-abs-diff for numeric columns and exact-match for key/string columns.

Usage:
  python3 compare_stage3_outputs.py --out_a <dirA> --out_b <dirB> --tol 1e-6

Notes:
- It aligns prediction rows by KEY_COLS to avoid false mismatches caused by row-order.
- If a parquet exists in one output but not the other, that's a failure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


KEY_COLS = ["run_id", "window_start_s", "window_end_s"]


def _fail(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        _fail(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _compare_csv(path_a: Path, path_b: Path, *, tol: float, sort_keys: List[str]) -> None:
    if not path_a.exists():
        _fail(f"Missing: {path_a}")
    if not path_b.exists():
        _fail(f"Missing: {path_b}")

    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)

    if list(a.columns) != list(b.columns):
        _fail(f"CSV columns differ for {path_a.name}")
    if len(a) != len(b):
        _fail(f"CSV row count differs for {path_a.name}: {len(a)} vs {len(b)}")

    a = a.sort_values(sort_keys).reset_index(drop=True)
    b = b.sort_values(sort_keys).reset_index(drop=True)

    # Exact match for non-numeric
    num_cols = a.select_dtypes(include=[np.number]).columns.tolist()
    non_num_cols = [c for c in a.columns if c not in num_cols]

    for c in non_num_cols:
        if not a[c].astype(str).equals(b[c].astype(str)):
            _fail(f"{path_a.name}: non-numeric column mismatch: {c}")

    # Numeric match within tol
    worst: List[Tuple[str, float]] = []
    for c in num_cols:
        da = a[c].to_numpy(dtype=float)
        db = b[c].to_numpy(dtype=float)
        if np.isnan(da).any() or np.isnan(db).any():
            _fail(f"{path_a.name}: NaN encountered in numeric column {c}")
        diff = _max_abs_diff(da, db)
        worst.append((c, diff))
        if diff > tol:
            _fail(f"{path_a.name}: column {c} max|diff|={diff:.3g} > tol={tol}")

    worst.sort(key=lambda x: x[1], reverse=True)
    top = ", ".join([f"{c}={d:.3g}" for c, d in worst[:6]])
    _ok(f"{path_a.name}: OK within tol={tol}. Worst diffs: {top}")


def _align_by_key(df_pred: pd.DataFrame, df_ref: pd.DataFrame) -> pd.DataFrame:
    if not all(c in df_pred.columns for c in KEY_COLS):
        _fail(f"Prediction parquet missing KEY_COLS: {KEY_COLS}")
    if not all(c in df_ref.columns for c in KEY_COLS):
        _fail(f"Reference parquet missing KEY_COLS: {KEY_COLS}")

    a = df_ref[KEY_COLS].copy()
    b = df_pred[KEY_COLS].copy()
    a["_i"] = np.arange(len(a), dtype=int)
    b["_j"] = np.arange(len(b), dtype=int)
    m = a.merge(b, on=KEY_COLS, how="inner")
    if len(m) != len(a) or len(m) != len(b):
        _fail("KEY_COLS mismatch when aligning predictions")
    m = m.sort_values("_i")
    j = m["_j"].to_numpy(dtype=int)
    return df_pred.iloc[j].reset_index(drop=True)


def _compare_parquet(pa: Path, pb: Path, *, tol: float) -> None:
    a = pd.read_parquet(pa)
    b = pd.read_parquet(pb)

    if list(a.columns) != list(b.columns):
        _fail(f"Parquet columns differ: {pa} vs {pb}")
    if len(a) != len(b):
        _fail(f"Parquet row count differs: {pa} ({len(a)}) vs {pb} ({len(b)})")

    # Align both by their own KEY_COLS order (use a as reference)
    b = _align_by_key(b, a)

    num_cols = a.select_dtypes(include=[np.number]).columns.tolist()
    non_num_cols = [c for c in a.columns if c not in num_cols]

    # exact match for key/string columns
    for c in non_num_cols:
        if not a[c].astype(str).equals(b[c].astype(str)):
            _fail(f"Parquet mismatch in column {c}: {pa}")

    # numeric within tol
    for c in num_cols:
        da = a[c].to_numpy(dtype=float)
        db = b[c].to_numpy(dtype=float)
        if np.isnan(da).any() or np.isnan(db).any():
            _fail(f"Parquet NaN in numeric column {c}: {pa}")
        diff = _max_abs_diff(da, db)
        if diff > tol:
            _fail(f"Parquet {pa.name}: column {c} max|diff|={diff:.3g} > tol={tol}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_a", required=True)
    ap.add_argument("--out_b", required=True)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--limit_parquets", type=int, default=0, help="If >0, only compare first N parquets")
    args = ap.parse_args()

    out_a = Path(args.out_a).expanduser()
    out_b = Path(args.out_b).expanduser()
    tol = float(args.tol)

    # CSV comparisons
    _compare_csv(
        out_a / "metrics" / "metrics_binary.csv",
        out_b / "metrics" / "metrics_binary.csv",
        tol=tol,
        sort_keys=["split", "seed", "time_ordered", "W", "S", "feature_ablation", "part", "task", "model", "system"],
    )
    _compare_csv(
        out_a / "metrics" / "metrics_multiclass.csv",
        out_b / "metrics" / "metrics_multiclass.csv",
        tol=tol,
        sort_keys=["split", "seed", "time_ordered", "W", "S", "feature_ablation", "part", "task", "model", "system"],
    )

    # Parquet comparisons
    pred_a = out_a / "predictions"
    pred_b = out_b / "predictions"
    if not pred_a.exists() or not pred_b.exists():
        _fail("Both outputs must contain predictions/ directory")

    files_a = sorted([p for p in pred_a.rglob("*.parquet")])
    files_b = sorted([p for p in pred_b.rglob("*.parquet")])

    rel_a = {p.relative_to(pred_a) for p in files_a}
    rel_b = {p.relative_to(pred_b) for p in files_b}

    only_a = sorted(rel_a - rel_b)
    only_b = sorted(rel_b - rel_a)
    if only_a or only_b:
        if only_a:
            print("[DIFF] Parquets only in A:")
            for r in only_a[:20]:
                print("  ", r)
        if only_b:
            print("[DIFF] Parquets only in B:")
            for r in only_b[:20]:
                print("  ", r)
        _fail("Prediction parquet set differs between outputs")

    rel_all = sorted(rel_a)
    if args.limit_parquets and args.limit_parquets > 0:
        rel_all = rel_all[: int(args.limit_parquets)]

    for i, rel in enumerate(rel_all, 1):
        pa = pred_a / rel
        pb = pred_b / rel
        _compare_parquet(pa, pb, tol=tol)
        if i % 50 == 0:
            print(f"[OK] Compared {i}/{len(rel_all)} parquets")

    _ok(f"Compared {len(rel_all)} parquet(s) successfully within tol={tol}")


if __name__ == "__main__":
    main()
