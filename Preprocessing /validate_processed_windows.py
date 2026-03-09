#!/usr/bin/env python3
"""
validate_processed_windows.py

Validator for Stage-2 processed datasets (outputs of preprocess_windows.py).

Behavior:
  - Prints a short status per split directory.
  - Exits with code 0 on success.
  - Raises an exception (non-zero exit) on any validation failure.
  - Writes a JSON report to:
        <out_dir>/processed/validation_processed_report.json

Validations performed (for each split_name and each stem):
  - required meta + label columns exist
  - no NaN/inf in feature columns
  - y_bin in {0,1}
  - y_cat non-empty
  - forbidden leakage columns absent
  - feature columns identical across train/val/test for each stem

"""

from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


META = {"run_id", "family", "window_start_s", "window_end_s"}
LABELS = {"y_bin", "y_cat"}
FORBIDDEN = {
    "attack_flow_count",
    "benign_flow_count",
    "attack_flow_frac",
    "window_has_attack_flow",
    # original labels should never be present post-processing
    "traffic_type_win",
    "attack_category_win",
    "attack_type_win",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stem_from_train_file(p: Path) -> str:
    name = p.name
    if not name.endswith("_train.parquet"):
        raise ValueError(f"Not a train parquet: {p}")
    return name[: -len("_train.parquet")]


def _hash_feature_names(cols: List[str]) -> str:
    # Stable hash for reproducibility without storing full list.
    h = hashlib.sha256()
    for c in cols:
        h.update(c.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _check_one_table(df: pd.DataFrame, context: str) -> None:
    # Required columns
    if not META.issubset(df.columns):
        missing = sorted(META - set(df.columns))
        raise AssertionError(f"{context}: missing META columns: {missing}")
    if not LABELS.issubset(df.columns):
        missing = sorted(LABELS - set(df.columns))
        raise AssertionError(f"{context}: missing LABEL columns: {missing}")

    # Forbidden columns
    bad = sorted(set(df.columns) & FORBIDDEN)
    if bad:
        raise AssertionError(f"{context}: forbidden/leakage columns present: {bad[:20]}")

    # Label sanity
    yb = df["y_bin"]
    if not pd.api.types.is_integer_dtype(yb) and not pd.api.types.is_bool_dtype(yb):
        # allow float if it still represents 0/1 exactly
        vals = set(pd.unique(yb.dropna()))
        if not vals.issubset({0, 1, 0.0, 1.0}):
            raise AssertionError(f"{context}: y_bin has unexpected values: {sorted(list(vals))[:10]}")
    else:
        vals = set(pd.unique(yb.dropna()))
        if not vals.issubset({0, 1}):
            raise AssertionError(f"{context}: y_bin has unexpected values: {sorted(list(vals))[:10]}")

    if df["y_cat"].astype("string").fillna("").str.len().min() == 0:
        raise AssertionError(f"{context}: y_cat contains empty/NaN values.")

    # Feature sanity: finite values
    feat_cols = [c for c in df.columns if c not in META and c not in LABELS]
    if not feat_cols:
        raise AssertionError(f"{context}: no feature columns found after excluding META/LABELS.")

    X = df[feat_cols].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(X).all():
        bad_mask = ~np.isfinite(X)
        i, j = np.where(bad_mask)
        raise AssertionError(f"{context}: non-finite value in feature '{feat_cols[int(j[0])]}' at row {int(i[0])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True, help="Stage-2 output directory containing processed/")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    proc_dir = out_dir / "processed"
    if not proc_dir.exists():
        raise FileNotFoundError(f"Missing: {proc_dir}")

    split_dirs = [p for p in proc_dir.iterdir() if p.is_dir()]
    if not split_dirs:
        raise FileNotFoundError(f"No split directories in {proc_dir}. Did preprocessing run?")

    report: Dict[str, object] = {
        "created_utc": _utc_now_iso(),
        "out_dir": str(out_dir),
        "processed_dir": str(proc_dir),
        "n_split_dirs": len(split_dirs),
        "splits": [],
    }

    total_stems = 0

    for sdir in sorted(split_dirs):
        train_files = sorted(sdir.glob("*_train.parquet"))
        if not train_files:
            raise FileNotFoundError(f"{sdir.name}: no *_train.parquet files found")

        split_entry: Dict[str, object] = {"split_name": sdir.name, "stems": []}
        print(f"\nChecking: {sdir.name}")

        for tr_path in train_files:
            stem = _stem_from_train_file(tr_path)
            va_path = sdir / f"{stem}_val.parquet"
            te_path = sdir / f"{stem}_test.parquet"
            if not va_path.exists() or not te_path.exists():
                raise FileNotFoundError(f"{sdir.name}/{stem}: missing val/test parquet")

            tr = pd.read_parquet(tr_path)
            va = pd.read_parquet(va_path)
            te = pd.read_parquet(te_path)

            _check_one_table(tr, f"{sdir.name}/{stem}/train")
            _check_one_table(va, f"{sdir.name}/{stem}/val")
            _check_one_table(te, f"{sdir.name}/{stem}/test")

            # Feature columns identical across partitions
            feat_tr = [c for c in tr.columns if c not in META and c not in LABELS]
            feat_va = [c for c in va.columns if c not in META and c not in LABELS]
            feat_te = [c for c in te.columns if c not in META and c not in LABELS]
            if feat_tr != feat_va or feat_tr != feat_te:
                raise AssertionError(f"{sdir.name}/{stem}: feature columns differ across train/val/test")

            stem_entry = {
                "stem": stem,
                "n_features": len(feat_tr),
                "feature_hash_sha256": _hash_feature_names(feat_tr),
                "n_train": int(tr.shape[0]),
                "n_val": int(va.shape[0]),
                "n_test": int(te.shape[0]),
                "y_bin_rate_train": float(tr["y_bin"].mean()) if tr.shape[0] else float("nan"),
                "y_bin_rate_val": float(va["y_bin"].mean()) if va.shape[0] else float("nan"),
                "y_bin_rate_test": float(te["y_bin"].mean()) if te.shape[0] else float("nan"),
                "n_y_cat_train": int(tr["y_cat"].nunique(dropna=True)),
                "n_y_cat_val": int(va["y_cat"].nunique(dropna=True)),
                "n_y_cat_test": int(te["y_cat"].nunique(dropna=True)),
            }
            split_entry["stems"].append(stem_entry)
            total_stems += 1

        print(f"OK: {sdir.name}")
        report["splits"].append(split_entry)

    report["n_stems_validated"] = total_stems

    out_path = proc_dir / "validation_processed_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nOK: all processed outputs validated.")
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
