#!/usr/bin/env python3
"""
validate_run_splits_v2.py

Validate run split JSON manifests created by make_run_splits.py.

What it checks (always):
  - train/val/test are disjoint
  - union equals all run_ids from paired_runs.csv
  - each split contains at least one benign run and at least one attack run
  - prints per-family counts

Optional strict check:
  --require-each-family
    Enforces that every family appears in train/val/test **only for stratified splits**
    (strategy == "stratified_by_family").

Why only stratified?
  Time-ordered splits are a realism stress test; requiring every family in every split can
  legitimately fail depending on chronology and is not required for valid binary evaluation.(not used in the paper)

If you *really* want strict family coverage on time-ordered splits too, pass:
  --require-each-family --apply-to-time-ordered

Example:
  python3 validate_run_splits_v2.py \
    --paired-runs /path/to/paired_runs.csv \
    --splits-dir /path/to/window-output/splits \
    --require-each-family
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd


def _load_runs(paired_runs_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(paired_runs_csv).copy()
    if "canon_stem" not in df.columns or "family" not in df.columns:
        raise ValueError("paired_runs.csv must contain columns: canon_stem, family")
    df["run_id"] = df["canon_stem"].astype(str)
    df["family"] = df["family"].astype(str)
    df = df.drop_duplicates("run_id").reset_index(drop=True)
    df["is_benign_run"] = df["family"].str.lower().eq("benign")
    df["is_attack_run"] = ~df["is_benign_run"]
    return df[["run_id", "family", "is_benign_run", "is_attack_run"]]


def _per_family_counts(runs_by_id: pd.DataFrame, ids: Set[str]) -> Dict[str, int]:
    sub = runs_by_id.loc[list(ids)]
    return sub["family"].value_counts().to_dict()


def _has_both_binary(runs_by_id: pd.DataFrame, ids: Set[str]) -> bool:
    sub = runs_by_id.loc[list(ids)]
    return bool(sub["is_benign_run"].any() and sub["is_attack_run"].any())


def _validate_one(payload: Dict[str, Any], runs: pd.DataFrame, require_each_family: bool, apply_to_time_ordered: bool) -> None:
    name = str(payload.get("split_name", "UNKNOWN"))
    strategy = str(payload.get("strategy", "")).strip().lower()

    tr = set(map(str, payload["train_run_ids"]))
    va = set(map(str, payload["val_run_ids"]))
    te = set(map(str, payload["test_run_ids"]))

    if tr & va or tr & te or va & te:
        raise AssertionError(f"{name}: overlaps detected among splits.")

    all_ids = set(runs["run_id"].astype(str).tolist())
    if (tr | va | te) != all_ids:
        missing = sorted(all_ids - (tr | va | te))
        extra = sorted((tr | va | te) - all_ids)
        raise AssertionError(f"{name}: coverage mismatch. missing={missing[:5]} extra={extra[:5]}")

    runs_by_id = runs.set_index("run_id")

    if not _has_both_binary(runs_by_id, tr) or not _has_both_binary(runs_by_id, va) or not _has_both_binary(runs_by_id, te):
        raise AssertionError(f"{name}: each split must include both benign and attack runs (binary evaluable).")

    fam_counts = {
        "train": _per_family_counts(runs_by_id, tr),
        "val": _per_family_counts(runs_by_id, va),
        "test": _per_family_counts(runs_by_id, te),
    }

    # Strict family coverage check
    if require_each_family:
        enforce = (strategy == "stratified_by_family") or (apply_to_time_ordered and strategy == "time_ordered")
        if enforce:
            families = set(runs["family"].tolist())
            for fam in families:
                for part_name, ids in [("train", tr), ("val", va), ("test", te)]:
                    sub = runs_by_id.loc[list(ids)]
                    if not (sub["family"] == fam).any():
                        raise AssertionError(f"{name}: family '{fam}' missing from {part_name}.")
        else:
            # Not an error; just a note.
            print(f"\nNOTE: {name} ({strategy}) skipping --require-each-family check (chronology-based).")

    print(f"\nOK: {name} ({payload.get('strategy')})")
    print(f"  sizes: train={len(tr)} val={len(va)} test={len(te)}")
    print(f"  by_family: {fam_counts}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-runs", type=str, required=True)
    ap.add_argument("--splits-dir", type=str, required=True)
    ap.add_argument("--require-each-family", action="store_true", help="Enforce each family appears in each split (stratified splits only by default)")
    ap.add_argument("--apply-to-time-ordered", action="store_true", help="Also enforce family coverage on time-ordered splits (often too strict)")
    args = ap.parse_args()

    runs = _load_runs(Path(args.paired_runs).expanduser())
    splits_dir = Path(args.splits_dir).expanduser()
    files = sorted(splits_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No split JSON files found in {splits_dir}")

    for p in files:
        payload = json.loads(p.read_text(encoding="utf-8"))
        _validate_one(payload, runs, require_each_family=args.require_each_family, apply_to_time_ordered=args.apply_to_time_ordered)

    print("\nOK: all split manifests validated.")


if __name__ == "__main__":
    main()
