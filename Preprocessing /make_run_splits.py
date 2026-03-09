#!/usr/bin/env python3
"""
make_run_splits.py

Stage-2A: Create run-level split manifests (train/val/test) for the paired 42 runs.

Inputs:
  - paired_runs.csv (must include canon_stem, family)
  - stage1 output dir (for time-ordered split) containing network_paired42.parquet (never used in the paper)

Outputs:
  - out_dir/splits/*.json  (one file per split, with full metadata + counts)
  - out_dir/splits/splits_index.csv (list of created split files)

Split strategies:
  1) stratified_by_family:
     - For each family, allocate train/val/test counts with minimum 1 in each split
       (valid here because each family has >=4 runs).
     - Random selection within each family controlled by seed.
  2) time_ordered:
     - Order runs by network run start time (min flow_start_ts in network_paired42).
     - Choose boundaries that best match target fractions while enforcing that each split
       contains at least one benign and at least one attack run (binary evaluable).

Written to support K seeds (for mean/std reporting).



Example:
  python3 make_run_splits.py \
    --paired-runs /path/to/paired_runs.csv \
    --stage1-out /path/to/window-output \
    --out-dir /path/to/window-output \
    --seeds 42,43,44 \
    --val-frac 0.15 --test-frac 0.15 \
    --strategies stratified,time_ordered
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required for --config. Install with: python3 -m pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping/dict at top level.")
    return data


def _deep_get(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _parse_seeds(arg: Optional[str], cfg: Dict[str, Any]) -> List[int]:
    if arg:
        return [int(x.strip()) for x in arg.split(",") if x.strip()]
    seeds = _deep_get(cfg, ["splits", "seeds"], None)
    if isinstance(seeds, list) and seeds:
        return [int(x) for x in seeds]
    # fallback: generate K seeds
    k = int(_deep_get(cfg, ["splits", "k_seeds"], 1))
    start = int(_deep_get(cfg, ["splits", "seed_start"], 42))
    return [start + i for i in range(k)]


def _load_runs(paired_runs_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(paired_runs_csv).copy()
    required = {"canon_stem", "family"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"paired_runs.csv missing columns: {sorted(missing)}")
    df["run_id"] = df["canon_stem"].astype(str)
    df["family"] = df["family"].astype(str)
    df = df.drop_duplicates("run_id").reset_index(drop=True)
    # binary class availability checks
    df["is_benign_run"] = df["family"].str.lower().eq("benign")
    df["is_attack_run"] = ~df["is_benign_run"]
    return df[["run_id", "family", "is_benign_run", "is_attack_run"]]


def _family_allocation(n: int, val_frac: float, test_frac: float) -> Tuple[int, int, int]:
    """
    Allocate (train_n, val_n, test_n) for a family of size n,
    enforcing at least 1 in each split (assumes n >= 4).
    """
    if n < 4:
        # For completeness; we can still do something deterministic.
        # Prefer: train >= 1, val >= 1 if possible, test >= 1 if possible.
        train = max(1, n - 2)
        val = 1 if n >= 2 else 0
        test = 1 if n >= 3 else 0
        return train, val, test

    test_n = max(1, int(round(n * test_frac)))
    val_n = max(1, int(round(n * val_frac)))

    # Ensure room for train
    if test_n + val_n > n - 1:
        # Reduce the larger of (val,test) until train >= 1
        while test_n + val_n > n - 1:
            if test_n >= val_n and test_n > 1:
                test_n -= 1
            elif val_n > 1:
                val_n -= 1
            else:
                break

    train_n = n - val_n - test_n
    if train_n < 1:
        # Force train = 1, adjust val/test (still keep at least 1 each if possible)
        train_n = 1
        remaining = n - 1
        # split remaining into val/test with at least 1 each
        val_n = max(1, min(val_n, remaining - 1))
        test_n = remaining - val_n
        if test_n < 1:
            test_n = 1
            val_n = remaining - test_n

    return train_n, val_n, test_n


def _split_stratified_by_family(
    runs: pd.DataFrame,
    seed: int,
    val_frac: float,
    test_frac: float,
    run_row_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Row-budget-aware stratified split.

    If run_row_counts is provided (run_id -> n_windows), uses greedy
    bin-packing to match target row fractions per family while keeping
    runs atomic (no leakage). Otherwise falls back to run-count allocation.
    """
    rng = np.random.default_rng(seed)
    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []
    by_family_counts: Dict[str, Dict[str, Any]] = {}

    for fam, g in runs.groupby("family", sort=True):
        ids = g["run_id"].tolist()
        n = len(ids)

        if run_row_counts is None or n < 4:
            # Fallback: old run-count allocation (unchanged from original)
            train_n, val_n, test_n = _family_allocation(n, val_frac=val_frac, test_frac=test_frac)
            perm = rng.permutation(ids).tolist()
            train_ids.extend(perm[:train_n])
            val_ids.extend(perm[train_n:train_n + val_n])
            test_ids.extend(perm[train_n + val_n:train_n + val_n + test_n])
            by_family_counts[fam] = {"n": n, "train": train_n, "val": val_n, "test": test_n}
            continue

        # --- Row-budget greedy packing ---
        # Shuffle for seed-dependent randomness
        perm = rng.permutation(ids).tolist()

        total_rows = sum(run_row_counts.get(rid, 0) for rid in ids)
        val_budget = total_rows * val_frac
        test_budget = total_rows * test_frac

        # Guarantee: at least 1 run in each split (pick first 3 from shuffled order)
        fam_train = [perm[0]]
        fam_val = [perm[1]] if n > 1 else []
        fam_test = [perm[2]] if n > 2 else []
        val_rows = run_row_counts.get(perm[1], 0) if n > 1 else 0
        test_rows = run_row_counts.get(perm[2], 0) if n > 2 else 0

        # Assign remaining runs greedily to the split most under-budget
        for rid in perm[3:]:
            rc = run_row_counts.get(rid, 0)
            val_deficit = val_budget - val_rows
            test_deficit = test_budget - test_rows
            if val_deficit <= 0 and test_deficit <= 0:
                fam_train.append(rid)
            elif val_deficit >= test_deficit:
                fam_val.append(rid)
                val_rows += rc
            else:
                fam_test.append(rid)
                test_rows += rc

        train_ids.extend(fam_train)
        val_ids.extend(fam_val)
        test_ids.extend(fam_test)
        by_family_counts[fam] = {
            "n": n,
            "train": len(fam_train), "val": len(fam_val), "test": len(fam_test),
            "train_rows": sum(run_row_counts.get(r, 0) for r in fam_train),
            "val_rows": sum(run_row_counts.get(r, 0) for r in fam_val),
            "test_rows": sum(run_row_counts.get(r, 0) for r in fam_test),
        }

    # Validate coverage/disjointness
    all_ids = set(runs["run_id"])
    tr, va, te = map(set, (train_ids, val_ids, test_ids))
    if tr & va or tr & te or va & te:
        raise AssertionError("Stratified split produced overlapping run_ids.")
    if (tr | va | te) != all_ids:
        missing = sorted(all_ids - (tr | va | te))
        extra = sorted((tr | va | te) - all_ids)
        raise AssertionError(f"Coverage mismatch. missing={missing[:5]} extra={extra[:5]}")

    def _has_both(split_ids):
        sub = runs.set_index("run_id").loc[list(split_ids)]
        return bool(sub["is_benign_run"].any() and sub["is_attack_run"].any())

    if not _has_both(train_ids) or not _has_both(val_ids) or not _has_both(test_ids):
        raise AssertionError("Each split must include both benign and attack runs.")

    return {
        "strategy": "stratified_by_family_rowaware",
        "seed": int(seed),
        "val_frac": float(val_frac),
        "test_frac": float(test_frac),
        "train_run_ids": sorted(train_ids),
        "val_run_ids": sorted(val_ids),
        "test_run_ids": sorted(test_ids),
        "by_family_counts": by_family_counts,
    }
###################################################################

def _run_start_times_from_network(stage1_out: Path) -> pd.DataFrame:
    net_path = stage1_out / "network_paired42.parquet"
    if not net_path.exists():
        raise FileNotFoundError(f"Missing {net_path}. Run Stage-1 first.")
    net = pd.read_parquet(net_path, columns=["pcap_id", "flow_start_ts"])
    net["run_id"] = net["pcap_id"].astype(str)
    net["flow_start_ts"] = pd.to_numeric(net["flow_start_ts"], errors="coerce")
    if net["flow_start_ts"].isna().any():
        raise ValueError("flow_start_ts contains NaNs in network_paired42.parquet.")
    start = net.groupby("run_id", sort=False)["flow_start_ts"].min().reset_index()
    start = start.rename(columns={"flow_start_ts": "run_start_ts"})
    return start


def _split_time_ordered(runs: pd.DataFrame, stage1_out: Path, seed: int, val_frac: float, test_frac: float) -> Dict[str, Any]:
    # Merge run start times
    start = _run_start_times_from_network(stage1_out)
    df = runs.merge(start, on="run_id", how="left")
    if df["run_start_ts"].isna().any():
        missing = df.loc[df["run_start_ts"].isna(), "run_id"].tolist()
        raise ValueError(f"Missing run_start_ts for runs: {missing[:10]}")

    # Sort by run_start_ts, then tie-break by hashed run_id with seed for determinism
    # (ties are unlikely but we keep deterministic behavior).
    tie = df["run_id"].apply(lambda s: (hash((int(seed), str(s))) & 0xFFFFFFFF)).astype(np.int64)
    df = df.assign(_tie=tie).sort_values(["run_start_ts", "_tie"], ascending=[True, True], kind="mergesort").reset_index(drop=True)

    N = len(df)
    target_test = max(1, int(round(N * test_frac)))
    target_val = max(1, int(round(N * val_frac)))
    target_train = max(1, N - target_val - target_test)
    # Ensure at least 1 in each
    if target_train < 1:
        target_train = 1

    # Constraints: each split must have at least 1 benign and 1 attack
    def ok(i: int, j: int) -> bool:
        train = df.iloc[:i]
        val = df.iloc[i:j]
        test = df.iloc[j:]
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            return False
        for part in (train, val, test):
            if not (part["is_benign_run"].any() and part["is_attack_run"].any()):
                return False
        return True

    # Objective: match targets
    def score(i: int, j: int) -> float:
        return float((abs(i - target_train) + abs((j - i) - target_val) + abs((N - j) - target_test)))

    best: Optional[Tuple[float, int, int]] = None
    # Brute force search all boundaries; N is small (42).
    for i in range(1, N - 1):
        for j in range(i + 1, N):
            if not ok(i, j):
                continue
            sc = score(i, j)
            if best is None or sc < best[0]:
                best = (sc, i, j)

    if best is None:
        raise AssertionError("Could not find a time-ordered split satisfying benign+attack presence in all splits.")

    _, i, j = best
    train_ids = df.iloc[:i]["run_id"].tolist()
    val_ids = df.iloc[i:j]["run_id"].tolist()
    test_ids = df.iloc[j:]["run_id"].tolist()

    # By-family counts (for reporting only)
    def count_ids(ids: List[str]) -> Dict[str, int]:
        sub = runs.set_index("run_id").loc[ids]
        return sub["family"].value_counts().to_dict()

    by_family = {
        "train": count_ids(train_ids),
        "val": count_ids(val_ids),
        "test": count_ids(test_ids),
    }

    return {
        "strategy": "time_ordered",
        "seed": int(seed),
        "val_frac": float(val_frac),
        "test_frac": float(test_frac),
        "targets": {"train": int(target_train), "val": int(target_val), "test": int(target_test)},
        "train_run_ids": train_ids,
        "val_run_ids": val_ids,
        "test_run_ids": test_ids,
        "by_family_counts": by_family,
    }


def _write_split(out_dir: Path, name: str, payload: Dict[str, Any]) -> Path:
    payload = dict(payload)
    payload["split_name"] = name
    payload["created_utc"] = _utc_now_iso()
    p = out_dir / "splits" / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML config (optional)")
    ap.add_argument("--paired-runs", type=str, default=None, help="paired_runs.csv (overrides config)")
    ap.add_argument("--stage1-out", type=str, default=None, help="Stage-1 output dir (overrides config)")
    ap.add_argument("--out-dir", type=str, default=None, help="Output dir (defaults to stage1-out)")
    ap.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 42,43,44")
    ap.add_argument("--val-frac", type=float, default=None)
    ap.add_argument("--test-frac", type=float, default=None)
    ap.add_argument("--strategies", type=str, default=None, help="Comma-separated: stratified,time_ordered")
    ap.add_argument("--force", action="store_true", help="Overwrite split files if exist")
    args = ap.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = _read_yaml(Path(args.config).expanduser())

    paired_runs_csv = Path(args.paired_runs or _deep_get(cfg, ["paths", "paired_runs_csv"])).expanduser()
    stage1_out = Path(args.stage1_out or _deep_get(cfg, ["paths", "stage1_out_dir"], _deep_get(cfg, ["paths", "out_dir"], "./window-output"))).expanduser()
    out_dir = Path(args.out_dir or _deep_get(cfg, ["paths", "out_dir"], str(stage1_out))).expanduser()

    val_frac = float(args.val_frac if args.val_frac is not None else _deep_get(cfg, ["splits", "val_frac"], 0.15))
    test_frac = float(args.test_frac if args.test_frac is not None else _deep_get(cfg, ["splits", "test_frac"], 0.15))
    if val_frac <= 0 or test_frac <= 0 or (val_frac + test_frac) >= 0.8:
        raise ValueError("Invalid val/test fractions. Ensure 0<val,test and val+test<0.8.")

    strategies_arg = args.strategies or _deep_get(cfg, ["splits", "strategies"], "stratified,time_ordered")
    strategies = [s.strip().lower() for s in str(strategies_arg).split(",") if s.strip()]
    for s in strategies:
        if s not in {"stratified", "time_ordered"}:
            raise ValueError(f"Unknown strategy: {s}")

    seeds = _parse_seeds(args.seeds, cfg)
    if not seeds:
        raise ValueError("No seeds provided.")

    _ensure_dir(out_dir / "splits")

    runs = _load_runs(paired_runs_csv)
    # --- Load row counts from Stage-1 run summary for row-aware splitting ---
    run_summary_path = stage1_out / "network_run_summary_all.csv"
    run_row_counts: Optional[Dict[str, int]] = None
    if run_summary_path.exists():
        _rs = pd.read_csv(run_summary_path)
        run_row_counts = dict(zip(_rs["run_id"].astype(str), _rs["n_windows"].astype(int)))
        print(f"[row-aware] Loaded row counts for {len(run_row_counts)} runs from {run_summary_path}")
    else:
        print(f"[row-aware] WARNING: {run_summary_path} not found — falling back to run-count allocation.")
    
    all_run_ids = sorted(runs["run_id"].tolist())

    created: List[Dict[str, Any]] = []

    for seed in seeds:
        if "stratified" in strategies:
            payload = _split_stratified_by_family(runs, seed=seed, val_frac=val_frac, test_frac=test_frac, run_row_counts=run_row_counts)
            name = f"stratified_seed{seed}"
            path = out_dir / "splits" / f"{name}.json"
            if path.exists() and not args.force:
                pass
            else:
                p = _write_split(out_dir, name, payload)
                created.append({"split_name": name, "path": str(p), "strategy": payload["strategy"], "seed": seed})

        if "time_ordered" in strategies:
            payload = _split_time_ordered(runs, stage1_out=stage1_out, seed=seed, val_frac=val_frac, test_frac=test_frac)
            name = f"time_ordered_seed{seed}"
            path = out_dir / "splits" / f"{name}.json"
            if path.exists() and not args.force:
                pass
            else:
                p = _write_split(out_dir, name, payload)
                created.append({"split_name": name, "path": str(p), "strategy": payload["strategy"], "seed": seed})

    # Write index
    idx_path = out_dir / "splits" / "splits_index.csv"
    idx_df = pd.DataFrame(created)
    if not idx_df.empty:
        idx_df.to_csv(idx_path, index=False)
        print(f"Wrote: {idx_path} (n={len(idx_df)})")
    else:
        print("No split files written (maybe they already exist and --force not set).")

    # Quick summary
    print(f"Runs: {len(all_run_ids)} total")
    print(f"Seeds: {seeds}")
    print(f"Strategies: {strategies}")
    print(f"Splits dir: {out_dir / 'splits'}")


if __name__ == "__main__":
    main()
