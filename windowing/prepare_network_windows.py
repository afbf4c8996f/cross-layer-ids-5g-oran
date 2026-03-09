#!/usr/bin/env python3
"""
prepare_network_windows.py

Stage-1 data preparation for the network modality:
  1) Load paired run manifest (paired_runs.csv) and keep only those run_ids.
  2) Filter Network_Dataset_with_ts_iat.parquet to the paired runs (42 runs).
  3) Add per-run relative time (t_rel_s) = flow_start_ts - min(flow_start_ts) per run.
  4) Build fixed-stride window features (W,S) on the network timeline.
  5) Produce:
      - network_paired42.parquet
      - network_windows_W{W}_S{S}.parquet (features + labels)
      - window_index_W{W}_S{S}.parquet (labels + metadata, no features)
      - summaries (run durations, onset times, occupancy)

Design choices:
  - Primary unit is a "run" identified by run_id == canon_stem == pcap_id.
  - We do NOT use packet-level tables for mainline modeling here. we use the
    authors'(dataset curator) flow table with reconstructed timestamps.
  - Window labels for training (binary): majority-of-flows in window.
  - Attack onset for TTD (evaluation later): first attack flow time (run-level), not
    "first attack window" (to avoid labeling-rule bias).
  - All time is relative to run start. we do not rely on cross-machine epoch coherence.



Example:
  python3 prepare_network_windows.py --config config.yaml
or:
  python3 prepare_network_windows.py \
    --paired-runs paired_runs.csv \
    --network Network_Dataset_with_ts_iat.parquet \
    --out-dir stage1_out \
    --windows 10:2,5:2
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore


# ------------------------- Config helpers -------------------------

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


def _parse_windows_spec(spec: str) -> List[Tuple[int, int]]:
    """
    Parse windows spec like: "10:2,5:2" -> [(10,2),(5,2)]
    """
    out: List[Tuple[int, int]] = []
    spec = spec.strip()
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad window spec '{part}'. Expected W:S.")
        w_s = part.split(":")
        if len(w_s) != 2:
            raise ValueError(f"Bad window spec '{part}'. Expected W:S.")
        W = int(w_s[0])
        S = int(w_s[1])
        if W <= 0 or S <= 0:
            raise ValueError("Window W and stride S must be positive integers.")
        out.append((W, S))
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ------------------------- Core logic -------------------------

def _load_paired_runs(paired_runs_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(paired_runs_csv)
    required = {"family", "canon_stem", "pcap_path", "txt_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"paired_runs.csv missing columns: {sorted(missing)}")
    df = df.copy()
    df["run_id"] = df["canon_stem"].astype(str)
    return df[["run_id", "family", "pcap_path", "txt_path"]].drop_duplicates("run_id").reset_index(drop=True)


def _read_network_filtered(network_parquet: Path, run_ids: Sequence[str], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Read parquet and filter to pcap_id in run_ids. Uses pyarrow.dataset if available for efficiency.
    """
    run_ids = list(map(str, run_ids))
    if network_parquet.suffix.lower() not in {".parquet"}:
        # Fallback: CSV
        df = pd.read_csv(network_parquet)
        if "pcap_id" not in df.columns:
            raise ValueError("Network file must contain 'pcap_id'.")
        df = df[df["pcap_id"].astype(str).isin(run_ids)]
        if columns is not None:
            keep = [c for c in columns if c in df.columns]
            df = df[keep]
        return df.reset_index(drop=True)

    try:
        import pyarrow.dataset as ds  # type: ignore
        import pyarrow.compute as pc  # type: ignore
    except Exception:
        # Fallback: pandas
        df = pd.read_parquet(network_parquet, columns=columns)
        if "pcap_id" not in df.columns:
            raise ValueError("Network parquet must contain 'pcap_id'.")
        return df[df["pcap_id"].astype(str).isin(run_ids)].reset_index(drop=True)

    dataset = ds.dataset(str(network_parquet), format="parquet")
    if columns is None:
        columns = dataset.schema.names
    # Ensure pcap_id is included for filtering and grouping
    if "pcap_id" not in columns:
        columns = ["pcap_id"] + columns
    filt = pc.field("pcap_id").isin(run_ids)
    table = dataset.to_table(columns=columns, filter=filt)
    return table.to_pandas()


def _compute_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    if "pcap_id" not in df.columns or "flow_start_ts" not in df.columns:
        raise ValueError("Network data must include 'pcap_id' and 'flow_start_ts'.")
    out = df.copy()
    out["pcap_id"] = out["pcap_id"].astype(str)

    # Ensure numeric
    out["flow_start_ts"] = pd.to_numeric(out["flow_start_ts"], errors="coerce")
    if out["flow_start_ts"].isna().any():
        bad = out["flow_start_ts"].isna().sum()
        raise ValueError(f"flow_start_ts has {bad} NaNs after coercion. Fix input.")
    run_start = out.groupby("pcap_id", sort=False)["flow_start_ts"].transform("min")
    out["t_rel_s"] = out["flow_start_ts"] - run_start
    return out


def _traffic_is_attack(series: pd.Series) -> np.ndarray:
    """
    Robustly map traffic_type to boolean attack mask.

    Handles common dataset encodings:
      - String labels: 'Attack', 'Attack_', 'Benign', etc.  -> attack if startswith('attack')
      - Numeric / bool labels: 1/0 or True/False           -> attack if > 0 (or True)

    This is intentionally conservative: only clear "attack" signals map to True.
    """
    # Fast path for boolean
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=bool)

    # Numeric encodings (0/1, 1/0, etc.)
    if pd.api.types.is_numeric_dtype(series):
        s_num = pd.to_numeric(series, errors="coerce").fillna(0.0)
        return (s_num.to_numpy(dtype=float) > 0.0)

    # String-like encodings
    s = series.astype(str).str.strip().str.lower()

    # Accept 'attack', 'attack_', 'attack '...
    is_attack = s.str.startswith("attack")

    # Also accept common numeric-as-string
    is_attack = is_attack | s.isin(["1", "true", "yes"])

    return is_attack.to_numpy(dtype=bool)



def _safe_mode_int(codes: np.ndarray) -> int:
    """Return mode code (>=0) for integer codes array. Assumes at least one element and all >=0."""
    if codes.size == 1:
        return int(codes[0])
    bc = np.bincount(codes)
    return int(bc.argmax())


def _factorize_run(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Factorize a categorical series into integer codes and unique labels.
    Missing values become code -1.
    """
    codes, uniques = pd.factorize(series.astype("string"), sort=False)
    return codes.astype(np.int32, copy=False), uniques.to_numpy()


def _window_starts(run_end_s: float, W: int, S: int) -> np.ndarray:
    """
    Compute window start times in seconds (float) from 0 to (run_end_s - W) inclusive, step S.
    """
    if not np.isfinite(run_end_s) or run_end_s <= 0:
        return np.array([], dtype=np.int64)
    last_start = math.floor((run_end_s - W) / S) * S
    if last_start < 0:
        return np.array([], dtype=np.int64)
    return np.arange(0, last_start + 1, S, dtype=np.int64)


def _make_network_windows_for_run(
    run_df: pd.DataFrame,
    W: int,
    S: int,
    majority_threshold: float,
    keep_empty_windows: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build window features for one run. Returns (windows_df, run_summary).
    """
    run_id = str(run_df["pcap_id"].iloc[0])
    family = str(run_df["family"].iloc[0])

    # Sort by relative time
    run_df = run_df.sort_values("t_rel_s", kind="mergesort").reset_index(drop=True)

    t = run_df["t_rel_s"].to_numpy(dtype=np.float64)
    if t.size == 0:
        return pd.DataFrame(), {"run_id": run_id, "family": family, "n_flows": 0}

    # Define run end: include flow duration if available
    if "flow_duration" in run_df.columns:
        dur = pd.to_numeric(run_df["flow_duration"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        run_end = float(np.max(t + np.maximum(dur, 0.0)))
    else:
        run_end = float(np.max(t))

    starts = _window_starts(run_end, W, S)
    if starts.size == 0:
        return pd.DataFrame(), {"run_id": run_id, "family": family, "n_flows": int(t.size), "run_end_s": run_end, "n_windows": 0}

    # Prepare arrays for fast aggregations
    attack_mask = _traffic_is_attack(run_df["traffic_type"])

    # Numeric columns to aggregate (safe subset)
    num_cols = [
        "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts",
        "src_ip_bytes", "dst_ip_bytes", "missed_bytes",
        "http_trans_depth", "files_total_bytes"
    ]
    # Keep only columns that exist
    num_cols = [c for c in num_cols if c in run_df.columns]
    num_arrays: Dict[str, np.ndarray] = {}
    for c in num_cols:
        num_arrays[c] = pd.to_numeric(run_df[c], errors="coerce").to_numpy(dtype=np.float64)

    # Factorize categorical columns we may want to keep as window-level mode (strings kept for later encoding)
    cat_cols = ["proto", "service", "conn_state", "history"]
    cat_cols = [c for c in cat_cols if c in run_df.columns]
    cat_codes: Dict[str, np.ndarray] = {}
    cat_uniques: Dict[str, np.ndarray] = {}
    for c in cat_cols:
        codes, uniques = _factorize_run(run_df[c])
        cat_codes[c] = codes
        cat_uniques[c] = uniques

    # For unique counts (inter-flow behavior)
    # Factorize IPs for speed
    ip_cols = ["src_ip", "dst_ip"]
    ip_cols = [c for c in ip_cols if c in run_df.columns]
    ip_codes: Dict[str, np.ndarray] = {}
    for c in ip_cols:
        codes, _ = _factorize_run(run_df[c])
        ip_codes[c] = codes

    # Ports for unique counts
    port_cols = ["src_port", "dst_port"]
    port_cols = [c for c in port_cols if c in run_df.columns]
    port_arrays: Dict[str, np.ndarray] = {}
    for c in port_cols:
        port_arrays[c] = pd.to_numeric(run_df[c], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)

    # Attack category/type factorization (for window mode among attack flows)
    atk_cat_codes, atk_cat_uniques = _factorize_run(run_df["attack_category"]) if "attack_category" in run_df.columns else (None, None)
    atk_type_codes, atk_type_uniques = _factorize_run(run_df["attack_type"]) if "attack_type" in run_df.columns else (None, None)

    # Binary indicators
    bin_cols = ["is_GET_mthd", "http_status_error", "is_file_transfered"]
    bin_cols = [c for c in bin_cols if c in run_df.columns]
    bin_arrays: Dict[str, np.ndarray] = {}
    for c in bin_cols:
        bin_arrays[c] = pd.to_numeric(run_df[c], errors="coerce").fillna(0).to_numpy(dtype=np.float64)

    rows: List[Dict[str, Any]] = []

    # Run-level onset for TTD (first attack flow)
    if attack_mask.any():
        first_attack_t = float(t[np.argmax(attack_mask)]) if attack_mask.dtype == bool else float(t[attack_mask][0])
        # The above np.argmax trick assumes boolean; keep safe:
        first_attack_t = float(t[np.where(attack_mask)[0][0]])
    else:
        first_attack_t = float("nan")

    # Build windows
    for w0 in starts:
        w1 = w0 + W
        l = int(np.searchsorted(t, w0, side="left"))
        r = int(np.searchsorted(t, w1, side="left"))
        n = r - l

        if n == 0 and not keep_empty_windows:
            continue

        # Base record
        rec: Dict[str, Any] = {
            "run_id": run_id,
            "family": family,
            "window_start_s": int(w0),
            "window_end_s": int(w1),
            "n_flows": int(n),
            "empty_window": int(n == 0),
        }

        # Label counts (flow-based)
        if n > 0:
            atk_n = int(attack_mask[l:r].sum())
        else:
            atk_n = 0
        benign_n = int(n - atk_n)
        rec["attack_flow_count"] = atk_n
        rec["benign_flow_count"] = benign_n
        rec["attack_flow_frac"] = float(atk_n / n) if n > 0 else 0.0
        rec["window_has_attack_flow"] = int(atk_n > 0)

        # Training binary label (majority)
        is_attack_win = (rec["attack_flow_frac"] >= majority_threshold) if n > 0 else False
        rec["traffic_type_win"] = "Attack" if is_attack_win else "Benign"

        # Window mode labels (attack_category/type) based on attack flows only
        if atk_n > 0 and atk_cat_codes is not None:
            codes = atk_cat_codes[l:r]
            codes = codes[attack_mask[l:r] & (codes >= 0)]
            rec["attack_category_win"] = str(atk_cat_uniques[_safe_mode_int(codes)]) if codes.size > 0 else None
        else:
            rec["attack_category_win"] = None

        if atk_n > 0 and atk_type_codes is not None:
            codes = atk_type_codes[l:r]
            codes = codes[attack_mask[l:r] & (codes >= 0)]
            rec["attack_type_win"] = str(atk_type_uniques[_safe_mode_int(codes)]) if codes.size > 0 else None
        else:
            rec["attack_type_win"] = None

        # Numeric aggregates
        # Use nan-safe operations; for empty windows produce zeros.
        for c, arr in num_arrays.items():
            if n > 0:
                x = arr[l:r]
                rec[f"{c}_sum"] = float(np.nansum(x))
                rec[f"{c}_mean"] = float(np.nanmean(x)) if np.isfinite(np.nanmean(x)) else float("nan")
                rec[f"{c}_std"] = float(np.nanstd(x)) if np.isfinite(np.nanstd(x)) else float("nan")
            else:
                rec[f"{c}_sum"] = 0.0
                rec[f"{c}_mean"] = float("nan")
                rec[f"{c}_std"] = float("nan")

        # Derived totals/rates
        if "src_bytes" in num_arrays and "dst_bytes" in num_arrays:
            total_bytes = rec["src_bytes_sum"] + rec["dst_bytes_sum"]
            rec["total_bytes_sum"] = float(total_bytes)
            rec["bytes_per_s"] = float(total_bytes / W)
        if "src_pkts" in num_arrays and "dst_pkts" in num_arrays:
            total_pkts = rec["src_pkts_sum"] + rec["dst_pkts_sum"]
            rec["total_pkts_sum"] = float(total_pkts)
            rec["pkts_per_s"] = float(total_pkts / W)
        rec["flows_per_s"] = float(n / W)

        # Unique counts (IPs/ports)
        for c, codes in ip_codes.items():
            if n > 0:
                sl = codes[l:r]
                sl = sl[sl >= 0]
                rec[f"n_unique_{c}"] = int(np.unique(sl).size) if sl.size > 0 else 0
            else:
                rec[f"n_unique_{c}"] = 0

        for c, parr in port_arrays.items():
            if n > 0:
                sl = parr[l:r]
                sl = sl[sl >= 0]
                rec[f"n_unique_{c}"] = int(np.unique(sl).size) if sl.size > 0 else 0
            else:
                rec[f"n_unique_{c}"] = 0

        # Categorical window modes (stored as strings)
        for c in cat_cols:
            codes = cat_codes[c]
            if n > 0:
                sl = codes[l:r]
                sl = sl[sl >= 0]
                if sl.size > 0:
                    rec[f"{c}_mode"] = str(cat_uniques[c][_safe_mode_int(sl)])
                else:
                    rec[f"{c}_mode"] = None
                rec[f"{c}_n_unique"] = int(np.unique(sl).size) if sl.size > 0 else 0
            else:
                rec[f"{c}_mode"] = None
                rec[f"{c}_n_unique"] = 0

        # Binary indicator aggregates
        for c, arr in bin_arrays.items():
            if n > 0:
                rec[f"{c}_sum"] = float(np.nansum(arr[l:r]))
                rec[f"{c}_mean"] = float(np.nanmean(arr[l:r])) if np.isfinite(np.nanmean(arr[l:r])) else float("nan")
            else:
                rec[f"{c}_sum"] = 0.0
                rec[f"{c}_mean"] = float("nan")

        rows.append(rec)

    win_df = pd.DataFrame(rows)

    run_summary = {
        "run_id": run_id,
        "family": family,
        "n_flows_total": int(t.size),
        "run_end_s": float(run_end),
        "n_windows": int(win_df.shape[0]),
        "window_occupancy": float((win_df["n_flows"] > 0).mean()) if win_df.shape[0] > 0 else float("nan"),
        "t_first_attack_flow_s": first_attack_t,
    }

    return win_df, run_summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML config file (optional)")
    ap.add_argument("--paired-runs", type=str, default=None, help="paired_runs.csv path (overrides config)")
    ap.add_argument("--network", type=str, default=None, help="Network_Dataset_with_ts_iat.parquet (overrides config)")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory (overrides config)")
    ap.add_argument("--windows", type=str, default=None, help="Window specs '10:2,5:2' (overrides config)")
    ap.add_argument("--majority-threshold", type=float, default=None, help="Majority threshold for Attack label (default 0.5)")
    ap.add_argument("--keep-empty-windows", action="store_true", help="Keep windows with zero flows (default True)")
    ap.add_argument("--drop-empty-windows", action="store_true", help="Drop windows with zero flows")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = _read_yaml(Path(args.config).expanduser())

    paired_runs_csv = Path(args.paired_runs or _deep_get(cfg, ["paths", "paired_runs_csv"])).expanduser()
    network_path = Path(args.network or _deep_get(cfg, ["paths", "network_parquet"])).expanduser()
    out_dir = Path(args.out_dir or _deep_get(cfg, ["paths", "out_dir"], "./stage1_out")).expanduser()

    win_spec = args.windows or _deep_get(cfg, ["windows_spec"], None)
    if win_spec is None:
        # allow list in config: windows: [[10,2],[5,2]]
        win_list = _deep_get(cfg, ["windows"], [])
        windows: List[Tuple[int, int]] = []
        if isinstance(win_list, list) and win_list:
            for item in win_list:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    windows.append((int(item[0]), int(item[1])))
        if not windows:
            windows = [(10, 2), (5, 2)]
    else:
        windows = _parse_windows_spec(str(win_spec))

    majority_threshold = float(args.majority_threshold if args.majority_threshold is not None else _deep_get(cfg, ["labels", "majority_threshold"], 0.5))

    keep_empty = True
    if args.drop_empty_windows:
        keep_empty = False
    if args.keep_empty_windows:
        keep_empty = True

    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "windows")
    _ensure_dir(out_dir / "summaries")

    paired = _load_paired_runs(paired_runs_csv)
    run_ids = paired["run_id"].tolist()
    run_meta = paired.set_index("run_id")[["family", "pcap_path", "txt_path"]].to_dict(orient="index")

    # Load and filter network
    print(f"Loading network data from: {network_path}")
    net_df = _read_network_filtered(network_path, run_ids=run_ids, columns=None)
    if net_df.empty:
        raise RuntimeError("Filtered network dataframe is empty. Check pcap_id values and input paths.")

    # Add family column (from paired runs)
    net_df["pcap_id"] = net_df["pcap_id"].astype(str)
    net_df["family"] = net_df["pcap_id"].map(lambda rid: run_meta.get(rid, {}).get("family", None))
    if net_df["family"].isna().any():
        missing = net_df.loc[net_df["family"].isna(), "pcap_id"].unique().tolist()
        raise ValueError(f"Some pcap_id values not found in paired_runs manifest: {missing[:10]} ...")

    # Relative time
    net_df = _compute_relative_time(net_df)

    # Save paired42 parquet
    paired_out = out_dir / "network_paired42.parquet"
    if paired_out.exists() and not args.force:
        print(f"Exists, skipping write (use --force to overwrite): {paired_out}")
    else:
        net_df.to_parquet(paired_out, index=False)
        print(f"Wrote: {paired_out} (rows={len(net_df):,})")

    # Build windows per run
    run_summaries: List[Dict[str, Any]] = []
    for (W, S) in windows:
        print(f"\nBuilding network windows W={W}, S={S} ...")
        win_rows: List[pd.DataFrame] = []
        per_run_summaries: List[Dict[str, Any]] = []

        for run_id, g in net_df.groupby("pcap_id", sort=False):
            # Attach family for this run already present
            wdf, rs = _make_network_windows_for_run(
                run_df=g,
                W=W, S=S,
                majority_threshold=majority_threshold,
                keep_empty_windows=keep_empty,
            )
            if not wdf.empty:
                win_rows.append(wdf)
            per_run_summaries.append(rs)

        if win_rows:
            all_wins = pd.concat(win_rows, ignore_index=True)
        else:
            all_wins = pd.DataFrame()

        # Write window files
        net_win_out = out_dir / "windows" / f"network_windows_W{W}_S{S}.parquet"
        idx_out = out_dir / "windows" / f"window_index_W{W}_S{S}.parquet"
        summ_out = out_dir / "summaries" / f"network_run_summary_W{W}_S{S}.csv"

        if (net_win_out.exists() or idx_out.exists() or summ_out.exists()) and not args.force:
            print(f"Outputs exist for W={W},S={S}; skipping writes (use --force to overwrite).")
        else:
            all_wins.to_parquet(net_win_out, index=False)
            # Window index contains only metadata + labels (no feature sums)
            idx_cols = [
                "run_id","family","window_start_s","window_end_s",
                "n_flows","empty_window",
                "attack_flow_count","benign_flow_count","attack_flow_frac","window_has_attack_flow",
                "traffic_type_win","attack_category_win","attack_type_win"
            ]
            idx_df = all_wins[idx_cols].copy() if not all_wins.empty else pd.DataFrame(columns=idx_cols)
            idx_df.to_parquet(idx_out, index=False)
            pd.DataFrame(per_run_summaries).to_csv(summ_out, index=False)

            print(f"Wrote: {net_win_out} (rows={len(all_wins):,})")
            print(f"Wrote: {idx_out} (rows={len(idx_df):,})")
            print(f"Wrote: {summ_out}")

        run_summaries.extend(per_run_summaries)

    # Write a single run summary (dedup by run_id, keep max run_end maybe)
    run_sum_df = pd.DataFrame(run_summaries)
    if not run_sum_df.empty:
        # Keep the most informative row per run (largest n_windows from any W,S)
        run_sum_df = run_sum_df.sort_values(["run_id", "n_windows"], ascending=[True, False]).drop_duplicates("run_id")
        run_sum_path = out_dir / "summaries" / "network_run_summary_all.csv"
        if not run_sum_path.exists() or args.force:
            run_sum_df.to_csv(run_sum_path, index=False)
            print(f"\nWrote: {run_sum_path}")

    # Write config snapshot
    snap = {
        "paired_runs_csv": str(paired_runs_csv),
        "network_path": str(network_path),
        "out_dir": str(out_dir),
        "windows": [{"W": W, "S": S} for (W, S) in windows],
        "majority_threshold": majority_threshold,
        "keep_empty_windows": keep_empty,
    }
    snap_path = out_dir / "summaries" / "stage1_network_config.json"
    if not snap_path.exists() or args.force:
        snap_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        print(f"Wrote: {snap_path}")


if __name__ == "__main__":
    main()
