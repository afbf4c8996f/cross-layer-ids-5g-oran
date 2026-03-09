#!/usr/bin/env python3
"""
validate_stage1_outputs.py

Sanity/consistency checks for Stage-1 window preparation outputs: (always run!)

  - network_paired42.parquet
  - radio_seconds_paired42.parquet
  - windows/network_windows_W{W}_S{S}.parquet
  - windows/radio_windows_W{W}_S{S}.parquet
  - windows/window_index_W{W}_S{S}.parquet

Checks performed:
  1) Basic schema presence (required columns).
  2) Run counts (expect 42 paired runs).
  3) Window grids match exactly between network/radio/index for each (W,S).
  4) Labels copied into radio windows match window_index exactly.
  5) Radio constraints: rate_total >= 0, reset_indicator in {0,1}, sec_missing in {0,1},
     radio_missing_seconds within [0,W], missing_frac == missing/W.
  6) Ensure 'pmi' is not present in radio_seconds (we keep pmi_0 and pmi_1 instead).
  7) Produce a validation report JSON summarizing results.

This script is intentionally strict and will raise on mismatch.

Example:
  python3 validate_stage1_outputs.py --out-dir /path/to/window-output
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _parse_ws_from_filename(name: str) -> Optional[Tuple[int, int]]:
    # match ..._W10_S2...
    m = re.search(r"_W(\d+)_S(\d+)", name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _discover_windows(out_dir: Path) -> List[Tuple[int, int]]:
    win_dir = out_dir / "windows"
    ws: List[Tuple[int, int]] = []
    for p in win_dir.glob("window_index_W*_S*.parquet"):
        parsed = _parse_ws_from_filename(p.name)
        if parsed:
            ws.append(parsed)
    ws = sorted(set(ws))
    return ws


def _assert_cols(df: pd.DataFrame, required: List[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise AssertionError(f"{context}: missing columns: {missing}")


def _keys_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[["run_id", "window_start_s", "window_end_s"]].copy()


def _check_keys_equal(a: pd.DataFrame, b: pd.DataFrame, context: str) -> None:
    a2 = _keys_df(a).sort_values(["run_id", "window_start_s", "window_end_s"], kind="mergesort").reset_index(drop=True)
    b2 = _keys_df(b).sort_values(["run_id", "window_start_s", "window_end_s"], kind="mergesort").reset_index(drop=True)
    if len(a2) != len(b2):
        raise AssertionError(f"{context}: key rowcount mismatch {len(a2)} vs {len(b2)}")
    if not a2.equals(b2):
        # Find first mismatch for debugging
        neq = ~(a2 == b2).all(axis=1)
        idx = int(np.where(neq.to_numpy())[0][0]) if neq.any() else -1
        raise AssertionError(f"{context}: window keys mismatch at row {idx}:\nA={a2.iloc[idx].to_dict()}\nB={b2.iloc[idx].to_dict()}")


def _check_label_equal(index_df: pd.DataFrame, other_df: pd.DataFrame, context: str) -> None:
    cols = ["traffic_type_win", "attack_category_win", "attack_type_win"]
    for c in cols:
        if c not in index_df.columns or c not in other_df.columns:
            raise AssertionError(f"{context}: missing label column '{c}' in one of the dataframes")
        a = index_df[c].astype("string").fillna(pd.NA)
        b = other_df[c].astype("string").fillna(pd.NA)
        if not a.equals(b):
            # Find first mismatch
            neq = (a != b) & ~(a.isna() & b.isna())
            if neq.any():
                i = int(np.where(neq.to_numpy())[0][0])
                raise AssertionError(f"{context}: label mismatch in '{c}' at row {i}: index='{a.iloc[i]}', other='{b.iloc[i]}'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True, help="Stage-1 output directory (window-output)")
    ap.add_argument("--report", type=str, default=None, help="Optional path to write report JSON (default out-dir/validation_report.json)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    win_dir = out_dir / "windows"
    if not win_dir.exists():
        raise FileNotFoundError(f"Missing windows dir: {win_dir}")

    report_path = Path(args.report).expanduser() if args.report else (out_dir / "validation_report.json")

    # Load core tables
    net_path = out_dir / "network_paired42.parquet"
    rad_sec_path = out_dir / "radio_seconds_paired42.parquet"

    if not net_path.exists():
        raise FileNotFoundError(f"Missing: {net_path}")
    if not rad_sec_path.exists():
        raise FileNotFoundError(f"Missing: {rad_sec_path}")

    net = pd.read_parquet(net_path)
    rad = pd.read_parquet(rad_sec_path)

    _assert_cols(net, ["pcap_id", "flow_start_ts", "t_rel_s", "traffic_type"], "network_paired42")
    _assert_cols(rad, ["run_id", "t_rel_s", "rate_total", "reset_indicator", "sec_missing", "row_count", "time_mode"], "radio_seconds_paired42")

    # Run counts
    n_runs_net = net["pcap_id"].astype(str).nunique()
    n_runs_rad = rad["run_id"].astype(str).nunique()

    if n_runs_net != 42:
        raise AssertionError(f"Expected 42 runs in network_paired42, found {n_runs_net}")
    if n_runs_rad != 42:
        raise AssertionError(f"Expected 42 runs in radio_seconds_paired42, found {n_runs_rad}")

    # Radio 'pmi' check
    if "pmi" in rad.columns:
        raise AssertionError("radio_seconds_paired42 contains 'pmi' column. It should be dropped (keep pmi_0 and pmi_1).")

    # Basic radio constraints
    if (rad["rate_total"] < -1e-9).any():
        raise AssertionError("radio_seconds_paired42 has negative rate_total values. Should be clamped to >= 0.")
    if not set(pd.unique(rad["reset_indicator"].astype(int))).issubset({0, 1}):
        raise AssertionError("radio_seconds_paired42 reset_indicator must be in {0,1}.")
    if not set(pd.unique(rad["sec_missing"].astype(int))).issubset({0, 1}):
        raise AssertionError("radio_seconds_paired42 sec_missing must be in {0,1}.")
    if (rad["t_rel_s"] < 0).any():
        raise AssertionError("radio_seconds_paired42 has negative t_rel_s. Should be >=0.")

    # Discover windows
    ws = _discover_windows(out_dir)
    if not ws:
        raise FileNotFoundError(f"No window_index files found in {win_dir}")

    results: Dict[str, object] = {
        "out_dir": str(out_dir),
        "n_runs_network": int(n_runs_net),
        "n_runs_radio": int(n_runs_rad),
        "windows": [],
    }

    for (W, S) in ws:
        idx_path = win_dir / f"window_index_W{W}_S{S}.parquet"
        netw_path = win_dir / f"network_windows_W{W}_S{S}.parquet"
        radw_path = win_dir / f"radio_windows_W{W}_S{S}.parquet"

        if not idx_path.exists():
            raise FileNotFoundError(f"Missing: {idx_path}")
        if not netw_path.exists():
            raise FileNotFoundError(f"Missing: {netw_path}")
        if not radw_path.exists():
            raise FileNotFoundError(f"Missing: {radw_path}")

        idx = pd.read_parquet(idx_path)
        netw = pd.read_parquet(netw_path)
        radw = pd.read_parquet(radw_path)

        # Required columns for keys + labels
        _assert_cols(idx, ["run_id", "family", "window_start_s", "window_end_s", "traffic_type_win", "attack_category_win", "attack_type_win"], f"window_index_W{W}_S{S}")
        _assert_cols(netw, ["run_id", "family", "window_start_s", "window_end_s", "traffic_type_win", "attack_category_win", "attack_type_win"], f"network_windows_W{W}_S{S}")
        _assert_cols(radw, ["run_id", "family", "window_start_s", "window_end_s", "traffic_type_win", "attack_category_win", "attack_type_win"], f"radio_windows_W{W}_S{S}")

        # Window keys must match across all three
        _check_keys_equal(idx, netw, f"W{W}S{S}: index vs network")
        _check_keys_equal(idx, radw, f"W{W}S{S}: index vs radio")

        # Labels must match index exactly in radio
        idx_sorted = idx.sort_values(["run_id", "window_start_s", "window_end_s"], kind="mergesort").reset_index(drop=True)
        radw_sorted = radw.sort_values(["run_id", "window_start_s", "window_end_s"], kind="mergesort").reset_index(drop=True)
        _check_label_equal(idx_sorted, radw_sorted, f"W{W}S{S}: index vs radio labels")

        # Radio window constraints
        for col in ["radio_present_seconds", "radio_missing_seconds", "radio_missing_frac", "radio_window_missing"]:
            if col not in radw.columns:
                raise AssertionError(f"W{W}S{S}: radio_windows missing '{col}'")
        if (radw["radio_missing_seconds"] < 0).any() or (radw["radio_missing_seconds"] > W).any():
            raise AssertionError(f"W{W}S{S}: radio_missing_seconds out of [0,W]")
        if (radw["radio_present_seconds"] < 0).any() or (radw["radio_present_seconds"] > W).any():
            raise AssertionError(f"W{W}S{S}: radio_present_seconds out of [0,W]")
        # Check frac
        frac = radw["radio_missing_seconds"].astype(float) / float(W)
        if not np.allclose(frac.to_numpy(), radw["radio_missing_frac"].to_numpy(dtype=float), equal_nan=True, atol=1e-9):
            raise AssertionError(f"W{W}S{S}: radio_missing_frac mismatch (should be missing/W)")

        # radio_window_missing should imply present==0 (or missing==W)
        bad = (radw["radio_window_missing"].astype(int) == 1) & (radw["radio_present_seconds"].astype(int) != 0)
        if bad.any():
            i = int(np.where(bad.to_numpy())[0][0])
            raise AssertionError(f"W{W}S{S}: radio_window_missing=1 but present_seconds!=0 at row {i}")

        # Summary stats
        missing_rate = float(radw["radio_window_missing"].mean())
        median_missing_frac = float(radw["radio_missing_frac"].median())
        results["windows"].append({
            "W": int(W), "S": int(S),
            "n_windows": int(idx.shape[0]),
            "radio_window_missing_rate": missing_rate,
            "radio_missing_frac_median": median_missing_frac,
        })

    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"OK: validation passed for {len(ws)} window configs.")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    import re
    main()
