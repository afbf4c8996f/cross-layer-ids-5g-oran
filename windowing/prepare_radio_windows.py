#!/usr/bin/env python3
"""
prepare_radio_windows_v2.py

Stage-1 data preparation for the radio modality (paired-run .txt telemetry):

  1) Read paired_runs.csv (run_id, family, txt_path).
  2) Parse per-run telemetry JSON-lines.
     - If a timestamp field exists: treat as timestamped mode.
     - If no timestamp exists: treat as implicit 1 Hz mode (row index => seconds).
  3) Aggregate to per-second features (t_rel_s as integer seconds since run start):
     - Aggregate across multiple rows per second (multi-UE):
         * Counters (dlBytes, ulBytes): sum
         * KPIs: median (robust)
         * row_count: number of rows at that second
         * ue_count: nunique ue_id (if present) [identity not retained]
     - Convert counters to per-second rates:
         rate_total = Δ(dlBytes_sum + ulBytes_sum) with negative deltas treated as counter resets.
         reset_indicator = 1 if delta < 0 before clamping; rate_total is clamped to >=0.
     - Create a dense per-second index (0..max_sec) to support window missingness stats.
  4) Build radio window features for the SAME windows used on the network timeline by
     consuming window_index_W{W}_S{S}.parquet produced by prepare_network_windows.py.
     (This ensures identical window grid and label assignment.)
  5) Write:
      - radio_seconds_paired42.parquet
      - radio_windows_W{W}_S{S}.parquet
      - summaries

Decision here reflect:
  - Drop 'pmi' (keep pmi_0, pmi_1).
  - Do not use UE identifiers as model inputs (only aggregate ue_count / row_count).
  - Handle missing seconds; radio windows include missingness indicators.
  - Time is relative to run start; no reliance on CU/DU epoch coherence. (this is important)

Example:
  python3 prepare_radio_windows.py --config config.yaml
or:
  python3 prepare_radio_windows.py \
    --paired-runs paired_runs.csv \
    --out-dir stage1_out \
    --windows 10:2,5:2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import orjson  # type: ignore
except Exception:
    orjson = None  # type: ignore

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
        W = int(w_s[0]); S = int(w_s[1])
        if W <= 0 or S <= 0:
            raise ValueError("Window W and stride S must be positive integers.")
        out.append((W, S))
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ------------------------- Parsing -------------------------

def _loads_json_line(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        if orjson is not None:
            return orjson.loads(line)
        return json.loads(line)
    except Exception:
        return None


def _detect_timestamp_key(rec: Dict[str, Any], ts_keys: Sequence[str]) -> Optional[str]:
    for k in ts_keys:
        if k in rec:
            return k
    return None


def _coerce_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float):
            if not np.isfinite(x):
                return None
            return int(x)
        s = str(x).strip()
        if not s:
            return None
        # allow float strings
        return int(float(s))
    except Exception:
        return None


def _to_rel_seconds_from_timestamp(ts_vals: np.ndarray) -> np.ndarray:
    """
    Convert absolute timestamps to integer seconds since start.
    Supports ms epoch or seconds epoch.
    """
    ts0 = int(np.nanmin(ts_vals))
    if ts0 > 10**11:  # ms epoch
        rel = (ts_vals.astype(np.int64) - ts0) // 1000
    else:  # seconds
        rel = ts_vals.astype(np.int64) - ts0
    rel = np.maximum(rel, 0)
    return rel.astype(np.int64)


# ------------------------- Per-run processing -------------------------

def _parse_run_txt(txt_path: Path, ts_keys: Sequence[str]) -> Tuple[pd.DataFrame, str]:
    """
    Parse one telemetry txt to a raw dataframe of dicts.
    Returns (df, time_mode) where time_mode is 'timestamp' or 'implicit'.
    """
    records: List[Dict[str, Any]] = []
    first_ts_key: Optional[str] = None

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = _loads_json_line(line)
            if not isinstance(rec, dict):
                continue
            if first_ts_key is None:
                first_ts_key = _detect_timestamp_key(rec, ts_keys)
            records.append(rec)

    if not records:
        raise ValueError(f"No JSON objects parsed from {txt_path}")

    df = pd.DataFrame(records)

    # Drop pmi if present (keep pmi_0, pmi_1)
    if "pmi" in df.columns:
        df = df.drop(columns=["pmi"])

    if first_ts_key is not None and first_ts_key in df.columns:
        df["_timestamp_raw"] = df[first_ts_key].apply(_coerce_int)
        if df["_timestamp_raw"].isna().all():
            # Treat as implicit if timestamps are non-coercible
            time_mode = "implicit"
        else:
            time_mode = "timestamp"
    else:
        time_mode = "implicit"

    return df, time_mode


def _aggregate_to_seconds(df: pd.DataFrame, run_id: str, family: str, time_mode: str) -> pd.DataFrame:
    """
    Convert raw df to per-second aggregates with a dense second index.
    """
    # Required counters (if missing, raise)
    for k in ["dlBytes", "ulBytes"]:
        if k not in df.columns:
            raise ValueError(f"Telemetry missing required counter '{k}' for run {run_id}")

    df = df.copy()
    df["dlBytes"] = pd.to_numeric(df["dlBytes"], errors="coerce")
    df["ulBytes"] = pd.to_numeric(df["ulBytes"], errors="coerce")

    if time_mode == "timestamp":
        ts = pd.to_numeric(df["_timestamp_raw"], errors="coerce").to_numpy(dtype=np.float64)
        # drop rows with missing timestamp
        keep = np.isfinite(ts)
        df = df.loc[keep].reset_index(drop=True)
        ts = ts[keep]
        if df.empty:
            raise ValueError(f"No valid timestamp rows after coercion for run {run_id}")
        t_sec = _to_rel_seconds_from_timestamp(ts.astype(np.int64))
        df["t_rel_s"] = t_sec
    else:
        # Implicit 1Hz: each row is one second sample (already per-second).
        df["t_rel_s"] = np.arange(len(df), dtype=np.int64)

    # Identify KPI columns (exclude ids and non-numeric objects)
    id_like = {"_timestamp_raw", "t_rel_s", "ue_id", "rnti", "cellid", "in_sync"}
    # Keep numeric columns only for KPI aggregation
    # We'll aggregate: counters by sum; KPIs by median.
    # Count ue_id unique if present.
    cols = list(df.columns)
    kpi_cols: List[str] = []
    for c in cols:
        if c in id_like:
            continue
        if c in {"dlBytes", "ulBytes"}:
            continue
        # keep numeric-like columns; attempt coercion
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            # try numeric coercion; if mostly NaN then ignore
            tmp = pd.to_numeric(df[c], errors="coerce")
            if tmp.notna().mean() >= 0.2:  # heuristic
                df[c] = tmp
                kpi_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            kpi_cols.append(c)

    group = df.groupby("t_rel_s", sort=True)

    agg_dict: Dict[str, Any] = {
        "dlBytes": "sum",
        "ulBytes": "sum",
    }
    for c in kpi_cols:
        agg_dict[c] = "median"

    sec = group.agg(agg_dict).reset_index()

    # Row count per second
    sec["row_count"] = group.size().to_numpy(dtype=np.int64)

    # UE count per second (if ue_id exists)
    if "ue_id" in df.columns:
        sec["ue_count"] = group["ue_id"].nunique(dropna=True).to_numpy(dtype=np.int64)
    else:
        sec["ue_count"] = np.nan

    # Dense seconds index
    max_sec = int(sec["t_rel_s"].max()) if not sec.empty else -1
    dense = pd.DataFrame({"t_rel_s": np.arange(max_sec + 1, dtype=np.int64)})
    dense = dense.merge(sec, on="t_rel_s", how="left")

    dense["run_id"] = run_id
    dense["family"] = family
    dense["time_mode"] = time_mode

    # Missing second indicator: if row_count is NaN -> no record
    dense["sec_has_record"] = dense["row_count"].notna()
    dense["sec_missing"] = (~dense["sec_has_record"]).astype(np.int8)

    # Fill counters for rate computation (ffill/bfill then 0)
    # Use ffill/bfill explicitly (no deprecated fillna(method=...))
    dl_filled = dense["dlBytes"].ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
    ul_filled = dense["ulBytes"].ffill().bfill().fillna(0.0).to_numpy(dtype=np.float64)
    tot_filled = dl_filled + ul_filled

    # Delta
    delta = np.diff(tot_filled, prepend=tot_filled[0])
    reset = (delta < 0).astype(np.int8)
    rate = np.maximum(delta, 0.0)

    dense["totalBytes_counter_filled"] = tot_filled
    dense["rate_total"] = rate
    dense["reset_indicator"] = reset
    dense["reset_delta_raw"] = delta  # keep for debugging; can drop later

    # For missing seconds, keep KPIs as NaN; for model later we can impute on train only.
    # Ensure row_count/ue_count missing seconds become 0 (more interpretable)
    dense["row_count"] = dense["row_count"].fillna(0).astype(np.int64)
    if "ue_count" in dense.columns:
        dense["ue_count"] = dense["ue_count"].fillna(0).astype(np.int64)

    return dense


def _window_features_from_seconds(sec_df: pd.DataFrame, win_df: pd.DataFrame, W: int) -> pd.DataFrame:
    """
    Compute per-window radio features for windows given in win_df (run_id, window_start_s, window_end_s).
    Assumes sec_df has dense seconds for this run.
    """
    # Extract numeric KPI columns for window aggregation (exclude identifiers and debug columns)
    exclude = {"run_id","family","time_mode","t_rel_s","sec_has_record","sec_missing"}
    exclude |= {"totalBytes_counter_filled","reset_delta_raw"}
    label_cols = {"traffic_type_win","attack_category_win","attack_type_win"}
    # sec_df doesn't contain labels; win_df does.
    kpi_cols = [c for c in sec_df.columns if c not in exclude and c not in {"dlBytes","ulBytes"} and c not in label_cols]
    # Make sure these are numeric
    for c in kpi_cols:
        if not pd.api.types.is_numeric_dtype(sec_df[c]):
            sec_df[c] = pd.to_numeric(sec_df[c], errors="coerce")

    # For each window, slice seconds [start, end)
    max_t = int(sec_df["t_rel_s"].max()) if not sec_df.empty else -1
    sec_df = sec_df.set_index("t_rel_s", drop=False)

    out_rows: List[Dict[str, Any]] = []
    for _, w in win_df.iterrows():
        run_id = w["run_id"]
        family = w["family"]
        s0 = int(w["window_start_s"]); s1 = int(w["window_end_s"])
        # Clip to available seconds
        s0c = max(0, s0); s1c = min(max_t + 1, s1)
        if s1c <= s0c:
            # No seconds available; mark missing
            present = 0
            missing = W
            slice_df = None
        else:
            slice_df = sec_df.loc[s0c:s1c-1]
            # W expected seconds; if shorter (end clipped), treat missing for remainder
            expected = W
            present = int(slice_df["sec_has_record"].sum())
            # missing seconds inside clipped slice + clipped-away seconds
            missing_inside = int(slice_df["sec_missing"].sum())
            missing = int(missing_inside + max(0, expected - (s1c - s0c)))

        rec: Dict[str, Any] = {
            "run_id": run_id,
            "family": family,
            "window_start_s": s0,
            "window_end_s": s1,
            "radio_present_seconds": present,
            "radio_missing_seconds": missing,
            "radio_missing_frac": float(missing / W),
        }

        # Carry labels from window index (these are derived from network flows)
        rec["traffic_type_win"] = w.get("traffic_type_win", None)
        rec["attack_category_win"] = w.get("attack_category_win", None)
        rec["attack_type_win"] = w.get("attack_type_win", None)
        rec["n_flows"] = int(w.get("n_flows", 0))
        rec["attack_flow_frac"] = float(w.get("attack_flow_frac", 0.0))

        if slice_df is None or slice_df.empty:
            rec["radio_window_missing"] = 1
            # Set numeric aggregates to NaN/0 as appropriate
            rec["rate_total_sum"] = 0.0
            rec["rate_total_mean"] = float("nan")
            rec["reset_count"] = 0
            rec["row_count_sum"] = 0
            rec["ue_count_sum"] = 0
            # KPIs: NaN
            for c in kpi_cols:
                rec[f"{c}_mean"] = float("nan")
                rec[f"{c}_std"] = float("nan")
        else:
            rec["radio_window_missing"] = int(present == 0)
            # Rates/resets
            rec["rate_total_sum"] = float(slice_df["rate_total"].sum())
            rec["rate_total_mean"] = float(slice_df["rate_total"].mean())
            rec["reset_count"] = int(slice_df["reset_indicator"].sum())
            rec["row_count_sum"] = int(slice_df["row_count"].sum())
            if "ue_count" in slice_df.columns:
                rec["ue_count_sum"] = int(slice_df["ue_count"].sum())
            else:
                rec["ue_count_sum"] = 0

            # KPI aggregates over seconds (ignore NaNs)
            for c in kpi_cols:
                x = slice_df[c].to_numpy(dtype=np.float64)
                if np.isfinite(x).any():
                    rec[f"{c}_mean"] = float(np.nanmean(x))
                    rec[f"{c}_std"] = float(np.nanstd(x))
                else:
                    rec[f"{c}_mean"] = float("nan")
                    rec[f"{c}_std"] = float("nan")

        out_rows.append(rec)

    return pd.DataFrame(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML config file (optional)")
    ap.add_argument("--paired-runs", type=str, default=None, help="paired_runs.csv path (overrides config)")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory (overrides config)")
    ap.add_argument("--windows", type=str, default=None, help="Window specs '10:2,5:2' (overrides config)")
    ap.add_argument("--timestamp-keys", type=str, default=None, help="Comma-separated timestamp keys to check (default: timestamp,ts)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = ap.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = _read_yaml(Path(args.config).expanduser())

    paired_runs_csv = Path(args.paired_runs or _deep_get(cfg, ["paths", "paired_runs_csv"])).expanduser()
    out_dir = Path(args.out_dir or _deep_get(cfg, ["paths", "out_dir"], "./stage1_out")).expanduser()

    win_spec = args.windows or _deep_get(cfg, ["windows_spec"], None)
    if win_spec is None:
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

    ts_keys = ["timestamp", "ts"]
    if args.timestamp_keys:
        ts_keys = [s.strip() for s in args.timestamp_keys.split(",") if s.strip()]
    else:
        cfg_ts = _deep_get(cfg, ["radio", "timestamp_keys"], None)
        if isinstance(cfg_ts, list) and cfg_ts:
            ts_keys = [str(x) for x in cfg_ts]

    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "windows")
    _ensure_dir(out_dir / "summaries")

    paired = pd.read_csv(paired_runs_csv).copy()
    if "canon_stem" not in paired.columns or "txt_path" not in paired.columns or "family" not in paired.columns:
        raise ValueError("paired_runs.csv must contain columns: canon_stem, txt_path, family")
    paired["run_id"] = paired["canon_stem"].astype(str)
    paired = paired.drop_duplicates("run_id").reset_index(drop=True)

    # Per-second outputs collected across runs
    sec_rows: List[pd.DataFrame] = []
    run_summaries: List[Dict[str, Any]] = []

    print(f"Parsing telemetry for {len(paired)} runs ...")
    for i, row in paired.iterrows():
        run_id = str(row["run_id"]); family = str(row["family"])
        txt_path = Path(row["txt_path"]).expanduser()
        if not txt_path.exists():
            raise FileNotFoundError(f"Telemetry txt not found for run {run_id}: {txt_path}")
        df_raw, mode = _parse_run_txt(txt_path, ts_keys=ts_keys)
        sec = _aggregate_to_seconds(df_raw, run_id=run_id, family=family, time_mode=mode)
        sec_rows.append(sec)

        run_summaries.append({
            "run_id": run_id,
            "family": family,
            "time_mode": mode,
            "n_raw_rows": int(df_raw.shape[0]),
            "n_seconds": int(sec.shape[0]),
            "sec_missing_frac": float(sec["sec_missing"].mean()) if sec.shape[0] > 0 else float("nan"),
            "multi_rows_same_second": bool((sec["row_count"] > 1).any()),
            "resets_count": int(sec["reset_indicator"].sum()),
        })

    radio_seconds = pd.concat(sec_rows, ignore_index=True) if sec_rows else pd.DataFrame()
    out_sec = out_dir / "radio_seconds_paired42.parquet"
    if out_sec.exists() and not args.force:
        print(f"Exists, skipping write (use --force to overwrite): {out_sec}")
    else:
        radio_seconds.to_parquet(out_sec, index=False)
        print(f"Wrote: {out_sec} (rows={len(radio_seconds):,})")

    # Save run summary
    out_run = out_dir / "summaries" / "radio_run_summary_stage1.csv"
    if not out_run.exists() or args.force:
        pd.DataFrame(run_summaries).to_csv(out_run, index=False)
        print(f"Wrote: {out_run}")

    # Build radio windows aligned to network window index for each (W,S)
    for (W, S) in windows:
        idx_path = out_dir / "windows" / f"window_index_W{W}_S{S}.parquet"
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing window index file: {idx_path}. Run prepare_network_windows.py first.")
        win_index = pd.read_parquet(idx_path)
        # Safety checks
        needed = {"run_id","family","window_start_s","window_end_s","traffic_type_win","attack_category_win","attack_type_win"}
        missing = needed - set(win_index.columns)
        if missing:
            raise ValueError(f"window_index missing columns: {sorted(missing)}")

        print(f"\nBuilding radio window features W={W}, S={S} using {idx_path.name} ...")

        win_parts: List[pd.DataFrame] = []
        for run_id, wdf in win_index.groupby("run_id", sort=False):
            sec = radio_seconds[radio_seconds["run_id"] == run_id].copy()
            if sec.empty:
                # Shouldn't happen in paired runs; but be robust
                part = wdf.copy()
                part["radio_window_missing"] = 1
                win_parts.append(part)
                continue
            part = _window_features_from_seconds(sec, wdf, W=W)
            win_parts.append(part)

        all_radio_win = pd.concat(win_parts, ignore_index=True) if win_parts else pd.DataFrame()
        out_win = out_dir / "windows" / f"radio_windows_W{W}_S{S}.parquet"
        if out_win.exists() and not args.force:
            print(f"Exists, skipping write (use --force to overwrite): {out_win}")
        else:
            all_radio_win.to_parquet(out_win, index=False)
            print(f"Wrote: {out_win} (rows={len(all_radio_win):,})")

        # Summary
        summ = {
            "W": W, "S": S,
            "n_windows": int(all_radio_win.shape[0]),
            "radio_window_missing_rate": float(all_radio_win["radio_window_missing"].mean()) if all_radio_win.shape[0] > 0 else float("nan"),
            "radio_missing_frac_median": float(all_radio_win["radio_missing_frac"].median()) if all_radio_win.shape[0] > 0 else float("nan"),
        }
        out_json = out_dir / "summaries" / f"radio_window_summary_W{W}_S{S}.json"
        if not out_json.exists() or args.force:
            out_json.write_text(json.dumps(summ, indent=2), encoding="utf-8")
            print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()