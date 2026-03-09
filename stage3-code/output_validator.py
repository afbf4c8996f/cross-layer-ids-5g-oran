#!/usr/bin/env python3
"""Validate Stage-3 output directory for consistency + numeric sanity.

This is intended to be run after *any* training run.

It checks:
- metrics CSVs exist and load
- run_artifact.json files exist
- prediction parquets exist AND contain sane probability-like values
- TTD summaries:
    * inf allowed only in ttd_* columns AND only when detect_rate == 0
    * NaNs are allowed only in places that are structurally "not applicable":
        - delta_window_minus_flow: n_attack_runs/detect_rate/ttd_* are NaN
        - delta_* are NaN when n_delta is NaN (delta not defined) or n_delta==0

Prediction validation (new vs v2):
- Binary parquets must contain:
    * y_true in {0,1}
    * score finite and within [0,1]
- Multiclass parquets must contain:
    * y_true, y_pred strings
    * p_max finite and within [0,1]
  If full per-class probability columns are present (p_<class>), it additionally checks:
    * each prob in [0,1]
    * rows sum to ~1 (within tolerance)
    * p_max matches row-wise max(p_<class>)

Usage:
  python3 output_validator.py --out_dir <stage3-out>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def _fail(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _load_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        _fail(f"Missing file: {p}")
    df = pd.read_csv(p)
    _ok(f"Loaded {p.name}: {len(df)} rows, {df.shape[1]} cols")
    return df


def _assert_no_nan_inf_numeric(
    df: pd.DataFrame,
    *,
    context: str,
    allow_nan_cols: Iterable[str] = (),
    allow_inf_cols: Iterable[str] = (),
) -> None:
    allow_nan_cols = set(allow_nan_cols)
    allow_inf_cols = set(allow_inf_cols)

    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return

    # INF
    inf_cols = num.columns[np.isinf(num.to_numpy()).any(axis=0)].tolist()
    bad_inf = [c for c in inf_cols if c not in allow_inf_cols]
    if bad_inf:
        _fail(f"{context}: found inf in numeric columns not allowed: {bad_inf}")

    # NaN
    nan_cols = num.columns[num.isna().any(axis=0)].tolist()
    bad_nan = [c for c in nan_cols if c not in allow_nan_cols]
    if bad_nan:
        _fail(f"{context}: found NaN in numeric columns not allowed: {bad_nan}")


def _validate_ttd_summary(df: pd.DataFrame, *, name: str) -> None:
    required = {
        "ttd_mode",
        "detect_rate",
        "ttd_median",
        "ttd_p25",
        "ttd_p75",
        "n_attack_runs",
        "n_delta",
        "delta_median",
        "delta_p25",
        "delta_p75",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        _fail(f"{name}: missing columns: {missing}")

    ttd_cols = ["ttd_median", "ttd_p25", "ttd_p75"]
    delta_cols = ["delta_median", "delta_p25", "delta_p75"]

    # Rule 1: inf allowed only in ttd cols, and only when detect_rate == 0.
    num = df.select_dtypes(include=[np.number])
    inf_any = np.isinf(num.to_numpy())
    if inf_any.any():
        inf_cols = num.columns[np.isinf(num.to_numpy()).any(axis=0)].tolist()
        # Allow subset (some summaries may not contain inf), but disallow other cols.
        extra = [c for c in inf_cols if c not in ttd_cols]
        if extra:
            _fail(f"{name}: inf appeared in non-ttd columns: {extra}")

        bad = df[np.isinf(df[ttd_cols].to_numpy()).any(axis=1)]
        if not (bad["detect_rate"].to_numpy() == 0).all():
            _fail(f"{name}: found inf TTD rows where detect_rate != 0")

    # Rule 2: delta_window_minus_flow mode uses ONLY delta columns.
    is_delta_mode = df["ttd_mode"].astype(str) == "delta_window_minus_flow"
    if is_delta_mode.any():
        must_nan = ["n_attack_runs", "detect_rate"] + ttd_cols
        not_nan = df.loc[is_delta_mode, must_nan].notna().any(axis=None)
        if bool(not_nan):
            _fail(f"{name}: delta_window_minus_flow rows must have NaN in {must_nan}")

    # Rule 3: non-delta modes must define attack/ttd fields.
    not_delta_mode = ~is_delta_mode
    if not_delta_mode.any():
        if df.loc[not_delta_mode, "n_attack_runs"].isna().any():
            _fail(f"{name}: non-delta rows have NaN n_attack_runs")
        if df.loc[not_delta_mode, "detect_rate"].isna().any():
            _fail(f"{name}: non-delta rows have NaN detect_rate")

        dr = df.loc[not_delta_mode, "detect_rate"].to_numpy(dtype=float)
        ttd = df.loc[not_delta_mode, ttd_cols].to_numpy(dtype=float)
        finite_mask = np.isfinite(ttd)
        ok_mask = (dr == 0) | finite_mask.all(axis=1)
        if not ok_mask.all():
            _fail(f"{name}: found non-delta rows where detect_rate>0 but ttd_* not finite")

    # Rule 4: delta columns are allowed to be NaN only when n_delta is NaN (not applicable)
    #         or when n_delta == 0.
    n_delta = df["n_delta"]
    n_delta_nan = n_delta.isna()
    if n_delta_nan.any():
        if df.loc[n_delta_nan, delta_cols].notna().any(axis=None):
            _fail(f"{name}: n_delta is NaN but delta_* has non-NaN values")

    n_delta_zero = (~n_delta_nan) & (n_delta.to_numpy(dtype=float) == 0)
    if n_delta_zero.any():
        if df.loc[n_delta_zero, delta_cols].notna().any(axis=None):
            _fail(f"{name}: n_delta==0 but delta_* has non-NaN values")

    n_delta_pos = (~n_delta_nan) & (n_delta.to_numpy(dtype=float) > 0)
    if n_delta_pos.any():
        vals = df.loc[n_delta_pos, delta_cols].to_numpy(dtype=float)
        if (np.isnan(vals).any() or np.isinf(vals).any()):
            _fail(f"{name}: n_delta>0 but delta_* contains NaN/inf")


def _validate_pred_binary(df: pd.DataFrame, *, name: str, tol: float = 1e-6) -> None:
    need = {"y_true", "score"}
    missing = sorted(need - set(df.columns))
    if missing:
        _fail(f"{name}: missing required columns: {missing}")

    y = df["y_true"].to_numpy()
    # Allow bool/int; enforce set {0,1}
    try:
        y_int = y.astype(int)
    except Exception:
        _fail(f"{name}: y_true is not int/bool-like")
    bad_y = ~np.isin(y_int, [0, 1])
    if bad_y.any():
        _fail(f"{name}: y_true contains values outside {{0,1}} (example={y_int[bad_y][0]!r})")

    s = df["score"].to_numpy(dtype=float)
    if not np.isfinite(s).all():
        _fail(f"{name}: score contains NaN/inf")
    if float(np.min(s)) < -tol or float(np.max(s)) > 1.0 + tol:
        _fail(
            f"{name}: score out of [0,1] range (min={float(np.min(s)):.6g}, max={float(np.max(s)):.6g}, tol={tol})"
        )


def _validate_pred_multiclass(df: pd.DataFrame, *, name: str, tol: float = 1e-6) -> None:
    need = {"y_true", "y_pred", "p_max"}
    missing = sorted(need - set(df.columns))
    if missing:
        _fail(f"{name}: missing required columns: {missing}")

    # Strings must be non-null
    if df["y_true"].isna().any():
        _fail(f"{name}: y_true contains NaN")
    if df["y_pred"].isna().any():
        _fail(f"{name}: y_pred contains NaN")

    pmax = df["p_max"].to_numpy(dtype=float)
    if not np.isfinite(pmax).all():
        _fail(f"{name}: p_max contains NaN/inf")
    if float(np.min(pmax)) < -tol or float(np.max(pmax)) > 1.0 + tol:
        _fail(
            f"{name}: p_max out of [0,1] range (min={float(np.min(pmax)):.6g}, max={float(np.max(pmax)):.6g}, tol={tol})"
        )

    # Optional: full probability columns present?
    prob_cols = [c for c in df.columns if c.startswith("p_") and c not in {"p_max"}]
    if prob_cols:
        P = df[prob_cols].to_numpy(dtype=float)
        if not np.isfinite(P).all():
            _fail(f"{name}: probability columns contain NaN/inf")
        if float(np.min(P)) < -tol or float(np.max(P)) > 1.0 + tol:
            _fail(f"{name}: probability columns out of [0,1] range")

        row_sum = P.sum(axis=1)
        if not np.allclose(row_sum, 1.0, atol=max(1e-4, tol * 100), rtol=0.0):
            worst = float(np.max(np.abs(row_sum - 1.0)))
            _fail(f"{name}: probability rows do not sum to 1 (worst |sum-1|={worst:.6g})")

        pmax_from_P = P.max(axis=1)
        if not np.allclose(pmax_from_P, pmax, atol=max(1e-4, tol * 100), rtol=0.0):
            worst = float(np.max(np.abs(pmax_from_P - pmax)))
            _fail(f"{name}: p_max does not match max(prob_cols) (worst |diff|={worst:.6g})")


def _validate_prediction_parquet(p: Path, *, tol: float) -> None:
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        _fail(f"Failed to read parquet {p}: {e}")

    if "score" in df.columns:
        _validate_pred_binary(df, name=str(p), tol=tol)
        return
    if "p_max" in df.columns:
        _validate_pred_multiclass(df, name=str(p), tol=tol)
        return

    _fail(f"{p}: could not infer prediction type (missing 'score' and 'p_max')")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument(
        "--pred_tol",
        type=float,
        default=1e-6,
        help="Tolerance for probability range checks (defaults to 1e-6).",
    )
    ap.add_argument(
        "--max_pred_files",
        type=int,
        default=0,
        help="If >0, validate only first N prediction parquets (useful for quick smoke checks).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    pred_tol = float(args.pred_tol)
    max_pred = int(args.max_pred_files)

    # metrics
    mdir = out_dir / "metrics"
    df_bin = _load_csv(mdir / "metrics_binary.csv")
    df_mc = _load_csv(mdir / "metrics_multiclass.csv")

    _assert_no_nan_inf_numeric(df_bin, context="metrics_binary.csv")
    _assert_no_nan_inf_numeric(df_mc, context="metrics_multiclass.csv")

    # run artifacts
    art_files = list(out_dir.rglob("run_artifact.json"))
    if not art_files:
        _fail("No run_artifact.json files found")
    _ok(f"Found {len(art_files)} run_artifact.json files")

    # prediction parquets
    pred_root = out_dir / "predictions"
    pred_files = list(pred_root.rglob("*.parquet")) if pred_root.exists() else []
    if not pred_files:
        _fail("No prediction parquet files found under predictions/")
    _ok(f"Found {len(pred_files)} prediction parquet files")

    pred_files = sorted(pred_files)
    if max_pred > 0:
        pred_files = pred_files[:max_pred]
        _ok(f"Validating first {len(pred_files)} prediction parquet(s) (max_pred_files={max_pred})")

    for i, p in enumerate(pred_files, 1):
        _validate_prediction_parquet(p, tol=pred_tol)
        if i % 50 == 0:
            _ok(f"Validated {i}/{len(pred_files)} prediction parquets")
    _ok(f"Validated {len(pred_files)} prediction parquet(s)")

    # TTD summaries (optional but should exist if ttd enabled)
    ttd_summaries: List[Path] = sorted(out_dir.rglob("ttd_summary_fpr*.csv"))
    if not ttd_summaries:
        print("[WARN] No ttd_summary_fpr*.csv found (TTD may be disabled).")
    else:
        for p in ttd_summaries:
            df = pd.read_csv(p)
            _validate_ttd_summary(df, name=p.name)
        _ok(f"Validated {len(ttd_summaries)} TTD summary file(s)")

    print("[ALL CHECKS PASSED]")


if __name__ == "__main__":
    main()
