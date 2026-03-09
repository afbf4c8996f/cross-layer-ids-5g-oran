#!/usr/bin/env python3
"""Replay & verify saved *tabular/ML* predictions against saved joblib models.

This is a contract test analogous to replay_predictions_check_v4_fixed.py (torch/DL),
but for the classical/tabular models (logreg / xgboost / random forest).

What it checks
--------------
For each run unit under <out_dir>:
  - Load joblib model(s) from out_dir/models/<split>/<W*_S*>/<model>/
  - Re-run predict_proba(...) on the processed VAL/TEST inputs
  - Compare to saved prediction parquet(s) in out_dir/predictions/...

It validates:
  - model ↔ processed features compatibility
  - row-order / KEY_COLS alignment
  - multiclass label-order correctness (p_<class> columns)
  - fusion_mean correctness (if fusion parquet exists)
  - fusion_stacked correctness (binary; if fusion_head + parquet exist)

Usage
-----
python3 replay_tabular_predictions_check.py --out_dir <stage3-out> --tol 1e-5

Notes
-----
- This script targets the *joblib* artifact layout produced by run_stage3_tabular.py:

    out_dir/models/<split>/W10_S2/<model>/
        network_model.joblib
        radio_model.joblib
        fusion_head_binary.joblib              (optional)
        network_model_multiclass.joblib        (optional)
        radio_model_multiclass.joblib          (optional)
        multiclass_classes.json                (optional)

  (Older layouts that included a /binary or /multiclass subdir are also handled.)

- It is READ-ONLY. It does not modify out_dir.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from stage3_io import KEY_COLS, align_modalities, load_processed
from stage3_fusion import fusion_mean_binary, fusion_mean_multiclass

try:
    from joblib import load as joblib_load  # type: ignore
except Exception as e:  # pragma: no cover
    print(f"[FATAL] joblib import failed: {e}")
    raise


def _clip01(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.where(np.isfinite(p), p, 0.5)
    return np.clip(p, eps, 1.0 - eps)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip01(np.asarray(p, dtype=np.float64))
    return np.log(p / (1.0 - p))


def _stacked_binary_predict(head: Any, p_net: np.ndarray, p_rad: np.ndarray) -> np.ndarray:
    """Compute stacked-fusion probability given a saved fusion head.

    IMPORTANT (numerics):
      The Stage-3 pipeline feeds float32 probability scores into the fusion head.
      If we upcast to float64 in replay, logit() near 0/1 can change enough to
      create false replay failures. So we keep the fusion inputs in float32.
    """
    p_net = np.asarray(p_net, dtype=np.float32).reshape(-1)
    p_rad = np.asarray(p_rad, dtype=np.float32).reshape(-1)

    # Preferred: Stage-3 StackedFusionBinary wrapper
    if hasattr(head, "predict") and callable(getattr(head, "predict")):
        try:
            out = head.predict(p_net, p_rad)
            return np.asarray(out, dtype=np.float32).reshape(-1)
        except TypeError:
            # Some wrappers may have a different signature; fall through.
            pass

    # Otherwise: sklearn estimator directly (or wrapper with .model.predict_proba)
    one = np.float32(1.0)
    eps = np.float32(1e-6)

    def _logit32(p: np.ndarray) -> np.ndarray:
        p = np.clip(p.astype(np.float32), eps, one - eps)
        return np.log(p / (one - p)).astype(np.float32)

    X = np.stack([_logit32(p_net), _logit32(p_rad)], axis=1).astype(np.float32)

    if hasattr(head, "predict_proba") and callable(getattr(head, "predict_proba")):
        proba = head.predict_proba(X)
        return np.asarray(proba, dtype=np.float32)[:, 1]

    mdl = getattr(head, "model", None)
    if mdl is not None and hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(X)
        return np.asarray(proba, dtype=np.float32)[:, 1]

    raise TypeError(f"Unsupported fusion head type: {type(head)}")


def _align_pred_df_to_processed(pred_df: pd.DataFrame, proc_df: pd.DataFrame) -> pd.DataFrame:
    """Reorder pred_df rows to match proc_df order by KEY_COLS."""
    if len(pred_df) != len(proc_df):
        raise ValueError(f"Row-count mismatch: pred={len(pred_df)} proc={len(proc_df)}")

    a = proc_df[KEY_COLS].copy()
    b = pred_df[KEY_COLS].copy()
    a["_i"] = np.arange(len(a), dtype=int)
    b["_j"] = np.arange(len(b), dtype=int)
    m = a.merge(b, on=KEY_COLS, how="inner")
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError("KEY_COLS mismatch between processed data and predictions parquet.")
    m = m.sort_values("_i")
    j = m["_j"].to_numpy(dtype=int)
    return pred_df.iloc[j].reset_index(drop=True)


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def _predict_proba_with_optional_groups(est: Any, X: np.ndarray, groups: Any) -> np.ndarray:
    """Call predict_proba with (optional) groups kwarg."""
    try:
        sig = inspect.signature(est.predict_proba)
        if "groups" in sig.parameters:
            return est.predict_proba(X, groups=groups)
    except Exception:
        pass
    return est.predict_proba(X)


def _unwrap_estimator(obj: Any) -> Any:
    """ModelBundles store the sklearn estimator under .model."""
    return getattr(obj, "model", obj)


def _binary_col_for_positive(est: Any, P: np.ndarray) -> int:
    """Return column index for class==1 when possible; otherwise fall back to last col."""
    if P.ndim != 2 or P.shape[1] < 2:
        return -1
    cls = getattr(est, "classes_", None)
    if cls is not None:
        try:
            cls_list = list(cls)
            if 1 in cls_list:
                return int(cls_list.index(1))
        except Exception:
            pass
    return 1 if P.shape[1] == 2 else (P.shape[1] - 1)


def _align_multiclass_proba(P: np.ndarray, est: Any, n_classes: int) -> np.ndarray:
    """Align proba columns to class index order 0..K-1 when estimator.classes_ is permuted."""
    if P.ndim != 2 or P.shape[1] != n_classes:
        return P
    cls = getattr(est, "classes_", None)
    if cls is None:
        return P
    try:
        cls_list = list(cls)
    except Exception:
        return P

    # If classes are ints covering 0..K-1 but permuted, reorder.
    if all(isinstance(c, (int, np.integer)) for c in cls_list):
        cls_int = [int(c) for c in cls_list]
        if set(cls_int) == set(range(n_classes)):
            P2 = np.zeros_like(P)
            for j, lab in enumerate(cls_int):
                P2[:, lab] = P[:, j]
            return P2
    return P


def _class_to_colnames(classes: list[str]) -> list[str]:
    """Must match run_stage3_tabular.py sanitization."""
    used = set()
    out: list[str] = []
    for j, raw in enumerate(classes):
        col = re.sub(r"[^0-9A-Za-z_]+", "_", str(raw).strip())
        if col == "":
            col = f"cls{j}"
        col = f"p_{col}"
        base = col
        k = 1
        while col in used:
            k += 1
            col = f"{base}__{k}"
        used.add(col)
        out.append(col)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--tol", default=1e-5, type=float)
    ap.add_argument("--limit", default=0, type=int, help="If >0, limit number of run units checked.")
    ap.add_argument("--processed_dir", default="", type=str, help="Fallback processed_dir if stage3_run_config.json is missing.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    tol = float(args.tol)

    run_cfg_path = out_dir / "stage3_run_config.json"
    processed_dir: Path
    feature_ablation_cfg: Dict[str, Any]
    if run_cfg_path.exists():
        run_cfg = json.loads(run_cfg_path.read_text())
        processed_dir = Path(run_cfg["paths"]["processed_dir"]).expanduser()
        feature_ablation_cfg = run_cfg.get("feature_ablation", {}) or {}
    else:
        if args.processed_dir:
            processed_dir = Path(args.processed_dir).expanduser()
            feature_ablation_cfg = {}
            print(f"[WARN] {run_cfg_path} missing; using --processed_dir={processed_dir} and assuming feature_ablation=full.")
        else:
            raise FileNotFoundError(
                f"Missing {run_cfg_path}. Pass an out_dir produced by run_stage3_tabular.py, "
                "or provide --processed_dir to replay against an older output directory."
            )

    models_root = out_dir / "models"
    pred_root = out_dir / "predictions"
    if not models_root.exists():
        raise FileNotFoundError(f"Missing models dir: {models_root}")
    if not pred_root.exists():
        raise FileNotFoundError(f"Missing predictions dir: {pred_root}")

    # Discover run units by scanning model joblibs.
    # Supported layouts:
    #   - models/<split>/W10_S2/<model>/<file>.joblib
    #   - models/<split>/W10_S2/<model>/<task>/<file>.joblib   (older)
    grouped: Dict[Tuple[str, str, str], Dict[str, Path]] = {}
    for p in sorted(models_root.rglob("*.joblib")):
        try:
            model_dir = p.parent
            if model_dir.name in ("binary", "multiclass"):
                model_dir = model_dir.parent
            model = model_dir.name
            ws = model_dir.parent.name
            split = model_dir.parent.parent.name
        except Exception:
            continue
        key = (split, ws, model)
        grouped.setdefault(key, {})[p.name] = p

    keys = sorted(grouped.keys())
    if args.limit and args.limit > 0:
        keys = keys[: int(args.limit)]

    n_ok = 0
    n_fail = 0

    for (split, ws, model) in keys:
        # Parse W,S
        try:
            W = int(ws.split("_")[0].lstrip("W"))
            S = int(ws.split("_")[1].lstrip("S"))
        except Exception:
            print(f"[SKIP] Unparseable ws={ws!r}")
            continue

        # Load processed data (val/test) and align modalities
        net_va = load_processed(processed_dir, split, "network", W, S, "val", feature_ablation=feature_ablation_cfg)
        net_te = load_processed(processed_dir, split, "network", W, S, "test", feature_ablation=feature_ablation_cfg)
        rad_va = load_processed(processed_dir, split, "radio", W, S, "val", feature_ablation=feature_ablation_cfg)
        rad_te = load_processed(processed_dir, split, "radio", W, S, "test", feature_ablation=feature_ablation_cfg)
        net_va, rad_va = align_modalities(net_va, rad_va)
        net_te, rad_te = align_modalities(net_te, rad_te)

        def _load_estimator(path: Path) -> Any:
            obj = joblib_load(str(path))
            return _unwrap_estimator(obj)

        def _check_binary() -> None:
            nonlocal n_ok, n_fail

            task = "binary"
            task_pred_dir = pred_root / split / ws / model / task

            # Base models: network_only + radio_only
            ests: Dict[str, Any] = {}
            for sys_name, proc_val, proc_test in [
                ("network", net_va, net_te),
                ("radio", rad_va, rad_te),
            ]:
                model_file = f"{sys_name}_model.joblib"
                model_path = grouped[(split, ws, model)].get(model_file)
                if model_path is None:
                    continue

                est = _load_estimator(model_path)
                ests[sys_name] = est

                for part, proc_part in [("val", proc_val), ("test", proc_test)]:
                    name = f"{sys_name}_only"
                    pred_path = task_pred_dir / f"{name}_{part}.parquet"
                    if not pred_path.exists():
                        continue

                    df = pd.read_parquet(pred_path)
                    df = _align_pred_df_to_processed(df, proc_part.df)

                    P = _predict_proba_with_optional_groups(est, proc_part.X, proc_part.groups)
                    P = np.asarray(P, dtype=float)
                    p = P[:, _binary_col_for_positive(est, P)]
                    p = np.where(np.isfinite(p), p, 0.5).astype(np.float32)
                    diff = _max_abs_diff(p, df["score"].to_numpy(dtype=float))
                    if diff > tol:
                        print(f"[FAIL] {split}/{ws}/{model} {name} {part} max|diff|={diff:.3g} > tol={tol}")
                        n_fail += 1
                    else:
                        n_ok += 1

            # Mean fusion
            if "network" in ests and "radio" in ests:
                for part, (proc_net, proc_rad, proc_ref) in [
                    ("val", (net_va, rad_va, net_va)),
                    ("test", (net_te, rad_te, net_te)),
                ]:
                    pred_path = task_pred_dir / f"fusion_mean_{part}.parquet"
                    if not pred_path.exists():
                        continue

                    # Prefer saved base parquets for fusion replay (robust to model divergence).
                    net_base_path = task_pred_dir / f"network_only_{part}.parquet"
                    rad_base_path = task_pred_dir / f"radio_only_{part}.parquet"

                    if net_base_path.exists() and rad_base_path.exists():
                        df_net_base = pd.read_parquet(net_base_path)
                        df_rad_base = pd.read_parquet(rad_base_path)
                        df_net_base = _align_pred_df_to_processed(df_net_base, proc_ref.df)
                        df_rad_base = _align_pred_df_to_processed(df_rad_base, proc_ref.df)
                        p_net = df_net_base["score"].to_numpy(dtype=np.float32)
                        p_rad = df_rad_base["score"].to_numpy(dtype=np.float32)
                        p_mean = fusion_mean_binary(p_net, p_rad)
                    else:
                        Pn = _predict_proba_with_optional_groups(ests["network"], proc_net.X, proc_net.groups)
                        Pr = _predict_proba_with_optional_groups(ests["radio"], proc_rad.X, proc_rad.groups)
                        Pn = np.asarray(Pn, dtype=float)
                        Pr = np.asarray(Pr, dtype=float)
                        p_mean = fusion_mean_binary(Pn, Pr)[:, 1]

                    df = pd.read_parquet(pred_path)
                    df = _align_pred_df_to_processed(df, proc_ref.df)
                    diff = _max_abs_diff(p_mean, df["score"].to_numpy(dtype=float))
                    if diff > tol:
                        print(f"[FAIL] {split}/{ws}/{model} fusion_mean {part} max|diff|={diff:.3g} > tol={tol}")
                        n_fail += 1
                    else:
                        n_ok += 1

                # Stacked fusion (optional)
                head_path = grouped[(split, ws, model)].get("fusion_head_binary.joblib")
                if head_path is not None:
                    try:
                        head = joblib_load(str(head_path))
                    except Exception as e:  # pragma: no cover
                        print(f"[WARN] Could not load fusion head {head_path}: {e}")
                    else:
                        for part, (proc_net, proc_rad, proc_ref) in [
                            ("val", (net_va, rad_va, net_va)),
                            ("test", (net_te, rad_te, net_te)),
                        ]:
                            pred_path = task_pred_dir / f"fusion_stacked_{part}.parquet"
                            if not pred_path.exists():
                                continue

                            # IMPORTANT (replay robustness): stacked fusion applies a logit transform.
                            # Even tiny numerical differences between *replayed* base probabilities and
                            # the *saved* base parquets can be amplified after logit, producing false
                            # replay failures (this is most common for XGBoost when probs saturate).
                            #
                            # Contract we want to verify here:
                            #   fusion_head_binary.joblib + (network_only/radio_only parquets) -> fusion_stacked parquet.
                            #
                            # So prefer feeding the fusion head with the saved base parquets when present.
                            net_base_path = task_pred_dir / f"network_only_{part}.parquet"
                            rad_base_path = task_pred_dir / f"radio_only_{part}.parquet"

                            if net_base_path.exists() and rad_base_path.exists():
                                df_net_base = pd.read_parquet(net_base_path)
                                df_rad_base = pd.read_parquet(rad_base_path)
                                df_net_base = _align_pred_df_to_processed(df_net_base, proc_ref.df)
                                df_rad_base = _align_pred_df_to_processed(df_rad_base, proc_ref.df)
                                p_net = df_net_base["score"].to_numpy(dtype=np.float32)
                                p_rad = df_rad_base["score"].to_numpy(dtype=np.float32)
                            else:
                                Pn = _predict_proba_with_optional_groups(ests["network"], proc_net.X, proc_net.groups)
                                Pr = _predict_proba_with_optional_groups(ests["radio"], proc_rad.X, proc_rad.groups)
                                Pn = np.asarray(Pn, dtype=np.float32)
                                Pr = np.asarray(Pr, dtype=np.float32)
                                p_net = Pn[:, _binary_col_for_positive(ests["network"], Pn)]
                                p_rad = Pr[:, _binary_col_for_positive(ests["radio"], Pr)]
                                p_net = np.where(np.isfinite(p_net), p_net, 0.5).astype(np.float32)
                                p_rad = np.where(np.isfinite(p_rad), p_rad, 0.5).astype(np.float32)

                            if p_net.dtype != np.float32 or p_rad.dtype != np.float32:
                                raise AssertionError(
                                    f"Expected float32 for stacked fusion inputs, got {p_net.dtype} / {p_rad.dtype} "
                                    f"({split}/{ws}/{model} part={part})"
                                )

                            p_stack = _stacked_binary_predict(head, p_net, p_rad)

                            df = pd.read_parquet(pred_path)
                            df = _align_pred_df_to_processed(df, proc_ref.df)
                            diff = _max_abs_diff(p_stack, df["score"].to_numpy(dtype=float))
                            if diff > tol:
                                print(
                                    f"[FAIL] {split}/{ws}/{model} fusion_stacked {part} "
                                    f"max|diff|={diff:.3g} > tol={tol}"
                                )
                                n_fail += 1
                            else:
                                n_ok += 1

        def _check_multiclass() -> None:
            nonlocal n_ok, n_fail

            task = "multiclass"
            task_pred_dir = pred_root / split / ws / model / task
            if not task_pred_dir.exists():
                return

            # classes list (LabelEncoder order)
            classes_path = models_root / split / ws / model / "multiclass_classes.json"
            if not classes_path.exists():
                # older layout fallback
                hits = list((models_root / split / ws / model).rglob("multiclass_classes.json"))
                if not hits:
                    print(f"[WARN] Missing classes file for {split}/{ws}/{model}; skipping multiclass replay.")
                    return
                classes_path = hits[0]
            classes = json.loads(classes_path.read_text())
            n_classes = int(len(classes))
            colnames = _class_to_colnames([str(c) for c in classes])

            ests: Dict[str, Any] = {}
            for sys_name, proc_val, proc_test in [
                ("network", net_va, net_te),
                ("radio", rad_va, rad_te),
            ]:
                model_file = f"{sys_name}_model_multiclass.joblib"
                model_path = grouped[(split, ws, model)].get(model_file)
                if model_path is None:
                    continue

                est = _load_estimator(model_path)
                ests[sys_name] = est

                for part, proc_part in [("val", proc_val), ("test", proc_test)]:
                    name = f"{sys_name}_only"
                    pred_path = task_pred_dir / f"{name}_{part}.parquet"
                    if not pred_path.exists():
                        continue

                    df = pd.read_parquet(pred_path)
                    df = _align_pred_df_to_processed(df, proc_part.df)

                    P = _predict_proba_with_optional_groups(est, proc_part.X, proc_part.groups)
                    P = np.asarray(P, dtype=float)
                    if P.ndim != 2 or P.shape[1] != n_classes:
                        raise ValueError(f"Bad predict_proba shape for {split}/{ws}/{model}/{sys_name}: {P.shape} (expected Nx{n_classes})")
                    P = _align_multiclass_proba(P, est, n_classes)
                    # Sanitize NaN/Inf from diverged models (must match training pipeline)
                    P = np.where(np.isfinite(P), P, 0.0).astype(np.float32)
                    row_sums = P.sum(axis=1, keepdims=True)
                    uniform = np.float32(1.0 / max(P.shape[1], 1))
                    P = np.where(row_sums > 0, P / np.where(row_sums > 0, row_sums, 1.0), uniform)

                    p_max = np.max(P, axis=1)
                    y_pred = np.asarray(classes, dtype=str)[np.argmax(P, axis=1)]
                    diff = _max_abs_diff(p_max, df["p_max"].to_numpy(dtype=float))
                    same = np.array_equal(y_pred.astype(str), df["y_pred"].astype(str).to_numpy())

                    # Compare per-class probability columns if present.
                    for j, col in enumerate(colnames):
                        if col in df.columns:
                            dcol = _max_abs_diff(P[:, j], df[col].to_numpy(dtype=float))
                            diff = max(diff, dcol)

                    if (diff > tol) or (not same):
                        print(
                            f"[FAIL] {split}/{ws}/{model} {name} {part} "
                            f"max|diff|={diff:.3g} (tol={tol}) y_pred_match={same}"
                        )
                        n_fail += 1
                    else:
                        n_ok += 1

            # Mean fusion
            if "network" in ests and "radio" in ests:
                for part, (proc_net, proc_rad, proc_ref) in [
                    ("val", (net_va, rad_va, net_va)),
                    ("test", (net_te, rad_te, net_te)),
                ]:
                    pred_path = task_pred_dir / f"fusion_mean_{part}.parquet"
                    if not pred_path.exists():
                        continue

                    # Prefer saved base parquets for fusion replay (robust to model divergence).
                    # Contract: fusion_mean = fusion_mean_multiclass(network_only, radio_only)
                    net_base_path = task_pred_dir / f"network_only_{part}.parquet"
                    rad_base_path = task_pred_dir / f"radio_only_{part}.parquet"

                    if net_base_path.exists() and rad_base_path.exists():
                        df_net_base = pd.read_parquet(net_base_path)
                        df_rad_base = pd.read_parquet(rad_base_path)
                        df_net_base = _align_pred_df_to_processed(df_net_base, proc_ref.df)
                        df_rad_base = _align_pred_df_to_processed(df_rad_base, proc_ref.df)
                        Pn = df_net_base[colnames].to_numpy(dtype=np.float32)
                        Pr = df_rad_base[colnames].to_numpy(dtype=np.float32)
                    else:
                        # Fallback: re-run model
                        Pn = _predict_proba_with_optional_groups(ests["network"], proc_net.X, proc_net.groups)
                        Pr = _predict_proba_with_optional_groups(ests["radio"], proc_rad.X, proc_rad.groups)
                        Pn = np.asarray(Pn, dtype=float)
                        Pr = np.asarray(Pr, dtype=float)
                        Pn = _align_multiclass_proba(Pn, ests["network"], n_classes)
                        Pr = _align_multiclass_proba(Pr, ests["radio"], n_classes)
                        Pn = np.where(np.isfinite(Pn), Pn, 0.0).astype(np.float32)
                        Pr = np.where(np.isfinite(Pr), Pr, 0.0).astype(np.float32)
                        row_sums_n = Pn.sum(axis=1, keepdims=True)
                        row_sums_r = Pr.sum(axis=1, keepdims=True)
                        uniform = np.float32(1.0 / max(Pn.shape[1], 1))
                        Pn = np.where(row_sums_n > 0, Pn / np.where(row_sums_n > 0, row_sums_n, 1.0), uniform)
                        Pr = np.where(row_sums_r > 0, Pr / np.where(row_sums_r > 0, row_sums_r, 1.0), uniform)

                    Pm = fusion_mean_multiclass(Pn, Pr)
                    p_max = np.max(Pm, axis=1)
                    y_pred = np.asarray(classes, dtype=str)[np.argmax(Pm, axis=1)]

                    df = pd.read_parquet(pred_path)
                    df = _align_pred_df_to_processed(df, proc_ref.df)
                    diff = _max_abs_diff(p_max, df["p_max"].to_numpy(dtype=float))
                    same = np.array_equal(y_pred.astype(str), df["y_pred"].astype(str).to_numpy())

                    for j, col in enumerate(colnames):
                        if col in df.columns:
                            dcol = _max_abs_diff(Pm[:, j], df[col].to_numpy(dtype=float))
                            diff = max(diff, dcol)

                    if (diff > tol) or (not same):
                        print(
                            f"[FAIL] {split}/{ws}/{model} fusion_mean {part} "
                            f"max|diff|={diff:.3g} (tol={tol}) y_pred_match={same}"
                        )
                        n_fail += 1
                    else:
                        n_ok += 1

        _check_binary()
        _check_multiclass()

    if n_fail:
        print(f"[SUMMARY] FAIL={n_fail} OK={n_ok}")
        sys.exit(2)

    print(f"[SUMMARY] ALL OK ({n_ok} checks, tol={tol})")


if __name__ == "__main__":
    main()
