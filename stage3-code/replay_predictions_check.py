#!/usr/bin/env python3
"""Replay & verify saved predictions against saved checkpoints.

This is a *contract test* for your Stage-3 outputs.

What it checks
--------------
For each run unit under <out_dir>:
  - Reload checkpoint(s)
  - Re-run wrapper.predict_proba(...) on the processed VAL/TEST inputs
  - Compare to saved prediction parquet(s)

It validates:
  - checkpoint ↔ model architecture compatibility
  - row-order / KEY_COLS alignment
  - multiclass label-order correctness
  - fusion_mean correctness (if fusion parquet exists)
  - fusion_stacked correctness (binary; if fusion_head + parquet exist)

It is READ-ONLY. It does not modify out_dir.

Usage
-----
python3 replay_predictions_check_v4_fixed.py --out_dir <stage3-out> --tol 1e-5

Notes
-----
- This script assumes the repo's checkpoint format:
    {"model_state_dict": ..., "cfg": ..., "meta": ...}
  but it also supports legacy state-dict-only bundles.
- If a particular parquet doesn't exist, it's skipped (not an error),
  because fusion outputs are optional depending on config.
- If <out_dir>/stage3_run_config.json is missing (older runs), you may pass
  --processed_dir to point at your preprocessing output.
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
from stage3_models import (
    make_gru_binary,
    make_gru_multiclass,
    make_resmlp_binary,
    make_resmlp_multiclass,
    make_tcn_binary,
    make_tcn_multiclass,
    make_transformer_binary,
    make_transformer_multiclass,
)
from stage3_fusion import fusion_mean_binary, fusion_mean_multiclass

try:
    import torch
except Exception as e:  # pragma: no cover
    print(f"[FATAL] torch import failed: {e}")
    raise

try:
    from joblib import load as joblib_load  # type: ignore
except Exception:  # pragma: no cover
    joblib_load = None  # type: ignore


def _clip01(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.where(np.isfinite(p), p, 0.5)
    return np.clip(p, eps, 1.0 - eps)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip01(np.asarray(p, dtype=np.float64))
    return np.log(p / (1.0 - p))


def _stacked_binary_predict(head: Any, p_net: np.ndarray, p_rad: np.ndarray) -> np.ndarray:
    """Compute stacked-fusion probability given a saved fusion head.

    Supports:
      - stage3_fusion.StackedFusionBinary (has .predict(p_net, p_rad))
      - sklearn LogisticRegression (has .predict_proba(X))
      - wrappers with .model.predict_proba

    IMPORTANT:
      The Stage-3 pipeline feeds float32 probability scores into the fusion head.
      Upcasting to float64 before the logit transform can shift outputs by ~1e-3
      when base probabilities are saturated near 0/1 (logit is extremely steep),
      creating false replay failures. So we keep the fusion inputs in float32.
    """
    p_net = np.asarray(p_net).reshape(-1)
    p_rad = np.asarray(p_rad).reshape(-1)

    # Preferred: Stage-3 StackedFusionBinary wrapper
    if hasattr(head, "predict") and callable(getattr(head, "predict")):
        try:
            out = head.predict(p_net.astype(np.float32), p_rad.astype(np.float32))
            return np.asarray(out, dtype=np.float32).reshape(-1)
        except TypeError:
            pass

    # sklearn estimator directly
    if hasattr(head, "predict_proba") and callable(getattr(head, "predict_proba")):
        X = np.stack([_logit(p_net).astype(np.float32), _logit(p_rad).astype(np.float32)], axis=1).astype(np.float32)
        proba = head.predict_proba(X)
        return np.asarray(proba, dtype=np.float32)[:, 1]

    # wrapper with .model
    mdl = getattr(head, "model", None)
    if mdl is not None and hasattr(mdl, "predict_proba"):
        X = np.stack([_logit(p_net).astype(np.float32), _logit(p_rad).astype(np.float32)], axis=1).astype(np.float32)
        proba = mdl.predict_proba(X)
        return np.asarray(proba, dtype=np.float32)[:, 1]

    raise TypeError(f"Unsupported fusion head type: {type(head)}")

def _safe_torch_load(path: Path) -> Any:
    """Safer torch.load defaults (future compatible)."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _extract_state_dict(ckpt_obj: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, Any]]:
    """Return (state_dict, cfg_dict, meta_dict) from checkpoint."""
    if isinstance(ckpt_obj, dict):
        if "model_state_dict" in ckpt_obj:
            state = ckpt_obj["model_state_dict"]
            cfg = ckpt_obj.get("cfg", {}) or {}
            meta = ckpt_obj.get("meta", {}) or {}
            return state, cfg, meta
        if "state_dict" in ckpt_obj:
            return ckpt_obj["state_dict"], {}, {}

    # plain state_dict
    if isinstance(ckpt_obj, dict) and all(isinstance(k, str) for k in ckpt_obj.keys()):
        # type: ignore[return-value]
        return ckpt_obj, {}, {}

    raise TypeError(f"Unrecognized checkpoint format: {type(ckpt_obj)}")


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


def _predict_proba_with_optional_groups(model: Any, X: np.ndarray, groups: Any) -> np.ndarray:
    """Call predict_proba with (optional) groups kwarg."""
    try:
        sig = inspect.signature(model.predict_proba)
        if "groups" in sig.parameters:
            return model.predict_proba(X, groups=groups)
    except Exception:
        pass
    return model.predict_proba(X)


def _bundle_factory(model_name: str, task: str):
    model_name = str(model_name).strip().lower()
    task = str(task).strip().lower()
    if model_name == "gru":
        return make_gru_binary if task == "binary" else make_gru_multiclass
    if model_name == "tcn":
        return make_tcn_binary if task == "binary" else make_tcn_multiclass
    if model_name == "transformer":
        return make_transformer_binary if task == "binary" else make_transformer_multiclass
    if model_name == "resmlp":
        return make_resmlp_binary if task == "binary" else make_resmlp_multiclass
    raise ValueError(f"Unsupported model_name={model_name!r}")


def _ensure_wrapper_net(wrapper: Any, *, input_dim: int, out_dim: int) -> None:
    """Initialize wrapper.net so we can load weights without calling fit()."""
    import stage3_torch as st

    # ResMLP wrappers
    if wrapper.__class__.__name__ in ("TorchResMLPBinary", "TorchResMLPMulticlass"):
        wrapper.net = st.ResMLPTabular(
            input_dim=int(input_dim),
            d_model=int(wrapper.cfg.d_model),
            n_blocks=int(wrapper.cfg.n_blocks),
            mlp_ratio=float(wrapper.cfg.mlp_ratio),
            dropout=float(wrapper.cfg.dropout),
            out_dim=int(out_dim),
            use_grn=bool(getattr(wrapper.cfg, "use_grn", False)),
            grn_eps=float(getattr(wrapper.cfg, "grn_eps", 1e-3)),
        ).to(wrapper.device)
        wrapper._is_fitted = True
        return

    # Sequence wrappers (GRU/TCN/Transformer)
    pad = 1 if bool(getattr(wrapper.cfg, "add_pad_indicator", True)) else 0
    seq_input_dim = int(input_dim) + pad

    if wrapper.__class__.__name__ in ("TorchGRUBinary", "TorchGRUMulticlass"):
        wrapper.net = st.GRUSeqClassifier(
            input_dim=seq_input_dim,
            d_model=int(wrapper.cfg.d_model),
            n_layers=int(wrapper.cfg.n_layers),
            dropout=float(wrapper.cfg.dropout),
            out_dim=int(out_dim),
        ).to(wrapper.device)
        wrapper._is_fitted = True
        return

    if wrapper.__class__.__name__ in ("TorchTCNBinary", "TorchTCNMulticlass"):
        wrapper.net = st.TCNSeqClassifier(
            input_dim=seq_input_dim,
            channels=int(wrapper.cfg.d_model),
            n_blocks=int(wrapper.cfg.n_blocks),
            kernel_size=int(wrapper.cfg.kernel_size),
            dropout=float(wrapper.cfg.dropout),
            out_dim=int(out_dim),
            use_weight_norm=bool(getattr(wrapper.cfg, "use_weight_norm", True)),
            norm=str(getattr(wrapper.cfg, "norm", "group")),
        ).to(wrapper.device)
        wrapper._is_fitted = True
        return

    if wrapper.__class__.__name__ in ("TorchTransformerBinary", "TorchTransformerMulticlass"):
        wrapper.net = st.TransformerSeqClassifier(
            input_dim=seq_input_dim,
            d_model=int(wrapper.cfg.d_model),
            n_layers=int(wrapper.cfg.n_layers),
            n_heads=int(getattr(wrapper.cfg, "n_heads", 4)),
            ff_mult=float(getattr(wrapper.cfg, "ff_mult", 2.0)),
            dropout=float(wrapper.cfg.dropout),
            out_dim=int(out_dim),
            seq_len=int(getattr(wrapper.cfg, "seq_len", 8)),
            has_pad_indicator=bool(getattr(wrapper.cfg, "add_pad_indicator", True)),
        ).to(wrapper.device)
        wrapper._is_fitted = True
        return

    raise ValueError(f"Don't know how to init net for wrapper class {wrapper.__class__.__name__}")


def _find_fusion_head_binary(out_dir: Path, split: str, ws: str, model: str) -> Path | None:
    """Locate fusion_head_binary.joblib for the given run unit."""
    base = out_dir / "models" / split / ws / model
    candidates = [
        base / "binary" / "fusion_head_binary.joblib",
        base / "fusion_head_binary.joblib",
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback: any nested match
    hits = sorted(base.rglob("fusion_head_binary.joblib"))
    return hits[0] if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--tol", default=1e-5, type=float)
    ap.add_argument("--limit", default=0, type=int, help="If >0, limit number of run units checked.")
    ap.add_argument(
        "--processed_dir",
        default="",
        type=str,
        help="Fallback processed_dir if stage3_run_config.json is missing.",
    )
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
            print(
                f"[WARN] {run_cfg_path} missing; using --processed_dir={processed_dir} "
                "and assuming feature_ablation=full."
            )
        else:
            raise FileNotFoundError(
                f"Missing {run_cfg_path}. Pass an out_dir produced by run_stage3_tabular.py, "
                "or provide --processed_dir to replay against an older output directory."
            )

    ckpt_root = out_dir / "checkpoints"
    pred_root = out_dir / "predictions"
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Missing checkpoints dir: {ckpt_root}")
    if not pred_root.exists():
        raise FileNotFoundError(f"Missing predictions dir: {pred_root}")

    ckpt_files = sorted(ckpt_root.rglob("*.pt"))
    if not ckpt_files:
        print("[WARN] No checkpoints found.")
        return

    # Group checkpoints by (split, W_S, model)
    grouped: Dict[Tuple[str, str, str], Dict[str, Path]] = {}
    for p in ckpt_files:
        # .../checkpoints/<split>/W10_S2/<model>/<system>_<task>.pt
        try:
            split = p.parents[2].name
            ws = p.parents[1].name
            model = p.parents[0].name
            stem = p.stem  # e.g., "network_binary"
            system, task = stem.split("_", 1)
        except Exception:
            continue
        key = (split, ws, model)
        grouped.setdefault(key, {})[f"{system}_{task}"] = p

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

        # Load processed data once (val/test) and align modalities.
        net_va = load_processed(processed_dir, split, "network", W, S, "val", feature_ablation=feature_ablation_cfg)
        net_te = load_processed(processed_dir, split, "network", W, S, "test", feature_ablation=feature_ablation_cfg)
        rad_va = load_processed(processed_dir, split, "radio", W, S, "val", feature_ablation=feature_ablation_cfg)
        rad_te = load_processed(processed_dir, split, "radio", W, S, "test", feature_ablation=feature_ablation_cfg)
        net_va, rad_va = align_modalities(net_va, rad_va)
        net_te, rad_te = align_modalities(net_te, rad_te)

        def _load_wrapper(ckpt_path: Path, proc_split, *, task: str, n_classes: int | None) -> Any:
            ckpt_obj = _safe_torch_load(ckpt_path)
            state, cfg_dict, _meta = _extract_state_dict(ckpt_obj)

            if task == "multiclass":
                assert n_classes is not None
                bundle = _bundle_factory(model, task)(cfg_dict, n_classes=int(n_classes), seed=0)
                out_dim = int(n_classes)
            else:
                bundle = _bundle_factory(model, task)(cfg_dict, seed=0)
                out_dim = 1

            w = bundle.model
            _ensure_wrapper_net(w, input_dim=int(proc_split.X.shape[1]), out_dim=int(out_dim))
            w.net.load_state_dict(state, strict=True)
            return w

        def _check_task(task: str) -> None:
            nonlocal n_ok, n_fail

            task = task.lower().strip()
            task_pred_dir = pred_root / split / ws / model / task

            # For multiclass we need the class list to reconstruct y_pred.
            classes: list[str] | None = None
            n_classes: int | None = None
            if task == "multiclass":
                classes_path = out_dir / "models" / split / ws / model / "multiclass_classes.json"
                if not classes_path.exists():
                    raise FileNotFoundError(f"Missing classes file: {classes_path}")
                classes = json.loads(classes_path.read_text())
                n_classes = int(len(classes))

            # Base models: network_only + radio_only
            wrappers: Dict[str, Any] = {}
            for sys_name, proc_val, proc_test in [
                ("network", net_va, net_te),
                ("radio", rad_va, rad_te),
            ]:
                ckpt_key = f"{sys_name}_{task}"
                ckpt_path = grouped[(split, ws, model)].get(ckpt_key)
                if ckpt_path is None:
                    continue

                w = _load_wrapper(ckpt_path, proc_val, task=task, n_classes=n_classes)
                wrappers[sys_name] = w

                # Compare VAL and TEST
                for part, proc_part in [("val", proc_val), ("test", proc_test)]:
                    name = f"{sys_name}_only"
                    pred_path = task_pred_dir / f"{name}_{part}.parquet"
                    if not pred_path.exists():
                        continue

                    df = pd.read_parquet(pred_path)
                    df = _align_pred_df_to_processed(df, proc_part.df)

                    if task == "binary":
                        p = _predict_proba_with_optional_groups(w, proc_part.X, proc_part.groups)[:, 1]
                        p = np.where(np.isfinite(p), p, 0.5).astype(np.float32)
                        diff = _max_abs_diff(p, df["score"].to_numpy(dtype=float))
                        if diff > tol:
                            print(f"[FAIL] {split}/{ws}/{model} {name} {part} max|diff|={diff:.3g} > tol={tol}")
                            n_fail += 1
                        else:
                            n_ok += 1
                    else:
                        assert classes is not None
                        P = _predict_proba_with_optional_groups(w, proc_part.X, proc_part.groups)
                        P = np.where(np.isfinite(P), P, 0.0).astype(np.float32)
                        row_sums = P.sum(axis=1, keepdims=True)
                        uniform = np.float32(1.0 / max(P.shape[1], 1))
                        P = np.where(row_sums > 0, P / np.where(row_sums > 0, row_sums, 1.0), uniform)
                        p_max = np.max(P, axis=1)
                        y_pred = np.asarray(classes, dtype=str)[np.argmax(P, axis=1)]
                        diff = _max_abs_diff(p_max, df["p_max"].to_numpy(dtype=float))
                        same = np.array_equal(y_pred.astype(str), df["y_pred"].astype(str).to_numpy())

                        # If per-class probability columns are present, compare them too.
                        proba_cols = [c for c in df.columns if c.startswith("p_") and c not in ("p_max",)]
                        if proba_cols:
                            used = set()
                            exp_cols: list[str] = []
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
                                exp_cols.append(col)
                            for j, col in enumerate(exp_cols):
                                if col in df.columns:
                                    dcol = _max_abs_diff(P[:, j], df[col].to_numpy(dtype=float))
                                    diff = max(diff, dcol)

                        if (diff > tol) or (not same):
                            print(
                                f"[FAIL] {split}/{ws}/{model} {name} {part} "
                                f"p_max max|diff|={diff:.3g} (tol={tol}) y_pred_match={same}"
                            )
                            n_fail += 1
                        else:
                            n_ok += 1

            # Mean fusion check (if parquet exists and both base wrappers exist)
            if "network" in wrappers and "radio" in wrappers:
                for part, (proc_net, proc_rad, proc_ref) in [
                    ("val", (net_va, rad_va, net_va)),
                    ("test", (net_te, rad_te, net_te)),
                ]:
                    pred_path = task_pred_dir / f"fusion_mean_{part}.parquet"
                    if not pred_path.exists():
                        continue

                    w_net = wrappers["network"]
                    w_rad = wrappers["radio"]

                    if task == "binary":
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
                            Pn = _predict_proba_with_optional_groups(w_net, proc_net.X, proc_net.groups)
                            Pr = _predict_proba_with_optional_groups(w_rad, proc_rad.X, proc_rad.groups)
                            p_mean = fusion_mean_binary(Pn, Pr)[:, 1]

                        df = pd.read_parquet(pred_path)
                        df = _align_pred_df_to_processed(df, proc_ref.df)
                        diff = _max_abs_diff(p_mean, df["score"].to_numpy(dtype=float))
                        if diff > tol:
                            print(f"[FAIL] {split}/{ws}/{model} fusion_mean {part} max|diff|={diff:.3g} > tol={tol}")
                            n_fail += 1
                        else:
                            n_ok += 1
                    else:
                        assert classes is not None
                        # Use saved base parquets for fusion replay (robust to model divergence/GPU non-determinism).
                        # Contract: fusion_mean = fusion_mean_multiclass(network_only, radio_only)
                        net_base_path = task_pred_dir / f"network_only_{part}.parquet"
                        rad_base_path = task_pred_dir / f"radio_only_{part}.parquet"

                        # Build expected per-class column names (must match training pipeline)
                        used_cols: set[str] = set()
                        exp_cols: list[str] = []
                        for j, raw in enumerate(classes):
                            col = re.sub(r"[^0-9A-Za-z_]+", "_", str(raw).strip())
                            if col == "":
                                col = f"cls{j}"
                            col = f"p_{col}"
                            base = col
                            k = 1
                            while col in used_cols:
                                k += 1
                                col = f"{base}__{k}"
                            used_cols.add(col)
                            exp_cols.append(col)

                        if net_base_path.exists() and rad_base_path.exists():
                            df_net_base = pd.read_parquet(net_base_path)
                            df_rad_base = pd.read_parquet(rad_base_path)
                            df_net_base = _align_pred_df_to_processed(df_net_base, proc_ref.df)
                            df_rad_base = _align_pred_df_to_processed(df_rad_base, proc_ref.df)
                            Pn = df_net_base[exp_cols].to_numpy(dtype=np.float32)
                            Pr = df_rad_base[exp_cols].to_numpy(dtype=np.float32)
                        else:
                            # Fallback: re-run model (may fail for diverged models)
                            Pn = _predict_proba_with_optional_groups(w_net, proc_net.X, proc_net.groups)
                            Pr = _predict_proba_with_optional_groups(w_rad, proc_rad.X, proc_rad.groups)
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

                        proba_cols = [c for c in df.columns if c.startswith("p_") and c not in ("p_max",)]
                        if proba_cols:
                            for j, col in enumerate(exp_cols):
                                if col in df.columns:
                                    dcol = _max_abs_diff(Pm[:, j], df[col].to_numpy(dtype=float))
                                    diff = max(diff, dcol)

                        if (diff > tol) or (not same):
                            print(
                                f"[FAIL] {split}/{ws}/{model} fusion_mean {part} "
                                f"p_max max|diff|={diff:.3g} (tol={tol}) y_pred_match={same}"
                            )
                            n_fail += 1
                        else:
                            n_ok += 1

                        # Stacked fusion replay (binary only)
            # NOTE: Stacked fusion is *very* sensitive to tiny differences in base probabilities
            # because it operates in logit space. For a strict artifact contract, we validate that
            # the saved stacked parquet equals head.predict() applied to the saved base parquets.
            # (This avoids false failures when checkpoint-replayed base probs differ by ~1e-6..1e-5.)
            if task == "binary":
                head_path = out_dir / "models" / split / ws / model / "fusion_head_binary.joblib"
                if head_path.exists() and ("network" in wrappers and "radio" in wrappers):
                    if joblib_load is None:
                        print(f"[WARN] joblib not available; skipping fusion_stacked replay for {split}/{ws}/{model}")
                    else:
                        try:
                            head = joblib_load(str(head_path))
                        except Exception as e:  # pragma: no cover
                            print(f"[WARN] Could not load fusion head {head_path}: {e}")
                        else:
                            w_net = wrappers["network"]
                            w_rad = wrappers["radio"]

                            for part, (proc_net, proc_rad, proc_ref) in [
                                ("val", (net_va, rad_va, net_va)),
                                ("test", (net_te, rad_te, net_te)),
                            ]:
                                pred_path = task_pred_dir / f"fusion_stacked_{part}.parquet"
                                if not pred_path.exists():
                                    continue

                                # Prefer saved base parquets as fusion inputs (strict artifact consistency).
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
                                    # Fallback (should be rare): use replayed base probs.
                                    Pn = _predict_proba_with_optional_groups(w_net, proc_net.X, proc_net.groups)
                                    Pr = _predict_proba_with_optional_groups(w_rad, proc_rad.X, proc_rad.groups)
                                    p_net = np.where(np.isfinite(Pn[:, 1]), Pn[:, 1], 0.5).astype(np.float32)
                                    p_rad = np.where(np.isfinite(Pr[:, 1]), Pr[:, 1], 0.5).astype(np.float32)
                                
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
# Binary always expected
        _check_task("binary")

        # Multiclass checkpoints may not exist (time_ordered splits intentionally skip multiclass).
        if any(k.endswith("_multiclass") for k in grouped[(split, ws, model)].keys()):
            _check_task("multiclass")

    if n_fail:
        print(f"[SUMMARY] FAIL={n_fail} OK={n_ok}")
        sys.exit(2)

    print(f"[SUMMARY] ALL OK ({n_ok} checks, tol={tol})")


if __name__ == "__main__":
    main()
