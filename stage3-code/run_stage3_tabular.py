#!/usr/bin/env python3
"""run_stage3_tabular.py

Stage 3 v3 (Tabular training + evaluation)

What this script does
- Loads Stage-2 processed window datasets (network + radio) for each split and window config.
- Trains tabular baselines (LogReg, XGBoost).
- Evaluates BOTH VAL and TEST so you can:
    * select headline configs on VAL (pre-declared rule)
    * report TEST mean±std across seeds without test-set cherry picking

Binary task
- Threshold-free metrics (ROC-AUC, PR-AUC, ECE, Brier, log-loss)
- Operating-point metrics at FPR targets using TRAIN OOF score quantile thresholds
  (with configurable threshold policies)
- Time-to-detect (TTD) at the same operating point
  * Primary TTD: flow-onset (run-level onset from earliest attack flow)
  * Sensitivity TTD: window-onset (first positive window)
  * Export ΔTTD distribution (window - flow) over detected runs
- Late fusion: network-only, radio-only, mean-prob, (optional) proper stacked fusion

Multiclass task
- Macro-F1 / weighted-F1 / accuracy / log-loss
- Late fusion: network-only, radio-only, mean-prob

Outputs (out_dir)
- metrics/metrics_binary.csv
- metrics/binary_operating_metrics_fpr0.01.csv (and one file per fpr target)
- metrics/ttd_summary_fpr0.01.csv (and one file per fpr target)
- metrics/metrics_multiclass.csv
- predictions/<split>/W{W}_S{S}/{model}/{binary|multiclass}/*_{val|test}.parquet
- models/<split>/W{W}_S{S}/{model}/... (optional)
- artifacts/<split>/W{W}_S{S}/{model}/run_artifact.json

Optional extra outputs (if enabled)
- ttd_runs/<split>/W{W}_S{S}/{model}/binary/<system>_...csv (per-run TTD details)

Note
This script assumes Stage-2 preprocessing already produced:
  <processed_dir>/processed/<split_name>/{network|radio}_W{W}_S{S}_{train|val|test}.parquet
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stage3_utils import ensure_dir, read_yaml, write_json
from stage3_io import load_processed, align_modalities, join_meta
from stage3_models import (
    make_logreg_binary,
    make_logreg_multiclass,
    make_xgb_binary,
    make_xgb_multiclass,
    make_rf_binary,
    make_rf_multiclass,
    make_resmlp_binary,
    make_resmlp_multiclass,
    make_gru_binary,
    make_gru_multiclass,
    make_tcn_binary,
    make_tcn_multiclass,
    make_transformer_binary,
    make_transformer_multiclass,
    predict_proba_binary,
    predict_proba_multiclass,
)
from stage3_metrics import (
    binary_metrics,
    multiclass_metrics,
    sanitize_multiclass_proba,
    operating_point,
    threshold_from_oof_policy,
)
from stage3_fusion import (
    fusion_mean_binary,
    fusion_mean_multiclass,
    train_stacked_binary,
)
from stage3_oof import make_run_folds_binary, iter_fold_indices
from stage3_ttd import (
    compute_ttd_flow_onset,
    compute_ttd_window_onset,
    summarize_ttd,
    summarize_ttd_delta,
)
from stage3_onset import OnsetMap, load_onset_map_from_run_summary, validate_onset_map_against_meta


# ----------------------------
# Small helpers
# ----------------------------


# ----------------------------
# Config resolution (task/modality overrides)
# ----------------------------

# This mirrors stage3_hpo.resolve_model_cfg so a single YAML can hold per-task/per-modality
# hyperparameters (e.g., Optuna best YAML), while remaining backward compatible.

_RESERVED_CFG_KEYS = {"binary", "multiclass", "network", "radio"}


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _resolve_model_cfg_fallback(model_cfg: Dict[str, Any], *, task: str, modality: str) -> Dict[str, Any]:
    base = {k: v for k, v in dict(model_cfg).items() if k not in _RESERVED_CFG_KEYS}

    task_cfg = _as_dict(model_cfg.get(task))
    task_base = {k: v for k, v in dict(task_cfg).items() if k not in ("network", "radio")}

    mod_cfg = _as_dict(task_cfg.get(modality)) if isinstance(task_cfg, dict) else {}
    if not mod_cfg:
        mod_cfg = _as_dict(model_cfg.get(modality))

    out = dict(base)
    out.update(task_base)
    out.update(mod_cfg)
    return out


# Prefer the canonical resolver if stage3_hpo is present; else fall back.
try:  # pragma: no cover
    from stage3_hpo import resolve_model_cfg as _resolve_model_cfg  # type: ignore
except Exception:  # pragma: no cover
    _resolve_model_cfg = None

def _sanitize(p: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0.5 (neutral probability) — guards against model divergence."""
    p = np.asarray(p, dtype=np.float32)
    return np.where(np.isfinite(p), p, np.float32(0.5))


def _sanitize_matrix(P: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf in multiclass probability matrix and renormalize rows.

    Single source of truth: stage3_metrics.sanitize_multiclass_proba.
    We pass eps=0.0 here to preserve exact 0/1 values when present; the
    multiclass_metrics() function applies eps-clipping when computing log-loss.
    """
    return sanitize_multiclass_proba(P, eps=0.0).astype(np.float32, copy=False)


def resolve_model_cfg(model_cfg: Dict[str, Any], *, task: str, modality: str) -> Dict[str, Any]:
    if _resolve_model_cfg is not None:
        try:
            return _resolve_model_cfg(model_cfg, task=task, modality=modality)  # type: ignore
        except Exception:
            return _resolve_model_cfg_fallback(model_cfg, task=task, modality=modality)
    return _resolve_model_cfg_fallback(model_cfg, task=task, modality=modality)


def _safe_seed_from_split(split_name: str, default: int = 0) -> int:
    m = re.search(r"seed(\d+)", split_name)
    if m:
        return int(m.group(1))
    return int(default)


def _discover_splits(processed_dir: Path, glob_pat: str) -> List[str]:
    base = processed_dir / "processed"
    if not base.exists():
        raise FileNotFoundError(f"processed_dir must contain a 'processed/' subdir: {base}")
    out = [p.name for p in base.glob(glob_pat) if p.is_dir()]
    out = sorted(set(out))
    return out


def _bundle_is_torch(bundle: Any) -> bool:
    """Heuristic: whether this ModelBundle wraps a torch model (stage3_torch wrappers).

    We intentionally avoid relying on a 'kind' attribute because ModelBundle in this repo
    does not define one consistently.

    Returns True when:
    - bundle.model lives in the 'stage3_torch' module (our wrappers), OR
    - bundle.model has a '.net' attribute that looks like a torch.nn.Module.
    """
    m = getattr(bundle, "model", None)
    mod = getattr(m, "__module__", "") or ""
    if (mod.split(".")[-1] == "stage3_torch") or mod.startswith("stage3_torch"):
        return True
    net = getattr(m, "net", None)
    if net is None:
        return False
    try:
        import torch  # type: ignore
        return isinstance(net, torch.nn.Module)
    except Exception:
        # If torch isn't importable here but a '.net' exists, treat as torch-like.
        return True


def _oof_predict_binary_with_row_folds(
    X: np.ndarray,
    y: np.ndarray,
    row_folds: List[Tuple[np.ndarray, np.ndarray]],
    build_model_fn,
    groups_all: Optional[np.ndarray] = None,
    group_strata_all: Optional[np.ndarray] = None,
) -> np.ndarray:
    """OOF scores for TRAIN using precomputed row folds."""
    oof = np.zeros((X.shape[0],), dtype=np.float32)
    for fi, (tr_rows, te_rows) in enumerate(row_folds):
        m = build_model_fn()
        _fit_with_optional_groups(
            m.model,
            X[tr_rows],
            y[tr_rows],
            groups=(groups_all[tr_rows] if groups_all is not None else None),
            group_strata=(group_strata_all[tr_rows] if group_strata_all is not None else None),
        )
        oof[te_rows] = predict_proba_binary(m, X[te_rows], groups=(groups_all[te_rows] if groups_all is not None else None))
    return oof


def _fit_with_optional_groups(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
    group_strata: Optional[np.ndarray] = None,
) -> None:
    """Call model.fit with optional group-aware kwargs if supported.

    - sklearn estimators generally do not accept a 'groups' or 'group_strata' kwarg.
    - Torch wrappers *do* (used for run-level early stopping split).
    """

    if (groups is None) and (group_strata is None):
        model.fit(X, y)
        return

    try:
        sig = inspect.signature(model.fit)
        kwargs: Dict[str, Any] = {}
        if (groups is not None) and ("groups" in sig.parameters):
            kwargs["groups"] = groups
        if (group_strata is not None) and ("group_strata" in sig.parameters):
            kwargs["group_strata"] = group_strata

        if kwargs:
            model.fit(X, y, **kwargs)
        else:
            model.fit(X, y)
    except TypeError as e:
        msg = str(e)
        # Only fall back when the error is due to unsupported group kwargs.
        if ('unexpected keyword argument' in msg) and (('groups' in msg) or ('group_strata' in msg)):
            model.fit(X, y)
        else:
            raise


def _oof_stacked_scores_binary(
    p_net_oof: np.ndarray,
    p_rad_oof: np.ndarray,
    y: np.ndarray,
    row_folds: List[Tuple[np.ndarray, np.ndarray]],
    C: float,
    seed: int,
) -> np.ndarray:
    """Cross-fitted stacked scores on TRAIN.

    For each fold:
      - train fusion head on other folds' OOF features
      - predict on this fold

    This keeps stacked scores out-of-sample for BOTH the base models and the fusion head.
    """

    oof_stack = np.zeros_like(p_net_oof, dtype=np.float32)
    for fi, (tr_rows, te_rows) in enumerate(row_folds):
        f = train_stacked_binary(
            p_net_oof[tr_rows],
            p_rad_oof[tr_rows],
            y[tr_rows],
            C=C,
            seed=seed + fi,
        )
        oof_stack[te_rows] = f.predict(p_net_oof[te_rows], p_rad_oof[te_rows])
    return oof_stack


def _save_predictions_binary(
    out_dir: Path,
    meta: pd.DataFrame,
    y_true: np.ndarray,
    scores: np.ndarray,
    name: str,
    part: str,
) -> None:
    df = meta.copy()
    df["y_true"] = y_true.astype(np.int8)
    df["score"] = scores.astype(np.float32)
    ensure_dir(out_dir)
    df.to_parquet(out_dir / f"{name}_{part}.parquet", index=False)


def _save_predictions_multiclass(
    out_dir: Path,
    meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pmax: np.ndarray,
    name: str,
    part: str,
    *,
    proba: np.ndarray | None = None,
    classes: np.ndarray | None = None,
) -> None:
    """Save multiclass predictions.

    Always saves:
      - y_true (string)
      - y_pred (string)
      - p_max  (float32)

    Optionally (analysis-ready): if `proba` (N,K) and `classes` (K,) are provided,
    saves per-class probabilities in columns named p_<class>.

    We sanitize class names for column-safety and deduplicate if needed.
    """

    df = meta.copy()
    df["y_true"] = y_true.astype(str)
    df["y_pred"] = y_pred.astype(str)
    df["p_max"] = pmax.astype(np.float32)

    if proba is not None and classes is not None:
        P = np.asarray(proba)
        cls = np.asarray(classes).astype(str)
        if P.ndim != 2:
            raise ValueError(f"proba must be 2D (N,K); got shape {P.shape}")
        if P.shape[1] != cls.shape[0]:
            raise ValueError(f"proba K mismatch: P has {P.shape[1]} cols but classes has {cls.shape[0]}")

        used = set()
        for j, raw in enumerate(cls.tolist()):
            # keep columns stable & parquet-friendly
            col = re.sub(r"[^0-9A-Za-z_]+", "_", str(raw).strip())
            if col == "":
                col = f"cls{j}"
            col = f"p_{col}"
            # de-duplicate if sanitization collides
            base = col
            k = 1
            while col in used:
                k += 1
                col = f"{base}__{k}"
            used.add(col)
            df[col] = P[:, j].astype(np.float32)

    ensure_dir(out_dir)
    df.to_parquet(out_dir / f"{name}_{part}.parquet", index=False)


def _ttd_list_to_df(ttd: List[Any], prefix: str) -> pd.DataFrame:
    """Convert List[TTDResult] to a DataFrame with a prefix."""
    rows = []
    for r in ttd:
        rows.append({
            "run_id": r.run_id,
            f"{prefix}_detected": bool(r.detected),
            f"{prefix}_onset_s": float(r.onset_s),
            f"{prefix}_detect_s": float(r.detect_s) if np.isfinite(r.detect_s) else np.nan,
            f"{prefix}_ttd_s": float(r.ttd_s),
        })
    return pd.DataFrame(rows)


# ----------------------------
# Core runner for one split/W,S/model
# ----------------------------


def run_one(
    processed_dir: Path,
    out_dir: Path,
    split_name: str,
    W: int,
    S: int,
    model_name: str,
    model_cfg: Dict[str, Any],
    run_binary: bool,
    run_multiclass: bool,
    fpr_targets: List[float],
    threshold_policies: List[str],
    benign_family_name: str,
    folds_binary: int,
    fusion_cfg: Dict[str, Any],
    outputs_cfg: Dict[str, Any],
    feature_ablation_cfg: Optional[Dict[str, Any]] = None,
    onset_map: Optional[OnsetMap] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (binary_metric_rows, multiclass_metric_rows, operating_rows, ttd_rows)."""

    split_seed = _safe_seed_from_split(split_name, default=0)
    is_time_ordered = split_name.startswith("time_ordered")

    # Feature ablation tag (recorded in metrics/artifacts).
    fa_cfg = feature_ablation_cfg or {}
    fa_enabled = bool(fa_cfg.get("enabled", False))
    fa_variant = str(fa_cfg.get("variant", "full"))
    # If not enabled, treat as 'full' for clean grouping.
    fa_tag = fa_variant if fa_enabled else "full"

    # Resolved hyperparameters per task/modality (Optuna best YAML support).
    # These are flat dicts safe to pass into both ML builders (e.g., XGBoost) and Torch builders.
    cfg_bin_net = resolve_model_cfg(model_cfg, task="binary", modality="network")
    cfg_bin_rad = resolve_model_cfg(model_cfg, task="binary", modality="radio")
    cfg_mc_net = resolve_model_cfg(model_cfg, task="multiclass", modality="network")
    cfg_mc_rad = resolve_model_cfg(model_cfg, task="multiclass", modality="radio")

    # Load processed data (train/val/test)
    net_tr = load_processed(processed_dir, split_name, "network", W, S, "train", feature_ablation=feature_ablation_cfg)
    net_va = load_processed(processed_dir, split_name, "network", W, S, "val", feature_ablation=feature_ablation_cfg)
    net_te = load_processed(processed_dir, split_name, "network", W, S, "test", feature_ablation=feature_ablation_cfg)

    rad_tr = load_processed(processed_dir, split_name, "radio", W, S, "train", feature_ablation=feature_ablation_cfg)
    rad_va = load_processed(processed_dir, split_name, "radio", W, S, "val", feature_ablation=feature_ablation_cfg)
    rad_te = load_processed(processed_dir, split_name, "radio", W, S, "test", feature_ablation=feature_ablation_cfg)

    # Align modalities for each split_part
    net_tr, rad_tr = align_modalities(net_tr, rad_tr)
    net_va, rad_va = align_modalities(net_va, rad_va)
    net_te, rad_te = align_modalities(net_te, rad_te)

    # Meta joins for outputs
    meta_tr = join_meta(net_tr.df, rad_tr.df)
    meta_va = join_meta(net_va.df, rad_va.df)
    meta_te = join_meta(net_te.df, rad_te.df)

    # Onset sanity checks (fast)
    if onset_map is not None:
        validate_onset_map_against_meta(meta_tr, onset_map, strict=True)

    bin_rows: List[Dict[str, Any]] = []
    mc_rows: List[Dict[str, Any]] = []
    op_rows: List[Dict[str, Any]] = []
    ttd_rows: List[Dict[str, Any]] = []

    # DL bookkeeping (logs/checkpoints/history) for reproducibility
    dl_logs: Dict[str, str] = {}
    dl_checkpoints: Dict[str, str] = {}
    dl_history: Dict[str, Any] = {}

    def _dl_paths(system: str, task: str) -> Tuple[str, str]:
        """Return (log_path, checkpoint_path) for this split/W/S/model/system/task."""
        log_path = out_dir / "logs" / split_name / f"W{W}_S{S}" / model_name / f"{system}_{task}.log"
        ckpt_path = out_dir / "checkpoints" / split_name / f"W{W}_S{S}" / model_name / f"{system}_{task}.pt"
        ensure_dir(log_path.parent)
        ensure_dir(ckpt_path.parent)
        return str(log_path), str(ckpt_path)

    onset_s_by_run = onset_map.onset_s_by_run if onset_map is not None else None

    folds_binary_used = int(folds_binary)

    # ---------------- Binary ----------------
    if run_binary:
        y_tr = net_tr.y_bin
        y_va = net_va.y_bin
        y_te = net_te.y_bin
        def build_base_binary(cfg_base: Dict[str, Any], *, enable_io: bool = False, system: str = "", task: str = "binary"):
            """Build a base binary model bundle.

            Important: we must NOT inject DL-only keys into ML configs (e.g., XGBoost), because
            XGBoost will error on unexpected kwargs. Therefore log/checkpoint paths are only
            attached for the Torch models.
            """
            if model_name == "logreg":
                return make_logreg_binary(cfg_base)
            if model_name == "xgboost":
                return make_xgb_binary(cfg_base)
            if model_name == "rf":
                return make_rf_binary(cfg_base)
            if model_name in ("resmlp", "gru", "tcn", "transformer"):
                cfg_use = dict(cfg_base)
                if enable_io:
                    lp, cp = _dl_paths(system, task)
                else:
                    lp, cp = None, None
                cfg_use['log_path'] = lp
                cfg_use['checkpoint_path'] = cp
                if enable_io:
                    dl_logs[f"{system}_{task}"] = lp
                    dl_checkpoints[f"{system}_{task}"] = cp
                if enable_io and cfg_use.get('log_path'):
                    print(f"[TRAIN] {split_name} W{W}_S{S} {model_name} {system} {task} -> log: {cfg_use['log_path']}")
                if model_name == 'resmlp':
                    return make_resmlp_binary(cfg_use, seed=split_seed)
                if model_name == 'gru':
                    return make_gru_binary(cfg_use, seed=split_seed)
                if model_name == 'tcn':
                    return make_tcn_binary(cfg_use, seed=split_seed)
                if model_name == 'transformer':
                    return make_transformer_binary(cfg_use, seed=split_seed)
            raise ValueError(f"Unknown base model: {model_name}")

        def build_net_binary(*, enable_io: bool = False, task: str = "binary"):
            return build_base_binary(cfg_bin_net, enable_io=enable_io, system="network", task=task)

        def build_rad_binary(*, enable_io: bool = False, task: str = "binary"):
            return build_base_binary(cfg_bin_rad, enable_io=enable_io, system="radio", task=task)

        def build_net_binary_fold():
            return build_base_binary(cfg_bin_net, enable_io=False, system="network", task="binary")

        def build_rad_binary_fold():
            return build_base_binary(cfg_bin_rad, enable_io=False, system="radio", task="binary")
        


        # Binary run-level strata (benign vs attack) for Torch early stopping split.
        # This matches the stratification logic used by make_run_folds_binary (family==benign_family_name -> 0, else -> 1).
        group_strata_bin_tr: Optional[np.ndarray] = None
        if "family" in net_tr.df.columns:
            fam_norm = net_tr.df["family"].astype(str).str.strip().str.lower()
            benign_norm = str(benign_family_name).strip().lower()
            group_strata_bin_tr = (~fam_norm.eq(benign_norm)).astype(np.int64).to_numpy()
        # Allow per-model override of OOF fold count (useful: ML=5 folds, DL=3 folds).
        # IMPORTANT: fold count must be the same for network and radio because we share the same row_folds.
        folds_binary_eff_net = int(cfg_bin_net.get("folds_binary", cfg_bin_net.get("oof_folds_binary", folds_binary)))
        folds_binary_eff_rad = int(cfg_bin_rad.get("folds_binary", cfg_bin_rad.get("oof_folds_binary", folds_binary)))
        if folds_binary_eff_net != folds_binary_eff_rad:
            raise ValueError(
                "folds_binary mismatch between network and radio configs: "
                f"network={folds_binary_eff_net} radio={folds_binary_eff_rad}. "
                "Please set folds_binary at the root or task-level (not per modality)."
            )
        folds_binary_eff = int(folds_binary_eff_net)

        folds_binary_used = int(folds_binary_eff)

        # Run-level folds for OOF calibration
        run_folds = make_run_folds_binary(net_tr.df, n_splits=int(folds_binary_eff), seed=int(split_seed), benign_family_name=benign_family_name)
        row_folds = iter_fold_indices(net_tr.df, run_folds)

        # OOF base predictions (TRAIN only)
        p_net_oof = _oof_predict_binary_with_row_folds(
            net_tr.X,
            y_tr,
            row_folds,
            build_net_binary_fold,
            groups_all=net_tr.groups,
            group_strata_all=group_strata_bin_tr,
        )
        p_rad_oof = _oof_predict_binary_with_row_folds(
            rad_tr.X,
            y_tr,
            row_folds,
            build_rad_binary_fold,
            groups_all=rad_tr.groups,
            group_strata_all=group_strata_bin_tr,
        )
        # Ensure stacked-fusion uses the exact same float32 base scores that get written to parquet.
        # Without this, sklearn/XGBoost often produce float64, and the logit transform becomes dtype-dependent,
        # causing tiny but contract-breaking drift in fusion_stacked replay.
        p_net_oof = _sanitize(np.asarray(p_net_oof, dtype=np.float32))
        p_rad_oof = _sanitize(np.asarray(p_rad_oof, dtype=np.float32))
        p_mean_oof = fusion_mean_binary(p_net_oof, p_rad_oof)

        # Optional stacked fusion (binary)
        stacked_cfg = fusion_cfg.get("stacked_logreg", {}) if isinstance(fusion_cfg, dict) else {}
        stacked_enabled = bool(stacked_cfg.get("enabled", True))
        C_stack = float(stacked_cfg.get("C", 1.0))
        if stacked_enabled:
            p_stack_oof = _sanitize(_oof_stacked_scores_binary(p_net_oof, p_rad_oof, y_tr, row_folds=row_folds, C=C_stack, seed=split_seed))
            fusion_head = train_stacked_binary(p_net_oof, p_rad_oof, y_tr, C=C_stack, seed=split_seed)
        else:
            p_stack_oof = None
            fusion_head = None

        # Fit base models on full TRAIN
        net_model = build_net_binary(enable_io=True, task="binary")
        _fit_with_optional_groups(net_model.model, net_tr.X, y_tr, groups=net_tr.groups, group_strata=group_strata_bin_tr)
        rad_model = build_rad_binary(enable_io=True, task="binary")
        _fit_with_optional_groups(rad_model.model, rad_tr.X, y_tr, groups=rad_tr.groups, group_strata=group_strata_bin_tr)

        if model_name == "resmlp":
            dl_history["network_binary"] = getattr(net_model.model, "history_", None)
            dl_history["radio_binary"] = getattr(rad_model.model, "history_", None)

        # Predict on VAL/TEST
        p_net_va = predict_proba_binary(net_model, net_va.X, groups=net_va.groups)
        p_net_te = predict_proba_binary(net_model, net_te.X, groups=net_te.groups)

        p_rad_va = predict_proba_binary(rad_model, rad_va.X, groups=rad_va.groups)
        p_rad_te = predict_proba_binary(rad_model, rad_te.X, groups=rad_te.groups)

        # Keep VAL/TEST base probabilities in float32 before any fusion (mean/stacked) so saved parquets and
        # fusion_head.predict(...) see identical inputs (critical for strict replay).
        p_net_va = _sanitize(p_net_va)
        p_net_te = _sanitize(p_net_te)
        p_rad_va = _sanitize(p_rad_va)
        p_rad_te = _sanitize(p_rad_te)
        p_mean_va = fusion_mean_binary(p_net_va, p_rad_va)
        p_mean_te = fusion_mean_binary(p_net_te, p_rad_te)

        if stacked_enabled and fusion_head is not None:
            p_stack_va = _sanitize(fusion_head.predict(p_net_va, p_rad_va))
            p_stack_te = _sanitize(fusion_head.predict(p_net_te, p_rad_te))
        else:
            p_stack_va = None
            p_stack_te = None

        score_map: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
            "network_only": (p_net_oof, p_net_va, p_net_te),
            "radio_only": (p_rad_oof, p_rad_va, p_rad_te),
            "fusion_mean": (p_mean_oof, p_mean_va, p_mean_te),
        }
        if stacked_enabled and p_stack_oof is not None and p_stack_va is not None and p_stack_te is not None:
            score_map["fusion_stacked"] = (p_stack_oof, p_stack_va, p_stack_te)

        # Binary metrics (threshold-free) on VAL + TEST
        for system, (p_oof, p_va, p_te) in score_map.items():
            for part, y_part, p_part, meta_part in [
                ("val", y_va, p_va, meta_va),
                ("test", y_te, p_te, meta_te),
            ]:
                m = binary_metrics(y_part, p_part)
                pr_base = float(m.get("pos_rate", float("nan")))
                pr_auc = float(m.get("pr_auc", float("nan")))
                pr_lift = pr_auc - pr_base if (pr_auc == pr_auc and pr_base == pr_base) else float("nan")
                pr_ratio = (pr_auc / pr_base) if (pr_auc == pr_auc and pr_base == pr_base and pr_base > 0) else float("nan")

                row = {
                    "split": split_name,
                    "seed": split_seed,
                    "time_ordered": bool(is_time_ordered),
                    "W": int(W),
                    "S": int(S),
                    "feature_ablation": fa_tag,
                    "part": part,
                    "task": "binary",
                    "model": model_name,
                    "system": system,
                    **m,
                    "pr_baseline": pr_base,
                    "pr_lift": pr_lift,
                    "pr_ratio": pr_ratio,
                }
                bin_rows.append(row)

                if outputs_cfg.get("save_predictions", True):
                    pred_dir = out_dir / "predictions" / split_name / f"W{W}_S{S}" / model_name / "binary"
                    _save_predictions_binary(pred_dir, meta_part, y_part, p_part, name=system, part=part)

            # Operating point + TTD (VAL + TEST) for each FPR target and threshold policy
            for fpr_t in fpr_targets:
                for thr_policy in threshold_policies:
                    thr, thr_info = threshold_from_oof_policy(
                        meta_train=meta_tr,
                        oof_scores=p_oof,
                        y_train=y_tr,
                        fpr_target=float(fpr_t),
                        policy=str(thr_policy),
                        onset_s_by_run=onset_s_by_run,
                        benign_family_name=str(benign_family_name),
                    )

                    for part, y_part, p_part, meta_part in [
                        ("val", y_va, p_va, meta_va),
                        ("test", y_te, p_te, meta_te),
                    ]:
                        opm = operating_point(y_part, p_part, thr)
                        op_rows.append({
                            "split": split_name,
                            "seed": split_seed,
                            "time_ordered": bool(is_time_ordered),
                            "W": int(W),
                            "S": int(S),
                            "feature_ablation": fa_tag,
                            "part": part,
                            "task": "binary",
                            "model": model_name,
                            "system": system,
                            "fpr_target": float(fpr_t),
                            **{k: thr_info.get(k) for k in ["threshold_policy", "thr_source", "n_selected", "n_total_train_windows"] if k in thr_info},
                            **opm,
                        })

                        # TTD at the same operating point
                        df_ttd = meta_part.copy()
                        if "y_bin" not in df_ttd.columns:
                            df_ttd["y_bin"] = y_part.astype(int)
                        df_ttd["score"] = p_part.astype(float)

                        # window-onset (always available)
                        ttd_win = compute_ttd_window_onset(
                            df_ttd,
                            score_col="score",
                            thr=float(thr),
                            benign_family_name=str(benign_family_name),
                            label_col="y_bin",
                        )
                        win_summ = summarize_ttd(ttd_win)

                        # flow-onset (requires onset map)
                        if onset_s_by_run is not None:
                            ttd_flow = compute_ttd_flow_onset(
                                df_ttd,
                                score_col="score",
                                thr=float(thr),
                                onset_s_by_run=onset_s_by_run,
                                benign_family_name=str(benign_family_name),
                            )
                            flow_summ = summarize_ttd(ttd_flow)
                            delta_summ = summarize_ttd_delta(ttd_flow, ttd_win)
                        else:
                            # If onset map isn't provided, we can still run window-onset.
                            ttd_flow = []
                            flow_summ = {
                                "n_attack_runs": float("nan"),
                                "detect_rate": float("nan"),
                                "ttd_median": float("nan"),
                                "ttd_p25": float("nan"),
                                "ttd_p75": float("nan"),
                            }
                            delta_summ = {
                                "n_delta": float("nan"),
                                "delta_median": float("nan"),
                                "delta_p25": float("nan"),
                                "delta_p75": float("nan"),
                            }

                        base_ttd_row = {
                            "split": split_name,
                            "seed": split_seed,
                            "time_ordered": bool(is_time_ordered),
                            "W": int(W),
                            "S": int(S),
                            "feature_ablation": fa_tag,
                            "part": part,
                            "task": "binary",
                            "model": model_name,
                            "system": system,
                            "fpr_target": float(fpr_t),
                            "threshold": float(thr),
                            "threshold_policy": str(thr_info.get("threshold_policy", thr_policy)),
                            "thr_source": str(thr_info.get("thr_source", "train_oof_quantile")),
                            "thr_n_selected": float(thr_info.get("n_selected", float("nan"))),
                        }

                        ttd_rows.append({
                            **base_ttd_row,
                            "ttd_mode": "flow_onset",
                            **flow_summ,
                            **delta_summ,
                        })
                        ttd_rows.append({
                            **base_ttd_row,
                            "ttd_mode": "window_onset",
                            **win_summ,
                            **{k: float("nan") for k in ["n_delta", "delta_median", "delta_p25", "delta_p75"]},
                        })
                        ttd_rows.append({
                            **base_ttd_row,
                            "ttd_mode": "delta_window_minus_flow",
                            **{k: float("nan") for k in ["n_attack_runs", "detect_rate", "ttd_median", "ttd_p25", "ttd_p75"]},
                            **delta_summ,
                        })

                        # Optional: per-run TTD details (CSV; avoids parquet dependency)
                        if outputs_cfg.get("save_ttd_runs", False) and (onset_s_by_run is not None):
                            try:
                                ttd_dir = ensure_dir(
                                    out_dir
                                    / "ttd_runs"
                                    / split_name
                                    / f"W{W}_S{S}"
                                    / model_name
                                    / "binary"
                                )
                                df_flow = _ttd_list_to_df(ttd_flow, prefix="flow")
                                df_win = _ttd_list_to_df(ttd_win, prefix="window")
                                merged = df_flow.merge(df_win, on="run_id", how="outer")
                                merged["delta_window_minus_flow_s"] = merged["window_ttd_s"] - merged["flow_ttd_s"]
                                fname = f"{system}_{part}_fpr{float(fpr_t)}_{str(thr_info.get('threshold_policy', thr_policy))}.csv"
                                merged.to_csv(ttd_dir / fname, index=False)
                            except Exception as e:
                                # Non-fatal: TTD summaries are still computed above.
                                print(f"[WARN] Failed to save per-run TTD details CSV: {e}")

        # Save models (optional)
        if outputs_cfg.get("save_models", True):
            mdir = ensure_dir(out_dir / "models" / split_name / f"W{W}_S{S}" / model_name)
            # joblib is optional; only save if installed
            try:
                from joblib import dump  # type: ignore

                # Torch model bundles (kind=='dl') are not reliably picklable (e.g., weight_norm parametrizations).
                # For DL, we persist checkpoints/state_dict instead; so only joblib-dump ML bundles here.
                if not _bundle_is_torch(net_model):
                    dump(net_model, mdir / "network_model.joblib")
                if not _bundle_is_torch(rad_model):
                    dump(rad_model, mdir / "radio_model.joblib")
                if stacked_enabled and fusion_head is not None:
                    dump(fusion_head, mdir / "fusion_head_binary.joblib")
            except Exception as e:
                print(f"[WARN] Could not joblib-dump models for {model_name} ({split_name} W{W}_S{S}): {e}")

            # For Torch bundles, also write a .pt with state_dict (joblib often fails / is brittle).
            try:
                import torch  # type: ignore

                def _torch_save_bundle(bundle: Any, path: Path) -> None:
                    if not _bundle_is_torch(bundle):
                        return
                    m = getattr(bundle, "model", None)
                    net = getattr(m, "net", None)
                    if net is None:
                        return
                    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                    torch.save({"state_dict": state}, path)

                _torch_save_bundle(net_model, mdir / "network_model.pt")
                _torch_save_bundle(rad_model, mdir / "radio_model.pt")
            except Exception as e:
                # Do not hard-fail the pipeline on saving issues.
                print(f"[WARN] Could not torch-save model state_dicts for {model_name} ({split_name} W{W}_S{S}): {e}")

    # ---------------- Multiclass ----------------
    # Per locked protocol:
    #   - multiclass only on stratified splits (time-ordered used for binary robustness)
    #   - fusion uses mean-prob only (no multiclass stacking)
    if run_multiclass and (not is_time_ordered):
        y_tr_str = net_tr.y_cat.astype(str)
        y_va_str = net_va.y_cat.astype(str)
        y_te_str = net_te.y_cat.astype(str)

        # Encode labels explicitly so BOTH network and radio models share identical class order.
        from sklearn.preprocessing import LabelEncoder  # local import keeps deps minimal

        le = LabelEncoder()
        le.fit(y_tr_str)
        classes = le.classes_.astype(str)
        n_classes = int(len(classes))

        y_tr = le.transform(y_tr_str)
        y_va = le.transform(y_va_str)
        y_te = le.transform(y_te_str)
        def build_base_mc(cfg_base: Dict[str, Any], n_classes_: int, *, enable_io: bool = False, system: str = "", task: str = "multiclass"):
            """Build a base multiclass model bundle.

            Same caution as binary: do not inject DL-only keys into ML configs that pass kwargs
            (e.g., XGBoost).
            """
            if model_name == "logreg":
                return make_logreg_multiclass(cfg_base)
            if model_name == "xgboost":
                return make_xgb_multiclass(cfg_base, n_classes=int(n_classes_))
            if model_name == "rf":
                return make_rf_multiclass(cfg_base, n_classes=int(n_classes_))
            if model_name in ("resmlp", "gru", "tcn", "transformer"):
                cfg_use = dict(cfg_base)
                if enable_io:
                    lp, cp = _dl_paths(system, task)
                else:
                    lp, cp = None, None
                cfg_use['log_path'] = lp
                cfg_use['checkpoint_path'] = cp
                if enable_io and cfg_use.get('log_path'):
                    print(f"[TRAIN] {split_name} W{W}_S{S} {model_name} {system} {task} -> log: {cfg_use['log_path']}")
                if model_name == 'resmlp':
                    return make_resmlp_multiclass(cfg_use, n_classes=n_classes, seed=split_seed)
                if model_name == 'gru':
                    return make_gru_multiclass(cfg_use, n_classes=n_classes, seed=split_seed)
                if model_name == 'tcn':
                    return make_tcn_multiclass(cfg_use, n_classes=n_classes, seed=split_seed)
                if model_name == 'transformer':
                    return make_transformer_multiclass(cfg_use, n_classes=n_classes, seed=split_seed)
            raise ValueError(f"Unknown base model: {model_name}")

        def build_net_mc(n_classes_: int, *, enable_io: bool = False, task: str = "multiclass"):
            return build_base_mc(cfg_mc_net, n_classes_, enable_io=enable_io, system="network", task=task)

        def build_rad_mc(n_classes_: int, *, enable_io: bool = False, task: str = "multiclass"):
            return build_base_mc(cfg_mc_rad, n_classes_, enable_io=enable_io, system="radio", task=task)

        # Fit on TRAIN (same y encoding for both modalities)
        net_mc = build_net_mc(int(n_classes), enable_io=True, task="multiclass")
        _fit_with_optional_groups(net_mc.model, net_tr.X, y_tr, groups=net_tr.groups, group_strata=y_tr)
        rad_mc = build_rad_mc(int(n_classes), enable_io=True, task="multiclass")
        _fit_with_optional_groups(rad_mc.model, rad_tr.X, y_tr, groups=rad_tr.groups, group_strata=y_tr)

        if model_name == "resmlp":
            dl_history["network_multiclass"] = getattr(net_mc.model, "history_", None)
            dl_history["radio_multiclass"] = getattr(rad_mc.model, "history_", None)

        # Predict on VAL/TEST (prob matrices aligned to LabelEncoder order)
        P_net_va = _sanitize_matrix(predict_proba_multiclass(net_mc, net_va.X, groups=net_va.groups))
        P_net_te = _sanitize_matrix(predict_proba_multiclass(net_mc, net_te.X, groups=net_te.groups))

        P_rad_va = _sanitize_matrix(predict_proba_multiclass(rad_mc, rad_va.X, groups=rad_va.groups))
        P_rad_te = _sanitize_matrix(predict_proba_multiclass(rad_mc, rad_te.X, groups=rad_te.groups))

        # Basic sanity checks
        if P_net_va.shape[1] != n_classes or P_rad_va.shape[1] != n_classes:
            raise ValueError(
                f"Multiclass proba columns mismatch: expected {n_classes}, "
                f"got net={P_net_va.shape[1]} rad={P_rad_va.shape[1]}"
            )

        P_mean_va = fusion_mean_multiclass(P_net_va, P_rad_va)
        P_mean_te = fusion_mean_multiclass(P_net_te, P_rad_te)

        mc_score_map = {
            "network_only": (P_net_va, P_net_te),
            "radio_only": (P_rad_va, P_rad_te),
            "fusion_mean": (P_mean_va, P_mean_te),
        }

        for system, (P_va, P_te) in mc_score_map.items():
            for part, y_part_str, P_part, meta_part in [
                ("val", y_va_str, P_va, meta_va),
                ("test", y_te_str, P_te, meta_te),
            ]:
                mm = multiclass_metrics(y_part_str, P_part, classes=classes)
                row = {
                    "split": split_name,
                    "seed": split_seed,
                    "time_ordered": False,
                    "W": int(W),
                    "S": int(S),
                    "feature_ablation": fa_tag,
                    "part": part,
                    "task": "multiclass",
                    "model": model_name,
                    "system": system,
                    **mm,
                }
                mc_rows.append(row)

                if outputs_cfg.get("save_predictions", True):
                    pred_dir = out_dir / "predictions" / split_name / f"W{W}_S{S}" / model_name / "multiclass"
                    y_pred = classes[np.argmax(P_part, axis=1)]
                    pmax = np.max(P_part, axis=1)
                    _save_predictions_multiclass(
                        pred_dir,
                        meta_part,
                        y_true=y_part_str,
                        y_pred=y_pred,
                        pmax=pmax,
                        name=system,
                        part=part,
                        proba=P_part,
                        classes=classes,
                    )

        if outputs_cfg.get("save_models", True):
            mdir = ensure_dir(out_dir / "models" / split_name / f"W{W}_S{S}" / model_name)
            # Save label encoder classes to guarantee reproducibility (even if model saving fails)
            try:
                with open(mdir / "multiclass_classes.json", "w", encoding="utf-8") as f:
                    json.dump(list(classes), f, indent=2)
            except Exception as e:
                print(f"[WARN] Could not write multiclass_classes.json: {e}")

            try:
                from joblib import dump  # type: ignore

                # Only joblib-dump ML bundles; DL bundles are saved via state_dict/checkpoints.
                if not _bundle_is_torch(net_mc):
                    dump(net_mc, mdir / "network_model_multiclass.joblib")
                if not _bundle_is_torch(rad_mc):
                    dump(rad_mc, mdir / "radio_model_multiclass.joblib")
            except Exception as e:
                print(f"[WARN] Could not joblib-dump multiclass models for {model_name} ({split_name} W{W}_S{S}): {e}")

            # Torch fallback for DL bundles
            try:
                import torch  # type: ignore

                def _torch_save_bundle(bundle: Any, path: Path) -> None:
                    if not _bundle_is_torch(bundle):
                        return
                    m = getattr(bundle, "model", None)
                    net = getattr(m, "net", None)
                    if net is None:
                        return
                    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                    torch.save({"state_dict": state}, path)

                _torch_save_bundle(net_mc, mdir / "network_model_multiclass.pt")
                _torch_save_bundle(rad_mc, mdir / "radio_model_multiclass.pt")
            except Exception as e:
                print(f"[WARN] Could not torch-save multiclass model state_dicts for {model_name} ({split_name} W{W}_S{S}): {e}")

    # ---------------- Artifacts (per split/W/S/model) ----------------

    if bool(outputs_cfg.get("write_run_artifacts", True)):
        try:
            art_dir = ensure_dir(out_dir / "artifacts" / split_name / f"W{W}_S{S}" / model_name)

            def _pos_rate(y: np.ndarray) -> float:
                y = np.asarray(y)
                if y.size == 0:
                    return float("nan")
                return float(np.mean(y.astype(float)))

            payload: Dict[str, Any] = {
                "split": split_name,
                "seed": split_seed,
                "time_ordered": bool(is_time_ordered),
                "W": int(W),
                "S": int(S),
                "feature_ablation": fa_tag,
                "feature_ablation_cfg": fa_cfg if fa_enabled else {},
                "model": model_name,
                "model_cfg": model_cfg,
                # Resolved hyperparameters per task/modality (useful when YAML contains overrides).
                "model_cfg_resolved": {
                    "binary": {
                        "network": cfg_bin_net if run_binary else {},
                        "radio": cfg_bin_rad if run_binary else {},
                    },
                    "multiclass": {
                        "network": cfg_mc_net if (run_multiclass and (not is_time_ordered)) else {},
                        "radio": cfg_mc_rad if (run_multiclass and (not is_time_ordered)) else {},
                    },
                },
                "n_features_network": int(len(net_tr.feature_cols)),
                "n_features_radio": int(len(rad_tr.feature_cols)),
                "feature_cols_network": list(net_tr.feature_cols),
                "feature_cols_radio": list(rad_tr.feature_cols),
                "n_train": int(net_tr.X.shape[0]),
                "n_val": int(net_va.X.shape[0]),
                "n_test": int(net_te.X.shape[0]),
            }
            if run_binary:
                payload.update({
                    "pos_rate_train": _pos_rate(net_tr.y_bin),
                    "pos_rate_val": _pos_rate(net_va.y_bin),
                    "pos_rate_test": _pos_rate(net_te.y_bin),
                    "fpr_targets": [float(x) for x in fpr_targets],
                    "threshold_policies": [str(x) for x in threshold_policies],
                    "folds_binary": int(folds_binary_used),
                    "stacked_enabled": bool(fusion_cfg.get("stacked_logreg", {}).get("enabled", True)),
                    "benign_family_name": str(benign_family_name),
                    "onset_map_provided": bool(onset_map is not None),
                })
            if run_multiclass and (not is_time_ordered):
                payload.update({
                    "multiclass_label": "y_cat",
                    "classes_train": sorted(pd.Series(net_tr.y_cat).dropna().unique().tolist()),
                })

            if model_name == "resmlp":
                # Ensure JSON-serializable types (no numpy scalars)
                def _sanitize_json(obj: Any) -> Any:
                    if obj is None or isinstance(obj, (str, int, float, bool)):
                        return obj
                    if isinstance(obj, np.generic):
                        return obj.item()
                    if isinstance(obj, dict):
                        return {str(k): _sanitize_json(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [_sanitize_json(v) for v in obj]
                    return str(obj)

                payload["dl_logs"] = _sanitize_json(dl_logs)
                payload["dl_checkpoints"] = _sanitize_json(dl_checkpoints)
                payload["dl_history"] = _sanitize_json(dl_history)

            write_json(art_dir / "run_artifact.json", payload)
        except Exception:
            # Artifacts are helpful but must not break the main pipeline.
            pass

    return bin_rows, mc_rows, op_rows, ttd_rows


# ----------------------------
# CLI
# ----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str, help="Path to config_stage3.yaml")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))

    processed_dir = Path(cfg["paths"]["processed_dir"]).expanduser()
    out_dir = ensure_dir(Path(cfg["paths"]["out_dir"]).expanduser())

    windows = cfg["experiment"]["windows"]
    strat_glob = cfg["experiment"].get("stratified_splits_glob", "stratified_seed*")
    time_glob = cfg["experiment"].get("time_ordered_splits_glob", "time_ordered_seed*")

    run_binary = bool(cfg.get("tasks", {}).get("run_binary", True))
    run_multiclass = bool(cfg.get("tasks", {}).get("run_multiclass", True))

    thr_cfg = cfg.get("thresholds", {})
    fpr_targets = list(thr_cfg.get("fpr_targets", [0.01]))
    # You can run all three policies in one pass.
    threshold_policies = list(thr_cfg.get("policies", [thr_cfg.get("policy", "all_benign_labeled")]))

    benign_family_name = str(thr_cfg.get("benign_family_name", "Benign"))

    folds_binary = int(cfg.get("oof", {}).get("folds_binary", 5))

    # Models
    tab_cfg = cfg.get("models", {}).get("tabular", {})
    dl_cfg = cfg.get("models", {}).get("dl", {})
    models_to_run: List[Tuple[str, Dict[str, Any]]] = []
    for mname, mcfg in tab_cfg.items():
        if isinstance(mcfg, dict) and bool(mcfg.get("enabled", False)):
            models_to_run.append((mname, mcfg))

    # Optional deep-learning models (torch backends are imported lazily).
    for mname, mcfg in dl_cfg.items():
        if isinstance(mcfg, dict) and bool(mcfg.get("enabled", False)):
            models_to_run.append((mname, mcfg))

    fusion_cfg = cfg.get("fusion", {})
    outputs_cfg = cfg.get("outputs", {})

    feature_ablation_cfg = cfg.get("feature_ablation", {})

    # Onset map (optional but recommended for flow-onset TTD + pre-attack threshold policy)
    onset_map: Optional[OnsetMap] = None
    onset_cfg = cfg.get("onset", {})
    onset_path = onset_cfg.get("run_summary_path", None)
    if onset_path:
        onset_map = load_onset_map_from_run_summary(
            Path(onset_path).expanduser(),
            onset_col=str(onset_cfg.get("onset_col", "t_first_attack_flow_s")),
            benign_family_name=str(onset_cfg.get("benign_family_name", benign_family_name)),
        )

    # Split discovery
    strat_splits = _discover_splits(processed_dir, strat_glob)
    time_splits = _discover_splits(processed_dir, time_glob)

    splits_all = [(s, False) for s in strat_splits] + [(s, True) for s in time_splits]

    if not splits_all:
        raise RuntimeError("No splits discovered. Check processed_dir and glob patterns.")

    all_bin: List[Dict[str, Any]] = []
    all_mc: List[Dict[str, Any]] = []
    all_op: List[Dict[str, Any]] = []
    all_ttd: List[Dict[str, Any]] = []

    for split_name, is_time in splits_all:
        for ws in windows:
            W = int(ws["W"])
            S = int(ws["S"])
            for model_name, model_cfg in models_to_run:
                # Multiclass only on stratified
                mc_flag = run_multiclass and (not is_time)
                bin_flag = run_binary

                b, m, o, t = run_one(
                    processed_dir=processed_dir,
                    out_dir=out_dir,
                    split_name=split_name,
                    W=W,
                    S=S,
                    model_name=model_name,
                    model_cfg=model_cfg,
                    run_binary=bin_flag,
                    run_multiclass=mc_flag,
                    fpr_targets=fpr_targets,
                    threshold_policies=threshold_policies,
                    benign_family_name=benign_family_name,
                    folds_binary=folds_binary,
                    fusion_cfg=fusion_cfg,
                    outputs_cfg=outputs_cfg,
                    feature_ablation_cfg=feature_ablation_cfg,
                    onset_map=onset_map,
                )
                all_bin.extend(b)
                all_mc.extend(m)
                all_op.extend(o)
                all_ttd.extend(t)

    # Write outputs
    mdir = ensure_dir(out_dir / "metrics")

    if all_bin:
        pd.DataFrame(all_bin).to_csv(mdir / "metrics_binary.csv", index=False)
    if all_mc:
        pd.DataFrame(all_mc).to_csv(mdir / "metrics_multiclass.csv", index=False)
    if all_op:
        op_df = pd.DataFrame(all_op)
        for fpr_t in sorted(set(op_df["fpr_target"].astype(float).tolist())):
            op_df[op_df["fpr_target"] == fpr_t].to_csv(mdir / f"binary_operating_metrics_fpr{fpr_t}.csv", index=False)
    if all_ttd:
        ttd_df = pd.DataFrame(all_ttd)
        for fpr_t in sorted(set(ttd_df["fpr_target"].astype(float).tolist())):
            ttd_df[ttd_df["fpr_target"] == fpr_t].to_csv(mdir / f"ttd_summary_fpr{fpr_t}.csv", index=False)

    # Save run config snapshot
    write_json(out_dir / "stage3_run_config.json", cfg)

    print(f"Done. Out dir: {out_dir}")


if __name__ == "__main__":
    main()
