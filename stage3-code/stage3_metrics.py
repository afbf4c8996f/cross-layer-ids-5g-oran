"""stage3_metrics.py
Compute window-level metrics for binary and multiclass tasks.

Includes operating-point threshold utilities.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
    accuracy_score,
    f1_score,
)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Standard ECE for binary probabilities."""

    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(f"ECE: y_true len {y_true.shape[0]} != y_prob len {y_prob.shape[0]}")

    if y_true.size == 0:
        return float("nan")

    # Be robust to numerical pathologies (NaN/Inf/out-of-range).
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(f"binary_metrics: y_true len {y_true.shape[0]} != y_prob len {y_prob.shape[0]}")

    # Keep metrics stable even if a model emits NaNs/Infs or out-of-range scores.
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 0.0, 1.0)
    out: Dict[str, float] = {}
    out["n"] = float(len(y_true))
    out["pos_rate"] = float(np.mean(y_true))
    # These can fail if only one class present; guard
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["pr_auc"] = float("nan")
    try:
        out["brier"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        out["brier"] = float("nan")
    try:
        out["log_loss"] = float(log_loss(y_true, np.vstack([1.0 - y_prob, y_prob]).T, labels=[0, 1]))
    except Exception:
        out["log_loss"] = float("nan")
    out["ece"] = expected_calibration_error(y_true, y_prob)
    return out

def sanitize_multiclass_proba(P: np.ndarray, *, eps: float = 1e-15) -> np.ndarray:
    """Sanitize a multiclass probability matrix (N,K).

    Guarantees (for K>=1):
    - finite (no NaN/Inf)
    - non-negative
    - rows sum to 1 (within floating error)
    - values clipped to [eps, 1-eps] when eps>0 (stable for log-loss), then renormalized
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"sanitize_multiclass_proba: expected 2D (N,K), got {P.shape}")
    if P.size == 0:
        return P.astype(float, copy=True)

    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    P = np.maximum(P, 0.0)

    row_sums = P.sum(axis=1, keepdims=True)
    K = int(P.shape[1])
    uniform = 1.0 / max(K, 1)

    P = np.divide(
        P,
        row_sums,
        out=np.full_like(P, uniform, dtype=float),
        where=row_sums > 0,
    )

    eps = float(eps)
    if eps > 0.0:
        P = np.clip(P, eps, 1.0 - eps)
        P = P / P.sum(axis=1, keepdims=True)
    else:
        P = np.clip(P, 0.0, 1.0)

    return P

def multiclass_metrics(y_true: np.ndarray, P: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    """y_true: array of string labels; P: prob matrix aligned to classes."""
    y_true = np.asarray(y_true, dtype=str).reshape(-1)
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"multiclass_metrics: expected P to be 2D (N,K), got {P.shape}")
    if P.shape[0] != y_true.shape[0]:
        raise ValueError(f"multiclass_metrics: y_true len {y_true.shape[0]} != P rows {P.shape[0]}")
    if P.shape[1] != len(classes):
        raise ValueError(
            f"multiclass_metrics: P has {P.shape[1]} columns but classes has {len(classes)} entries"
        )
    P = sanitize_multiclass_proba(P)
    y_pred = classes[np.argmax(P, axis=1)]
    out: Dict[str, float] = {}
    out["n"] = float(len(y_true))
    out["acc"] = float(accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
    try:
        out["log_loss"] = float(log_loss(y_true, P, labels=list(classes)))
    except Exception:
        out["log_loss"] = float("nan")
    return out


def operating_point(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_hat = (y_prob >= float(thr)).astype(int)
    tn = int(np.sum((y_true == 0) & (y_hat == 0)))
    fp = int(np.sum((y_true == 0) & (y_hat == 1)))
    fn = int(np.sum((y_true == 1) & (y_hat == 0)))
    tp = int(np.sum((y_true == 1) & (y_hat == 1)))
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tpr
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and (prec + rec)) else float("nan")
    return {
        "threshold": float(thr),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def threshold_from_benign(scores_benign: np.ndarray, fpr_target: float) -> float:
    """Quantile threshold: FPR target means (1 - fpr_target) quantile."""

    s = np.asarray(scores_benign, dtype=float).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    q = 1.0 - float(fpr_target)
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(s, q))


# -------------------------------
# Threshold policies (OOF-based)
# -------------------------------

THRESHOLD_POLICIES = {
    # Recommended default: calibrate using benign *runs* only (scenario-level benign).
    "benign_runs_only",
    # Sensitivity variant: calibrate using strictly pre-attack windows (based on run-level onset).
    "pre_attack",
    # Backward-compatible: all windows with y_bin==0 in TRAIN (may include attack runs / label noise).
    "all_benign_labeled",
}


def _norm_policy(policy: str) -> str:
    p = str(policy).strip().lower()
    # allow a few aliases
    aliases = {
        "benign_runs": "benign_runs_only",
        "benign_runs_only": "benign_runs_only",
        "family_benign": "benign_runs_only",
        "preattack": "pre_attack",
        "pre_attack": "pre_attack",
        "all_benign": "all_benign_labeled",
        "all_benign_labeled": "all_benign_labeled",
        "window_label": "all_benign_labeled",
    }
    if p in aliases:
        return aliases[p]
    return p


def select_oof_scores_for_threshold(
    meta_train: pd.DataFrame,
    oof_scores: np.ndarray,
    y_train: np.ndarray,
    *,
    policy: str,
    onset_s_by_run: Optional[Dict[str, float]] = None,
    benign_family_name: str = "Benign",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Select TRAIN OOF scores to calibrate an operating-point threshold.

    Parameters
    - meta_train: per-window metadata for TRAIN (must align 1:1 with oof_scores)
    - oof_scores: TRAIN out-of-fold scores
    - y_train: window-level binary labels (for backward-compatible policy)
    - policy:
        * benign_runs_only (recommended)
        * pre_attack
        * all_benign_labeled (legacy)
    - onset_s_by_run: required for pre_attack

    Returns
    - scores_selected: 1D float array
    - info: dict with useful bookkeeping (counts, policy name)
    """

    pol = _norm_policy(policy)
    if pol not in THRESHOLD_POLICIES:
        raise ValueError(f"Unknown threshold policy: {policy!r}. Choose from: {sorted(THRESHOLD_POLICIES)}")

    if len(meta_train) != len(oof_scores) or len(y_train) != len(oof_scores):
        raise ValueError(
            "meta_train, oof_scores, y_train must have identical length. "
            f"Got meta={len(meta_train)}, oof={len(oof_scores)}, y={len(y_train)}"
        )

    # default: empty mask
    mask = np.zeros((len(oof_scores),), dtype=bool)

    benign_norm = str(benign_family_name).strip().lower()

    if pol == "all_benign_labeled":
        mask = (np.asarray(y_train).astype(int) == 0)

    elif pol == "benign_runs_only":
        if "family" not in meta_train.columns:
            raise ValueError("Threshold policy 'benign_runs_only' requires meta_train['family']")
        fam = meta_train["family"].astype(str).str.strip().str.lower().to_numpy()
        mask = (fam == benign_norm)

    elif pol == "pre_attack":
        if onset_s_by_run is None:
            raise ValueError("Threshold policy 'pre_attack' requires onset_s_by_run")
        for c in ("run_id", "window_end_s", "family"):
            if c not in meta_train.columns:
                raise ValueError(f"Threshold policy 'pre_attack' requires meta_train['{c}']")

        run_ids = meta_train["run_id"].astype(str).to_numpy()
        win_end = meta_train["window_end_s"].to_numpy(dtype=float)

        # onset = +inf for benign-family runs by construction; this keeps benign runs fully included.
        onset_arr = np.empty((len(run_ids),), dtype=float)
        for i, rid in enumerate(run_ids):
            onset_arr[i] = float(onset_s_by_run.get(rid, float("nan")))

        # strictly pre-attack windows
        mask = (win_end < onset_arr)

    scores = np.asarray(oof_scores, dtype=float)[mask]

    info: Dict[str, Any] = {
        "threshold_policy": pol,
        "n_total_train_windows": int(len(oof_scores)),
        "n_selected": int(np.sum(mask)),
    }

    # Extra diagnostics for reviewers / debugging
    sfin = scores[np.isfinite(scores)]
    if sfin.size:
        info["selected_score_min"] = float(np.min(sfin))
        info["selected_score_max"] = float(np.max(sfin))
    else:
        info["selected_score_min"] = float("nan")
        info["selected_score_max"] = float("nan")

    return scores, info


def threshold_from_oof_policy(
    meta_train: pd.DataFrame,
    oof_scores: np.ndarray,
    y_train: np.ndarray,
    *,
    fpr_target: float,
    policy: str,
    onset_s_by_run: Optional[Dict[str, float]] = None,
    benign_family_name: str = "Benign",
) -> Tuple[float, Dict[str, Any]]:
    """Compute a quantile threshold from TRAIN OOF scores according to a policy."""

    scores, info = select_oof_scores_for_threshold(
        meta_train,
        oof_scores,
        y_train,
        policy=policy,
        onset_s_by_run=onset_s_by_run,
        benign_family_name=benign_family_name,
    )

    thr = threshold_from_benign(scores, fpr_target=float(fpr_target))
    info = dict(info)
    info.update({
        "fpr_target": float(fpr_target),
        "threshold": float(thr),
        "thr_quantile": float(1.0 - float(fpr_target)),
        "thr_source": "train_oof_quantile",
    })
    return float(thr), info
