"""
stage3_oof.py
Out-of-fold prediction utilities with *run-level* folds (no group leakage).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

@dataclass
class RunFolds:
    folds: List[Tuple[np.ndarray, np.ndarray]]  # list of (train_run_idx, test_run_idx)
    run_ids: np.ndarray                         # runs array aligned to indices
    run_labels: np.ndarray                      # labels per run

def _mode_str(values: pd.Series) -> str:
    # Stable mode for strings
    vc = values.value_counts()
    if vc.empty:
        return ""
    return str(vc.index[0])

def make_run_folds_binary(df: pd.DataFrame, n_splits: int, seed: int, *, benign_family_name: str = "Benign") -> RunFolds:
    """
    Create run-level folds for binary OOF predictions.

    Uses GREEDY ROW-BALANCED bin-packing instead of StratifiedKFold.
    This ensures each fold has approximately equal numbers of benign AND
    attack *rows* (not just runs), preventing distribution shift across folds.

    Runs are still fully isolated (no group leakage).
    """
    # --- Step 1: Compute run-level labels and row counts ---
    if "family" in df.columns:
        fam = df.groupby("run_id")["family"].apply(_mode_str)
        run_ids = fam.index.astype(str).to_numpy()
        fam_norm = fam.astype(str).str.strip().str.lower()
        benign_norm = str(benign_family_name).strip().lower()
        run_labels = (~fam_norm.eq(benign_norm)).astype(int).to_numpy(dtype=int)
    else:
        run = (df.groupby("run_id")["y_bin"].mean() >= 0.5).astype(int)
        run_ids = run.index.astype(str).to_numpy()
        run_labels = run.to_numpy(dtype=int)

    # Row counts per run
    row_counts = df.groupby("run_id").size()
    run_row_counts = np.array([int(row_counts.get(rid, 0)) for rid in run_ids])

    # --- Step 2: Validate ---
    uniq = np.unique(run_labels)
    if len(uniq) < 2:
        raise ValueError(f"Binary TRAIN has only one class at run-level: {uniq}")

    min_count = int(np.min(np.bincount(run_labels)))
    n_splits_eff = min(int(n_splits), int(min_count))
    if n_splits_eff < 2:
        raise ValueError(
            "Binary TRAIN has too few runs for OOF folds: "
            f"minority-class runs={min_count} (need >=2). "
            "Increase benign/attack runs in TRAIN or reduce folds_binary."
        )

    # --- Step 3: Greedy row-balanced bin-packing ---
    rng = np.random.default_rng(int(seed))

    # For each class, sort runs by row count descending (with shuffled tie-breaking)
    fold_assignments = np.full(len(run_ids), -1, dtype=int)
    fold_row_totals = np.zeros((n_splits_eff, 2), dtype=np.int64)  # [fold, class] -> row count

    for cls in range(2):  # 0=benign, 1=attack
        cls_mask = (run_labels == cls)
        cls_indices = np.where(cls_mask)[0]

        # Shuffle for seed-dependent tie-breaking, then stable-sort by row count desc
        rng.shuffle(cls_indices)
        cls_indices = cls_indices[np.argsort(-run_row_counts[cls_indices], kind='mergesort')]

        for idx in cls_indices:
            # Assign to the fold with fewest rows of THIS class
            best_fold = int(np.argmin(fold_row_totals[:, cls]))
            fold_assignments[idx] = best_fold
            fold_row_totals[best_fold, cls] += run_row_counts[idx]

    # --- Step 4: Build fold tuples (train_idx, test_idx) ---
    folds = []
    for fi in range(n_splits_eff):
        te_mask = (fold_assignments == fi)
        tr_mask = ~te_mask
        folds.append((np.where(tr_mask)[0], np.where(te_mask)[0]))

    return RunFolds(folds=folds, run_ids=run_ids, run_labels=run_labels)

def make_run_folds_multiclass(df: pd.DataFrame, n_splits: int, seed: int) -> RunFolds:
    # Run label: mode of y_cat in run
    run = df.groupby("run_id")["y_cat"].apply(_mode_str)
    run_ids = run.index.astype(str).to_numpy()
    run_labels = run.to_numpy(dtype=str)

    # Count labels
    vals, counts = np.unique(run_labels, return_counts=True)
    if len(vals) < 2:
        raise ValueError(f"Multiclass TRAIN has <2 classes at run-level: {vals}")
    min_count = int(np.min(counts))
    n_splits_eff = min(int(n_splits), int(min_count))
    if n_splits_eff < 2:
        raise ValueError(
            "Multiclass TRAIN has too few runs in a class for OOF folds: "
            f"minority-class runs={min_count} (need >=2). "
            "Disable multiclass stacked fusion or reduce folds_multiclass."
        )

    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=int(seed))
    folds = []
    for tr, te in skf.split(run_ids, run_labels):
        folds.append((tr, te))
    return RunFolds(folds=folds, run_ids=run_ids, run_labels=run_labels)

def iter_fold_indices(df: pd.DataFrame, run_folds: RunFolds) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Convert run folds into row indices over df.
    Returns list of (train_row_idx, test_row_idx).
    """
    run_to_rows: Dict[str, np.ndarray] = {}
    # Precompute row indices per run
    runs = df["run_id"].astype(str).to_numpy()
    for rid in np.unique(runs):
        run_to_rows[rid] = np.flatnonzero(runs == rid)

    out = []
    for tr_run_idx, te_run_idx in run_folds.folds:
        tr_runs = set(run_folds.run_ids[tr_run_idx].tolist())
        te_runs = set(run_folds.run_ids[te_run_idx].tolist())
        # IMPORTANT for sequential models:
        # Sorting restores deterministic, time-consistent ordering because df is already sorted
        # by (run_id, window_start_s, window_end_s) in stage3_io.
        tr_rows = np.sort(np.concatenate([run_to_rows[r] for r in tr_runs]))
        te_rows = np.sort(np.concatenate([run_to_rows[r] for r in te_runs]))
        out.append((tr_rows, te_rows))
    return out
