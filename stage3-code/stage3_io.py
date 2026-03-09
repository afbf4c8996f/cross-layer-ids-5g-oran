"""
stage3_io.py
Load Stage-2 processed window parquets and split features/labels/meta.

Expected Stage-2 layout:
  <processed_dir>/processed/<split_name>/
      network_W{W}_S{S}_{train|val|test}.parquet
      radio_W{W}_S{S}_{train|val|test}.parquet
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

KEY_COLS = ["run_id", "window_start_s", "window_end_s"]

META_COLS = [
    "run_id",
    "family",
    "window_start_s",
    "window_end_s",
    "y_bin",
    "y_cat",
    # Optional / diagnostic:
    "n_flows",
    "empty_window",
    "window_start_ts",
    "window_end_ts",
]

DROP_COLS = [
    # Exact-constant radio features (audit_constant_features.py)
    "ri_mean",
    "ri_std",
    "pmi_0_mean",
    "pmi_0_std",
    "pmi_1_mean",
    "pmi_1_std",
]

DEFAULT_MODE_ONEHOT_PREFIXES = ["proto_mode_", "service_mode_", "conn_state_mode_", "history_mode_"]


@dataclass
class ProcessedSplit:
    df: pd.DataFrame
    X: np.ndarray
    y_bin: np.ndarray
    y_cat: np.ndarray
    groups: np.ndarray
    feature_cols: List[str]
    meta_cols_present: List[str]


def _find_file(processed_dir: Path, split_name: str, modality: str, W: int, S: int, part: str) -> Path:
    p = processed_dir / "processed" / split_name / f"{modality}_W{W}_S{S}_{part}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing processed file: {p}")
    return p


def _norm_variant(x: object) -> str:
    return str(x).strip().lower()


def _apply_feature_ablation(
    feature_cols: List[str],
    *,
    modality: str,
    feature_ablation: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return feature_cols filtered according to a simple ablation policy.

    This is deliberately strict and reversable: it never touches labels/meta,
    and it preserves column order.

    Current supported variants (intended for network modality):
      - full
      - no_mode_onehots
      - mode_onehots_only

    Notes
    - The "mode" one-hots are produced from Stage-1 categorical mode features
      (proto/service/conn_state/history) and one-hot encoded in Stage-2.
    - We apply this ablation to the *network* modality only by default, because
      radio does not contain these prefixes.
    """

    if not feature_ablation:
        return feature_cols

    enabled = bool(feature_ablation.get("enabled", False))
    if not enabled:
        return feature_cols

    # For now, apply only to network modality unless explicitly overridden.
    apply_modalities = feature_ablation.get("modalities", ["network"])
    apply_modalities = [str(m).strip().lower() for m in (apply_modalities or [])]
    if str(modality).strip().lower() not in apply_modalities:
        return feature_cols

    variant = _norm_variant(feature_ablation.get("variant", "full"))
    if variant in {"full", "none", "baseline"}:
        return feature_cols

    prefixes = feature_ablation.get("mode_onehot_prefixes", DEFAULT_MODE_ONEHOT_PREFIXES)
    prefixes = [str(p) for p in (prefixes or DEFAULT_MODE_ONEHOT_PREFIXES)]

    mode_cols = [c for c in feature_cols if any(c.startswith(p) for p in prefixes)]

    if variant in {"drop_history_onehots", "no_history_onehots"}:
        hist_set = set(c for c in feature_cols if c.startswith("history_mode_"))
        return [c for c in feature_cols if c not in hist_set]

    if variant in {"no_mode_onehots", "drop_mode_onehots"}:
        mode_set = set(mode_cols)
        return [c for c in feature_cols if c not in mode_set]

    if variant in {"mode_onehots_only", "only_mode_onehots"}:
        return mode_cols

    raise ValueError(
        f"Unknown feature_ablation.variant={feature_ablation.get('variant')!r}. "
        "Expected one of: full | no_mode_onehots | mode_onehots_only"
    )


def load_processed(
    processed_dir: Path,
    split_name: str,
    modality: str,
    W: int,
    S: int,
    part: str,
    *,
    feature_ablation: Optional[Dict[str, Any]] = None,
) -> ProcessedSplit:
    """Load one processed parquet and return (X, y, meta, groups).

    Parameters
    - modality: 'network' or 'radio'
    - part: 'train'|'val'|'test'
    - feature_ablation: optional dict from YAML (see config_stage3_v3_evalupgrade.yaml)
    """

    p = _find_file(processed_dir, split_name, modality, W, S, part)
    df = pd.read_parquet(p)

    # Validate required columns
    required = {"run_id", "y_bin", "y_cat", "window_start_s", "window_end_s"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{p}: missing required columns: {missing}")

    # Canonicalize types + sort for deterministic alignment
    df["run_id"] = df["run_id"].astype(str)
    df = df.sort_values(KEY_COLS).reset_index(drop=True)

    meta_present = [c for c in META_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if (c not in set(meta_present)) and (c not in set(DROP_COLS))]

    # Apply (optional) feature ablation BEFORE numeric filtering
    feature_cols = _apply_feature_ablation(feature_cols, modality=modality, feature_ablation=feature_ablation)
    # --- Feature ablation audit log ---
    n_hist = sum(1 for c in feature_cols if c.startswith("history_mode_"))
    print(f"[IO] {split_name}/{modality}/{part}: {len(feature_cols)} feats after ablation, "
          f"{n_hist} history_mode_ remaining, ablation={feature_ablation}")


    # Keep only numeric features
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError(
            f"{p}: no numeric feature columns found after excluding meta columns "
            f"(and applying feature_ablation={feature_ablation!r})."
        )

    X = df[num_cols].to_numpy(dtype=np.float32)
    # Ensure finite (LogReg can't handle NaN/inf)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y_bin = df["y_bin"].to_numpy(dtype=int)
    # y_cat can be string labels (recommended: attack_category)
    y_cat = df["y_cat"].astype(str).to_numpy()

    groups = df["run_id"].astype(str).to_numpy()

    return ProcessedSplit(
        df=df,
        X=X,
        y_bin=y_bin,
        y_cat=y_cat,
        groups=groups,
        feature_cols=num_cols,
        meta_cols_present=meta_present,
    )


def align_modalities(a: ProcessedSplit, b: ProcessedSplit) -> Tuple[ProcessedSplit, ProcessedSplit]:
    """Ensure a and b have identical row order by KEY_COLS.

    If one side has missing keys, we error (safer than silently dropping rows),
    because any mismatch would corrupt fusion.
    """

    a_keys = a.df[KEY_COLS].copy()
    b_keys = b.df[KEY_COLS].copy()
    a_keys["_idx_a"] = np.arange(len(a_keys), dtype=int)
    b_keys["_idx_b"] = np.arange(len(b_keys), dtype=int)

    m = a_keys.merge(b_keys, on=KEY_COLS, how="inner")
    if len(m) != len(a.df) or len(m) != len(b.df):
        raise ValueError(
            "Modality window keys do not match; cannot safely fuse. "
            f"intersection={len(m)} network_like={len(a.df)} radio_like={len(b.df)}"
        )

    ia = m["_idx_a"].to_numpy(dtype=int)
    ib = m["_idx_b"].to_numpy(dtype=int)

    a2 = ProcessedSplit(
        df=a.df.iloc[ia].reset_index(drop=True),
        X=a.X[ia],
        y_bin=a.y_bin[ia],
        y_cat=a.y_cat[ia],
        groups=a.groups[ia],
        feature_cols=a.feature_cols,
        meta_cols_present=a.meta_cols_present,
    )
    b2 = ProcessedSplit(
        df=b.df.iloc[ib].reset_index(drop=True),
        X=b.X[ib],
        y_bin=b.y_bin[ib],
        y_cat=b.y_cat[ib],
        groups=b.groups[ib],
        feature_cols=b.feature_cols,
        meta_cols_present=b.meta_cols_present,
    )

    # Safety check: after aligning by KEY_COLS, the ground-truth labels
    # should be identical across modalities. If this trips, the Stage-2
    # preprocessing pipeline produced inconsistent labels and fusion would
    # be invalid. (hence good place to raise)
    if not np.array_equal(a2.y_bin, b2.y_bin):
        raise ValueError("Label mismatch between modalities for y_bin after alignment.")
    if not np.array_equal(a2.y_cat.astype(str), b2.y_cat.astype(str)):
        raise ValueError("Label mismatch between modalities for y_cat after alignment.")
    if "family" in a2.df.columns and "family" in b2.df.columns:
        fa = a2.df["family"].astype(str).to_numpy()
        fb = b2.df["family"].astype(str).to_numpy()
        if not np.array_equal(fa, fb):
            raise ValueError("Label mismatch between modalities for family after alignment.")
    return a2, b2


def join_meta(df_a: pd.DataFrame, df_b: pd.DataFrame, on: Optional[List[str]] = None) -> pd.DataFrame:
    """Join two dfs on run_id + window_start_s + window_end_s, keeping only meta columns."""

    if on is None:
        on = KEY_COLS
    keep_a = [c for c in META_COLS if c in df_a.columns] + on
    keep_b = [c for c in META_COLS if c in df_b.columns] + on
    a = df_a[list(dict.fromkeys(keep_a))].copy()
    b = df_b[list(dict.fromkeys(keep_b))].copy()
    out = a.merge(b, on=on, how="inner", suffixes=("", "_b"))
    # Drop duplicate meta columns from b
    dup = [c for c in out.columns if c.endswith("_b")]
    if dup:
        out = out.drop(columns=dup)
    return out
