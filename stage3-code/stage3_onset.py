"""stage3_onset.py

Run-level onset utilities used by Stage-3 evaluation.

We want a *run-level* attack onset timestamp that is independent of windowing.
We use it for:
  (a) Time-to-detect (TTD) with a defensible "flow-onset" definition.
  (b) Optional threshold calibration variants that use strictly pre-attack windows.

Key design choice (reviewer-proof):
- Scenario-level `family` is the ground truth for whether a run is benign.
  If family matches `benign_family_name`, onset is defined as +inf ("no attack onset"), even if a few
  flow records happen to be labeled as attack.

This keeps onset-based evaluation/calibration from being derailed by sparse label noise
inside benign runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _norm_family(x: object) -> str:
    return str(x).strip().lower()


@dataclass(frozen=True)
class OnsetMap:
    """Run-level onsets in seconds relative to run start."""

    onset_s_by_run: Dict[str, float]
    family_by_run: Dict[str, str]
    benign_family_name: str = "Benign"

    def onset_s(self, run_id: str) -> float:
        return float(self.onset_s_by_run[str(run_id)])


def load_onset_map_from_run_summary(
    path: Path,
    *,
    run_id_col: str = "run_id",
    family_col: str = "family",
    onset_col: str = "t_first_attack_flow_s",
    benign_family_name: str = "Benign",
) -> OnsetMap:
    """Load a run summary table and produce an onset map.

    The file must include at least:
      - run_id (string-ish)
      - family (scenario label)
      - onset_col (float seconds relative to run start)

    For benign-family runs, onset is set to +inf by definition.

    Supported formats: .csv, .parquet
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Onset run summary not found: {path}")

    suf = path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(path)
    elif suf in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported onset summary format: {suf}. Use .csv or .parquet")

    for c in (run_id_col, family_col, onset_col):
        if c not in df.columns:
            raise ValueError(f"{path}: missing required column '{c}'. Columns={list(df.columns)}")

    benign_norm = _norm_family(benign_family_name)

    # Ensure unique per run_id (keep first)
    df = df.drop_duplicates(subset=[run_id_col]).copy()

    onset_s_by_run: Dict[str, float] = {}
    family_by_run: Dict[str, str] = {}

    for _, row in df.iterrows():
        rid = str(row[run_id_col])
        fam_raw = str(row[family_col])
        fam_norm = _norm_family(fam_raw)

        if fam_norm == benign_norm:
            onset = float("inf")
        else:
            try:
                onset = float(row[onset_col])
            except Exception as e:
                raise ValueError(f"{path}: bad onset value for run_id={rid}: {row[onset_col]!r}") from e
            if not np.isfinite(onset):
                raise ValueError(f"{path}: onset must be finite for attack-family run_id={rid}, got {onset}")
            if onset < 0:
                raise ValueError(f"{path}: onset must be >=0 seconds for run_id={rid}, got {onset}")

        onset_s_by_run[rid] = onset
        family_by_run[rid] = fam_raw

    return OnsetMap(
        onset_s_by_run=onset_s_by_run,
        family_by_run=family_by_run,
        benign_family_name=str(benign_family_name),
    )


def validate_onset_map_against_meta(
    meta: pd.DataFrame,
    onset: OnsetMap,
    *,
    run_id_col: str = "run_id",
    family_col: str = "family",
    window_end_col: str = "window_end_s",
    strict: bool = True,
) -> None:
    """Sanity checks to prevent silent evaluation bugs.

    - Every run_id in meta must exist in onset map.
    - For benign-family runs in meta, onset must be +inf.
    - For attack-family runs in meta, onset must be finite.
    - Basic units check: onset should not exceed max window_end by absurd amounts.

    If strict=False, prints warnings instead of raising.
    """

    def _fail(msg: str) -> None:
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}")

    for c in (run_id_col, family_col, window_end_col):
        if c not in meta.columns:
            _fail(f"Meta missing required column '{c}' for onset validation")
            return

    benign_norm = _norm_family(onset.benign_family_name)

    run_ids = meta[run_id_col].astype(str).unique().tolist()
    missing = [r for r in run_ids if r not in onset.onset_s_by_run]
    if missing:
        _fail(f"Onset map missing {len(missing)} run_ids present in windows: {missing[:10]}")

    fam_by_run = meta.groupby(run_id_col)[family_col].agg(lambda s: str(s.iloc[0])).to_dict()
    for rid, fam_raw in fam_by_run.items():
        fam_norm = _norm_family(fam_raw)
        onset_s = float(onset.onset_s_by_run.get(str(rid), float("nan")))
        if fam_norm == benign_norm:
            if onset_s != float("inf"):
                _fail(f"Benign run_id={rid} must have onset=+inf, got {onset_s}")
        else:
            if not np.isfinite(onset_s):
                _fail(f"Attack-family run_id={rid} must have finite onset, got {onset_s}")

    # Range check
    max_end_by_run = meta.groupby(run_id_col)[window_end_col].max().to_dict()
    for rid, max_end in max_end_by_run.items():
        onset_s = float(onset.onset_s_by_run.get(str(rid), float("nan")))
        if np.isfinite(onset_s):
            if onset_s > float(max_end) + 60.0:
                _fail(
                    f"Onset seems out of range for run_id={rid}: onset={onset_s:.3f}s, max_window_end={float(max_end):.3f}s"
                )


def onset_array_for_meta(
    meta: pd.DataFrame,
    onset: OnsetMap,
    *,
    run_id_col: str = "run_id",
) -> np.ndarray:
    """Vectorized onset lookup aligned to meta rows."""

    r = meta[run_id_col].astype(str).to_numpy()
    out = np.empty((len(r),), dtype=float)
    for i, rid in enumerate(r):
        out[i] = float(onset.onset_s_by_run.get(rid, float("nan")))
    return out
