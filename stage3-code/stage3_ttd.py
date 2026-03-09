"""stage3_ttd.py

Time-to-detect (TTD) evaluation for binary detection at a fixed threshold.

We support two onset definitions:
- flow-onset (recommended): run-level onset from earliest attack-labeled flow timestamp.
- window-onset (legacy/sensitivity): onset is the start time of the first attack-labeled window.

Detection time is defined at window_end_s (conservative, deployment-aligned).

TTD is computed only for *attack-family* runs. Benign runs have no attack onset by definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TTDResult:
    run_id: str
    detected: bool
    onset_s: float
    detect_s: float
    ttd_s: float


def _norm_family(x: object) -> str:
    return str(x).strip().lower()


def compute_ttd_flow_onset(
    df: pd.DataFrame,
    *,
    score_col: str,
    thr: float,
    onset_s_by_run: Dict[str, float],
    benign_family_name: str = "Benign",
) -> List[TTDResult]:
    """Compute TTD using a run-level onset map (flow-onset).

    Required df columns:
      - run_id
      - family
      - window_end_s
      - score_col

    For attack-family runs:
      onset_s = onset_s_by_run[run_id] (finite)
      detect_s = earliest window_end_s >= onset_s with score >= thr

    Benign-family runs are skipped.
    """

    req = {"run_id", "family", "window_end_s", score_col}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"TTD(flow) df missing columns: {missing}")

    benign_norm = _norm_family(benign_family_name)

    out: List[TTDResult] = []
    for rid, g in df.groupby("run_id"):
        g = g.sort_values("window_end_s")
        fam = _norm_family(g["family"].iloc[0])
        if fam == benign_norm:
            continue  # TTD defined on attack runs only

        rid_str = str(rid)
        if rid_str not in onset_s_by_run:
            raise ValueError(f"Missing onset for run_id={rid_str}")
        onset_s = float(onset_s_by_run[rid_str])
        if not np.isfinite(onset_s):
            raise ValueError(f"Attack-family run_id={rid_str} must have finite onset, got {onset_s}")

        scores = g[score_col].to_numpy(dtype=float)
        t_end = g["window_end_s"].to_numpy(dtype=float)

        # Windows whose *end* is after onset are usable for detection at their end.
        det_mask = (t_end >= onset_s) & (scores >= float(thr))
        if np.any(det_mask):
            det_i = int(np.argmax(det_mask))
            detect_s = float(t_end[det_i])
            ttd = float(max(detect_s - onset_s, 0.0))
            out.append(TTDResult(run_id=rid_str, detected=True, onset_s=onset_s, detect_s=detect_s, ttd_s=ttd))
        else:
            out.append(TTDResult(run_id=rid_str, detected=False, onset_s=onset_s, detect_s=float("nan"), ttd_s=float("inf")))

    return out


def compute_ttd_window_onset(
    df: pd.DataFrame,
    *,
    score_col: str,
    thr: float,
    benign_family_name: str = "Benign",
    label_col: str = "y_bin",
) -> List[TTDResult]:
    """Compute TTD using the first positive window as onset (legacy/sensitivity).

    Required df columns:
      - run_id
      - family
      - window_start_s
      - window_end_s
      - label_col (y_bin)
      - score_col

    Benign-family runs are skipped.
    """

    req = {"run_id", "family", "window_start_s", "window_end_s", label_col, score_col}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"TTD(window) df missing columns: {missing}")

    benign_norm = _norm_family(benign_family_name)

    out: List[TTDResult] = []
    for rid, g in df.groupby("run_id"):
        g = g.sort_values("window_start_s")
        fam = _norm_family(g["family"].iloc[0])
        if fam == benign_norm:
            continue

        rid_str = str(rid)
        y = g[label_col].to_numpy(dtype=int)

        if np.max(y) == 0:
            # Attack-family but no positive windows. Window-onset is undefined; treat as miss.
            out.append(TTDResult(run_id=rid_str, detected=False, onset_s=float("nan"), detect_s=float("nan"), ttd_s=float("inf")))
            continue

        onset_idx = int(np.argmax(y == 1))
        onset_s = float(g.iloc[onset_idx]["window_start_s"])

        scores = g[score_col].to_numpy(dtype=float)
        t_end = g["window_end_s"].to_numpy(dtype=float)

        det_mask = (t_end >= onset_s) & (scores >= float(thr))
        if np.any(det_mask):
            det_i = int(np.argmax(det_mask))
            detect_s = float(t_end[det_i])
            ttd = float(max(detect_s - onset_s, 0.0))
            out.append(TTDResult(run_id=rid_str, detected=True, onset_s=onset_s, detect_s=detect_s, ttd_s=ttd))
        else:
            out.append(TTDResult(run_id=rid_str, detected=False, onset_s=onset_s, detect_s=float("nan"), ttd_s=float("inf")))

    return out


def summarize_ttd(ttd: List[TTDResult]) -> Dict[str, float]:
    """Summary over attack-family runs."""

    if not ttd:
        return {
            "n_attack_runs": 0.0,
            "detect_rate": float("nan"),
            "ttd_median": float("nan"),
            "ttd_p25": float("nan"),
            "ttd_p75": float("nan"),
        }

    detected = [r for r in ttd if r.detected]
    n = len(ttd)
    n_det = len(detected)

    if n_det == 0:
        return {
            "n_attack_runs": float(n),
            "detect_rate": 0.0,
            "ttd_median": float("inf"),
            "ttd_p25": float("inf"),
            "ttd_p75": float("inf"),
        }

    vals = np.array([r.ttd_s for r in detected], dtype=float)
    return {
        "n_attack_runs": float(n),
        "detect_rate": float(n_det / n),
        "ttd_median": float(np.median(vals)),
        "ttd_p25": float(np.quantile(vals, 0.25)),
        "ttd_p75": float(np.quantile(vals, 0.75)),
    }


def summarize_ttd_delta(
    ttd_flow: List[TTDResult],
    ttd_window: List[TTDResult],
) -> Dict[str, float]:
    """Summarize delta between two TTD definitions.

    We define delta per run as:
      delta = ttd_window - ttd_flow

    and summarize over runs where both are detected (finite).
    """

    by_run_flow = {r.run_id: r for r in ttd_flow}
    by_run_win = {r.run_id: r for r in ttd_window}

    deltas: List[float] = []
    for rid, rf in by_run_flow.items():
        rw = by_run_win.get(rid)
        if rw is None:
            continue
        if (not rf.detected) or (not rw.detected):
            continue
        if not (np.isfinite(rf.ttd_s) and np.isfinite(rw.ttd_s)):
            continue
        deltas.append(float(rw.ttd_s - rf.ttd_s))

    if not deltas:
        return {
            "n_delta": 0.0,
            "delta_median": float("nan"),
            "delta_p25": float("nan"),
            "delta_p75": float("nan"),
        }

    v = np.asarray(deltas, dtype=float)
    return {
        "n_delta": float(len(v)),
        "delta_median": float(np.median(v)),
        "delta_p25": float(np.quantile(v, 0.25)),
        "delta_p75": float(np.quantile(v, 0.75)),
    }
