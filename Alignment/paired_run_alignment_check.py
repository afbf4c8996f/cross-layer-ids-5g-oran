#!/usr/bin/env python3
"""paired_run_alignment_check.py

Paired-run multimodal alignment feasibility check for NetsLab-5G-ORAN-IDD.


------
Some Lower_Layer telemetry TXT files in the Kaggle release are valid JSON-lines
but **do not contain an explicit timestamp field**. Example lines look like:
  {"dlBytes": 326..., "ulBytes": 659..., ...}

In those files, the row order implicitly represents ~1 Hz sampling, but the
timestamp itself is missing.

W support BOTH telemetry styles:
  1) Timestamped telemetry: epoch ms/s under keys like timestamp/ts
  2) Timestampless telemetry: uses an implicit 1 Hz index (0,1,2,...) based on
     valid-row order.

To make results comparable across both modes, perform alignment in a
**per-run relative time axis**:
  - Network time: seconds since first packet in the PCAP
  - Radio time: seconds since first telemetry sample (timestamped or implicit)

If telemetry provides epoch timestamps, the script also reports the implied
system-level start-time difference.

Outputs
-------
Writes to --out-dir:
  - results.csv / results.jsonl : per-run best lag (seconds), corr, acceptance
  - summary.json               : acceptance rates overall and by family
  - optional curves/           : corr-vs-lag curves per run
  - optional plots/            : overlays for selected runs
  - cache/                     : cached extracted series

PCAP Support
------------
Native parser supports classic PCAP (.pcap). For pcapng, use tshark reader.

"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import orjson
except Exception:  # pragma: no cover
    orjson = None  # type: ignore


# -----------------------------
# Utilities
# -----------------------------


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def sha1_short(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def which(cmd: str) -> Optional[str]:
    from shutil import which as _which

    return _which(cmd)


def is_pcapng(path: Path) -> bool:
    """Detect PCAPNG by the section header block magic."""
    try:
        with path.open("rb") as f:
            b = f.read(4)
        return b == b"\x0a\x0d\x0d\x0a"
    except Exception:
        return False


# -----------------------------
# PCAP readers
# -----------------------------


@dataclasses.dataclass
class PcapSeries:
    sec_index_epoch: np.ndarray  # int64 epoch seconds
    bytes_per_sec: np.ndarray  # int64
    start_ts: float
    end_ts: float
    packet_count: int


def read_pcap_classic_bytes_per_sec(path: Path, use_orig_len: bool = True) -> PcapSeries:
    """Minimal classic PCAP parser (no payload decode), bins bytes by epoch second."""
    import struct

    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open("rb") as f:
        gh = f.read(24)
        if len(gh) < 24:
            raise ValueError(f"{path} too small to be a pcap file")

        magic = gh[:4]
        if magic == b"\xd4\xc3\xb2\xa1":
            endian = "<"
            ts_unit = "usec"
        elif magic == b"\xa1\xb2\xc3\xd4":
            endian = ">"
            ts_unit = "usec"
        elif magic == b"\x4d\x3c\xb2\xa1":
            endian = "<"
            ts_unit = "nsec"
        elif magic == b"\xa1\xb2\x3c\x4d":
            endian = ">"
            ts_unit = "nsec"
        else:
            raise ValueError(
                f"{path} has unknown pcap magic {magic.hex()} (might be pcapng). "
                "Convert to pcap or use --pcap-reader tshark_fields."
            )

        ph_fmt = endian + "IIII"
        ph_size = 16

        agg: Dict[int, int] = {}
        packet_count = 0
        start_ts: Optional[float] = None
        end_ts: Optional[float] = None

        while True:
            ph = f.read(ph_size)
            if not ph:
                break
            if len(ph) < ph_size:
                break

            ts_sec, ts_frac, incl_len, orig_len = struct.unpack(ph_fmt, ph)
            f.seek(incl_len, os.SEEK_CUR)

            if ts_unit == "usec":
                ts = float(ts_sec) + float(ts_frac) / 1e6
            else:
                ts = float(ts_sec) + float(ts_frac) / 1e9

            if start_ts is None:
                start_ts = ts
            end_ts = ts
            packet_count += 1

            sec = int(ts_sec)
            plen = int(orig_len if use_orig_len else incl_len)
            agg[sec] = agg.get(sec, 0) + plen

        if packet_count == 0 or start_ts is None or end_ts is None:
            raise ValueError(f"No packets parsed from {path}")

        secs = np.fromiter(sorted(agg.keys()), dtype=np.int64)
        vals = np.fromiter((agg[int(s)] for s in secs), dtype=np.int64)

        return PcapSeries(
            sec_index_epoch=secs,
            bytes_per_sec=vals,
            start_ts=float(start_ts),
            end_ts=float(end_ts),
            packet_count=int(packet_count),
        )


def read_pcap_with_tshark_fields(path: Path) -> PcapSeries:
    """Fallback PCAP/PCAPNG reader using tshark to extract frame.time_epoch and frame.len."""
    tshark = which("tshark")
    if tshark is None:
        raise RuntimeError("tshark not found in PATH; install Wireshark/tshark or use classic PCAP")

    cmd = [
        tshark,
        "-r",
        str(path),
        "-T",
        "fields",
        "-E",
        "separator=,",
        "-e",
        "frame.time_epoch",
        "-e",
        "frame.len",
        "-n",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"tshark failed for {path}:\n{proc.stderr[:1000]}")

    agg: Dict[int, int] = {}
    packet_count = 0
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            t = float(parts[0])
            blen = int(parts[1])
        except Exception:
            continue
        sec = int(math.floor(t))
        agg[sec] = agg.get(sec, 0) + blen
        packet_count += 1
        if start_ts is None:
            start_ts = t
        end_ts = t

    if packet_count == 0 or start_ts is None or end_ts is None:
        raise ValueError(f"No packets parsed from {path} using tshark")

    secs = np.fromiter(sorted(agg.keys()), dtype=np.int64)
    vals = np.fromiter((agg[int(s)] for s in secs), dtype=np.int64)

    return PcapSeries(
        sec_index_epoch=secs,
        bytes_per_sec=vals,
        start_ts=float(start_ts),
        end_ts=float(end_ts),
        packet_count=int(packet_count),
    )


# -----------------------------
# Telemetry TXT parser
# -----------------------------


@dataclasses.dataclass
class RadioSeries:
    sec_index: np.ndarray  # int64 seconds (epoch seconds if timestamped, else implicit)
    rate_per_sec: np.ndarray  # float64
    start_ts: float
    end_ts: float
    row_count: int
    ue_count: int
    bytes_mode: str  # cumulative|instantaneous
    reset_events: int
    parse_errors: int
    time_mode: str  # timestamp|implicit
    missing_timestamp_rows: int


def _loads_json(line: str) -> Optional[dict]:
    """Robust per-line JSON/Python-dict parser."""
    line = line.strip()
    if not line:
        return None

    if line.endswith(","):
        line = line[:-1].rstrip()

    if (line.startswith("b'") and line.endswith("'")) or (line.startswith('b"') and line.endswith('"')):
        line = line[2:-1].strip()

    candidates: List[str] = [line]
    if "{" in line and "}" in line:
        i = line.find("{")
        j = line.rfind("}")
        if i >= 0 and j > i:
            sub = line[i : j + 1].strip()
            if sub and sub != line:
                candidates.insert(0, sub)

    def _json_try(s: str) -> Optional[dict]:
        if not s:
            return None
        try:
            obj = orjson.loads(s) if orjson is not None else json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    for s in candidates:
        obj = _json_try(s)
        if obj is not None:
            return obj

    for s in candidates:
        ss = s.strip()
        if ss.startswith("{") and ss.endswith("}"):
            try:
                obj2 = ast.literal_eval(ss)
                if isinstance(obj2, dict):
                    return obj2
            except Exception:
                pass
    return None


def _coerce_epoch_seconds(ts: float) -> int:
    # ms typically ~1e12-1e13; seconds ~1e9-1e10
    if ts > 1e11:
        return int(ts // 1000.0)
    return int(math.floor(ts))


def parse_radio_txt_to_series(
    path: Path,
    bytes_mode: str = "auto",  # auto|cumulative|instantaneous
    aggregate_over_ues: str = "sum",  # sum|median
) -> RadioSeries:
    """Parse telemetry TXT and produce a 1Hz throughput proxy.

    Supports:
      - timestamped JSON lines (timestamp/ts in ms or s)
      - timestampless JSON lines (implicit 1 Hz index via row order)
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    # Store both possibilities while reading
    rows_ts: List[Tuple[int, str, float]] = []   # (t_sec_epoch, ue, totBytes)
    rows_idx: List[Tuple[int, str, float]] = []  # (idx_sec, ue, totBytes)

    parse_errors = 0
    missing_ts_rows = 0
    valid_idx = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_s = line.strip()
            if not line_s:
                continue
            obj = _loads_json(line_s)
            if obj is None:
                parse_errors += 1
                continue

            d = {str(k).lower(): v for k, v in obj.items()}

            # bytes fields
            dl = d.get("dlbytes", d.get("dl_bytes", None))
            ul = d.get("ulbytes", d.get("ul_bytes", None))
            if dl is None or ul is None:
                parse_errors += 1
                continue

            ue = d.get("ue_id", d.get("ueid", d.get("ue", "unknown")))

            # timestamp (optional)
            ts = d.get("timestamp", d.get("ts", None))

            try:
                dl_f = float(dl)
                ul_f = float(ul)
                tot = dl_f + ul_f
                ue_s = str(ue)
            except Exception:
                parse_errors += 1
                continue

            if ts is None:
                # valid row but no timestamp
                rows_idx.append((valid_idx, ue_s, tot))
                missing_ts_rows += 1
            else:
                try:
                    ts_f = float(ts)
                    t_sec = _coerce_epoch_seconds(ts_f)
                    rows_ts.append((t_sec, ue_s, tot))
                except Exception:
                    # treat as missing timestamp rather than a parse error
                    rows_idx.append((valid_idx, ue_s, tot))
                    missing_ts_rows += 1

            valid_idx += 1

    # Decide time mode
    if len(rows_ts) > 0:
        time_mode = "timestamp"
        rows = rows_ts
        df = pd.DataFrame(rows, columns=["t", "ue_id", "totBytes"]).copy()
    elif len(rows_idx) > 0:
        time_mode = "implicit"
        rows = rows_idx
        df = pd.DataFrame(rows, columns=["t", "ue_id", "totBytes"]).copy()
    else:
        raise ValueError(f"No parsable telemetry rows in {path} (parse_errors={parse_errors})")

    # Reduce to one value per (ue_id, second): take max as end-of-second counter snapshot
    df = df.groupby(["ue_id", "t"], as_index=False)["totBytes"].max().sort_values(["ue_id", "t"])

    ue_ids = df["ue_id"].unique().tolist()
    ue_count = len(ue_ids)

    t_min = int(df["t"].min())
    t_max = int(df["t"].max())
    full_index = np.arange(t_min, t_max + 1, dtype=np.int64)

    # Infer bytes mode if auto: check monotonicity per UE
    inferred_mode = "cumulative"
    if bytes_mode == "auto":
        neg_fracs: List[float] = []
        for ue in ue_ids[: min(10, ue_count)]:
            s = df[df["ue_id"] == ue]["totBytes"].values
            if len(s) < 3:
                continue
            diffs = np.diff(s)
            neg_fracs.append(float(np.mean(diffs < 0)))
        inferred_mode = "instantaneous" if (len(neg_fracs) > 0 and float(np.mean(neg_fracs)) > 0.20) else "cumulative"
    else:
        inferred_mode = bytes_mode

    reset_events = 0

    if inferred_mode == "instantaneous":
        per_sec = df.groupby("t")["totBytes"].sum() if aggregate_over_ues == "sum" else df.groupby("t")["totBytes"].median()
        per_sec = per_sec.reindex(full_index, fill_value=0.0)
        rate = per_sec.values.astype(np.float64)

    elif inferred_mode == "cumulative":
        rates = np.zeros_like(full_index, dtype=np.float64)
        for ue in ue_ids:
            sub = df[df["ue_id"] == ue].set_index("t")["totBytes"].sort_index()
            sub_full = sub.reindex(full_index).ffill().fillna(0.0)
            diffs = sub_full.diff().fillna(0.0).values.astype(np.float64)
            neg = diffs < 0
            if np.any(neg):
                reset_events += int(np.sum(neg))
                diffs[neg] = 0.0
            rates += diffs
        rate = rates
    else:
        raise ValueError(f"bytes_mode must be auto|cumulative|instantaneous, got {bytes_mode}")

    return RadioSeries(
        sec_index=full_index,
        rate_per_sec=rate.astype(np.float64),
        start_ts=float(t_min),
        end_ts=float(t_max),
        row_count=int(len(rows)),
        ue_count=int(ue_count),
        bytes_mode=str(inferred_mode),
        reset_events=int(reset_events),
        parse_errors=int(parse_errors),
        time_mode=time_mode,
        missing_timestamp_rows=int(missing_ts_rows if time_mode == "implicit" else (missing_ts_rows)),
    )


# -----------------------------
# Alignment helpers
# -----------------------------


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(x, kernel, mode="same")


def corr_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 3:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def align_on_lag(
    net_secs: np.ndarray,
    net_vals: np.ndarray,
    rad_secs: np.ndarray,
    rad_vals: np.ndarray,
    lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align by shifting radio seconds by +lag and intersecting seconds."""
    rad_shifted = rad_secs + int(lag)
    # intersect
    common = np.intersect1d(net_secs, rad_shifted, assume_unique=False)
    if common.size == 0:
        return common, np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    net_map = {int(s): float(v) for s, v in zip(net_secs.tolist(), net_vals.tolist())}
    rad_map = {int(s): float(v) for s, v in zip(rad_shifted.tolist(), rad_vals.tolist())}

    x = np.array([net_map[int(t)] for t in common], dtype=np.float64)
    y = np.array([rad_map[int(t)] for t in common], dtype=np.float64)
    return common.astype(np.int64), x, y


@dataclasses.dataclass
class LagSearchResult:
    best_lag: int
    best_corr: float
    overlap_seconds: int
    corr_by_lag: Dict[int, float]
    peak_prominence: float


def search_best_lag(
    net_secs: np.ndarray,
    net_vals: np.ndarray,
    rad_secs: np.ndarray,
    rad_vals: np.ndarray,
    center_lag: int,
    window: int,
    coarse_step: int,
    fine_window: int,
    min_overlap: int,
    smooth_seconds: int,
    log1p: bool,
    peak_exclusion: int,
) -> LagSearchResult:
    """Coarse-to-fine lag search."""

    def _prep(arr: np.ndarray) -> np.ndarray:
        x = arr.astype(np.float64)
        if log1p:
            x = np.log1p(np.maximum(x, 0.0))
        if smooth_seconds > 1:
            x = moving_average(x, smooth_seconds)
        return x

    corr_by_lag: Dict[int, float] = {}

    # Coarse sweep
    coarse_lags = list(range(center_lag - window, center_lag + window + 1, max(1, coarse_step)))
    best_lag = coarse_lags[0]
    best_corr = float("-inf")
    best_overlap = 0

    for lag in coarse_lags:
        common, x, y = align_on_lag(net_secs, net_vals, rad_secs, rad_vals, lag)
        if common.size < min_overlap:
            corr_by_lag[int(lag)] = float("nan")
            continue
        x2 = _prep(x)
        y2 = _prep(y)
        c = corr_pearson(x2, y2)
        corr_by_lag[int(lag)] = float(c)
        if not math.isnan(c) and c > best_corr:
            best_corr = float(c)
            best_lag = int(lag)
            best_overlap = int(common.size)

    # Fine sweep around coarse best
    fine_lags = list(range(best_lag - fine_window, best_lag + fine_window + 1))
    for lag in fine_lags:
        if lag in corr_by_lag:
            continue
        common, x, y = align_on_lag(net_secs, net_vals, rad_secs, rad_vals, lag)
        if common.size < min_overlap:
            corr_by_lag[int(lag)] = float("nan")
            continue
        x2 = _prep(x)
        y2 = _prep(y)
        c = corr_pearson(x2, y2)
        corr_by_lag[int(lag)] = float(c)
        if not math.isnan(c) and c > best_corr:
            best_corr = float(c)
            best_lag = int(lag)
            best_overlap = int(common.size)

    # Peak prominence estimate
    # Exclude a small neighborhood around best lag and take next best
    vals = []
    for lag, c in corr_by_lag.items():
        if math.isnan(c):
            continue
        if abs(int(lag) - int(best_lag)) <= int(peak_exclusion):
            continue
        vals.append(float(c))
    next_best = max(vals) if len(vals) else float("nan")
    peak_prom = float(best_corr - next_best) if not math.isnan(next_best) else float("nan")

    if best_corr == float("-inf"):
        best_corr = float("nan")

    return LagSearchResult(
        best_lag=int(best_lag),
        best_corr=float(best_corr),
        overlap_seconds=int(best_overlap),
        corr_by_lag={int(k): float(v) for k, v in corr_by_lag.items()},
        peak_prominence=float(peak_prom),
    )


@dataclasses.dataclass
class AcceptanceResult:
    accepted: bool
    null_threshold: float
    half_corr_min: float
    reason: str


def null_calibrated_acceptance(
    aligned_x: np.ndarray,
    aligned_y: np.ndarray,
    observed_corr: float,
    rng: np.random.Generator,
    iters: int,
    quantile: float,
    min_shift: int,
    min_half_corr: float,
) -> AcceptanceResult:
    if aligned_x.size < 3 or aligned_y.size < 3 or math.isnan(observed_corr):
        return AcceptanceResult(False, float("nan"), float("nan"), "undefined_corr")

    n = int(aligned_x.size)
    if n < 2 * min_shift + 10:
        # too short for meaningful circular shifts
        return AcceptanceResult(False, float("nan"), float("nan"), "too_short")

    null_corrs = []
    for _ in range(int(iters)):
        s = int(rng.integers(min_shift, n - min_shift))
        y_shift = np.roll(aligned_y, s)
        c = corr_pearson(aligned_x, y_shift)
        if not math.isnan(c):
            null_corrs.append(float(c))

    if len(null_corrs) < max(20, iters // 4):
        return AcceptanceResult(False, float("nan"), float("nan"), "null_insufficient")

    thr = float(np.quantile(np.array(null_corrs, dtype=np.float64), quantile))

    # Stability: corr in both halves
    mid = n // 2
    c1 = corr_pearson(aligned_x[:mid], aligned_y[:mid])
    c2 = corr_pearson(aligned_x[mid:], aligned_y[mid:])
    half_min = float(np.nanmin([c1, c2]))

    if observed_corr < thr:
        return AcceptanceResult(False, thr, half_min, "below_null_threshold")
    if math.isnan(half_min) or half_min < min_half_corr:
        return AcceptanceResult(False, thr, half_min, "unstable_halves")
    return AcceptanceResult(True, thr, half_min, "accepted")


# -----------------------------
# IO: caching
# -----------------------------


def series_to_npz(path: Path, secs: np.ndarray, vals: np.ndarray, meta: dict) -> None:
    safe_mkdir(path.parent)
    np.savez_compressed(path, secs=secs.astype(np.int64), vals=vals.astype(np.float64), meta=json.dumps(meta))


def series_from_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    z = np.load(path, allow_pickle=False)
    secs = z["secs"].astype(np.int64)
    vals = z["vals"].astype(np.float64)
    meta = json.loads(str(z["meta"]))
    return secs, vals, meta


# -----------------------------
# Manifest
# -----------------------------


@dataclasses.dataclass
class PairedRun:
    family: str
    canon_stem: str
    pcap_path: Path
    txt_path: Path


def load_paired_runs(csv_path: Path) -> List[PairedRun]:
    df = pd.read_csv(csv_path)
    # Expect columns created by extract_paired_runs.py
    required = {"family", "canon_stem", "pcap_path", "txt_path"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"paired_runs.csv must contain columns {sorted(required)}")

    out: List[PairedRun] = []
    for _, r in df.iterrows():
        out.append(
            PairedRun(
                family=str(r["family"]),
                canon_stem=str(r["canon_stem"]),
                pcap_path=Path(str(r["pcap_path"])),
                txt_path=Path(str(r["txt_path"])),
            )
        )
    return out


# -----------------------------
# Per-run processing
# -----------------------------


@dataclasses.dataclass
class RunResult:
    family: str
    canon_stem: str
    pcap_path: str
    txt_path: str
    # Network
    net_t0_epoch: float
    net_seconds: int
    net_packet_count: int
    # Radio
    rad_time_mode: str
    rad_t0_epoch: Optional[float]
    rad_seconds: int
    rad_row_count: int
    rad_ue_count: int
    rad_bytes_mode: str
    rad_reset_events: int
    rad_parse_errors: int
    rad_missing_timestamp_rows: int
    # Alignment
    best_lag_seconds: int
    best_corr: float
    overlap_seconds: int
    accepted: bool
    null_threshold: float
    half_corr_min: float
    peak_prominence: float
    # Useful derived
    abs_start_diff_seconds: Optional[float]
    abs_offset_seconds: Optional[float]
    notes: str


def densify_series(secs: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if secs.size == 0:
        return secs, vals
    smin = int(np.min(secs))
    smax = int(np.max(secs))
    full = np.arange(smin, smax + 1, dtype=np.int64)
    mp = {int(s): float(v) for s, v in zip(secs.tolist(), vals.tolist())}
    full_vals = np.array([mp.get(int(t), 0.0) for t in full], dtype=np.float64)
    return full, full_vals


def process_one_run(
    run: PairedRun,
    out_dir: Path,
    cache_dir: Optional[Path],
    pcap_reader: str,
    telemetry_bytes_mode: str,
    search_window: int,
    coarse_step: int,
    fine_window: int,
    min_overlap: int,
    smooth_seconds: int,
    log1p: bool,
    peak_exclusion: int,
    null_iters: int,
    null_quantile: float,
    min_half_corr: float,
    seed: int,
    save_curves: bool,
) -> RunResult:
    # Cache keys
    cache_key = sha1_short(f"{run.pcap_path}|{run.txt_path}", 12)
    net_cache = None
    rad_cache = None
    if cache_dir is not None:
        safe_mkdir(cache_dir)
        net_cache = cache_dir / f"net__{run.family}__{run.canon_stem}__{cache_key}.npz"
        rad_cache = cache_dir / f"rad__{run.family}__{run.canon_stem}__{cache_key}.npz"

    # ---- Network series ----
    if net_cache is not None and net_cache.exists():
        net_secs_epoch, net_vals, net_meta = series_from_npz(net_cache)
        net_t0_epoch = float(net_meta["t0_epoch"])
        net_packet_count = int(net_meta["packet_count"])
    else:
        if pcap_reader == "auto":
            if is_pcapng(run.pcap_path):
                ps = read_pcap_with_tshark_fields(run.pcap_path)
            else:
                ps = read_pcap_classic_bytes_per_sec(run.pcap_path)
        elif pcap_reader == "classic":
            ps = read_pcap_classic_bytes_per_sec(run.pcap_path)
        elif pcap_reader == "tshark_fields":
            ps = read_pcap_with_tshark_fields(run.pcap_path)
        else:
            raise ValueError(f"Unknown pcap_reader: {pcap_reader}")

        # densify on epoch seconds
        net_secs_epoch, net_vals = densify_series(ps.sec_index_epoch, ps.bytes_per_sec.astype(np.float64))
        net_t0_epoch = float(net_secs_epoch[0])
        net_packet_count = int(ps.packet_count)

        if net_cache is not None:
            series_to_npz(
                net_cache,
                net_secs_epoch,
                net_vals,
                meta={"t0_epoch": net_t0_epoch, "packet_count": net_packet_count, "pcap_reader": pcap_reader},
            )

    # Convert to relative seconds (since first packet)
    net_secs = (net_secs_epoch - int(net_secs_epoch[0])).astype(np.int64)
    net_seconds = int(net_secs.size)

    # ---- Radio series ----
    if rad_cache is not None and rad_cache.exists():
        rad_secs_raw, rad_vals, rad_meta = series_from_npz(rad_cache)
        rad_time_mode = str(rad_meta["time_mode"])
        rad_t0_epoch = rad_meta.get("t0_epoch", None)
        rad_row_count = int(rad_meta["row_count"])
        rad_ue_count = int(rad_meta["ue_count"])
        rad_bytes_mode = str(rad_meta["bytes_mode"])
        rad_reset_events = int(rad_meta["reset_events"])
        rad_parse_errors = int(rad_meta["parse_errors"])
        rad_missing_ts_rows = int(rad_meta.get("missing_timestamp_rows", 0))
    else:
        rs = parse_radio_txt_to_series(run.txt_path, bytes_mode=telemetry_bytes_mode, aggregate_over_ues="sum")
        rad_secs_raw, rad_vals = rs.sec_index, rs.rate_per_sec
        rad_time_mode = rs.time_mode
        rad_t0_epoch = float(rad_secs_raw[0]) if rs.time_mode == "timestamp" else None
        rad_row_count = int(rs.row_count)
        rad_ue_count = int(rs.ue_count)
        rad_bytes_mode = str(rs.bytes_mode)
        rad_reset_events = int(rs.reset_events)
        rad_parse_errors = int(rs.parse_errors)
        rad_missing_ts_rows = int(rs.missing_timestamp_rows)

        if rad_cache is not None:
            series_to_npz(
                rad_cache,
                rad_secs_raw,
                rad_vals,
                meta={
                    "time_mode": rad_time_mode,
                    "t0_epoch": rad_t0_epoch,
                    "row_count": rad_row_count,
                    "ue_count": rad_ue_count,
                    "bytes_mode": rad_bytes_mode,
                    "reset_events": rad_reset_events,
                    "parse_errors": rad_parse_errors,
                    "missing_timestamp_rows": rad_missing_ts_rows,
                },
            )

    # Densify radio and convert to relative time
    rad_secs_raw, rad_vals = densify_series(rad_secs_raw.astype(np.int64), rad_vals.astype(np.float64))
    rad_secs = (rad_secs_raw - int(rad_secs_raw[0])).astype(np.int64)
    rad_seconds = int(rad_secs.size)

    # Notes can be populated before and after lag search
    notes = []

    # Cap min-overlap for lag search to the available run length.
    # This prevents "no candidate lags" when the user requests a large min_overlap
    # but the scenario itself is short (common for scans / short floods).
    min_len_possible = int(min(net_seconds, rad_seconds))
    min_overlap_search = int(min(min_overlap, min_len_possible))
    if min_overlap_search < 3:
        min_overlap_search = int(min_len_possible)
    if int(min_overlap_search) != int(min_overlap):
        notes.append(f"min_overlap_capped={min_overlap_search}")

    # ---- Lag search in relative domain ----
    center_lag = 0
    lag_res = search_best_lag(
        net_secs=net_secs,
        net_vals=net_vals,
        rad_secs=rad_secs,
        rad_vals=rad_vals,
        center_lag=center_lag,
        window=search_window,
        coarse_step=coarse_step,
        fine_window=fine_window,
        min_overlap=min_overlap_search,
        smooth_seconds=smooth_seconds,
        log1p=log1p,
        peak_exclusion=peak_exclusion,
    )

    best_lag = int(lag_res.best_lag)
    best_corr = float(lag_res.best_corr)
    overlap_seconds = int(lag_res.overlap_seconds)

    # ---- Acceptance ----
    rng = np.random.default_rng(seed=seed + int(sha1_short(cache_key, 6), 16) % 10_000)
    common, x_al, y_al = align_on_lag(net_secs, net_vals, rad_secs, rad_vals, best_lag)

    if common.size < 3:
        # No overlap for the chosen lag (usually because the run is shorter than the requested min_overlap).
        x_acc = np.array([], dtype=np.float64)
        y_acc = np.array([], dtype=np.float64)
        acc = AcceptanceResult(False, float("nan"), float("nan"), "no_overlap_for_best_lag")
    else:
        # apply same preprocessing
        x_acc = x_al.astype(np.float64)
        y_acc = y_al.astype(np.float64)
        if log1p:
            x_acc = np.log1p(np.maximum(x_acc, 0.0))
            y_acc = np.log1p(np.maximum(y_acc, 0.0))
        if smooth_seconds > 1:
            # Guard against empty arrays (shouldn't happen when common.size>=3, but keep safe)
            if x_acc.size > 0:
                x_acc = moving_average(x_acc, smooth_seconds)
            if y_acc.size > 0:
                y_acc = moving_average(y_acc, smooth_seconds)

        # Recompute observed corr for the best lag after preprocessing (more consistent than reusing search corr)
        observed_corr = corr_pearson(x_acc, y_acc)

        # Enforce the user-requested minimum overlap for *acceptance*.
        if int(common.size) < int(min_overlap):
            # Still record stability across halves for diagnostics
            mid = int(common.size) // 2
            c1 = corr_pearson(x_acc[:mid], y_acc[:mid]) if mid >= 3 else float("nan")
            c2 = corr_pearson(x_acc[mid:], y_acc[mid:]) if (int(common.size) - mid) >= 3 else float("nan")
            half_min = float(np.nanmin([c1, c2]))
            acc = AcceptanceResult(False, float("nan"), half_min, "below_min_overlap_req")
        else:
            acc = null_calibrated_acceptance(
                aligned_x=x_acc,
                aligned_y=y_acc,
                observed_corr=float(observed_corr),
                rng=rng,
                iters=null_iters,
                quantile=null_quantile,
                min_shift=max(10, min(60, overlap_seconds // 10 if overlap_seconds > 0 else 30)),
                min_half_corr=min_half_corr,
            )

        # Keep best_corr consistent with the actually-used preprocessing
        best_corr = float(observed_corr)

    if math.isnan(best_corr):
        notes.append("no_defined_corr")
    if overlap_seconds < min_overlap:
        notes.append("low_overlap")
    if not math.isnan(lag_res.peak_prominence) and lag_res.peak_prominence < 0.02:
        notes.append("flat_peak")
    notes.append(f"accept_reason={acc.reason}")

    # Save curves
    if save_curves:
        curves_dir = out_dir / "curves"
        safe_mkdir(curves_dir)
        curve_path = curves_dir / f"{run.family}__{run.canon_stem}__{cache_key}.json"
        with curve_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "family": run.family,
                    "canon_stem": run.canon_stem,
                    "pcap_path": str(run.pcap_path),
                    "txt_path": str(run.txt_path),
                    "time_mode": rad_time_mode,
                    "best_lag": best_lag,
                    "best_corr": best_corr,
                    "corr_by_lag": lag_res.corr_by_lag,
                },
                f,
                indent=2,
            )

    # Derived absolute offsets for timestamped telemetry
    abs_start_diff = None
    abs_offset = None
    if rad_time_mode == "timestamp" and rad_t0_epoch is not None:
        abs_start_diff = float(net_t0_epoch - float(rad_t0_epoch))
        abs_offset = float(abs_start_diff + float(best_lag))

    return RunResult(
        family=run.family,
        canon_stem=run.canon_stem,
        pcap_path=str(run.pcap_path),
        txt_path=str(run.txt_path),
        net_t0_epoch=float(net_t0_epoch),
        net_seconds=int(net_seconds),
        net_packet_count=int(net_packet_count),
        rad_time_mode=str(rad_time_mode),
        rad_t0_epoch=(float(rad_t0_epoch) if rad_t0_epoch is not None else None),
        rad_seconds=int(rad_seconds),
        rad_row_count=int(rad_row_count),
        rad_ue_count=int(rad_ue_count),
        rad_bytes_mode=str(rad_bytes_mode),
        rad_reset_events=int(rad_reset_events),
        rad_parse_errors=int(rad_parse_errors),
        rad_missing_timestamp_rows=int(rad_missing_ts_rows),
        best_lag_seconds=int(best_lag),
        best_corr=float(best_corr),
        overlap_seconds=int(overlap_seconds),
        accepted=bool(acc.accepted),
        null_threshold=float(acc.null_threshold),
        half_corr_min=float(acc.half_corr_min),
        peak_prominence=float(lag_res.peak_prominence),
        abs_start_diff_seconds=(float(abs_start_diff) if abs_start_diff is not None else None),
        abs_offset_seconds=(float(abs_offset) if abs_offset is not None else None),
        notes=";".join(notes) if notes else "",
    )


# -----------------------------
# Plotting
# -----------------------------


def maybe_plot_run(
    res: RunResult,
    out_dir: Path,
    net_secs: np.ndarray,
    net_vals: np.ndarray,
    rad_secs: np.ndarray,
    rad_vals: np.ndarray,
    lag: int,
    smooth_seconds: int,
    log1p: bool,
) -> None:
    import matplotlib.pyplot as plt

    plots_dir = out_dir / "plots"
    safe_mkdir(plots_dir)

    common, x, y = align_on_lag(net_secs, net_vals, rad_secs, rad_vals, lag)
    if common.size < 3:
        return

    x2 = x.copy()
    y2 = y.copy()
    if log1p:
        x2 = np.log1p(np.maximum(x2, 0.0))
        y2 = np.log1p(np.maximum(y2, 0.0))
    if smooth_seconds > 1:
        x2 = moving_average(x2, smooth_seconds)
        y2 = moving_average(y2, smooth_seconds)

    # Normalize for overlay readability
    def _z(a: np.ndarray) -> np.ndarray:
        s = np.std(a)
        return (a - np.mean(a)) / (s if s > 0 else 1.0)

    plt.figure(figsize=(11, 4))
    plt.plot(common, _z(x2), label="net (z)")
    plt.plot(common, _z(y2), label="radio (z)")
    # Include accept/reject context directly in the plot title
    reason = "accepted" if bool(res.accepted) else "rejected"
    if isinstance(res.notes, str) and "accept_reason=" in res.notes:
        reason = res.notes.split("accept_reason=", 1)[1].split(";", 1)[0]
    plt.title(
        f"{res.family}/{res.canon_stem}  accepted={bool(res.accepted)}  reason={reason}\n"
        f"lag={lag}s  corr={res.best_corr:.3f}  mode={res.rad_time_mode}"
    )
    plt.xlabel("seconds (relative)")
    plt.legend()
    plt.tight_layout()
    out = plots_dir / f"overlay__{res.family}__{res.canon_stem}__{'acc' if bool(res.accepted) else 'rej'}.png"
    plt.savefig(out, dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-runs", required=True, help="paired_runs.csv")
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--pcap-reader", default="auto", choices=["auto", "classic", "tshark_fields"])
    ap.add_argument("--telemetry-bytes-mode", default="auto", choices=["auto", "cumulative", "instantaneous"])
    ap.add_argument("--search-window-seconds", type=int, default=1800)
    ap.add_argument("--coarse-step", type=int, default=10)
    ap.add_argument("--fine-window", type=int, default=120)
    ap.add_argument("--min-overlap-seconds", type=int, default=300)
    ap.add_argument("--smooth-seconds", type=int, default=3)
    ap.add_argument("--log1p", action="store_true")
    ap.add_argument("--peak-exclusion", type=int, default=10)
    ap.add_argument("--null-iters", type=int, default=200)
    ap.add_argument("--null-quantile", type=float, default=0.99)
    ap.add_argument("--min-half-corr", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save-curves", action="store_true")
    ap.add_argument("--plots", default="none", choices=["none", "top_accepted", "all_accepted", "top_rejected", "all_rejected"])
    ap.add_argument("--max-plots", type=int, default=12)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    paired_runs = load_paired_runs(Path(args.paired_runs))
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)
    cache_dir = None if args.no_cache else out_dir / "cache"

    results: List[RunResult] = []
    errors: List[Tuple[str, str, str]] = []

    for i, run in enumerate(paired_runs, 1):
        tag = f"[{i}/{len(paired_runs)}] {run.family}/{run.canon_stem}"
        print(tag)
        try:
            res = process_one_run(
                run=run,
                out_dir=out_dir,
                cache_dir=cache_dir,
                pcap_reader=args.pcap_reader,
                telemetry_bytes_mode=args.telemetry_bytes_mode,
                search_window=int(args.search_window_seconds),
                coarse_step=int(args.coarse_step),
                fine_window=int(args.fine_window),
                min_overlap=int(args.min_overlap_seconds),
                smooth_seconds=int(args.smooth_seconds),
                log1p=bool(args.log1p),
                peak_exclusion=int(args.peak_exclusion),
                null_iters=int(args.null_iters),
                null_quantile=float(args.null_quantile),
                min_half_corr=float(args.min_half_corr),
                seed=int(args.seed),
                save_curves=bool(args.save_curves),
            )
            results.append(res)
        except Exception as ex:
            msg = repr(ex)
            print(f"  ERROR: {msg}")
            errors.append((run.family, run.canon_stem, msg))

    # Write per-run results
    df = pd.DataFrame([dataclasses.asdict(r) for r in results])
    results_csv = out_dir / "results.csv"
    results_jsonl = out_dir / "results.jsonl"
    df.to_csv(results_csv, index=False)
    with results_jsonl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(dataclasses.asdict(r)) + "\n")

    # Summary
    summary = {
        "n_runs_total": len(paired_runs),
        "n_runs_processed": len(results),
        "n_errors": len(errors),
        "acceptance_overall": float(df["accepted"].mean()) if len(df) else float("nan"),
        "by_family": {},
    }
    if len(df):
        for fam, sub in df.groupby("family"):
            summary["by_family"][str(fam)] = {
                "n": int(len(sub)),
                "accepted": int(sub["accepted"].sum()),
                "acceptance_rate": float(sub["accepted"].mean()),
                "best_corr_median": float(sub["best_corr"].median()),
                "telemetry_time_modes": dict(sub["rad_time_mode"].value_counts().to_dict()),
            }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Errors
    if errors:
        err_path = out_dir / "errors.txt"
        with err_path.open("w", encoding="utf-8") as f:
            for fam, stem, msg in errors:
                f.write(f"{fam},{stem}: {msg}\n")

    # Plot selection (requires re-loading cached series)
    if args.plots != "none" and len(df):
        try:
            import matplotlib  # noqa: F401
        except Exception:
            eprint("Matplotlib not available; skipping plots")
        else:
            plot_df = df.copy()
            if args.plots in ("top_accepted", "all_accepted"):
                plot_df = plot_df[plot_df["accepted"] == True].copy()  # noqa: E712
                plot_df = plot_df.sort_values("best_corr", ascending=False)
                if args.plots == "top_accepted":
                    plot_df = plot_df.head(int(args.max_plots))
            elif args.plots in ("top_rejected", "all_rejected"):
                plot_df = plot_df[plot_df["accepted"] == False].copy()  # noqa: E712
                plot_df = plot_df.sort_values("best_corr", ascending=False)
                if args.plots == "top_rejected":
                    plot_df = plot_df.head(int(args.max_plots))
            else:
                plot_df = plot_df[plot_df["accepted"] == True].copy()  # noqa: E712
                plot_df = plot_df.sort_values("best_corr", ascending=False).head(int(args.max_plots))


            # To plot, we need the extracted series; easiest is to recompute quickly without caching
            # But we can re-load from cache when available.
            for _, row in plot_df.iterrows():
                fam = str(row["family"])
                stem = str(row["canon_stem"])
                pcap_path = Path(str(row["pcap_path"]))
                txt_path = Path(str(row["txt_path"]))
                lag = int(row["best_lag_seconds"])

                # Rebuild relative series (no need to save)
                if args.pcap_reader == "auto":
                    ps = read_pcap_with_tshark_fields(pcap_path) if is_pcapng(pcap_path) else read_pcap_classic_bytes_per_sec(pcap_path)
                elif args.pcap_reader == "classic":
                    ps = read_pcap_classic_bytes_per_sec(pcap_path)
                else:
                    ps = read_pcap_with_tshark_fields(pcap_path)

                net_secs_epoch, net_vals = densify_series(ps.sec_index_epoch, ps.bytes_per_sec.astype(np.float64))
                net_secs = (net_secs_epoch - int(net_secs_epoch[0])).astype(np.int64)

                rs = parse_radio_txt_to_series(txt_path, bytes_mode=args.telemetry_bytes_mode, aggregate_over_ues="sum")
                rad_secs_raw, rad_vals = densify_series(rs.sec_index.astype(np.int64), rs.rate_per_sec.astype(np.float64))
                rad_secs = (rad_secs_raw - int(rad_secs_raw[0])).astype(np.int64)

                rr = RunResult(**row.to_dict())
                maybe_plot_run(
                    res=rr,
                    out_dir=out_dir,
                    net_secs=net_secs,
                    net_vals=net_vals,
                    rad_secs=rad_secs,
                    rad_vals=rad_vals,
                    lag=lag,
                    smooth_seconds=int(args.smooth_seconds),
                    log1p=bool(args.log1p),
                )

    print(f"Processed: {len(results)}/{len(paired_runs)}")
    print(f"Wrote: {results_csv}")
    print(f"Wrote: {summary_path}")
    if errors:
        print(f"Errors: {len(errors)} (see {out_dir / 'errors.txt'})")


if __name__ == "__main__":
    main()
