#!/usr/bin/env python3
"""
analyze_alignment_results.py

Post-process paired_run_alignment_check results.csv to:
  - create alignment-quality tiers (strong/weak/rejected)
  - write run lists for each tier
  - generate summary tables (overall + per family)
  - produce a small set of diagnostic plots

This is meant to support the paper's "multimodal feasibility" section and
to define a conservative "high-confidence aligned subset" for early fusion.

Usage:
  python3 analyze_alignment_results.py --results ./alignment_out/results.csv --out-dir ./alignment_out/analysis

Key outputs:
  - tiers.csv                   (per-run tier assignment + key metrics)
  - tier_summary_by_family.csv  (counts per family)
  - strong_runs.txt             (copy-paste list)
  - plots/*.png                 (figures)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results.csv from paired_run_alignment_check")
    ap.add_argument("--out-dir", required=True, help="Output directory for analysis artifacts")
    ap.add_argument("--strong-corr", type=float, default=0.80, help="Correlation threshold for strong tier")
    ap.add_argument("--strong-prom", type=float, default=0.10, help="Peak prominence threshold for strong tier")
    ap.add_argument("--require-nonflat", action="store_true", help="If set, flat_peak flagged runs cannot be strong")
    args = ap.parse_args()

    results_path = Path(args.results).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    safe_mkdir(out_dir)
    plots_dir = out_dir / "plots"
    safe_mkdir(plots_dir)

    df = pd.read_csv(results_path)
    if "notes" not in df.columns:
        df["notes"] = ""

    df["flat_peak"] = df["notes"].fillna("").astype(str).str.contains("flat_peak")
    df["corr_margin"] = df["best_corr"] - df["null_threshold"]
    df["corr_margin"] = df["corr_margin"].replace([np.inf, -np.inf], np.nan)

    # Tier definition
    strong = (
        (df["accepted"] == True)
        & (df["best_corr"] >= float(args.strong_corr))
        & (df["peak_prominence"].fillna(0.0) >= float(args.strong_prom))
    )
    if args.require_nonflat:
        strong = strong & (~df["flat_peak"])

    high_corr_ambiguous = (df["accepted"] == True) & (df["best_corr"] >= float(args.strong_corr)) & (~strong)
    accepted_weak = (df["accepted"] == True) & (~strong) & (~high_corr_ambiguous)
    rejected = (df["accepted"] == False)

    df["tier"] = np.select(
        [strong, high_corr_ambiguous, accepted_weak, rejected],
        ["strong", "high_corr_ambiguous", "accepted_weak", "rejected"],
        default="rejected",
    )

    # Save tier assignment
    tiers_csv = out_dir / "tiers.csv"
    df.sort_values(["tier", "family", "best_corr"], ascending=[True, True, False]).to_csv(tiers_csv, index=False)

    # Summary tables
    summary_family = (
        df.groupby(["family", "tier"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    summary_family["total"] = summary_family.drop(columns=["family"]).sum(axis=1)
    summary_family_csv = out_dir / "tier_summary_by_family.csv"
    summary_family.to_csv(summary_family_csv, index=False)

    summary_overall = df["tier"].value_counts().rename_axis("tier").reset_index(name="count")
    summary_overall_csv = out_dir / "tier_summary_overall.csv"
    summary_overall.to_csv(summary_overall_csv, index=False)

    # Write run lists (copy-paste friendly)
    strong_list = df[df["tier"] == "strong"].sort_values(["family", "best_corr"], ascending=[True, False])
    strong_txt = out_dir / "strong_runs.txt"
    with strong_txt.open("w", encoding="utf-8") as f:
        for _, r in strong_list.iterrows():
            f.write(f'{r["family"]}/{r["canon_stem"]}  corr={r["best_corr"]:.3f}  lag={int(r["best_lag_seconds"])}s\n')

    weak_list = df[df["tier"].isin(["high_corr_ambiguous", "accepted_weak"])].sort_values(["family", "best_corr"], ascending=[True, False])
    weak_txt = out_dir / "accepted_but_not_strong_runs.txt"
    with weak_txt.open("w", encoding="utf-8") as f:
        for _, r in weak_list.iterrows():
            f.write(f'{r["family"]}/{r["canon_stem"]}  tier={r["tier"]}  corr={r["best_corr"]:.3f}  lag={int(r["best_lag_seconds"])}s  notes={str(r["notes"])}\n')

    # -----------------
    # Plots (matplotlib)
    # -----------------
    import matplotlib.pyplot as plt

    # 1) Stacked bar: tier counts by family
    fams = summary_family["family"].tolist()
    tiers = [t for t in ["strong", "high_corr_ambiguous", "accepted_weak", "rejected"] if t in summary_family.columns]
    bottom = np.zeros(len(fams), dtype=float)
    plt.figure(figsize=(10, 4))
    for t in tiers:
        vals = summary_family[t].values.astype(float)
        plt.bar(fams, vals, bottom=bottom, label=t)
        bottom += vals
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("runs")
    plt.title("Alignment tiers by family")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "tiers_by_family_stacked.png", dpi=150)
    plt.close()

    # 2) Histogram: best_corr accepted vs rejected
    plt.figure(figsize=(8, 4))
    plt.hist(df[df["accepted"] == True]["best_corr"].values, bins=20, alpha=0.7, label="accepted")
    plt.hist(df[df["accepted"] == False]["best_corr"].values, bins=20, alpha=0.7, label="rejected")
    plt.xlabel("best_corr")
    plt.ylabel("count")
    plt.title("best_corr distribution (accepted vs rejected)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "best_corr_hist_accepted_vs_rejected.png", dpi=150)
    plt.close()

    # 3) Boxplot: best_corr by family (accepted only)
    acc = df[df["accepted"] == True].copy()
    plt.figure(figsize=(10, 4))
    data = [acc[acc["family"] == fam]["best_corr"].values for fam in sorted(acc["family"].unique())]
    plt.boxplot(data, labels=sorted(acc["family"].unique()))
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("best_corr")
    plt.title("best_corr by family (accepted runs)")
    plt.tight_layout()
    plt.savefig(plots_dir / "best_corr_box_by_family_accepted.png", dpi=150)
    plt.close()

    # 4) Lag histogram: strong runs
    if len(strong_list):
        plt.figure(figsize=(8, 4))
        plt.hist(strong_list["best_lag_seconds"].values.astype(int), bins=20)
        plt.xlabel("best_lag_seconds")
        plt.ylabel("count")
        plt.title("best_lag_seconds distribution (strong tier)")
        plt.tight_layout()
        plt.savefig(plots_dir / "lag_hist_strong.png", dpi=150)
        plt.close()

    # 5) Absolute offset histogram: strong + timestamp mode
    strong_ts = strong_list[strong_list["rad_time_mode"] == "timestamp"].copy()
    if len(strong_ts) and "abs_offset_seconds" in strong_ts.columns:
        vals = strong_ts["abs_offset_seconds"].dropna().values.astype(float)
        if len(vals):
            plt.figure(figsize=(8, 4))
            plt.hist(vals, bins=20)
            plt.xlabel("abs_offset_seconds")
            plt.ylabel("count")
            plt.title("abs_offset_seconds (strong tier, timestamped telemetry)")
            plt.tight_layout()
            plt.savefig(plots_dir / "abs_offset_hist_strong_timestamped.png", dpi=150)
            plt.close()

    # 6) Scatter: best_corr vs peak_prominence
    plt.figure(figsize=(8, 5))
    x = df["peak_prominence"].fillna(0.0).values
    y = df["best_corr"].values
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel("peak_prominence")
    plt.ylabel("best_corr")
    plt.title("best_corr vs peak_prominence (all runs)")
    plt.tight_layout()
    plt.savefig(plots_dir / "scatter_corr_vs_prominence.png", dpi=150)
    plt.close()

    print(f"Wrote: {tiers_csv}")
    print(f"Wrote: {summary_family_csv}")
    print(f"Wrote: {summary_overall_csv}")
    print(f"Wrote: {strong_txt}")
    print(f"Wrote plots under: {plots_dir}")


if __name__ == "__main__":
    main()
