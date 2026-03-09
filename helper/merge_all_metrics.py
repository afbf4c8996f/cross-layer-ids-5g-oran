"""
merge_all_metrics.py

Merge metrics_binary.csv and metrics_multiclass.csv from all 7 model
output directories into unified CSVs. Also produce a quick summary.

Usage:
    python merge_all_metrics.py \
        --root /path/to/dir/of/finished/optuna/tuning \
        --out  /where/to/write/merged
"""

import argparse
import pandas as pd
from pathlib import Path

MODELS = ["resmlp", "gru", "tcn", "transformer", "logreg", "xgboost", "rf"]

CSV_FILES = [
    "metrics_binary.csv",
    "metrics_multiclass.csv",
    "binary_operating_metrics_fpr0.01.csv",
    "ttd_summary_fpr0.01.csv",
]


def merge_task(root: Path, fname: str) -> pd.DataFrame:
    frames = []
    for m in MODELS:
        p = root / m / "metrics" / fname
        if not p.exists():
            print(f"  [WARN] Missing: {p}")
            continue
        df = pd.read_csv(p)
        # Add model column if not present
        if "model" not in df.columns:
            df.insert(0, "model", m)
        frames.append(df)
        print(f"  [OK] {m}/metrics/{fname}: {len(df)} rows")
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    return merged


def summary_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Best test metrics per model × modality × fusion, averaged across splits/seeds."""
    if df.empty:
        return df

    # Identify the test partition
    test_df = df[df["partition"] == "test"].copy() if "partition" in df.columns else df[df["part"] == "test"].copy()

    group_cols = ["model", "modality", "fusion"]
    group_cols = [c for c in group_cols if c in test_df.columns]

    metric_cols = [c for c in test_df.columns if c in [
        "roc_auc", "avg_precision", "f1", "accuracy", "balanced_accuracy",
        "precision", "recall", "log_loss", "brier_score",
    ]]

    if not group_cols or not metric_cols:
        return pd.DataFrame()

    summary = test_df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).round(4)
    # Flatten multi-level columns
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    return summary


def summary_multiclass(df: pd.DataFrame) -> pd.DataFrame:
    """Best test metrics per model × modality × fusion, averaged across splits/seeds."""
    if df.empty:
        return df

    test_df = df[df["partition"] == "test"].copy() if "partition" in df.columns else df[df["part"] == "test"].copy()

    group_cols = ["model", "modality", "fusion"]
    group_cols = [c for c in group_cols if c in test_df.columns]

    metric_cols = [c for c in test_df.columns if c in [
        "macro_f1", "weighted_f1", "accuracy", "balanced_accuracy",
        "macro_precision", "macro_recall", "log_loss", "cohen_kappa",
    ]]

    if not group_cols or not metric_cols:
        return pd.DataFrame()

    summary = test_df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).round(4)
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root dir containing per-model output folders")
    parser.add_argument("--out", required=True, help="Output directory for merged files")
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- Merge all 4 CSV types ---
    all_dfs = {}
    for fname in CSV_FILES:
        print(f"\n=== Merging {fname} ===")
        df = merge_task(root, fname)
        if not df.empty:
            out_name = "all_" + fname
            out_path = out / out_name
            df.to_csv(out_path, index=False)
            print(f"\n  Saved: {out_path} ({len(df)} rows)")
            all_dfs[fname] = df
        else:
            print(f"  [WARN] No data found for {fname}!")

    # --- Binary summary ---
    bin_df = all_dfs.get("metrics_binary.csv", pd.DataFrame())
    if not bin_df.empty:
        bin_summary = summary_binary(bin_df)
        if not bin_summary.empty:
            bin_sum_path = out / "summary_binary_test.csv"
            bin_summary.to_csv(bin_sum_path, index=False)
            print(f"\n  Saved: {bin_sum_path} ({len(bin_summary)} rows)")
            print("\n--- Binary Test Summary (mean across seeds/splits) ---")
            print(bin_summary.to_string(index=False))

    # --- Multiclass summary ---
    mc_df = all_dfs.get("metrics_multiclass.csv", pd.DataFrame())
    if not mc_df.empty:
        mc_summary = summary_multiclass(mc_df)
        if not mc_summary.empty:
            mc_sum_path = out / "summary_multiclass_test.csv"
            mc_summary.to_csv(mc_sum_path, index=False)
            print(f"\n  Saved: {mc_sum_path} ({len(mc_summary)} rows)")
            print("\n--- Multiclass Test Summary (mean across seeds/splits) ---")
            print(mc_summary.to_string(index=False))

    # --- Quick cross-model leaderboard ---
    print("\n\n=== LEADERBOARD ===")

    # Detect partition column name
    def get_test(df):
        if "partition" in df.columns:
            return df[df["partition"] == "test"]
        elif "part" in df.columns:
            return df[df["part"] == "test"]
        return df

    if not bin_df.empty:
        test_bin = get_test(bin_df)
        if not test_bin.empty and "roc_auc" in test_bin.columns:
            lb = test_bin.groupby("model")["roc_auc"].mean().sort_values(ascending=False)
            print("\nBinary Test ROC-AUC (mean across all configs):")
            for m, v in lb.items():
                print(f"  {m:15s} {v:.4f}")

    if not mc_df.empty:
        test_mc = get_test(mc_df)
        if not test_mc.empty and "macro_f1" in test_mc.columns:
            lb = test_mc.groupby("model")["macro_f1"].mean().sort_values(ascending=False)
            print("\nMulticlass Test Macro-F1 (mean across all configs):")
            for m, v in lb.items():
                print(f"  {m:15s} {v:.4f}")

    # --- TTD summary ---
    ttd_df = all_dfs.get("ttd_summary_fpr0.01.csv", pd.DataFrame())
    if not ttd_df.empty:
        print("\n\nTTD Summary (Time-to-Detect @ FPR=0.01):")
        group_cols = [c for c in ["model", "modality", "fusion"] if c in ttd_df.columns]
        ttd_metric = [c for c in ttd_df.columns if "ttd" in c.lower() or "median" in c.lower() or "mean" in c.lower()]
        if group_cols and ttd_metric:
            ttd_agg = ttd_df.groupby(group_cols)[ttd_metric[0]].mean().sort_values()
            print(ttd_agg.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
