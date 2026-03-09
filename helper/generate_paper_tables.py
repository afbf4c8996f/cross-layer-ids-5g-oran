"""
generate_paper_tables.py  (v2 — audited & corrected)
=====================================================
Reads the 4 merged CSVs from Training&Evaluaation dir and produces
all paper table data for Act B (Binary Detection) and Act C (Multiclass).

Filters applied to ALL tables:
  - W=10 (HPO setting)
  - part=test
  - stratified splits only (time_ordered=False)
  - 10 seeds per cell

Additional filters for operational metrics:
  - threshold_policy='benign_runs_only'
    (attacks start at t=0, so pre_attack is degenerate;
     all_benign_labeled includes windows from attack runs)
  - ttd_mode='flow_onset' (for TTD tables)

Audit log (what was checked and verified):
  1. time_ordered is boolean dtype — == False works correctly
  2. threshold_policy filter reduces ops/TTD from 3x to 1x per seed
  3. benign_runs_only != all_benign_labeled (different thresholds)
  4. benign_runs_only == pre_attack (confirms no pre-attack windows)
  5. No duplicate rows in binary or multiclass CSVs
  6. feature_ablation is consistently 'drop_history_onehots'
  7. f1_macro > f1_weighted is consistent (DoS=hardest + most samples)

Usage:
  python generate_paper_tables.py --merged /path/to/merged/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────
MODELS_ORDER = ['LogReg', 'XGBoost', 'RF', 'ResMLP', 'GRU', 'TCN', 'Transformer']
MODEL_MAP = {
    'logreg': 'LogReg', 'xgboost': 'XGBoost', 'rf': 'RF',
    'resmlp': 'ResMLP', 'gru': 'GRU', 'tcn': 'TCN', 'transformer': 'Transformer'
}
MODEL_TYPE = {
    'LogReg': 'Linear', 'XGBoost': 'Boosted Trees', 'RF': 'Bagged Trees',
    'ResMLP': 'Tabular DL', 'GRU': 'Recurrent DL', 'TCN': 'Conv. DL',
    'Transformer': 'Attention DL'
}
BINARY_SYSTEMS = ['network_only', 'radio_only', 'fusion_stacked', 'fusion_mean']
MC_SYSTEMS = ['network_only', 'radio_only', 'fusion_mean']
SYS_SHORT = {
    'network_only': 'Net', 'radio_only': 'Radio',
    'fusion_stacked': 'Fus.Stack', 'fusion_mean': 'Fus.Mean'
}

# The ONLY correct threshold policy for this dataset
THRESHOLD_POLICY = 'benign_runs_only'


def load_and_filter(merged_dir: Path):
    """Load all 4 CSVs and apply standard + policy filters."""
    bin_df = pd.read_csv(merged_dir / 'all_metrics_binary.csv')
    mc_df = pd.read_csv(merged_dir / 'all_metrics_multiclass.csv')
    ops_df = pd.read_csv(merged_dir / 'all_binary_operating_metrics_fpr0.01.csv')
    ttd_df = pd.read_csv(merged_dir / 'all_ttd_summary_fpr0.01.csv')

    # --- Standard filters: W=10, test, stratified ---
    def filt_base(df):
        return df[(df['W'] == 10) & (df['part'] == 'test') &
                  (df['time_ordered'] == False)].copy()

    bin_t = filt_base(bin_df)
    mc_t = filt_base(mc_df)

    # --- Operational metrics: ALSO filter by threshold_policy ---
    ops_t = ops_df[(ops_df['W'] == 10) & (ops_df['part'] == 'test') &
                   (ops_df['time_ordered'] == False) &
                   (ops_df['threshold_policy'] == THRESHOLD_POLICY)].copy()

    ttd_t = ttd_df[(ttd_df['W'] == 10) & (ttd_df['part'] == 'test') &
                   (ttd_df['time_ordered'] == False) &
                   (ttd_df['threshold_policy'] == THRESHOLD_POLICY) &
                   (ttd_df['ttd_mode'] == 'flow_onset')].copy()
    for col in ['detect_rate', 'ttd_median']:
        ttd_t[col] = pd.to_numeric(ttd_t[col], errors='coerce')

    return bin_t, mc_t, ops_t, ttd_t


def verify_filters(bin_t, mc_t, ops_t, ttd_t):
    """Verify expected row counts after filtering."""
    print("=" * 80)
    print("  DATA VERIFICATION")
    print("=" * 80)

    # Binary: 7 models x 4 systems x 10 seeds = 280
    combos_bin = bin_t.groupby(['model', 'system']).ngroups
    seeds_bin = bin_t.groupby(['model', 'system'])['seed'].count().unique()
    print(f"  Binary:     {len(bin_t)} rows | {combos_bin} combos (expect 28) | seeds/combo: {list(seeds_bin)}")
    assert combos_bin == 28, f"Expected 28 binary combos, got {combos_bin}"
    assert list(seeds_bin) == [10], f"Expected [10] seeds, got {list(seeds_bin)}"

    # Multiclass: 7 models x 3 systems x 10 seeds = 210
    combos_mc = mc_t.groupby(['model', 'system']).ngroups
    seeds_mc = mc_t.groupby(['model', 'system'])['seed'].count().unique()
    print(f"  Multiclass: {len(mc_t)} rows | {combos_mc} combos (expect 21) | seeds/combo: {list(seeds_mc)}")
    assert combos_mc == 21, f"Expected 21 multiclass combos, got {combos_mc}"
    assert list(seeds_mc) == [10], f"Expected [10] seeds, got {list(seeds_mc)}"

    # Ops: 7 models x 4 systems x 10 seeds x 1 policy = 280
    combos_ops = ops_t.groupby(['model', 'system']).ngroups
    seeds_ops = ops_t.groupby(['model', 'system'])['seed'].count().unique()
    print(f"  Ops@FPR1%:  {len(ops_t)} rows | {combos_ops} combos (expect 28) | seeds/combo: {list(seeds_ops)}")
    print(f"              threshold_policy='{THRESHOLD_POLICY}'")
    assert combos_ops == 28, f"Expected 28 ops combos, got {combos_ops}"
    assert list(seeds_ops) == [10], f"Expected [10] seeds, got {list(seeds_ops)}"

    # TTD: 7 models x 4 systems x 10 seeds x 1 policy x 1 mode = 280
    combos_ttd = ttd_t.groupby(['model', 'system']).ngroups
    seeds_ttd = ttd_t.groupby(['model', 'system'])['seed'].count().unique()
    print(f"  TTD:        {len(ttd_t)} rows | {combos_ttd} combos (expect 28) | seeds/combo: {list(seeds_ttd)}")
    print(f"              threshold_policy='{THRESHOLD_POLICY}', ttd_mode='flow_onset'")
    assert combos_ttd == 28, f"Expected 28 TTD combos, got {combos_ttd}"
    assert list(seeds_ttd) == [10], f"Expected [10] seeds, got {list(seeds_ttd)}"

    print(f"\n  Models:  {sorted(bin_t['model'].unique())}")
    print(f"  Seeds:   {sorted(bin_t['seed'].unique())}")
    print("\n  ALL VERIFICATION CHECKS PASSED.\n")


def get_stat(df, model_key, system, metric):
    """Get mean and std for a model x system x metric combination."""
    sub = df[(df['model'] == model_key) & (df['system'] == system)]
    if len(sub) == 0:
        return np.nan, np.nan
    return sub[metric].mean(), sub[metric].std()


def print_table(df, metric, title, systems_list, prec=3, lower_better=False):
    """Print a formatted table with mean +/- std."""
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"  W=10 | test | stratified | n=10 seeds per cell")
    if lower_better:
        print(f"  (lower is better)")
    print(f"{'=' * 90}")

    hdr = f"  {'Model':12s}"
    for s in systems_list:
        hdr += f"  {SYS_SHORT[s]:>15s}"
    print(hdr)
    print("  " + "-" * (14 + 17 * len(systems_list)))

    for model_name in MODELS_ORDER:
        model_key = [k for k, v in MODEL_MAP.items() if v == model_name][0]
        row = f"  {model_name:12s}"
        for sys in systems_list:
            m, s = get_stat(df, model_key, sys, metric)
            if np.isnan(m):
                cell = "---"
            else:
                cell = f"{m:.{prec}f}+/-{s:.{prec}f}"
            row += f"  {cell:>15s}"
        print(row)


def print_detection_rate(ttd_t):
    """Print detection rate and reliability summary."""
    print(f"\n{'=' * 90}")
    print(f"  DETECTION RATE -- flow_onset, benign_runs_only threshold")
    print(f"  W=10 | test | stratified | n=10 seeds per cell")
    print(f"{'=' * 90}")

    hdr = f"  {'Model':12s}"
    for s in BINARY_SYSTEMS:
        hdr += f"  {SYS_SHORT[s]:>10s}"
    hdr += "   Radio: blind seeds (DR=0)"
    print(hdr)
    print("  " + "-" * 80)

    for model_name in MODELS_ORDER:
        mk = [k for k, v in MODEL_MAP.items() if v == model_name][0]
        row = f"  {model_name:12s}"
        for sys in BINARY_SYSTEMS:
            sub = ttd_t[(ttd_t['model'] == mk) & (ttd_t['system'] == sys)]
            if len(sub) > 0:
                row += f"  {sub['detect_rate'].mean():>10.3f}"
            else:
                row += f"  {'---':>10s}"
        # Blind seeds for radio (detect_rate == 0 on any seed)
        radio = ttd_t[(ttd_t['model'] == mk) & (ttd_t['system'] == 'radio_only')]
        n_zero = (radio['detect_rate'] == 0).sum()
        n_total = len(radio)
        row += f"   {n_zero}/{n_total}"
        print(row)


def print_ttd_summary(ttd_t):
    """Print TTD for configs with reasonable detection rate."""
    print(f"\n{'=' * 90}")
    print(f"  TTD MEDIAN (seconds) -- configs with Detection Rate > 0.80")
    print(f"  threshold_policy='{THRESHOLD_POLICY}'")
    print(f"{'=' * 90}")
    print(f"  {'Model':12s} {'System':>10s}   {'Det.Rate':>8s}  {'TTD_med':>8s}")
    print("  " + "-" * 50)

    for model_name in MODELS_ORDER:
        mk = [k for k, v in MODEL_MAP.items() if v == model_name][0]
        for sys in ['radio_only', 'fusion_mean', 'network_only']:
            sub = ttd_t[(ttd_t['model'] == mk) & (ttd_t['system'] == sys)]
            if len(sub) == 0:
                continue
            dr = sub['detect_rate'].mean()
            ttd_med = sub['ttd_median'].replace([np.inf], np.nan).median()
            if dr > 0.80:
                ttd_str = f"{ttd_med:.1f}s" if not np.isnan(ttd_med) else "inf"
                print(f"  {model_name:12s} {SYS_SHORT[sys]:>10s}   {dr:>8.3f}  {ttd_str:>8s}")


def print_multiclass_fusion_gain(mc_t):
    """Print fusion gain summary for multiclass."""
    print(f"\n{'=' * 90}")
    print(f"  MULTICLASS FUSION GAIN: delta F1-macro (Fusion_mean - Radio)")
    print(f"{'=' * 90}")
    print(f"  {'Model':12s}  {'Radio':>8s}  {'Fusion':>8s}  {'delta':>8s}  {'pts':>8s}")
    print("  " + "-" * 50)

    for model_name in MODELS_ORDER:
        mk = [k for k, v in MODEL_MAP.items() if v == model_name][0]
        radio_m, _ = get_stat(mc_t, mk, 'radio_only', 'f1_macro')
        fus_m, _ = get_stat(mc_t, mk, 'fusion_mean', 'f1_macro')
        delta = fus_m - radio_m
        print(f"  {model_name:12s}  {radio_m:>8.3f}  {fus_m:>8.3f}  {delta:>+8.3f}  ({delta*100:+.1f})")


def print_summary(bin_t, mc_t, ops_t, ttd_t):
    """Print key verified findings."""
    print("\n\n" + "#" * 90)
    print("#  SUMMARY OF VERIFIED FINDINGS")
    print(f"#  Source: CSV, W=10, test, stratified, policy={THRESHOLD_POLICY}")
    print("#" * 90)

    # Radio dominance range
    gaps = []
    for mk in MODEL_MAP.keys():
        r, _ = get_stat(bin_t, mk, 'radio_only', 'roc_auc')
        n, _ = get_stat(bin_t, mk, 'network_only', 'roc_auc')
        gaps.append(r - n)
    print(f"\n  Radio dominance (ROC-AUC gap): {min(gaps)*100:.1f} to {max(gaps)*100:.1f} pts across 7 models")

    # Fusion gain range (multiclass)
    gains = []
    for mk in MODEL_MAP.keys():
        r, _ = get_stat(mc_t, mk, 'radio_only', 'f1_macro')
        f, _ = get_stat(mc_t, mk, 'fusion_mean', 'f1_macro')
        gains.append(f - r)
    print(f"  Fusion F1-macro gain: +{min(gains)*100:.1f} to +{max(gains)*100:.1f} pts across 7 models")
    print(f"  Fusion helps ALL 7 models: {all(g > 0 for g in gains)}")

    # Best configs
    best_bin = bin_t.groupby(['model', 'system'])['roc_auc'].mean()
    best_idx = best_bin.idxmax()
    print(f"  Best binary ROC-AUC: {MODEL_MAP[best_idx[0]]} {SYS_SHORT[best_idx[1]]} = {best_bin.max():.3f}")

    best_mc = mc_t.groupby(['model', 'system'])['f1_macro'].mean()
    best_mc_idx = best_mc.idxmax()
    print(f"  Best multiclass F1-macro: {MODEL_MAP[best_mc_idx[0]]} {SYS_SHORT[best_mc_idx[1]]} = {best_mc.max():.3f}")

    # Fusion_stacked instability
    print(f"\n  Fusion_stacked ROC-AUC std (instability check):")
    for model_name in MODELS_ORDER:
        mk = [k for k, v in MODEL_MAP.items() if v == model_name][0]
        _, s = get_stat(bin_t, mk, 'fusion_stacked', 'roc_auc')
        flag = " << UNSTABLE" if s > 0.10 else ""
        print(f"    {model_name:12s}: std={s:.3f}{flag}")

    # Fusion_stacked hurts detection rate
    print(f"\n  Fusion_stacked vs Radio detection rate:")
    for model_name in MODELS_ORDER:
        mk = [k for k, v in MODEL_MAP.items() if v == model_name][0]
        radio_dr = ttd_t[(ttd_t['model'] == mk) & (ttd_t['system'] == 'radio_only')]['detect_rate'].mean()
        fs_dr = ttd_t[(ttd_t['model'] == mk) & (ttd_t['system'] == 'fusion_stacked')]['detect_rate'].mean()
        delta = fs_dr - radio_dr
        flag = " << HURTS" if delta < -0.05 else ""
        print(f"    {model_name:12s}: radio={radio_dr:.3f}  stacked={fs_dr:.3f}  d={delta:+.3f}{flag}")

    # Classical ML vs DL reliability
    print(f"\n  Detection reliability (radio_only, seeds with DR=0 out of 10):")
    for model_name in MODELS_ORDER:
        mk = [k for k, v in MODEL_MAP.items() if v == model_name][0]
        radio = ttd_t[(ttd_t['model'] == mk) & (ttd_t['system'] == 'radio_only')]
        n_zero = (radio['detect_rate'] == 0).sum()
        n_total = len(radio)
        label = MODEL_TYPE[model_name]
        flag = " UNRELIABLE" if n_zero > 3 else (" CAUTION" if n_zero > 0 else " OK")
        print(f"    {model_name:12s} ({label:14s}): {n_zero}/{n_total} blind  [{flag}]")

    # Fusion_mean best log-loss
    print(f"\n  Fusion_mean best log-loss (binary) -- wins for how many models?")
    fm_wins = 0
    for mk in MODEL_MAP.keys():
        best_sys = None
        best_ll = np.inf
        for sys in BINARY_SYSTEMS:
            m, _ = get_stat(bin_t, mk, sys, 'log_loss')
            if m < best_ll:
                best_ll = m
                best_sys = sys
        if best_sys == 'fusion_mean':
            fm_wins += 1
        print(f"    {MODEL_MAP[mk]:12s}: best={SYS_SHORT[best_sys]} ({best_ll:.3f})")
    print(f"    Fusion_mean wins: {fm_wins}/7 models")

    print(f"\n  Script complete. All values verified from CSV source.")


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables from merged CSVs")
    parser.add_argument('--merged', type=str, required=True,
                        help='Path to merged/ directory containing all_*.csv files')
    args = parser.parse_args()
    merged_dir = Path(args.merged)

    # Load and Filter
    bin_t, mc_t, ops_t, ttd_t = load_and_filter(merged_dir)

    # Verification
    verify_filters(bin_t, mc_t, ops_t, ttd_t)

    # ACT B: BINARY DETECTION
    print("\n" + "#" * 90)
    print("#  ACT B: BINARY DETECTION -- All 7 models x 4 systems")
    print("#" * 90)

    print_table(bin_t, 'roc_auc', 'BINARY ROC-AUC', BINARY_SYSTEMS)
    print_table(bin_t, 'pr_auc', 'BINARY PR-AUC', BINARY_SYSTEMS)
    print_table(bin_t, 'log_loss', 'BINARY Log-Loss', BINARY_SYSTEMS, lower_better=True)
    print_table(ops_t, 'tpr',
                f'BINARY TPR @ 1% FPR (threshold_policy={THRESHOLD_POLICY})',
                BINARY_SYSTEMS)
    print_detection_rate(ttd_t)
    print_ttd_summary(ttd_t)

    # ACT C: MULTICLASS
    print("\n\n" + "#" * 90)
    print("#  ACT C: MULTICLASS -- All 7 models x 3 systems (no fusion_stacked)")
    print("#" * 90)

    print_table(mc_t, 'f1_macro', 'MULTICLASS F1-macro', MC_SYSTEMS)
    print_table(mc_t, 'acc', 'MULTICLASS Accuracy', MC_SYSTEMS)
    print_table(mc_t, 'log_loss', 'MULTICLASS Log-Loss', MC_SYSTEMS, lower_better=True)
    print_multiclass_fusion_gain(mc_t)

    # Summary
    print_summary(bin_t, mc_t, ops_t, ttd_t)


if __name__ == "__main__":
    main()