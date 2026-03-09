"""
Experiment I2+I3 Combined: Three-Factor Invariance of DoS→Benign Confusion
Purpose: Compute DoS→Benign confusion rate (%) for ALL combinations of:
    - 7 model architectures  (Factor 1: architecture invariance)
    - 3 feature sets          (Factor 2: feature-set invariance)
    - 2 window sizes          (Factor 3: window-size invariance)
Total: 7 × 3 × 2 = 42 configurations, each aggregated over 10 seeds.

Data source: Training&Evaluation prediction parquets
Ground truth: y_cat labels (window-level multiclass)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from datetime import datetime

# --- CONFIGURATION (identical to verified I2 and I3 scripts) ---
BASE = Path('/path/to/final/dir/of/final/Training & Evaluation/')
OUTPUT_DIR = Path('/where/to/write/audit_reports')

CLASSES = ['Benign', 'bruteforce', 'ddos', 'dos', 'probe', 'web']
SEEDS = range(42, 52)  # 42..51, 10 seeds

MODELS = ['rf', 'logreg', 'xgboost', 'gru', 'resmlp', 'tcn', 'transformer']
SYSTEMS = ['network_only', 'radio_only', 'fusion_mean']
WINDOWS = [5, 10]

# Indices for DoS→Benign extraction
DOS_IDX = CLASSES.index('dos')    # 3
BEN_IDX = CLASSES.index('Benign') # 0


def run_audit():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"I2_I3_combined_invariance_{timestamp}.txt"

    # Storage: (model, system, W) -> {'cm': ndarray, 'n_files': int}
    results = {}

    with open(report_path, 'w') as f:
        def dual_print(text):
            print(text)
            f.write(text + '\n')

        dual_print(f"THREE-FACTOR INVARIANCE AUDIT — {datetime.now()}")
        dual_print(f"Data Source: {BASE}")
        dual_print(f"Models: {MODELS}")
        dual_print(f"Systems: {SYSTEMS}")
        dual_print(f"Windows: {WINDOWS}")
        dual_print(f"Seeds: {list(SEEDS)}")
        dual_print(f"Target metric: DoS → Benign confusion rate (%)\n")

        # ---- PHASE 1: Read all prediction parquets ----
        dual_print("=" * 80)
        dual_print("PHASE 1: Loading prediction parquets")
        dual_print("=" * 80)

        for W in WINDOWS:
            for model in MODELS:
                for system in SYSTEMS:
                    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
                    n_files = 0

                    for seed in SEEDS:
                        path_obj = (BASE / model / 'predictions' /
                                    f'stratified_seed{seed}' /
                                    f'W{W}_S2' / model / 'multiclass' /
                                    f'{system}_test.parquet')

                        if path_obj.exists():
                            df = pd.read_parquet(path_obj)
                            label_col = [c for c in df.columns
                                         if c.lower() in ['label', 'y_true', 'y']][0]
                            pred_col = [c for c in df.columns
                                        if c.lower() in ['pred', 'y_pred', 'prediction']][0]
                            cm += confusion_matrix(
                                df[label_col], df[pred_col], labels=CLASSES
                            )
                            n_files += 1

                    results[(model, system, W)] = {
                        'cm': cm, 'n_files': n_files
                    }
                    status = f"OK ({n_files}/10 seeds)" if n_files > 0 else "MISSING"
                    dual_print(f"  W={W:2d} {model:12s} {system:14s}: {status}")

        # ---- PHASE 2: DoS→Benign rates, organized by window size ----
        dual_print(f"\n{'=' * 80}")
        dual_print("PHASE 2: DoS → Benign CONFUSION RATE (%) BY WINDOW SIZE")
        dual_print("=" * 80)

        for W in WINDOWS:
            dual_print(f"\n  W = {W}")
            dual_print(f"  {'Model':12s} {'Network':>10s} {'Radio':>10s} {'Fusion':>10s}")
            dual_print(f"  {'-' * 42}")

            for model in MODELS:
                row_vals = []
                for system in SYSTEMS:
                    r = results[(model, system, W)]
                    if r['n_files'] > 0:
                        cm = r['cm']
                        dos_total = cm[DOS_IDX, :].sum()
                        dos_to_ben = cm[DOS_IDX, BEN_IDX]
                        pct = 100.0 * dos_to_ben / dos_total if dos_total > 0 else 0
                        row_vals.append(f"{pct:8.1f}%")
                    else:
                        row_vals.append(f"{'N/A':>9s}")
                dual_print(f"  {model:12s} {row_vals[0]:>10s} {row_vals[1]:>10s} {row_vals[2]:>10s}")

        # ---- PHASE 3: Summary statistics ----
        dual_print(f"\n{'=' * 80}")
        dual_print("PHASE 3: INVARIANCE SUMMARY STATISTICS")
        dual_print("=" * 80)

        # Collect all valid DoS→Benign percentages
        all_pcts = []
        by_window = {W: [] for W in WINDOWS}
        by_system = {s: [] for s in SYSTEMS}
        by_model = {m: [] for m in MODELS}

        for (model, system, W), r in results.items():
            if r['n_files'] > 0:
                cm = r['cm']
                dos_total = cm[DOS_IDX, :].sum()
                if dos_total > 0:
                    pct = 100.0 * cm[DOS_IDX, BEN_IDX] / dos_total
                    all_pcts.append(pct)
                    by_window[W].append(pct)
                    by_system[system].append(pct)
                    by_model[model].append(pct)

        if all_pcts:
            dual_print(f"\n  OVERALL ({len(all_pcts)} configurations):")
            dual_print(f"    Range: {min(all_pcts):.1f}% – {max(all_pcts):.1f}%")
            dual_print(f"    Mean:  {np.mean(all_pcts):.1f}%")
            dual_print(f"    Std:   {np.std(all_pcts):.1f}%")

            dual_print(f"\n  BY WINDOW SIZE:")
            for W in WINDOWS:
                vals = by_window[W]
                if vals:
                    dual_print(f"    W={W:2d}: {min(vals):.1f}% – {max(vals):.1f}%  "
                               f"(mean {np.mean(vals):.1f}%, std {np.std(vals):.1f}%, "
                               f"n={len(vals)})")

            dual_print(f"\n  BY FEATURE SET:")
            for system in SYSTEMS:
                vals = by_system[system]
                if vals:
                    dual_print(f"    {system:14s}: {min(vals):.1f}% – {max(vals):.1f}%  "
                               f"(mean {np.mean(vals):.1f}%, std {np.std(vals):.1f}%, "
                               f"n={len(vals)})")

            dual_print(f"\n  BY MODEL (across all systems and windows):")
            for model in MODELS:
                vals = by_model[model]
                if vals:
                    dual_print(f"    {model:12s}: {min(vals):.1f}% – {max(vals):.1f}%  "
                               f"(mean {np.mean(vals):.1f}%, std {np.std(vals):.1f}%, "
                               f"n={len(vals)})")

        # ---- PHASE 4: Cross-check against verified I2 values ----
        dual_print(f"\n{'=' * 80}")
        dual_print("PHASE 4: CROSS-CHECK vs VERIFIED I2 VALUES (W=10)")
        dual_print("=" * 80)

        # These are the exact values from the I2 output we already verified
        i2_verified = {
            ('rf',          'network_only'):  38.8,
            ('rf',          'radio_only'):    39.3,
            ('rf',          'fusion_mean'):   38.8,
            ('logreg',      'network_only'):  40.6,
            ('logreg',      'radio_only'):    36.7,
            ('logreg',      'fusion_mean'):   39.8,
            ('xgboost',     'network_only'):  37.6,
            ('xgboost',     'radio_only'):    37.5,
            ('xgboost',     'fusion_mean'):   37.3,
            ('gru',         'network_only'):  32.4,
            ('gru',         'radio_only'):    36.1,
            ('gru',         'fusion_mean'):   36.3,
            ('resmlp',      'network_only'):  35.1,
            ('resmlp',      'radio_only'):    36.3,
            ('resmlp',      'fusion_mean'):   36.9,
            ('tcn',         'network_only'):  30.8,
            ('tcn',         'radio_only'):    36.0,
            ('tcn',         'fusion_mean'):   34.3,
            ('transformer', 'network_only'):  33.5,
            ('transformer', 'radio_only'):    35.3,
            ('transformer', 'fusion_mean'):   34.4,
        }

        all_match = True
        for (model, system), expected in i2_verified.items():
            r = results[(model, system, 10)]
            if r['n_files'] > 0:
                cm = r['cm']
                dos_total = cm[DOS_IDX, :].sum()
                actual = 100.0 * cm[DOS_IDX, BEN_IDX] / dos_total if dos_total > 0 else 0
                match = abs(actual - expected) < 0.1
                symbol = "✓" if match else "✗ MISMATCH"
                if not match:
                    all_match = False
                dual_print(f"  {model:12s} {system:14s}: expected {expected:.1f}%, "
                           f"got {actual:.1f}% {symbol}")

        dual_print(f"\n  CROSS-CHECK RESULT: {'ALL MATCH ✓' if all_match else 'MISMATCHES FOUND ✗'}")

        # ---- PHASE 5: Cross-check against verified I3 values ----
        dual_print(f"\n{'=' * 80}")
        dual_print("PHASE 5: CROSS-CHECK vs VERIFIED I3 VALUES (XGBoost fusion_mean)")
        dual_print("=" * 80)

        i3_verified = {5: 37.9, 10: 37.3}
        for W, expected in i3_verified.items():
            r = results[('xgboost', 'fusion_mean', W)]
            if r['n_files'] > 0:
                cm = r['cm']
                dos_total = cm[DOS_IDX, :].sum()
                actual = 100.0 * cm[DOS_IDX, BEN_IDX] / dos_total if dos_total > 0 else 0
                match = abs(actual - expected) < 0.1
                symbol = "✓" if match else "✗ MISMATCH"
                dual_print(f"  W={W}: expected {expected:.1f}%, got {actual:.1f}% {symbol}")

    # Final status
    print(f'\n' + '*' * 60)
    print(f'SUCCESS: Three-factor invariance audit complete.')
    print(f'REPORT: {report_path}')
    print('*' * 60 + '\n')


if __name__ == "__main__":
    run_audit()