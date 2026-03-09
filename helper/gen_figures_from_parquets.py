#!/usr/bin/env python3
"""
Figure: (a) Confusion Matrix + (b) Per-Class F1 Bars
XGBoost fusion_mean, W10, 10 seeds aggregated.

ALL data computed from prediction parquets — zero hardcoded values.
Cross-checks against Table B: seed-averaged F1-macro = 0.683.

Usage:
  python gen_figures_from_parquets.py \
    --stage3-dir /path/to/dir/where/Training&Evaluation/finishes

Outputs:
  fig_composite_2col.pdf   — double-column (figure*)
  fig_confusion_1col.pdf   — single-column standalone
  fig_perclass_f1_1col.pdf — single-column standalone
"""
import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# CONFIGURATION
# ============================================================
MODEL = 'xgboost'
SEEDS = list(range(42, 52))
W, S = 10, 2
SYSTEMS = ['network_only', 'radio_only', 'fusion_mean']

# Class labels as they appear in parquets (lowercase except Benign)
PARQUET_CLASSES = ['Benign', 'bruteforce', 'ddos', 'dos', 'probe', 'web']

# Display names for figures (title case)
DISPLAY_CLASSES = ['Benign', 'Bruteforce', 'DDoS', 'DoS', 'Probe', 'Web']

# Expected values for cross-check (from verified Table B and I2)
EXPECTED_FUSION_F1_MACRO_SEED_AVG = 0.683
TOLERANCE = 0.002  # allow ±0.002 for rounding

# Output directory (will be set from args or default)
OUTPUT_DIR = Path('.')

# ============================================================
# PHASE 1: LOAD AND COMPUTE
# ============================================================
def load_predictions(base_dir):
    """Load all prediction parquets, compute CM and per-class F1.
    
    Returns:
        cm_raw: (6,6) int array — aggregated confusion matrix (fusion_mean)
        per_class_f1: dict of system -> (6,) float array
        f1_macro_per_seed: dict of system -> list of 10 floats
    """
    base = Path(base_dir)

    # Collect predictions per system
    all_dfs = {sys: [] for sys in SYSTEMS}

    for seed in SEEDS:
        for sys in SYSTEMS:
            path = (base / MODEL / 'predictions' /
                    f'stratified_seed{seed}' / f'W{W}_S{S}' /
                    MODEL / 'multiclass' / f'{sys}_test.parquet')

            if not path.exists():
                print(f"ERROR: Missing {path}")
                sys_module = sys  # avoid shadowing
                raise FileNotFoundError(f"Missing parquet: {path}")

            df = pd.read_parquet(path)

            # Defensive column detection (same as verified I2 script)
            label_col = [c for c in df.columns
                         if c.lower() in ['label', 'y_true', 'y']][0]
            pred_col = [c for c in df.columns
                        if c.lower() in ['pred', 'y_pred', 'prediction']][0]

            # Rename for consistency
            df = df.rename(columns={label_col: 'y_true', pred_col: 'y_pred'})
            df['seed'] = seed
            all_dfs[sys].append(df)

    # Concatenate
    for sys in SYSTEMS:
        all_dfs[sys] = pd.concat(all_dfs[sys], ignore_index=True)

    # --- Confusion matrix (fusion_mean, all seeds pooled) ---
    df_fus = all_dfs['fusion_mean']
    cm_raw = confusion_matrix(df_fus['y_true'], df_fus['y_pred'],
                              labels=PARQUET_CLASSES)

    # --- Per-class F1 per system (all seeds pooled) ---
    per_class_f1 = {}
    for sys in SYSTEMS:
        df = all_dfs[sys]
        f1s = f1_score(df['y_true'], df['y_pred'],
                       labels=PARQUET_CLASSES, average=None)
        per_class_f1[sys] = f1s

    # --- Per-seed F1-macro (for cross-check against Table B) ---
    f1_macro_per_seed = {}
    for sys in SYSTEMS:
        seed_f1s = []
        df = all_dfs[sys]
        for seed in SEEDS:
            df_s = df[df['seed'] == seed]
            f1m = f1_score(df_s['y_true'], df_s['y_pred'],
                           labels=PARQUET_CLASSES, average='macro')
            seed_f1s.append(f1m)
        f1_macro_per_seed[sys] = seed_f1s

    return cm_raw, per_class_f1, f1_macro_per_seed, all_dfs


def cross_check(cm_raw, per_class_f1, f1_macro_per_seed):
    """Validate computed values against known verified results."""
    print("\n" + "=" * 70)
    print("  CROSS-CHECKS")
    print("=" * 70)

    # Check 1: Total windows = 31,286
    total = cm_raw.sum()
    print(f"  Total windows: {total} (expected 31286)", end='')
    assert total == 31286, f"MISMATCH: got {total}"
    print(" ✓")

    # Check 2: Seed-averaged F1-macro for fusion_mean ≈ 0.683
    fus_macro = np.mean(f1_macro_per_seed['fusion_mean'])
    print(f"  Fusion F1-macro (seed avg): {fus_macro:.3f} "
          f"(expected {EXPECTED_FUSION_F1_MACRO_SEED_AVG})", end='')
    assert abs(fus_macro - EXPECTED_FUSION_F1_MACRO_SEED_AVG) < TOLERANCE, \
        f"MISMATCH: got {fus_macro:.4f}"
    print(" ✓")

    # Check 3: DoS→Benign confusion = 37.3%
    dos_idx = PARQUET_CLASSES.index('dos')
    ben_idx = PARQUET_CLASSES.index('Benign')
    dos_to_ben_pct = 100.0 * cm_raw[dos_idx, ben_idx] / cm_raw[dos_idx, :].sum()
    print(f"  DoS→Benign: {dos_to_ben_pct:.1f}% (expected 37.3%)", end='')
    assert abs(dos_to_ben_pct - 37.3) < 0.1, f"MISMATCH: got {dos_to_ben_pct:.1f}"
    print(" ✓")

    # Check 4: Per-class F1 for fusion matches known values from I2
    i2_fusion_f1 = [0.6399, 0.6743, 0.6330, 0.5047, 0.7910, 0.8999]
    for i, (computed, expected) in enumerate(zip(per_class_f1['fusion_mean'],
                                                  i2_fusion_f1)):
        assert abs(computed - expected) < 0.001, \
            f"F1 mismatch for {PARQUET_CLASSES[i]}: {computed:.4f} vs {expected:.4f}"
    print(f"  Per-class F1 (fusion): all 6 match I2 values ✓")

    # Print summary
    print(f"\n  Per-class F1 computed from parquets:")
    print(f"  {'Class':12s} {'Network':>10s} {'Radio':>10s} {'Fusion':>10s}")
    print(f"  {'-' * 42}")
    for i, cls in enumerate(DISPLAY_CLASSES):
        print(f"  {cls:12s} "
              f"{per_class_f1['network_only'][i]:10.3f} "
              f"{per_class_f1['radio_only'][i]:10.3f} "
              f"{per_class_f1['fusion_mean'][i]:10.3f}")

    print(f"\n  Raw CM diagonal (recall %):")
    for i, cls in enumerate(DISPLAY_CLASSES):
        row_sum = cm_raw[i, :].sum()
        recall_pct = 100.0 * cm_raw[i, i] / row_sum
        print(f"  {cls:12s}: {recall_pct:.1f}%  ({cm_raw[i,i]}/{row_sum})")

    print("\n  ALL CROSS-CHECKS PASSED ✓")

def _detect_run_col(df: pd.DataFrame) -> str | None:
    """Try to find a run/group identifier column in meta."""
    candidates = [
        'run_id', 'run', 'run_name', 'runid',
        'group', 'group_id', 'gid',
        'scenario', 'trace', 'capture'
    ]
    cols = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in cols:
            return cols[k]
    return None


def report_paper_ready_stats(cm_raw, per_class_f1, f1_macro_per_seed, all_dfs):
    """Print all numbers needed to verify the Results section text."""
    print("\n" + "=" * 70)
    print("  PAPER-READY STATS (for Results section verification)")
    print("=" * 70)

    # ---------- 1) Macro F1 across seeds (mean ± std) ----------
    print("\n[Macro F1 across seeds (mean ± std)]")
    for sys in SYSTEMS:
        vals = np.asarray(f1_macro_per_seed[sys], dtype=float)
        mean = vals.mean()
        std_pop = vals.std(ddof=0)
        std_samp = vals.std(ddof=1)
        print(f"  {sys:12s}: mean={mean:.4f}  std(ddof=0)={std_pop:.4f}  std(ddof=1)={std_samp:.4f}")

    # ---------- 2) Confusion matrix: error concentration & top pairs ----------
    total = int(cm_raw.sum())
    correct = int(np.trace(cm_raw))
    errors = total - correct
    print("\n[Confusion summary]")
    print(f"  Total windows = {total}")
    print(f"  Total errors  = {errors} ({100*errors/total:.2f}%)")

    # Off-diagonal pairs sorted by COUNT (not row-%)
    pairs = []
    for i, t in enumerate(PARQUET_CLASSES):
        for j, p in enumerate(PARQUET_CLASSES):
            if i == j:
                continue
            cnt = int(cm_raw[i, j])
            if cnt > 0:
                pairs.append((cnt, t, p))
    pairs.sort(reverse=True, key=lambda x: x[0])

    topk = 6
    top = pairs[:topk]
    top_cnt = sum(c for c, _, _ in top)
    pct_top = 100.0 * top_cnt / errors if errors > 0 else 0.0

    print(f"\n  Top-{topk} confusion pairs by count:")
    for cnt, t, p in top:
        i = PARQUET_CLASSES.index(t)
        row_sum = int(cm_raw[i, :].sum())
        row_pct = 100.0 * cnt / row_sum if row_sum > 0 else 0.0
        err_pct = 100.0 * cnt / errors if errors > 0 else 0.0
        print(f"   - {t:10s} → {p:10s}: "
              f"count={cnt:5d} | row%={row_pct:5.1f}% | share-of-errors={err_pct:5.1f}%")

    print(f"\n  Top-{topk} pairs account for {pct_top:.1f}% of all misclassifications.")

    # Attack→Benign share of errors
    ben = PARQUET_CLASSES.index('Benign')
    attack_to_ben = 0
    for i, t in enumerate(PARQUET_CLASSES):
        if i == ben:
            continue
        attack_to_ben += int(cm_raw[i, ben])
    pct_attack_to_ben = 100.0 * attack_to_ben / errors if errors > 0 else 0.0
    print(f"\n  Attack→Benign errors: {attack_to_ben} "
          f"({pct_attack_to_ben:.1f}% of all errors)")

    # Specific row-normalized confusions often cited in text
    def row_norm_pct(true_cls, pred_cls):
        i = PARQUET_CLASSES.index(true_cls)
        j = PARQUET_CLASSES.index(pred_cls)
        row_sum = cm_raw[i, :].sum()
        return float(100.0 * cm_raw[i, j] / row_sum) if row_sum > 0 else float('nan')

    print("\n  Key row-normalized confusions (row%):")
    print(f"   DoS→Benign       : {row_norm_pct('dos','Benign'):.1f}%")
    print(f"   Bruteforce→Benign: {row_norm_pct('bruteforce','Benign'):.1f}%")
    print(f"   DDoS→Benign      : {row_norm_pct('ddos','Benign'):.1f}%")
    print(f"   DoS→DDoS         : {row_norm_pct('dos','ddos'):.1f}%")
    print(f"   DDoS→DoS         : {row_norm_pct('ddos','dos'):.1f}%")

    # ---------- 3) Per-class F1 + deltas ----------
    print("\n[Per-class F1 (pooled) and Fusion deltas]")
    print(f"  {'Class':12s} {'Net':>8s} {'Radio':>8s} {'Fusion':>8s} {'Fus-Rad':>9s} {'Rad-Net':>9s}")
    print(f"  {'-'*52}")
    for i, cls_disp in enumerate(DISPLAY_CLASSES):
        net = float(per_class_f1['network_only'][i])
        rad = float(per_class_f1['radio_only'][i])
        fus = float(per_class_f1['fusion_mean'][i])
        print(f"  {cls_disp:12s} {net:8.3f} {rad:8.3f} {fus:8.3f} {fus-rad:9.3f} {rad-net:9.3f}")

    # ---------- 4) Seed-wise class composition (verifies “DoS 9–30%, Benign 21–41%” etc.) ----------
    df_fus = all_dfs['fusion_mean']
    print("\n[Seed-wise class composition (% of windows)]")
    rows = []
    for seed in SEEDS:
        df_s = df_fus[df_fus['seed'] == seed]
        total_s = len(df_s)
        counts = df_s['y_true'].value_counts()
        row = {'seed': seed, 'n': total_s}
        for c in PARQUET_CLASSES:
            row[c] = 100.0 * float(counts.get(c, 0)) / total_s if total_s else 0.0
        rows.append(row)
    mix = pd.DataFrame(rows).sort_values('seed')

    # Print min/max for the two you cite most
    for c in ['Benign', 'dos']:
        print(f"  {c:8s}: min={mix[c].min():5.1f}%  max={mix[c].max():5.1f}%")

    # Optional: print compact table (Benign + attacks)
    print("\n  Seed  n     Benign  bruteforce  ddos   dos   probe   web")
    for _, r in mix.iterrows():
        print(f"  {int(r['seed']):4d} {int(r['n']):5d} "
              f"{r['Benign']:8.1f} {r['bruteforce']:10.1f} {r['ddos']:6.1f} "
              f"{r['dos']:5.1f} {r['probe']:7.1f} {r['web']:6.1f}")

    # ---------- 5) DoS windows per run (needs run_id in meta) ----------
    run_col = _detect_run_col(df_fus)
    print("\n[DoS windows per run]")
    if run_col is None:
        print("  No run_id/group column found in parquet meta — cannot compute per-run window ranges.")
    else:
        run_dos_counts = {}
        # Deduplicate across seeds safely: store first seen; assert consistent if seen again
        for seed in SEEDS:
            df_s = df_fus[df_fus['seed'] == seed]
            for rid, g in df_s.groupby(run_col):
                dos_windows = int((g['y_true'] == 'dos').sum())
                total_windows = int(len(g))
                if rid in run_dos_counts:
                    prev = run_dos_counts[rid]
                    # same run should have same window counts across seeds when it appears
                    if prev['dos_windows'] != dos_windows or prev['total_windows'] != total_windows:
                        print(f"  WARNING: run {rid} has inconsistent window counts across seeds "
                              f"(prev dos={prev['dos_windows']}, now dos={dos_windows})")
                else:
                    run_dos_counts[rid] = {'dos_windows': dos_windows, 'total_windows': total_windows}

        # Consider only runs that contain at least one DoS window
        dos_runs = [v['dos_windows'] for v in run_dos_counts.values() if v['dos_windows'] > 0]
        if not dos_runs:
            print("  No runs with DoS windows found in the union of test sets.")
        else:
            dos_runs = np.asarray(dos_runs, dtype=int)
            print(f"  Found {len(dos_runs)} unique runs with ≥1 DoS window.")
            print(f"  DoS windows per run: min={dos_runs.min()}  max={dos_runs.max()}  "
                  f"median={int(np.median(dos_runs))}")


# ============================================================
# PHASE 2: IEEE STYLE SETUP
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'grid.linewidth': 0.4,
})

# Colors — colorblind-friendly, print-safe
COLOR_NET = '#4393c3'   # Steel blue
COLOR_RAD = '#d6604d'   # Terra cotta
COLOR_FUS = '#2d004b'   # Deep purple
CMAP_CM = plt.cm.Blues


# ============================================================
# PHASE 3: DRAWING FUNCTIONS
# ============================================================
def draw_confusion_matrix(ax, cm_pct, fontsize_cells=6.5):
    """Draw row-normalized confusion matrix on given axes."""
    im = ax.imshow(cm_pct, cmap=CMAP_CM, vmin=0, vmax=100, aspect='equal')

    for i in range(6):
        for j in range(6):
            val = cm_pct[i, j]
            text_color = 'white' if val > 50 else 'black'

            if i == j:
                text = f'{val:.1f}%'
                weight = 'bold'
            elif val >= 15:
                text = f'{val:.1f}%'
                weight = 'bold'
                text_color = '#b2182b' if val < 50 else 'white'
            elif val >= 5:
                text = f'{val:.1f}%'
                weight = 'normal'
            elif val >= 1:
                text = f'{val:.1f}'
                weight = 'normal'
                text_color = '#666666'
            else:
                text = ''
                weight = 'normal'

            ax.text(j, i, text, ha='center', va='center',
                    fontsize=fontsize_cells, fontweight=weight, color=text_color)

    # Highlight DoS→Benign (largest off-diagonal)
    rect = Rectangle((-0.5, 2.5), 1, 1, linewidth=1.8,
                      edgecolor='#d73027', facecolor='none', zorder=10)
    ax.add_patch(rect)

    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(DISPLAY_CLASSES, fontsize=6.5, rotation=30, ha='right')
    ax.set_yticklabels(DISPLAY_CLASSES, fontsize=6.5)
    ax.set_xlabel('Predicted', fontsize=8, labelpad=5)
    ax.set_ylabel('True', fontsize=8, labelpad=5)

    return im


def draw_perclass_bars(ax, network_f1, radio_f1, fusion_f1, bar_fontsize=6):
    """Draw grouped vertical bars for per-class F1."""
    x = np.arange(len(DISPLAY_CLASSES))
    width = 0.24

    ax.bar(x - width, network_f1, width, label='Network',
           color=COLOR_NET, edgecolor='white', linewidth=0.4, alpha=0.88)
    ax.bar(x, radio_f1, width, label='Radio',
           color=COLOR_RAD, edgecolor='white', linewidth=0.4, alpha=0.88)
    bars_fus = ax.bar(x + width, fusion_f1, width, label='Fusion',
                      color=COLOR_FUS, edgecolor='white', linewidth=0.4, alpha=0.88)

    # Value labels on fusion bars only
    for bar, val in zip(bars_fus, fusion_f1):
        label = f'{val:.3f}'[1:]  # e.g. 0.8999 → ".900", drops leading 0
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                label, ha='center', va='bottom', fontsize=bar_fontsize,
                fontweight='bold', color=COLOR_FUS)

    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_CLASSES, fontsize=7.5, rotation=30, ha='right')
    ax.set_ylabel('F1 Score', fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.95),
              framealpha=0.9, edgecolor='#cccccc',
              fontsize=6.5, handlelength=1.2, handletextpad=0.4)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate CM + per-class F1 figures from prediction parquets')
    parser.add_argument('--stage3-dir', type=str, required=True,
                        help='Path to stage3-output-v2-no-history/')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory for output PDFs (default: current dir)')
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- LOAD & VALIDATE ----
    print("Loading predictions from parquets...")
    cm_raw, per_class_f1, f1_macro_per_seed, all_dfs = load_predictions(args.stage3_dir)
    cross_check(cm_raw, per_class_f1, f1_macro_per_seed)
    report_paper_ready_stats(cm_raw, per_class_f1, f1_macro_per_seed, all_dfs)

    # ---- DERIVED DATA ----
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    cm_pct = 100.0 * cm_raw / row_sums

    network_f1 = per_class_f1['network_only']
    radio_f1 = per_class_f1['radio_only']
    fusion_f1 = per_class_f1['fusion_mean']

    # ============================================================
    # 1. COMPOSITE FIGURE (double-column, figure*)
    # ============================================================
    fig, (ax_cm, ax_bar) = plt.subplots(
        1, 2, figsize=(7.16, 3.0),
        gridspec_kw={'width_ratios': [1.05, 1], 'wspace': 0.40}
    )

    # Panel (a): Confusion matrix — slightly smaller cell text
    im = draw_confusion_matrix(ax_cm, cm_pct, fontsize_cells=6.5)
    ax_cm.set_title('(a) Confusion Matrix', fontsize=9,
                    fontweight='bold', pad=6)
    cbar = fig.colorbar(im, ax=ax_cm, shrink=0.78, pad=0.06, aspect=20)
    cbar.set_label('Row-normalized (%)', fontsize=6.5, labelpad=4)
    cbar.ax.tick_params(labelsize=6)

    # Panel (b): Per-class F1
    draw_perclass_bars(ax_bar, network_f1, radio_f1, fusion_f1,
                       bar_fontsize=5.5)
    ax_bar.set_title('(b) Per-Class F1 by Feature Set', fontsize=9,
                     fontweight='bold', pad=6)
    
    fig.canvas.draw()
    pos_cm  = ax_cm.get_position()
    pos_bar = ax_bar.get_position()
    ax_bar.set_position([pos_bar.x0, pos_bar.y0,
                        pos_bar.width, pos_cm.y1 - pos_bar.y0])
    path_comp = out / 'fig_composite_2col.pdf'
    fig.savefig(path_comp, format='pdf')
    plt.close()
    print(f"\nSaved: {path_comp}")

    # ============================================================
    # 2. CONFUSION MATRIX STANDALONE (single-column)
    # ============================================================
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    im = draw_confusion_matrix(ax, cm_pct, fontsize_cells=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.07, aspect=20)
    cbar.set_label('Row-normalized (%)', fontsize=6.5, labelpad=4)
    cbar.ax.tick_params(labelsize=6)

    path_cm = out / 'fig_confusion_1col.pdf'
    fig.savefig(path_cm, format='pdf')
    plt.close()
    print(f"Saved: {path_cm}")

    # ============================================================
    # 3. PER-CLASS F1 STANDALONE (single-column)
    # ============================================================
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.6))
    draw_perclass_bars(ax, network_f1, radio_f1, fusion_f1, bar_fontsize=6)

    path_f1 = out / 'fig_perclass_f1_1col.pdf'
    fig.savefig(path_f1, format='pdf')
    plt.close()
    print(f"Saved: {path_f1}")

    print("\nAll figures generated — all data computed from parquets")


if __name__ == "__main__":
    main()