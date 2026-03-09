# End-to-End Pipeline Guide

This is the companion repository for the paper:

> **Cross-Layer Intrusion Detection in 5G O-RAN: Gains and Limits of Fusing Radio Telemetry with Network Flow Records**
> Submitted to IEEE CSR 2026

---

## Dataset

Download from Kaggle before starting — use the **updated version**, not the legacy one:

> https://www.kaggle.com/datasets/netslabdemo/netslab-5g-oran-idd

---

## Requirements

Install dependencies from the provided `environment.yml`:

```bash
conda env create -f stage3-code/environment.yml
conda activate MCVWP
```

For GPU-backed DL models, ensure a CUDA-compatible PyTorch build is installed. For `.pcapng` files, `tshark` must be available on your `PATH`.

---

## Repository layout

```
CodeArtifacts/
├── Alignment/
├── windowing/
├── Preprocessing/
├── stage3-code/
├── helper/
└── stage3-output-opt-tuned/   ← pre-tuned hyperparameters (see below)
    ├── {model}/artifacts/{split}/{W*_S*}/{model}/run_artifact.json
    └── merged/                ← aggregated results CSVs
```

---

## Pre-tuned hyperparameters

`stage3-output-opt-tuned/` contains the Optuna results used in the paper for all seven models (`gru`, `logreg`, `resmlp`, `rf`, `tcn`, `transformer`, `xgboost`) across 10 stratified seeds and both window configs. Each `run_artifact.json` holds the fully resolved hyperparameter configuration. To reproduce paper results, copy the relevant `model_cfg` values into your Stage 4 YAML before training.

---

## Pipeline

The four stages must be run in order. Each produces a key artefact consumed by the next.

### Stage 1 — Alignment

```bash
python3 Alignment/list_pairs.py \
    --root /path/to/archive --out-dir ./pairing_out

python3 Alignment/extract_paired_runs.py \
    --pairs ./pairing_out/pairs_collapsed.csv --out ./paired_runs.csv

python3 Alignment/paired_run_alignment_check.py \
    --paired-runs ./paired_runs.csv --out-dir ./alignment_out --log1p

python3 Alignment/analyze_alignment_results.py \
    --results ./alignment_out/results.csv --out-dir ./alignment_out/analysis
```

> **Checkpoint:** `alignment_out/analysis/strong_runs.txt` lists the runs that passed alignment. Filter `paired_runs.csv` to keep only those rows (match on `family` + `canon_stem`) before continuing. All downstream stages use this filtered manifest.

---

### Stage 2 — Windowing

Edit `windowing/config_stage1.yaml` to set your archive path, `paired_runs.csv` path, and output directory, then:

```bash
python3 windowing/prepare_network_windows.py --config windowing/config_stage1.yaml
python3 windowing/prepare_radio_windows.py   --config windowing/config_stage1.yaml
python3 windowing/validate_stage1_outputs.py --out-dir /path/to/window-output
```

---

### Stage 3 — Preprocessing

Edit `Preprocessing/config_stage2.yaml` to point to your `paired_runs.csv`, window-output directory, and desired output path, then:

```bash
python3 Preprocessing/make_run_splits.py          --config Preprocessing/config_stage2.yaml
python3 Preprocessing/validate_run_splits.py      \
    --paired-runs ./paired_runs.csv               \
    --splits-dir  /path/to/preprocessing-output/splits
python3 Preprocessing/preprocess_windows.py       --config Preprocessing/config_stage2.yaml
python3 Preprocessing/validate_processed_windows.py --out-dir /path/to/preprocessing-output
```

---

### Stage 4 — Training & Evaluation

Edit `stage3-code/config_stage3_optuna_base.yaml` to set `paths.processed_dir`, `paths.out_dir`, and `onset.run_summary_path` (pointing to `window-output/summaries/network_run_summary_all.csv`). Enable models by setting `enabled: true` in the config.

> **Note:** All Stage 4 scripts must be run from inside `stage3-code/` — they resolve sibling scripts by relative path.

```bash
cd stage3-code/
```

**Option A — HPO then training** (recommended for DL models):

```bash
python3 run_optuna_stage3.py  --config config_stage3_optuna_base.yaml
python3 run_stage3_tabular.py --config optuna-out/best_yaml/<config>__<model>__best.yaml
```

**Option B — Full automated sweep** (HPO + train + validate for all models at once):

```bash
python3 stage3_optuna_sweep.py \
    --base_config config_stage3_optuna_base.yaml \
    --optuna_root /path/to/optuna-out \
    --final_root  /path/to/stage3-output \
    --models xgboost gru resmlp tcn transformer logreg rf
```

**Option C — Direct training** (tabular models, or when using pre-tuned hyperparameters from `stage3-output-opt-tuned/`):

```bash
python3 run_stage3_tabular.py --config config_stage3_optuna_base.yaml
```

**Post-run validation:**

```bash
python3 output_validator.py                  --out_dir /path/to/stage3-output
python3 replay_tabular_predictions_check.py  --out_dir /path/to/stage3-output
python3 replay_predictions_check.py          --out_dir /path/to/stage3-output
```
