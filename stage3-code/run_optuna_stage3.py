#!/usr/bin/env python3
"""
run_optuna_stage3.py
CLI entrypoint for Stage-3 Optuna tuning.

Usage (recommended)
  python3 run_optuna_stage3.py --config config_stage3_v3_evalupgrade.yaml

This script expects a section 'optuna' inside the YAML config. Example:

optuna:
  out_dir: /ABS/PATH/optuna-out
  split: stratified_seed43
  W: 10
  S: 2
  model: resmlp
  tasks: [binary, multiclass]
  modalities: [network, radio]
  n_trials: 50
  timeout_s: 14400
  sampler: tpe
  pruner: median
  startup_trials: 10
  warmup_steps: 3

It tunes on TRAIN -> evaluates objective on official VAL only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from stage3_utils import read_yaml, ensure_dir
from stage3_hpo import (
    OptunaRunConfig,
    run_optuna_study,
    apply_best_configs_to_yaml,
    write_yaml,
    Task,
    Modality,
)


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Stage-3 YAML config path")
    ap.add_argument("--dry-run", action="store_true", help="Print planned studies and exit")
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))

    processed_dir = Path(cfg["paths"]["processed_dir"]).expanduser()
    opt = cfg.get("optuna", {})
    if not isinstance(opt, dict):
        raise ValueError("YAML: optuna must be a dict/section")

    out_dir = ensure_dir(Path(opt.get("out_dir", "optuna-out")).expanduser())
    split_name = str(opt.get("split", "stratified_seed43"))
    W = int(opt.get("W", 10))
    S = int(opt.get("S", 2))
    model = str(opt.get("model", "resmlp")).strip().lower()

    tasks = [str(t).strip().lower() for t in _as_list(opt.get("tasks", ["binary", "multiclass"]))]
    modalities = [str(m).strip().lower() for m in _as_list(opt.get("modalities", ["network", "radio"]))]

    n_trials = int(opt.get("n_trials", 50))
    timeout_s = opt.get("timeout_s", None)
    timeout_s = int(timeout_s) if timeout_s not in (None, "", False) else None

    sampler = str(opt.get("sampler", "tpe"))
    pruner = str(opt.get("pruner", "median"))
    startup_trials = int(opt.get("startup_trials", 10))
    warmup_steps = int(opt.get("warmup_steps", 3))

    seed = int(opt.get("seed", 42))
    benign_family_name = str(cfg.get("thresholds", {}).get("benign_family_name", "Benign"))

    feature_ablation = cfg.get("feature_ablation", None)
    if feature_ablation is not None and not isinstance(feature_ablation, dict):
        feature_ablation = None

    planned: List[Tuple[Task, Modality]] = []
    for t in tasks:
        if t not in ("binary", "multiclass"):
            raise ValueError(f"Unknown task: {t}")
        for m in modalities:
            if m not in ("network", "radio"):
                raise ValueError(f"Unknown modality: {m}")
            planned.append((t, m))

    if args.dry_run:
        print("Planned Optuna studies:")
        for t, m in planned:
            print(f"  model={model} task={t} modality={m} split={split_name} W={W} S={S}")
        return

    best_cfgs: Dict[Tuple[Task, Modality], Dict[str, Any]] = {}

    # Run studies one by one (no distributed Optuna; you can run different studies on different machines).
    for task, modality in planned:
        rcfg = OptunaRunConfig(
            processed_dir=processed_dir,
            out_dir=out_dir,
            split_name=split_name,
            W=W,
            S=S,
            seed=seed,
            model_name=model,
            task=task,          # type: ignore
            modality=modality,  # type: ignore
            n_trials=n_trials,
            timeout_s=timeout_s,
            sampler=sampler,
            pruner=pruner,
            startup_trials=startup_trials,
            warmup_steps=warmup_steps,
            benign_family_name=benign_family_name,
            feature_ablation=feature_ablation,
        )

        print(f"[Optuna] Starting: model={model} task={task} modality={modality} split={split_name} W={W} S={S}")
        res = run_optuna_study(cfg, rcfg)
        print(f"[Optuna] Done: best_value={res['best_value']:.6f}  dir={res['study_dir']}")

        # Load best_config.json produced by the study (authoritative)
        best_path = Path(res["study_dir"]) / "best_config.json"
        if best_path.exists():
            import json
            with open(best_path, "r", encoding="utf-8") as f:
                best = json.load(f)
            if isinstance(best, dict) and isinstance(best.get("cfg", None), dict):
                best_cfgs[(task, modality)] = dict(best["cfg"])

    # Write a "best YAML" that embeds task+modality overrides for the tuned model.
    if best_cfgs:
        best_yaml = apply_best_configs_to_yaml(cfg, model_name=model, best_cfgs=best_cfgs)
        out_path = out_dir / "best_yaml" / f"{Path(args.config).stem}__{model}__best.yaml"
        write_yaml(out_path, best_yaml)
        print(f"[Optuna] Wrote best YAML (overrides) to: {out_path}")

    print("[Optuna] All studies complete.")


if __name__ == "__main__":
    main()
