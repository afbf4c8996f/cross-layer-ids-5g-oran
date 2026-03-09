#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

DL_MODELS = ["resmlp", "gru", "tcn", "transformer"]
ML_MODELS = ["logreg", "xgboost", "rf"]

def _run(cmd: List[str], env: Dict[str, str] | None = None) -> None:
    # Force all child runs to use the SAME interpreter that launched this sweep.
    if cmd and cmd[0] in ("python3", "python"):
        cmd = [sys.executable] + cmd[1:]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _set_all_models_disabled(cfg: Dict[str, Any]) -> None:
    models = cfg.get("models", {}) or {}
    for family in ("dl", "tabular"):
        fam = models.get(family, {}) or {}
        for _, block in fam.items():
            if isinstance(block, dict) and "enabled" in block:
                block["enabled"] = False


def _enable_only(cfg: Dict[str, Any], model_name: str) -> str:
    models = cfg.get("models", {}) or {}
    dl = models.get("dl", {}) or {}
    tab = models.get("tabular", {}) or {}

    if model_name in dl:
        dl[model_name]["enabled"] = True
        return "dl"
    if model_name in tab:
        tab[model_name]["enabled"] = True
        return "tabular"
    raise KeyError(f"Model {model_name!r} not found in cfg.models.dl or cfg.models.tabular")


def _pick_best_yaml(best_yaml_dir: Path, model: str) -> Path:
    if not best_yaml_dir.exists():
        raise FileNotFoundError(f"Missing best_yaml dir: {best_yaml_dir}")

    ymls = sorted(best_yaml_dir.glob("*.yaml"))
    if not ymls:
        raise FileNotFoundError(f"No .yaml files found in {best_yaml_dir}")
    preferred = [p for p in ymls if p.name.endswith("__best.yaml")]
    if preferred:
        return max(preferred, key=lambda p: p.stat().st_mtime)

    return max(ymls, key=lambda p: p.stat().st_mtime)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", required=True, type=str)
    ap.add_argument("--optuna_root", required=True, type=str)
    ap.add_argument("--final_root", required=True, type=str)
    ap.add_argument("--models", nargs="*", default=[], help="Default: all models found in base config")
    ap.add_argument("--gen_dir", default="sweep_generated_configs", type=str)
    ap.add_argument("--n_trials", default=None, type=int)
    ap.add_argument("--timeout_s", default=None, type=int)
    ap.add_argument("--cuda_visible_devices", default="0", type=str)
    ap.add_argument("--tol", default="1e-5", type=str)
    ap.add_argument("--force", action="store_true", help="Overwrite existing per-model output dirs")
    args = ap.parse_args()

    # Variables defined within main()
    repo = Path(".").resolve()
    base_config = Path(args.base_config).expanduser().resolve()
    gen_dir = (repo / args.gen_dir).resolve()
    optuna_root = Path(args.optuna_root).expanduser().resolve()
    final_root = Path(args.final_root).expanduser().resolve()

    # 1. Decide which DL replay script to use
    dl_replay = repo / "replay_predictions_check.py"
    if not dl_replay.exists():
        dl_replay = repo / "replay_predictions_check_v5_fixed.py"

    # 2. Check for required files (now outside the if-check above)
    required = [
        repo / "run_optuna_stage3.py",
        repo / "run_stage3_tabular.py",
        repo / "output_validator.py",
        dl_replay,
        repo / "replay_tabular_predictions_check.py",
    ]
    
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("[FATAL] Missing required files:\n  " + "\n  ".join(missing))
        sys.exit(2)

    # 3. Load configuration
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")
        
    cfg = yaml.safe_load(base_config.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Base config did not parse to a dict.")

    # 4. Determine model list
    if args.models:
        models_to_run = args.models
    else:
        models_to_run = []
        m = cfg.get("models", {}) or {}
        models_to_run += [k for k in DL_MODELS if k in (m.get("dl", {}) or {})]
        models_to_run += [k for k in ML_MODELS if k in (m.get("tabular", {}) or {})]

    if not models_to_run:
        raise ValueError("No models found/selected.")

    # 5. Ensure root directories exist
    gen_dir.mkdir(parents=True, exist_ok=True)
    optuna_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)

    # 6. Iterate through models
    for model in models_to_run:
        print("\n" + "=" * 90)
        print(f"[SWEEP] model={model}")
        print("=" * 90)

        cfg_m = copy.deepcopy(cfg)

        # Enable only this model
        _set_all_models_disabled(cfg_m)
        family = _enable_only(cfg_m, model)

        # Set output paths
        model_final_dir = final_root / model
        model_optuna_dir = optuna_root / model

        if model_final_dir.exists() and any(model_final_dir.iterdir()) and not args.force:
            raise RuntimeError(
                f"Refusing to overwrite non-empty out_dir: {model_final_dir}. "
                f"Delete it or pass --force."
            )

        cfg_m["paths"]["out_dir"] = str(model_final_dir)
        cfg_m["optuna"]["out_dir"] = str(model_optuna_dir)
        cfg_m["optuna"]["model"] = model

        # Overrides for n_trials and timeout
        if args.n_trials is not None:
            cfg_m["optuna"]["n_trials"] = int(args.n_trials)
        if args.timeout_s is not None:
            cfg_m["optuna"]["timeout_s"] = int(args.timeout_s)

        # Write generated config
        cfg_path = gen_dir / f"config_stage3_v3_optuna__{model}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg_m, sort_keys=False))

        # Setup Environment (GPU logic)
        env = os.environ.copy()
        if family == "dl":
            env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""  # tabular usually stays on CPU

        # Execution Phase
        # Step A: Optuna Study
        _run(["python3", "run_optuna_stage3.py", "--config", str(cfg_path)], env=env)

        # Step B: Pick best result
        best_yaml_dir = model_optuna_dir / "best_yaml"
        best_yaml = _pick_best_yaml(best_yaml_dir, model)
        print(f"[SWEEP] best_yaml = {best_yaml}")

        # Step C: Final Evaluation
        _run(["python3", "run_stage3_tabular.py", "--config", str(best_yaml)], env=env)

        # Step D: Validation & Replay
        _run(["python3", "output_validator.py", "--out_dir", str(model_final_dir)], env=env)

        tol = str(args.tol)
        if family == "dl":
            _run(["python3", str(dl_replay), "--out_dir", str(model_final_dir), "--tol", tol], env=env)
        else:
            _run(["python3", "replay_tabular_predictions_check.py", "--out_dir", str(model_final_dir), "--tol", tol], env=env)

    print("\n[SWEEP] ✅ ALL MODELS COMPLETE — Optuna + final eval + validator + replay all passed.")


if __name__ == "__main__":
    main()