"""
stage3_hpo.py
Model-agnostic Optuna HPO harness for Stage-3.

Design goals
- Reuse existing Stage-3 I/O + metrics code (no duplicate evaluation logic).
- One Optuna study per (model, task, modality) as agreed.
- Keep outputs clean: SQLite DB + best_config JSON (+ optional best checkpoint).
- Future-proof: registry-based model specs; adding TCN/GRU/Transformer is "drop-in".

This module does NOT modify Stage-3 training/evaluation outputs; it is a separate
tuning harness that evaluates on the official VAL split only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

import json
import math
import inspect
from functools import lru_cache

import numpy as np
import pandas as pd

from stage3_io import load_processed, ProcessedSplit
from stage3_metrics import binary_metrics, multiclass_metrics
from stage3_models import (
    ModelBundle,
    make_logreg_binary,
    make_logreg_multiclass,
    make_resmlp_binary,
    make_resmlp_multiclass,
    make_gru_binary,
    make_gru_multiclass,
    make_tcn_binary,
    make_tcn_multiclass,
    make_transformer_binary,
    make_transformer_multiclass,
    make_xgb_binary,
    make_xgb_multiclass,
    make_rf_binary,
    make_rf_multiclass,
    predict_proba_binary,
    predict_proba_multiclass,
)

Task = Literal["binary", "multiclass"]
Modality = Literal["network", "radio"]


# ----------------------------
# Config resolution (task/modality overrides)
# ----------------------------

_RESERVED_KEYS = {"binary", "multiclass", "network", "radio"}


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def resolve_model_cfg(model_cfg: Dict[str, Any], *, task: Task, modality: Modality) -> Dict[str, Any]:
    """Resolve a model cfg with optional task/modality overrides.

    Supported patterns (all optional; fully backward compatible):

    1) Flat (legacy)
       resmlp:
         enabled: true
         d_model: 128
         ...

    2) Modality-only overrides
       resmlp:
         enabled: true
         d_model: 128
         network: {d_model: 64}
         radio:   {dropout: 0.2}

    3) Task-only overrides
       resmlp:
         enabled: true
         binary: {lr: 1e-3}
         multiclass: {lr: 3e-4, label_smoothing: 0.05}

    4) Task + modality overrides
       resmlp:
         enabled: true
         binary:
           network: {...}
           radio: {...}
         multiclass:
           network: {...}
           radio: {...}

    Merge order (later wins):
      base(root minus reserved keys) ->
      task overrides (task dict minus modality keys) ->
      modality overrides (from task dict if present, else root-level modality dict)
    """
    base = {k: v for k, v in dict(model_cfg).items() if k not in _RESERVED_KEYS}

    task_cfg = _as_dict(model_cfg.get(task))
    task_base = {k: v for k, v in dict(task_cfg).items() if k not in ("network", "radio")}

    # Prefer task-level modality override; fall back to root modality override.
    mod_cfg = _as_dict(task_cfg.get(modality)) if isinstance(task_cfg, dict) else {}
    if not mod_cfg:
        mod_cfg = _as_dict(model_cfg.get(modality))

    out = dict(base)
    out.update(task_base)
    out.update(mod_cfg)

    # Optional convenience: allow a combined override key at root level,
    # e.g.  binary__network: {dropout: 0.2}
    combo_key = f"{task}__{modality}"
    combo_cfg = _as_dict(model_cfg.get(combo_key))
    if combo_cfg:
        out.update(combo_cfg)
    return out

#--------------------------------
def _ensure_groups(split):
    """
    Return a non-None groups/run_id array for sequence models.
    Tries split.groups first, then falls back to split.df['run_id'].
    """
    g = getattr(split, "groups", None)
    if g is not None:
        return g
    df = getattr(split, "df", None)
    if df is not None and "run_id" in df.columns:
        return df["run_id"].to_numpy()
    return None
#-------------------------------------- 


# ----------------------------
# Data loading (cache per study)
# ----------------------------

def load_train_val(
    processed_dir: Path,
    split_name: str,
    modality: Modality,
    W: int,
    S: int,
    *,
    feature_ablation: Optional[Dict[str, Any]] = None,
) -> Tuple[ProcessedSplit, ProcessedSplit]:
    tr = load_processed(processed_dir, split_name, modality, W, S, "train", feature_ablation=feature_ablation)
    va = load_processed(processed_dir, split_name, modality, W, S, "val", feature_ablation=feature_ablation)
    return tr, va


def group_strata_binary(train_df: pd.DataFrame, *, benign_family_name: str = "Benign") -> np.ndarray:
    """Per-window group_strata for binary training (run-grouped early stopping).

    Matches Stage-3 runner logic:
    - If 'family' exists: benign_family -> 0, others -> 1
    - Else: run majority vote on y_bin (mean>=0.5) projected to windows
    """
    if "family" in train_df.columns:
        fam_norm = train_df["family"].astype(str).str.strip().str.lower()
        benign_norm = str(benign_family_name).strip().lower()
        return (~fam_norm.eq(benign_norm)).astype(np.int64).to_numpy()

    # Fallback: run-majority on window labels
    if "run_id" not in train_df.columns or "y_bin" not in train_df.columns:
        raise ValueError("Need run_id and y_bin to build group_strata fallback.")
    run_lab = (train_df.groupby("run_id")["y_bin"].mean() >= 0.5).astype(np.int64)
    return train_df["run_id"].map(run_lab).astype(np.int64).to_numpy()


# ----------------------------
# Fit helper (consistent with Stage-3 runner)
# ----------------------------

def _callable_base(fn: Any) -> Any:
    """Return a stable callable for signature inspection.

    For bound methods, we cache on the underlying function object (fn.__func__)
    so signatures are only inspected once per class method.
    """

    return getattr(fn, "__func__", fn)


@lru_cache(maxsize=512)
def _sig_params(fn_base: Any) -> Tuple[set[str], bool]:
    """Return (param_names, has_var_kw) for a callable.

    If signature inspection fails, returns (empty, False).
    """

    try:
        sig = inspect.signature(fn_base)
        params = sig.parameters
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        return set(params.keys()), bool(has_varkw)
    except Exception:
        return set(), False

def fit_with_optional_groups(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: Optional[np.ndarray] = None,
    group_strata: Optional[np.ndarray] = None,
    trial: Any = None,
) -> None:
    """Fit helper that passes groups / group_strata / trial only if supported.

    Important: we only fall back to a plain model.fit(X, y) call when the
    TypeError indicates an *unexpected keyword argument*; we do NOT mask
    genuine TypeErrors thrown inside model.fit.
    """

    kwargs: Dict[str, Any] = {}

    fn_base = _callable_base(model.fit)
    names, has_varkw = _sig_params(fn_base)

    if groups is not None and ("groups" in names or has_varkw):
        kwargs["groups"] = groups
    if group_strata is not None and ("group_strata" in names or has_varkw):
        kwargs["group_strata"] = group_strata
    if trial is not None and ("trial" in names or has_varkw):
        kwargs["trial"] = trial

    if not kwargs:
        model.fit(X, y)
        return

    try:
        model.fit(X, y, **kwargs)
    except TypeError as e:
        msg = str(e)
        if ("unexpected keyword" in msg) or ("got an unexpected keyword argument" in msg):
            # Estimator rejected one of the kwargs; retry without them.
            model.fit(X, y)
        else:
            raise

def binary_val_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    m = binary_metrics(y_true, y_prob)
    pr_base = float(m.get("pos_rate", float("nan")))
    pr_auc = float(m.get("pr_auc", float("nan")))
    pr_lift = pr_auc - pr_base if (pr_auc == pr_auc and pr_base == pr_base) else float("nan")
    pr_ratio = (pr_auc / pr_base) if (pr_auc == pr_auc and pr_base == pr_base and pr_base > 0) else float("nan")
    m["pr_baseline"] = pr_base
    m["pr_lift"] = float(pr_lift)
    m["pr_ratio"] = float(pr_ratio)
    return m


def multiclass_val_metrics(y_true_str: np.ndarray, P: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    """Validation metrics for multiclass tasks.

    NOTE: Do *not* renormalize here — multiclass_metrics() already performs
    robust sanitization via sanitize_multiclass_proba.
    """
    return multiclass_metrics(y_true_str, P, classes=classes)

# ----------------------------
# Model specs (registry)
# ----------------------------

@dataclass(frozen=True)
class StudyKey:
    model: str
    task: Task
    modality: Modality
    split: str
    W: int
    S: int

    @property
    def name(self) -> str:
        return f"{self.model}__{self.task}__{self.modality}__{self.split}__W{self.W}_S{self.S}"


class ModelSpec:
    """A model family that Optuna can tune."""

    name: str

    def suggest(self, trial: Any, *, base_cfg: Dict[str, Any], task: Task) -> Dict[str, Any]:
        raise NotImplementedError

    def build_binary(self, cfg: Dict[str, Any], *, seed: int) -> ModelBundle:
        raise NotImplementedError

    def build_multiclass(self, cfg: Dict[str, Any], *, n_classes: int, seed: int) -> ModelBundle:
        raise NotImplementedError

    def save_best_checkpoint(self, bundle: ModelBundle, path: Path, *, extra: Dict[str, Any]) -> None:
        """Optional: save a best checkpoint (torch state_dict recommended)."""
        # default: do nothing
        return


class ResMLPSpec(ModelSpec):
    name = "resmlp"

    def suggest(self, trial: Any, *, base_cfg: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """Bounded search space (small models; avoids wasting trials on huge configs)."""
        # Start from base config (YAML) and override suggested params.
        cfg = {k: v for k, v in dict(base_cfg).items() if k != "enabled"}

        # Capacity
        cfg["d_model"] = int(trial.suggest_categorical("d_model", [64, 128]))
        cfg["n_blocks"] = int(trial.suggest_categorical("n_blocks", [1, 2, 3, 4]))
        cfg["mlp_ratio"] = float(trial.suggest_categorical("mlp_ratio", [1.0, 2.0, 3.0]))

        # Regularization
        cfg["dropout"] = float(trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3]))
        cfg["weight_decay"] = float(trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))

        # Optimizer
        cfg["lr"] = float(trial.suggest_float("lr", 1e-4, 3e-3, log=True))
        cfg["batch_size"] = int(trial.suggest_categorical("batch_size", [128, 256, 512]))

        # Keep training loop bounded; these should generally be fixed in YAML, but we allow safe overrides.
        cfg.setdefault("max_epochs", int(base_cfg.get("max_epochs", 80)))
        cfg.setdefault("patience", int(base_cfg.get("patience", 10)))

        # Multiclass-only mild robustness
        if task == "multiclass":
            cfg["label_smoothing"] = float(trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1]))

        return cfg

    def build_binary(self, cfg: Dict[str, Any], *, seed: int) -> ModelBundle:
        return make_resmlp_binary(cfg, seed=int(seed))

    def build_multiclass(self, cfg: Dict[str, Any], *, n_classes: int, seed: int) -> ModelBundle:
        return make_resmlp_multiclass(cfg, n_classes=int(n_classes), seed=int(seed))

    def save_best_checkpoint(self, bundle: ModelBundle, path: Path, *, extra: Dict[str, Any]) -> None:
        """Save torch state_dict if the underlying model exposes .net."""
        try:
            import torch  # type: ignore
        except Exception:
            return
        try:
            net = getattr(bundle.model, "net", None)
            if net is None:
                return
            state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            ckpt = {"model_state_dict": state, **extra}
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, path)
        except Exception:
            return


class GRUSpec(ModelSpec):
    name = "gru"

    def suggest(self, trial, base_cfg: dict, task: str) -> dict:
        cfg = dict(base_cfg)

        cfg["d_model"] = int(trial.suggest_categorical("d_model", [32, 64, 128]))
        cfg["n_layers"] = int(trial.suggest_int("n_layers", 1, 2))
        cfg["dropout"] = float(trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3]))

        cfg["weight_decay"] = float(trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
        cfg["lr"] = float(trial.suggest_float("lr", 1e-4, 3e-3, log=True))
        cfg["batch_size"] = int(trial.suggest_categorical("batch_size", [128, 256, 512]))

        # Multiclass-only: optional label smoothing (low-risk calibration help)
        if task == "multiclass":
            cfg["label_smoothing"] = float(trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1]))

        return cfg

    def build_binary(self, cfg: dict, *, seed: int) -> ModelBundle:
        return make_gru_binary(cfg, seed=seed)

    def build_multiclass(self, cfg: dict, n_classes: int, *, seed: int) -> ModelBundle:
        return make_gru_multiclass(cfg, n_classes=n_classes, seed=seed)

    def save_best_checkpoint(self, bundle: ModelBundle, path, *, extra: Optional[dict] = None) -> None:
        # Save torch weights (if available)
        m = bundle.model
        net = getattr(m, "net", None)
        if net is None:
            return
        import torch
        torch.save({"model_state_dict": net.state_dict(), "extra": extra}, str(path))


class TCNSpec(ModelSpec):
    name = "tcn"

    def suggest(self, trial, base_cfg: dict, task: str) -> dict:
        cfg = dict(base_cfg)

        cfg["d_model"] = int(trial.suggest_categorical("d_model", [32, 64, 128]))
        cfg["n_blocks"] = int(trial.suggest_int("n_blocks", 2, 4))
        cfg["kernel_size"] = int(trial.suggest_categorical("kernel_size", [3, 5]))
        cfg["dropout"] = float(trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3]))

        cfg["weight_decay"] = float(trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
        cfg["lr"] = float(trial.suggest_float("lr", 1e-4, 3e-3, log=True))
        cfg["batch_size"] = int(trial.suggest_categorical("batch_size", [128, 256, 512]))

        if task == "multiclass":
            cfg["label_smoothing"] = float(trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1]))

        return cfg

    def build_binary(self, cfg: dict, *, seed: int) -> ModelBundle:
        return make_tcn_binary(cfg, seed=seed)

    def build_multiclass(self, cfg: dict, n_classes: int, *, seed: int) -> ModelBundle:
        return make_tcn_multiclass(cfg, n_classes=n_classes, seed=seed)

    def save_best_checkpoint(self, bundle: ModelBundle, path, *, extra: Optional[dict] = None) -> None:
        m = bundle.model
        net = getattr(m, "net", None)
        if net is None:
            return
        import torch
        torch.save({"model_state_dict": net.state_dict(), "extra": extra}, str(path))


class TransformerSpec(ModelSpec):
    name = "transformer"

    def suggest(self, trial, base_cfg: dict, task: str) -> dict:
        cfg = dict(base_cfg)

        d_model = int(trial.suggest_categorical("d_model", [32, 64, 128]))
        cfg["d_model"] = d_model

        # IMPORTANT (Optuna): categorical distributions must be constant within a study.
        # Do NOT make the choice set depend on d_model ("dynamic value space" error).
        # All d_model choices above are divisible by 2/4/8, so this stays valid.
        # Keep choices compatible across trials (and compatible with any already-started studies).
        # NOTE: If you start a *fresh* Optuna study and want to explore 8 heads, you can extend
        # this list to [2, 4, 8]. For resuming an existing study that started with [2, 4],
        # this must remain [2, 4] to avoid distribution-compatibility errors.
        cfg["n_heads"] = int(trial.suggest_categorical("n_heads", [2, 4,8]))

        cfg["n_layers"] = int(trial.suggest_int("n_layers", 1, 3))
        cfg["ff_mult"] = float(trial.suggest_categorical("ff_mult", [1.0, 2.0, 3.0]))
        cfg["dropout"] = float(trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3]))

        cfg["weight_decay"] = float(trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True))
        cfg["lr"] = float(trial.suggest_float("lr", 1e-4, 3e-3, log=True))
        cfg["batch_size"] = int(trial.suggest_categorical("batch_size", [128, 256, 512]))

        if task == "multiclass":
            cfg["label_smoothing"] = float(trial.suggest_categorical("label_smoothing", [0.0, 0.05, 0.1]))

        return cfg

    def build_binary(self, cfg: dict, *, seed: int) -> ModelBundle:
        return make_transformer_binary(cfg, seed=seed)

    def build_multiclass(self, cfg: dict, n_classes: int, *, seed: int) -> ModelBundle:
        return make_transformer_multiclass(cfg, n_classes=n_classes, seed=seed)

    def save_best_checkpoint(self, bundle: ModelBundle, path, *, extra: Optional[dict] = None) -> None:
        m = bundle.model
        net = getattr(m, "net", None)
        if net is None:
            return
        import torch
        torch.save({"model_state_dict": net.state_dict(), "extra": extra}, str(path))

# Registry (extend here)

class LogRegSpec(ModelSpec):
    model_name = "logreg"

    def suggest(self, trial: optuna.Trial, base_cfg: Dict[str, Any], task: str) -> Dict[str, Any]:
        cfg = dict(base_cfg)

        # Core regularization strength (most important hyperparam for logreg)
        cfg["C"] = trial.suggest_float("C", 1e-4, 1e2, log=True)
        cfg["max_iter"] = trial.suggest_int("max_iter", 300, 5000, log=True)

        # Class imbalance handling is often a big lever for binary.
        cfg["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced"])
        return cfg

    def build_binary(self, cfg: Dict[str, Any], seed: int) -> ModelBundle:
        cfg = dict(cfg)
        cfg.setdefault("random_state", int(seed))
        return make_logreg_binary(cfg)

    def build_multiclass(self, cfg: Dict[str, Any], n_classes: int, seed: int) -> ModelBundle:
        cfg = dict(cfg)
        cfg.setdefault("random_state", int(seed))
        return make_logreg_multiclass(cfg, n_classes=int(n_classes))

    def save_best_checkpoint(self, bundle: ModelBundle, path: Path, extra: Dict[str, Any]) -> None:
        # For sklearn models, persist via joblib.
        try:
            import joblib  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"joblib is required to save sklearn models: {e}") from e

        out_path = path.with_suffix(".joblib")
        joblib.dump(bundle.model, str(out_path))


class XGBSpec(ModelSpec):
    model_name = "xgboost"

    def suggest(self, trial: optuna.Trial, base_cfg: Dict[str, Any], task: str) -> Dict[str, Any]:
        cfg = dict(base_cfg)

        # High-impact tree / regularization knobs
        cfg["n_estimators"] = trial.suggest_int("n_estimators", 200, 2000, log=True)
        cfg["max_depth"] = trial.suggest_int("max_depth", 3, 10)
        cfg["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        cfg["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        cfg["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        cfg["min_child_weight"] = trial.suggest_float("min_child_weight", 1.0, 20.0, log=True)
        cfg["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)
        cfg["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        cfg["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True)

        return cfg

    def build_binary(self, cfg: Dict[str, Any], seed: int) -> ModelBundle:
        cfg = dict(cfg)
        cfg.setdefault("random_state", int(seed))
        return make_xgb_binary(cfg)

    def build_multiclass(self, cfg: Dict[str, Any], n_classes: int, seed: int) -> ModelBundle:
        cfg = dict(cfg)
        cfg.setdefault("random_state", int(seed))
        return make_xgb_multiclass(cfg, n_classes=int(n_classes))

    def save_best_checkpoint(self, bundle: ModelBundle, path: Path, extra: Dict[str, Any]) -> None:
        try:
            import joblib  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"joblib is required to save sklearn models: {e}") from e

        out_path = path.with_suffix(".joblib")
        joblib.dump(bundle.model, str(out_path))


class RFSpec(ModelSpec):
    model_name = "rf"

    def suggest(self, trial: optuna.Trial, base_cfg: Dict[str, Any], task: str) -> Dict[str, Any]:
        cfg = dict(base_cfg)

        cfg["n_estimators"] = trial.suggest_int("n_estimators", 200, 2500, log=True)
        cfg["criterion"] = trial.suggest_categorical("criterion", ["gini", "entropy"])

        # Depth is a bias/variance lever; allow None + some bounded options.
        cfg["max_depth"] = trial.suggest_categorical("max_depth", [None, 6, 10, 20, 40])

        cfg["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 50, log=True)
        cfg["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 20, log=True)

        cfg["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 1.0])
        cfg["bootstrap"] = trial.suggest_categorical("bootstrap", [True, False])

        # Only meaningful when bootstrap=True; otherwise keep it None.
        if cfg["bootstrap"]:
            cfg["max_samples"] = trial.suggest_float("max_samples", 0.5, 1.0)
        else:
            cfg["max_samples"] = None

        # Class imbalance handling
        cfg["class_weight"] = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])

        cfg["n_jobs"] = -1
        return cfg

    def build_binary(self, cfg: Dict[str, Any], seed: int) -> ModelBundle:
        cfg = dict(cfg)
        cfg.setdefault("random_state", int(seed))
        return make_rf_binary(cfg)

    def build_multiclass(self, cfg: Dict[str, Any], n_classes: int, seed: int) -> ModelBundle:
        cfg = dict(cfg)
        cfg.setdefault("random_state", int(seed))
        return make_rf_multiclass(cfg, n_classes=int(n_classes))

    def save_best_checkpoint(self, bundle: ModelBundle, path: Path, extra: Dict[str, Any]) -> None:
        try:
            import joblib  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"joblib is required to save sklearn models: {e}") from e

        out_path = path.with_suffix(".joblib")
        joblib.dump(bundle.model, str(out_path))

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "resmlp": ResMLPSpec(),
    "gru": GRUSpec(),
    "tcn": TCNSpec(),
    "transformer": TransformerSpec(),
    # Tabular baselines
    "logreg": LogRegSpec(),
    "xgboost": XGBSpec(),
    "rf": RFSpec(),
}


# ----------------------------
# Optuna runner
# ----------------------------

@dataclass
class OptunaRunConfig:
    processed_dir: Path
    out_dir: Path
    split_name: str
    W: int
    S: int
    seed: int

    model_name: str
    task: Task
    modality: Modality

    # HPO limits
    n_trials: int = 50
    timeout_s: Optional[int] = None

    # Threshold-free objective
    objective_metric: str = "log_loss"  # for binary + multiclass

    # Optuna knobs
    sampler: str = "tpe"
    pruner: str = "median"
    startup_trials: int = 10
    warmup_steps: int = 3

    benign_family_name: str = "Benign"
    feature_ablation: Optional[Dict[str, Any]] = None


def _require_optuna() -> Any:
    try:
        import optuna  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Optuna is required for HPO. Install with: pip install optuna"
        ) from e
    return optuna


def _make_storage_url(sqlite_path: Path) -> str:
    # Ensure absolute path for sqlite:///...
    p = sqlite_path.expanduser().resolve()
    return f"sqlite:///{p.as_posix()}"


def _select_base_cfg(stage3_cfg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    # Prefer models.dl, fall back to models.tabular
    m = stage3_cfg.get("models", {})
    dl = m.get("dl", {}) if isinstance(m, dict) else {}
    tab = m.get("tabular", {}) if isinstance(m, dict) else {}
    if model_name in dl:
        return dict(dl[model_name])
    if model_name in tab:
        return dict(tab[model_name])
    raise KeyError(f"Model {model_name!r} not found under models.dl or models.tabular in YAML.")


def _study_dir(base_out: Path, key: StudyKey) -> Path:
    return base_out / "studies" / key.name


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_optuna_study(stage3_cfg: Dict[str, Any], rcfg: OptunaRunConfig) -> Dict[str, Any]:
    """Run one Optuna study and persist outputs.

    Returns a dict with best_value, best_params, and paths written.
    """
    optuna = _require_optuna()

    model_name = str(rcfg.model_name).strip().lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model for Optuna: {model_name!r}. Known: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name]

    key = StudyKey(model=model_name, task=rcfg.task, modality=rcfg.modality, split=rcfg.split_name, W=rcfg.W, S=rcfg.S)
    sdir = _study_dir(rcfg.out_dir, key)
    sdir.mkdir(parents=True, exist_ok=True)

    sqlite_path = sdir / "study.sqlite3"
    storage = _make_storage_url(sqlite_path)

    # Sampler/pruner
    if str(rcfg.sampler).strip().lower() == "tpe":
        sampler = optuna.samplers.TPESampler(seed=int(rcfg.seed))
    else:
        sampler = optuna.samplers.RandomSampler(seed=int(rcfg.seed))

    pruner_name = str(rcfg.pruner).strip().lower()
    if pruner_name == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=int(rcfg.startup_trials), n_warmup_steps=int(rcfg.warmup_steps))
    elif pruner_name in {"none", "off", "disabled"}:
        pruner = optuna.pruners.NopPruner()
    else:
        # Keep it simple; Hyperband can be added later if needed
        pruner = optuna.pruners.MedianPruner(n_startup_trials=int(rcfg.startup_trials), n_warmup_steps=int(rcfg.warmup_steps))

    study = optuna.create_study(
        study_name=key.name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    # Cache data (loaded once)
    tr, va = load_train_val(
        rcfg.processed_dir,
        rcfg.split_name,
        rcfg.modality,
        int(rcfg.W),
        int(rcfg.S),
        feature_ablation=rcfg.feature_ablation,
    )

    # ---- Ensure groups (run_id) are available (seq models need this) ----
    groups_tr = _ensure_groups(tr)
    groups_va = _ensure_groups(va)

    # Sequence models REQUIRE run_id groups to build (left-padded) sequences.
    if model_name in ("gru", "tcn", "transformer"):
        if groups_tr is None:
            raise RuntimeError(
                "HPO: TRAIN split has no groups/run_id; sequence models require run_id groups."
            )
        if groups_va is None:
            raise RuntimeError(
                "HPO: VAL split has no groups/run_id; sequence models require run_id groups."
            )

    # Normalize to numpy arrays (stable dtype for grouping + consistent length checks)
    if groups_tr is not None:
        groups_tr = np.asarray(groups_tr).astype(str).reshape(-1)
    if groups_va is not None:
        groups_va = np.asarray(groups_va).astype(str).reshape(-1)

    # Sanity (prevents silent misalignment bugs)
    if groups_tr is not None and len(groups_tr) != tr.X.shape[0]:
        raise RuntimeError(f"HPO: groups_tr len {len(groups_tr)} != tr.X rows {tr.X.shape[0]}")
    if groups_va is not None and len(groups_va) != va.X.shape[0]:
        raise RuntimeError(f"HPO: groups_va len {len(groups_va)} != va.X rows {va.X.shape[0]}")


    # Build labels
    if rcfg.task == "binary":
        y_tr = tr.y_bin.astype(int)
        y_va = va.y_bin.astype(int)
        group_strata = group_strata_binary(tr.df, benign_family_name=rcfg.benign_family_name)
        classes = None
        y_va_str = None
        n_classes = None
    else:
        # Match Stage-3 multiclass protocol: LabelEncoder fit on TRAIN strings
        from sklearn.preprocessing import LabelEncoder  # local import
        le = LabelEncoder()
        y_tr_str = tr.y_cat.astype(str)
        y_va_str = va.y_cat.astype(str)
        le.fit(y_tr_str)
        classes = le.classes_.astype(str)
        y_tr = le.transform(y_tr_str).astype(int)
        y_va = le.transform(y_va_str).astype(int)
        group_strata = y_tr  # per-window; model will mode() per run
        n_classes = int(len(classes))

    base_cfg = resolve_model_cfg(_select_base_cfg(stage3_cfg, model_name), task=rcfg.task, modality=rcfg.modality)

    # Track best so we can save config/checkpoint without relying on Optuna callbacks.
    best_val = float("inf")

    def objective(trial: Any) -> float:
        nonlocal best_val
        # Build trial cfg (start from base, then suggest)
        trial_cfg = spec.suggest(trial, base_cfg=base_cfg, task=rcfg.task)

        # Build model bundle
        if rcfg.task == "binary":
            bundle = spec.build_binary(trial_cfg, seed=int(rcfg.seed))
            fit_with_optional_groups(bundle.model, tr.X, y_tr, groups=groups_tr, group_strata=group_strata, trial=trial)
            p_va = predict_proba_binary(bundle, va.X, groups=groups_va)
            # Guard: prune trials with invalid probabilities
            if not np.isfinite(p_va).all() or p_va.size == 0:
                raise optuna.TrialPruned(
                    f"Invalid binary proba: finite={bool(np.isfinite(p_va).all())}, n={int(p_va.size)}"
                )
                    # Guard: prune trials with out-of-range probabilities
            p_min = float(np.min(p_va))
            p_max = float(np.max(p_va))
            if (p_min < -1e-6) or (p_max > 1.0 + 1e-6):
                raise optuna.TrialPruned(f"Binary proba out of [0,1]: min={p_min}, max={p_max}")
            m = binary_val_metrics(y_va, p_va)
            val = float(m.get(rcfg.objective_metric, float("nan")))
        else:
            assert classes is not None and n_classes is not None and y_va_str is not None
            bundle = spec.build_multiclass(trial_cfg, n_classes=int(n_classes), seed=int(rcfg.seed))
            fit_with_optional_groups(bundle.model, tr.X, y_tr, groups=groups_tr, group_strata=group_strata, trial=trial)
            P_va = predict_proba_multiclass(bundle, va.X, groups=groups_va)
            # Guard: prune trials with invalid probabilities
            if not np.isfinite(P_va).all() or P_va.size == 0:
                raise optuna.TrialPruned(
                    f"Invalid multiclass proba: finite={bool(np.isfinite(P_va).all())}, n={int(P_va.size)}"
                )
                    # Guard: prune trials with out-of-range probabilities
            p_min = float(np.min(P_va))
            p_max = float(np.max(P_va))
            if (p_min < -1e-6) or (p_max > 1.0 + 1e-6):
                raise optuna.TrialPruned(f"Multiclass proba out of [0,1]: min={p_min}, max={p_max}")
            row_sums = P_va.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-3, rtol=0.0):
                rs_min = float(np.min(row_sums))
                rs_max = float(np.max(row_sums))
                raise optuna.TrialPruned(f"Multiclass proba rows not normalized: sum in [{rs_min}, {rs_max}]")
            mm = multiclass_val_metrics(y_va_str, P_va, classes=classes)
            m = dict(mm)
            val = float(m.get(rcfg.objective_metric, float("nan")))

        # Attach tracked metrics for later analysis (not part of objective)
        for k, v in m.items():
            if isinstance(v, (float, int)) and (not math.isnan(float(v))):
                trial.set_user_attr(k, float(v))
        # Keep the final cfg for reproducibility
        trial.set_user_attr("resolved_cfg", dict(trial_cfg))

        # Save best config + checkpoint (best-so-far)
        if val == val and val < (best_val - 1e-12):
            best_val = float(val)
            _write_json(sdir / "best_config.json", {"model": model_name, "task": rcfg.task, "modality": rcfg.modality, "split": rcfg.split_name, "W": rcfg.W, "S": rcfg.S, "value": best_val, "cfg": trial_cfg})
            # Optional checkpoint
            try:
                extra = {"task": rcfg.task, "modality": rcfg.modality, "split": rcfg.split_name, "W": int(rcfg.W), "S": int(rcfg.S), "value": best_val, "cfg": dict(trial_cfg)}
                spec.save_best_checkpoint(bundle, sdir / "best_checkpoint.pt", extra=extra)
            except Exception as e:
                # Non-fatal: save error should not kill the study.
                try:
                    trial.set_user_attr("best_checkpoint_save_error", str(e))
                except Exception:
                    pass

        # Free GPU memory between trials (best-effort)
        try:
            import torch  # type: ignore
            del bundle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            # Best-effort cleanup only.
            try:
                trial.set_user_attr("trial_cleanup_error", str(e))
            except Exception:
                pass

        return float(val)

    study.optimize(
        objective,
        n_trials=int(rcfg.n_trials),
        timeout=rcfg.timeout_s,
        catch=(RuntimeError, ValueError, FloatingPointError),
    )

    # Persist summary table
    try:
        df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
        _write_csv(sdir / "study_summary.csv", df)
    except Exception as e:
        # Non-fatal: summary CSV is for convenience.
        print(f"[WARN] Failed to write study_summary.csv for {key.name}: {e}")

    # Persist best trial (robust even if all trials fail)
    if (not math.isfinite(best_val)) or (best_val == float("inf")):
        best_obj = {
            "study": key.name,
            "error": "No successful trial produced a finite objective value.",
            "n_trials": int(len(study.trials)),
        }
        _write_json(sdir / "best_trial.json", best_obj)
        return {
            "study_name": key.name,
            "study_dir": str(sdir),
            "storage": storage,
            "best_value": float("nan"),
            "best_params": {},
        }

    try:
        best = study.best_trial
        best_value = float(best.value) if best.value is not None else float("nan")
        best_obj = {
            "study": key.name,
            "best_value": best_value,
            "best_params": dict(best.params),
            "best_user_attrs": dict(best.user_attrs),
        }
        _write_json(sdir / "best_trial.json", best_obj)

        return {
            "study_name": key.name,
            "study_dir": str(sdir),
            "storage": storage,
            "best_value": best_value,
            "best_params": dict(best.params),
        }
    except Exception as e:
        # Fallback: best-so-far config written by our objective loop.
        fallback: Dict[str, Any] = {}
        best_path = sdir / "best_config.json"
        if best_path.exists():
            try:
                with open(best_path, "r", encoding="utf-8") as f:
                    fallback = json.load(f)
            except Exception:
                fallback = {}

        best_value_fb = float("nan")
        if isinstance(fallback, dict) and "value" in fallback:
            try:
                best_value_fb = float(fallback.get("value"))
            except Exception:
                best_value_fb = float("nan")

        best_obj = {
            "study": key.name,
            "error": f"study.best_trial failed: {e}",
            "n_trials": int(len(study.trials)),
            "fallback_best_config": fallback,
        }
        _write_json(sdir / "best_trial.json", best_obj)

        return {
            "study_name": key.name,
            "study_dir": str(sdir),
            "storage": storage,
            "best_value": best_value_fb,
            "best_params": {},
        }


def apply_best_configs_to_yaml(
    base_cfg: Dict[str, Any],
    *,
    model_name: str,
    best_cfgs: Dict[Tuple[Task, Modality], Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a NEW YAML dict with nested best configs applied under models.<group>[model_name].

    - For DL models, <group> is models.dl
    - For tabular baselines (logreg/xgboost/rf), <group> is models.tabular

    We write nested overrides under:
        models.<group>.<model_name>.<task>.<modality> = <best_hparams>

    This matches run_stage3_tabular.py's config resolution:
        cfg0.get(task, {}).get(modality, cfg0.get("base", cfg0))
    """
    cfg = json.loads(json.dumps(base_cfg))  # deep copy (safe for YAML)

    m = cfg.setdefault("models", {})
    if not isinstance(m, dict):
        raise ValueError("Config: models must be a dict")

    dl = m.setdefault("dl", {})
    tab = m.setdefault("tabular", {})
    if not isinstance(dl, dict):
        raise ValueError("Config: models.dl must be a dict")
    if not isinstance(tab, dict):
        raise ValueError("Config: models.tabular must be a dict")

    # Decide where to write: prefer whichever group already contains the model.
    if model_name in tab:
        group = tab
    elif model_name in dl:
        group = dl
    else:
        # Fall back to registry convention if model isn't present in base YAML.
        group = tab if model_name in ("logreg", "xgboost", "rf") else dl

    mdl = group.setdefault(model_name, {})
    if not isinstance(mdl, dict):
        raise ValueError(f"Config: models.<group>.{model_name} must be a dict")

    # Keep existing keys; we only set nested overrides.
    for (task, modality), hps in best_cfgs.items():
        tnode = mdl.setdefault(task, {})
        if not isinstance(tnode, dict):
            tnode = {}
            mdl[task] = tnode
        mnode = tnode.setdefault(modality, {})
        if not isinstance(mnode, dict):
            mnode = {}
            tnode[modality] = mnode

        # Overwrite with best cfg
        mnode.clear()
        mnode.update(dict(hps))

    return cfg


def write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    """Write YAML if PyYAML is available; else write JSON with .json suffix."""
    try:
        import yaml  # type: ignore
    except Exception:
        _write_json(path.with_suffix(".json"), obj)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
