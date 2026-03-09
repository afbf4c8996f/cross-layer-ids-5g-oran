"""
stage3_models.py
Base models for Stage-3 tabular experiments.
"""
from __future__ import annotations

import inspect
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def _callable_base(fn: Any) -> Any:
    """Return a stable callable for signature inspection.

    For bound methods, cache on the underlying function (fn.__func__).
    """

    return getattr(fn, "__func__", fn)


@lru_cache(maxsize=512)
def _sig_accepts_kw(fn_base: Any, kw: str) -> bool:
    """Whether callable accepts kw directly or via **kwargs."""

    try:
        sig = inspect.signature(fn_base)
        if kw in sig.parameters:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except Exception:
        return False

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None  # type: ignore

@dataclass
class ModelBundle:
    name: str
    model: Any
    is_multiclass: bool

def make_logreg_binary(cfg: Dict[str, Any]) -> ModelBundle:
    # Avoid deprecated 'multi_class' arg (sklearn>=1.5 warns)
    C = float(cfg.get("C", 1.0))
    max_iter = int(cfg.get("max_iter", 5000))
    solver = str(cfg.get("solver", "lbfgs"))
    n_jobs = int(cfg.get("n_jobs", -1))
    class_weight = cfg.get("class_weight", "balanced")
    m = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        n_jobs=n_jobs,
        class_weight=class_weight,
        # multi_class not set on purpose
    )
    return ModelBundle(name="logreg", model=m, is_multiclass=False)

def make_logreg_multiclass(cfg: Dict[str, Any], *, n_classes: int | None = None) -> ModelBundle:
    """Multiclass Logistic Regression baseline.

    ``n_classes`` is accepted for API compatibility with the Optuna HPO
    builders, which pass it for all multiclass models. LogisticRegression does
    not require it, so it is intentionally unused.
    """
    C = float(cfg.get("C", 1.0))
    max_iter = int(cfg.get("max_iter", 5000))
    solver = str(cfg.get("solver", "lbfgs"))
    n_jobs = int(cfg.get("n_jobs", -1))
    m = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        n_jobs=n_jobs,
        # multi_class not set; sklearn will use multinomial when appropriate
    )
    return ModelBundle(name="logreg", model=m, is_multiclass=True)

def make_xgb_binary(cfg: Dict[str, Any]) -> ModelBundle:
    if xgb is None:
        raise RuntimeError("xgboost is not installed. Install with: pip install xgboost")
    params = {k: v for k, v in dict(cfg).items() if k != 'enabled'}
    # Some sane defaults
    params.setdefault("n_estimators", 600)
    params.setdefault("max_depth", 6)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("subsample", 0.9)
    params.setdefault("colsample_bytree", 0.9)
    params.setdefault("reg_lambda", 1.0)
    params.setdefault("tree_method", "hist")
    params.setdefault("random_state", 0)

    m = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        **params,
    )
    return ModelBundle(name="xgboost", model=m, is_multiclass=False)

def make_xgb_multiclass(cfg: Dict[str, Any], n_classes: int) -> ModelBundle:
    if xgb is None:
        raise RuntimeError("xgboost is not installed. Install with: pip install xgboost")
    params = {k: v for k, v in dict(cfg).items() if k != 'enabled'}
    params.setdefault("n_estimators", 600)
    params.setdefault("max_depth", 6)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("subsample", 0.9)
    params.setdefault("colsample_bytree", 0.9)
    params.setdefault("reg_lambda", 1.0)
    params.setdefault("tree_method", "hist")
    params.setdefault("random_state", 0)

    m = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=int(n_classes),
        eval_metric="mlogloss",
        **params,
    )
    return ModelBundle(name="xgboost", model=m, is_multiclass=True)


def make_rf_binary(cfg: Dict[str, Any]) -> ModelBundle:
    """RandomForest baseline (binary)."""
    params = {k: v for k, v in dict(cfg).items() if k != "enabled"}

    # Sane defaults (robust baseline)
    params.setdefault("n_estimators", 800)
    params.setdefault("max_depth", None)
    params.setdefault("min_samples_split", 2)
    params.setdefault("min_samples_leaf", 1)
    params.setdefault("max_features", "sqrt")
    params.setdefault("bootstrap", True)
    params.setdefault("class_weight", "balanced")
    params.setdefault("n_jobs", -1)
    params.setdefault("random_state", 0)

    m = RandomForestClassifier(**params)
    return ModelBundle(name="rf", model=m, is_multiclass=False)


def make_rf_multiclass(cfg: Dict[str, Any], n_classes: int) -> ModelBundle:
    """RandomForest baseline (multiclass)."""
    params = {k: v for k, v in dict(cfg).items() if k != "enabled"}

    params.setdefault("n_estimators", 800)
    params.setdefault("max_depth", None)
    params.setdefault("min_samples_split", 2)
    params.setdefault("min_samples_leaf", 1)
    params.setdefault("max_features", "sqrt")
    params.setdefault("bootstrap", True)
    params.setdefault("class_weight", None)
    params.setdefault("n_jobs", -1)
    params.setdefault("random_state", 0)

    m = RandomForestClassifier(**params)
    return ModelBundle(name="rf", model=m, is_multiclass=True)


# ----------------------------
# PyTorch (optional) models
# ----------------------------


def make_resmlp_binary(cfg: Dict[str, Any], *, seed: int = 0) -> ModelBundle:
    """ResMLP binary classifier (PyTorch).

    The underlying model is built lazily at fit-time so input_dim is inferred
    from X and never hardcoded.
    """
    try:
        from stage3_torch import TorchResMLPBinary  # local import keeps torch optional
    except Exception as e:
        raise RuntimeError(
            "Failed to import PyTorch backend (stage3_torch). "
            "Install torch or disable models.dl.resmlp in the YAML. "
            f"Import error: {e}"
        )
    m = TorchResMLPBinary(cfg, seed=int(seed))
    return ModelBundle(name="resmlp", model=m, is_multiclass=False)


def make_resmlp_multiclass(cfg: Dict[str, Any], n_classes: int, *, seed: int = 0) -> ModelBundle:
    """ResMLP multiclass classifier (PyTorch)."""
    try:
        from stage3_torch import TorchResMLPMulticlass
    except Exception as e:
        raise RuntimeError(
            "Failed to import PyTorch backend (stage3_torch). "
            "Install torch or disable models.dl.resmlp in the YAML. "
            f"Import error: {e}"
        )
    m = TorchResMLPMulticlass(cfg, n_classes=int(n_classes), seed=int(seed))
    return ModelBundle(name="resmlp", model=m, is_multiclass=True)


def make_gru_binary(cfg: dict, *, seed: int = 0) -> ModelBundle:
    from stage3_torch import TorchGRUBinary
    return ModelBundle(name="gru", model=TorchGRUBinary(cfg, seed=int(seed)), is_multiclass=False)


def make_gru_multiclass(cfg: dict, n_classes: int, *, seed: int = 0) -> ModelBundle:
    from stage3_torch import TorchGRUMulticlass
    return ModelBundle(name="gru", model=TorchGRUMulticlass(cfg, n_classes=int(n_classes), seed=int(seed)), is_multiclass=True)


def make_tcn_binary(cfg: dict, *, seed: int = 0) -> ModelBundle:
    from stage3_torch import TorchTCNBinary
    return ModelBundle(name="tcn", model=TorchTCNBinary(cfg, seed=int(seed)), is_multiclass=False)


def make_tcn_multiclass(cfg: dict, n_classes: int, *, seed: int = 0) -> ModelBundle:
    from stage3_torch import TorchTCNMulticlass
    return ModelBundle(name="tcn", model=TorchTCNMulticlass(cfg, n_classes=int(n_classes), seed=int(seed)), is_multiclass=True)


def make_transformer_binary(cfg: dict, *, seed: int = 0) -> ModelBundle:
    from stage3_torch import TorchTransformerBinary
    return ModelBundle(name="transformer", model=TorchTransformerBinary(cfg, seed=int(seed)), is_multiclass=False)


def make_transformer_multiclass(cfg: dict, n_classes: int, *, seed: int = 0) -> ModelBundle:
    from stage3_torch import TorchTransformerMulticlass
    return ModelBundle(
        name="transformer",
        model=TorchTransformerMulticlass(cfg, n_classes=int(n_classes), seed=int(seed)),
        is_multiclass=True,
    )

def predict_proba_binary(bundle: ModelBundle, X: np.ndarray, *, groups: Optional[np.ndarray] = None) -> np.ndarray:
    """Return p(attack)=proba[:,1].

    For sequence models, `groups` (run_id array) is required to build per-run causal sequences.
    For classic tabular models, `groups` is ignored.
    """
    m = bundle.model

    # If the estimator supports groups at inference, pass them.
    if groups is not None and _sig_accepts_kw(_callable_base(m.predict_proba), "groups"):
        proba = m.predict_proba(X, groups=groups)
    else:
        proba = m.predict_proba(X)

    # Expected shapes:
    #  - sklearn-style: (N,2) with columns [p0, p1]
    #  - torch wrappers here also return (N,2)
    # very good place to raise
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(f"predict_proba expected shape (N,2+). Got {proba.shape} from {type(m)}")
    return proba[:, 1]


def predict_proba_multiclass(bundle: ModelBundle, X: np.ndarray, *, groups: Optional[np.ndarray] = None) -> np.ndarray:
    """Return class probability matrix (N,K)."""
    m = bundle.model

    if groups is not None and _sig_accepts_kw(_callable_base(m.predict_proba), "groups"):
        proba = m.predict_proba(X, groups=groups)
    else:
        proba = m.predict_proba(X)
    if proba.ndim != 2:
        raise ValueError(f"predict_proba expected 2D (N,K). Got {proba.shape} from {type(m)}")
    return proba
