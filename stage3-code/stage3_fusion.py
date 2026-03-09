"""
stage3_fusion.py
Late fusion:
- mean probability (binary) / mean per-class prob (multiclass)
- stacked logistic regression with proper OOF training
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

def _clip01(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.where(np.isfinite(p), p, 0.5)
    return np.clip(p, eps, 1.0 - eps)

def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip01(p)
    return np.log(p / (1.0 - p))

def fusion_mean_binary(p_net: np.ndarray, p_rad: np.ndarray) -> np.ndarray:
    p_net = np.where(np.isfinite(p_net), p_net, 0.5)
    p_rad = np.where(np.isfinite(p_rad), p_rad, 0.5)
    return ((p_net + p_rad) / 2.0).astype(np.float32)

def fusion_mean_multiclass(P_net: np.ndarray, P_rad: np.ndarray) -> np.ndarray:
    P_net = np.where(np.isfinite(P_net), P_net, 0.0)
    P_rad = np.where(np.isfinite(P_rad), P_rad, 0.0)
    P = (P_net + P_rad) / 2.0
    # Renormalize (tiny numeric drift)
    s = P.sum(axis=1, keepdims=True)
    s[s == 0.0] = 1.0
    return (P / s).astype(np.float32)

@dataclass
class StackedFusionBinary:
    model: LogisticRegression

    def predict(self, p_net: np.ndarray, p_rad: np.ndarray) -> np.ndarray:
        X = np.stack([_logit(p_net), _logit(p_rad)], axis=1).astype(np.float32)
        proba = self.model.predict_proba(X)
        return proba[:, 1].astype(np.float32)

def train_stacked_binary(p_net: np.ndarray, p_rad: np.ndarray, y: np.ndarray, C: float = 1.0, seed: int = 0) -> StackedFusionBinary:
    X = np.stack([_logit(p_net), _logit(p_rad)], axis=1).astype(np.float32)
    y = y.astype(int)
    m = LogisticRegression(
        C=float(C),
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=int(seed),
    )
    m.fit(X, y)
    return StackedFusionBinary(model=m)

@dataclass
class StackedFusionMulticlass:
    model: LogisticRegression
    classes_: np.ndarray  # label strings

    def predict_proba(self, P_net: np.ndarray, P_rad: np.ndarray) -> np.ndarray:
        # Use log-probabilities as features (stable)
        eps = 1e-6
        Pn = np.where(np.isfinite(P_net), P_net, eps)
        Pr = np.where(np.isfinite(P_rad), P_rad, eps)
        Pn = np.clip(Pn, eps, 1.0)
        Pr = np.clip(Pr, eps, 1.0)
        X = np.concatenate([np.log(Pn), np.log(Pr)], axis=1).astype(np.float32)
        P = self.model.predict_proba(X).astype(np.float32)
        return P

def train_stacked_multiclass(P_net: np.ndarray, P_rad: np.ndarray, y_str: np.ndarray, C: float = 1.0, seed: int = 0) -> StackedFusionMulticlass:
    eps = 1e-6
    Pn = np.where(np.isfinite(P_net), P_net, eps)
    Pr = np.where(np.isfinite(P_rad), P_rad, eps)
    Pn = np.clip(Pn, eps, 1.0)
    Pr = np.clip(Pr, eps, 1.0)
    X = np.concatenate([np.log(Pn), np.log(Pr)], axis=1).astype(np.float32)

    # sklearn expects y as strings ok
    m = LogisticRegression(
        C=float(C),
        max_iter=5000,
        solver="lbfgs",
        random_state=int(seed),
    )
    m.fit(X, y_str.astype(str))
    return StackedFusionMulticlass(model=m, classes_=m.classes_)
