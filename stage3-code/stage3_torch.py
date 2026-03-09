"""
stage3_torch.py

PyTorch helpers for Stage-3 tabular experiments.

Design goals (aligned with your Stage-3 protocol):
  - sklearn-like API: .fit(X, y, groups=None) + .predict_proba(X)
  - NO hardcoded input shapes (infer input_dim from X at fit-time)
  - Run-level (group) early-stopping split when groups are provided
  - Stratified-by-run-label early-stopping split for BOTH binary and multiclass
  - Optional AMP on CUDA
  - Small, regularized ResMLP for tabular
  - Torch is an optional dependency unless DL models are enabled
"""

from __future__ import annotations

import logging
import random
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Tuple, TYPE_CHECKING

import numpy as np

try:
    from typing import Self  # Py3.11+
except Exception:  # pragma: no cover
    from typing_extensions import Self  # type: ignore

# ---- typing-only aliases (avoid optional-import confusion in type checkers) ----
# NOTE: We keep torch as an optional dependency (imported lazily by stage3_models),
# but we avoid referencing `torch.*` in annotations on the runtime path to keep
# type checkers (Pylance/Mypy) happy.
if TYPE_CHECKING:  # pragma: no cover
    import torch
    from torch import Tensor as TorchTensor
    Tensor = TorchTensor
    TorchDeviceHint = torch.device
else:  # pragma: no cover
    Tensor = Any  # type: ignore
    TorchDeviceHint = Any  # type: ignore


try:  # pragma: no cover
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import amp
    from torch.utils.data import DataLoader, TensorDataset, Subset
    try:
        from torch.nn.utils.parametrizations import weight_norm as _weight_norm  # type: ignore
    except Exception:  # pragma: no cover
        from torch.nn.utils import weight_norm as _weight_norm  # type: ignore
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    amp = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore
    Subset = None  # type: ignore
    _TORCH_IMPORT_ERROR = e
else:
    _TORCH_IMPORT_ERROR = None


# ---------------------------
# helpers
# ---------------------------

def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for DL models but is not available in this environment. "
            "Install torch (and CUDA build if desired). "
            f"Original import error: {_TORCH_IMPORT_ERROR}"
        )


def _apply_weight_norm(module: nn.Module) -> nn.Module:
    """Apply weight norm without noisy deprecation warnings.

    Newer PyTorch versions prefer torch.nn.utils.parametrizations.weight_norm.
    Some builds still emit a FutureWarning about the legacy API when applying
    weight norm.

    This helper uses whatever weight_norm implementation was imported at module
    import time (parametrizations.weight_norm when available, otherwise the
    legacy one) and suppresses only the deprecation FutureWarning.
    """

    _require_torch()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r".*torch\.nn\.utils\.weight_norm.*deprecated.*",
        )
        return _weight_norm(module)


def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(device: str) -> TorchDeviceHint:
    _require_torch()
    d = str(device).strip().lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _as_float32(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return np.ascontiguousarray(X)


def _as_int64(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    if y.dtype != np.int64:
        y = y.astype(np.int64, copy=False)
    return np.ascontiguousarray(y)


def _assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr))
        raise ValueError(
            f"{name} contains NaN/Inf (first bad index={bad[0][0] if len(bad[0]) else 'unknown'}). "
            "Fix upstream preprocessing; refusing to train with non-finite inputs."
        )


def _mode_value(values: np.ndarray) -> Any:
    # Works for ints/strings; small arrays (per-run) so np.unique is fine.
    u, c = np.unique(values, return_counts=True)
    return u[int(np.argmax(c))]


def _train_val_split_by_groups_stratified(
    y: np.ndarray,
    groups: np.ndarray,
    val_frac: float,
    seed: int,
    *,
    group_strata: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (train_idx, val_idx) using a run-level split, stratified by run label.

    Stratification label per-run is:
      - mode(group_strata in the run), if group_strata is provided (recommended: family)
      - else mode(y in the run)

    Guarantees (when possible):
      - no stratum with >=2 runs will be fully moved into val (prevents train starvation)
      - always keeps at least 1 run in train if n_groups > 1
    """
    n = int(len(y))
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    val_frac = float(val_frac)
    if not (0.0 < val_frac < 1.0):
        idx = np.arange(n, dtype=int)
        return idx, np.array([], dtype=int)

    groups = np.asarray(groups).astype(str)
    uniq_groups = np.unique(groups)
    n_groups = int(len(uniq_groups))

    if n_groups <= 1:
        idx = np.arange(n, dtype=int)
        return idx, np.array([], dtype=int)

    rng = np.random.default_rng(int(seed))

    # build per-group stratum label
    strata_by_group = {}
    for g in uniq_groups:
        mask = (groups == g)
        if group_strata is not None:
            s = _mode_value(np.asarray(group_strata)[mask])
        else:
            s = _mode_value(y[mask])
        strata_by_group[g] = s

    # group lists per stratum
    strata_vals = np.array([strata_by_group[g] for g in uniq_groups], dtype=object)
    unique_strata = np.unique(strata_vals)

    # choose val groups per stratum (keep >=1 group in train whenever possible)
    val_groups: list[str] = []
    for s in unique_strata:
        g_s = uniq_groups[strata_vals == s]
        g_s = np.array(g_s, dtype=str)

        if len(g_s) <= 1:
            # cannot split this stratum across train+val; keep it in train
            continue

        n_val_s = int(round(val_frac * len(g_s)))
        n_val_s = max(1, n_val_s)
        n_val_s = min(n_val_s, len(g_s) - 1)

        rng.shuffle(g_s)
        val_groups.extend(g_s[:n_val_s].tolist())

    # If we ended up with empty val, it means no stratum had >=2 groups.
    # In that case, *any* non-empty val would necessarily move an entire stratum
    # out of TRAIN, starving the model of that class/run-type.
    # For correctness, return an empty val split (disables early stopping).
    if len(val_groups) == 0:
        idx = np.arange(n, dtype=int)
        return idx, np.array([], dtype=int)

    val_groups_set = set(val_groups)
    val_mask = np.isin(groups, list(val_groups_set))
    tr_idx = np.flatnonzero(~val_mask).astype(int)
    va_idx = np.flatnonzero(val_mask).astype(int)

    # final safety: never allow empty train
    if len(tr_idx) == 0:
        # move one val group back to train
        any_val_group = next(iter(val_groups_set))
        val_mask = np.isin(groups, [g for g in val_groups if g != any_val_group])
        tr_idx = np.flatnonzero(~val_mask).astype(int)
        va_idx = np.flatnonzero(val_mask).astype(int)

    return tr_idx, va_idx


# ---------------------------
# Model: ResMLP (+ optional GRN)
# ---------------------------

class GRN(nn.Module):
    """
    Global Response Normalization (lightweight).
    For tabular, we normalize across feature dim (last dim).
    Kept optional via cfg.use_grn.
    """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.zeros(1, int(dim)))
        self.beta = nn.Parameter(torch.zeros(1, int(dim)))

    def forward(self, x: Tensor) -> Tensor:
        gx = x.abs()
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return x + (x * nx) * self.gamma + self.beta


class ResMLPBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_ratio: float,
        dropout: float,
        *,
        use_grn: bool = False,
        grn_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        hidden = int(max(4, round(d_model * float(mlp_ratio))))
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()
        self.grn = GRN(hidden, eps=grn_eps) if use_grn else nn.Identity()
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: Tensor) -> Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.grn(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h


class ResMLPTabular(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_blocks: int,
        mlp_ratio: float,
        dropout: float,
        out_dim: int,
        *,
        use_grn: bool = False,
        grn_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(int(input_dim), int(d_model))
        self.blocks = nn.ModuleList(
            [
                ResMLPBlock(
                    d_model=int(d_model),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                    use_grn=bool(use_grn),
                    grn_eps=float(grn_eps),
                )
                for _ in range(int(n_blocks))
            ]
        )
        self.ln_out = nn.LayerNorm(int(d_model))
        self.head = nn.Linear(int(d_model), int(out_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_out(x)
        return self.head(x)


# ---------------------------
# Training config
# ---------------------------

@dataclass
class TorchTrainCfg:
    d_model: int = 128
    n_blocks: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    use_grn: bool = True
    grn_eps: float = 1e-6

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256

    max_epochs: int = 80
    patience: int = 10
    grad_clip_norm: float = 1.0
    val_frac: float = 0.2

    # Multiclass-only regularization (applies to training loss only)
    label_smoothing: float = 0.0

    device: str = "cuda"
    amp: bool = True
    num_workers: int = 0

    # Optional I/O (only used when set; caller can keep them None for OOF folds)
    log_path: Optional[str] = None
    checkpoint_path: Optional[str] = None

    seed: int = 0


def _cfg_from_dict(d: dict, *, seed: int) -> TorchTrainCfg:
    c = TorchTrainCfg()
    for k, v in d.items():
        if not hasattr(c, k):
            continue
        cur = getattr(c, k)
        # Optional fields default to None; do not attempt to cast via NoneType.
        if cur is None:
            setattr(c, k, v)
        else:
            try:
                setattr(c, k, type(cur)(v))
            except Exception:
                setattr(c, k, v)

    # Handle aliases
    if "n_layers" in d:
        c.n_blocks = int(d["n_layers"])

    # Normalize optional I/O paths
    if getattr(c, "log_path", None) in ("", "none", "null", False):
        c.log_path = None
    if getattr(c, "checkpoint_path", None) in ("", "none", "null", False):
        c.checkpoint_path = None

    c.seed = int(seed)
    return c


def _make_file_logger(log_path: str) -> logging.Logger:
    """Create a per-fit file logger (no stdout spam)."""
    p = Path(str(log_path))
    p.parent.mkdir(parents=True, exist_ok=True)

    # Unique logger name per log file (prevents handler duplication).
    logger = logging.getLogger(f"stage3_torch.{p.stem}.{id(p)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear any existing handlers (important in notebooks / repeated runs).
    for h in list(logger.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            logging.getLogger(__name__).debug("Failed to close log handler", exc_info=True)
        logger.removeHandler(h)

    fh = logging.FileHandler(p, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    return logger


def _close_logger(logger: Optional[logging.Logger]) -> None:
    if logger is None:
        return
    for h in list(logger.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            logging.getLogger(__name__).debug("Failed to close log handler", exc_info=True)
        logger.removeHandler(h)


def _save_checkpoint(path: str, *, state_dict: dict, cfg: TorchTrainCfg, meta: dict) -> None:
    """Save a robust Torch checkpoint (state_dict + cfg + meta)."""
    _require_torch()
    p = Path(str(path))
    p.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_state_dict": state_dict,
        "cfg": asdict(cfg),
        "meta": meta,
    }
    torch.save(ckpt, p)


# ---------------------------
# sklearn-like wrappers
# ---------------------------

class TorchResMLPBinary:
    def __init__(self, cfg: dict, *, seed: int = 0) -> None:
        _require_torch()
        self.cfg = _cfg_from_dict(cfg, seed=seed)
        self.device = _resolve_device(self.cfg.device)
        self.net: Optional[ResMLPTabular] = None
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        *,
        group_strata: Optional[np.ndarray] = None,
    ) -> Self:
        _require_torch()
        _set_global_seed(self.cfg.seed)

        X_np = _as_float32(X)
        y_np = _as_int64(y)

        _assert_finite("X", X_np)
        _assert_finite("y", y_np.astype(np.float64, copy=False))

        n = int(len(y_np))
        if n != int(X_np.shape[0]):
            raise ValueError(f"X/y length mismatch: X={X_np.shape[0]} y={n}")

        # split indices
        if groups is not None:
            tr_idx, va_idx = _train_val_split_by_groups_stratified(
                y_np, np.asarray(groups), self.cfg.val_frac, self.cfg.seed, group_strata=group_strata
            )
        else:
            rng = np.random.default_rng(self.cfg.seed)
            idx = np.arange(n, dtype=int)
            rng.shuffle(idx)
            n_val = max(1, int(round(self.cfg.val_frac * n)))
            n_val = min(n_val, n - 1)
            va_idx = idx[:n_val]
            tr_idx = idx[n_val:]

        # tensors once
        X_all = torch.from_numpy(X_np)
        y_all = torch.from_numpy(y_np).to(dtype=torch.float32)  # int64; cast to float in loss

        ds_all = TensorDataset(X_all, y_all)
        ds_tr = Subset(ds_all, tr_idx.tolist())
        ds_va = Subset(ds_all, va_idx.tolist()) if len(va_idx) > 0 else None

        pin = (getattr(self.device, "type", "") == "cuda")


        dl_tr = DataLoader(
            ds_tr,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=pin,
        )
        dl_va = (
            DataLoader(
                ds_va,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=pin,
            )
            if ds_va is not None
            else None
        )

        self.net = ResMLPTabular(
            input_dim=int(X_np.shape[1]),
            d_model=self.cfg.d_model,
            n_blocks=self.cfg.n_blocks,
            mlp_ratio=self.cfg.mlp_ratio,
            dropout=self.cfg.dropout,
            out_dim=1,
            use_grn=self.cfg.use_grn,
            grn_eps=self.cfg.grn_eps,
        ).to(self.device)

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        use_amp = bool(self.cfg.amp) and (getattr(self.device, "type", "") == "cuda")
        scaler = amp.GradScaler(enabled=use_amp)

        log_path = getattr(self.cfg, "log_path", None)
        ckpt_path = getattr(self.cfg, "checkpoint_path", None)
        logger = _make_file_logger(log_path) if log_path else None

        # Per-epoch training curve (JSON-friendly)
        history: dict = {"epoch": [], "train_loss": [], "val_loss": []}

        if logger is not None:
            logger.info(
                "start | task=binary | n=%d | n_train=%d | n_val=%d | input_dim=%d | device=%s | amp=%s",
                int(n),
                int(len(tr_idx)),
                int(len(va_idx)),
                int(X_np.shape[1]),
                str(self.device),
                str(use_amp),
            )
            logger.info("cfg | %s", str(asdict(self.cfg)))
        t_fit0 = time.time()

        best_val = float("inf")
        bad_epochs = 0
        best_state = None
        best_epoch = -1
        stopped_early = False

        for _epoch in range(self.cfg.max_epochs):
            t_epoch0 = time.time()
            self.net.train()

            train_sum = 0.0
            train_n = 0

            for xb, yb in dl_tr:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                with amp.autocast(device_type=self.device.type, enabled=use_amp):
                    logits = self.net(xb).squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(logits, yb)

                # Track train loss (weighted by batch size)
                bs = int(xb.shape[0])
                train_sum += float(loss.detach().item()) * bs
                train_n += bs

                scaler.scale(loss).backward()
                if float(self.cfg.grad_clip_norm) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), float(self.cfg.grad_clip_norm))
                scaler.step(opt)
                scaler.update()

            train_loss = train_sum / max(1, train_n)

            val_loss: Optional[float] = None
            improved = False

            if dl_va is not None:
                self.net.eval()
                val_sum = 0.0
                val_n = 0
                with torch.no_grad():
                    for xb, yb in dl_va:
                        xb = xb.to(self.device, non_blocking=True)
                        yb = yb.to(self.device, non_blocking=True)
                        with amp.autocast(device_type=self.device.type, enabled=use_amp):
                            logits = self.net(xb).squeeze(-1)
                            l = F.binary_cross_entropy_with_logits(logits, yb)
                        bs = int(xb.shape[0])
                        val_sum += float(l.item()) * bs
                        val_n += bs
                val_loss = val_sum / max(1, val_n)

                if val_loss < best_val - 1e-6:
                    best_val = float(val_loss)
                    best_epoch = int(_epoch)
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                    bad_epochs = 0
                    improved = True

                    if ckpt_path:
                        _save_checkpoint(
                            ckpt_path,
                            state_dict=best_state,
                            cfg=self.cfg,
                            meta={
                                "task": "binary",
                                "input_dim": int(X_np.shape[1]),
                                "out_dim": 1,
                                "epoch": int(_epoch),
                                "best_val_loss": float(best_val),
                                "seed": int(self.cfg.seed),
                                "n_train": int(len(tr_idx)),
                                "n_val": int(len(va_idx)),
                                "saved_at": time.time(),
                            },
                        )
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(self.cfg.patience):
                        stopped_early = True
                        if logger is not None:
                            logger.info(
                                "early_stop | epoch=%d | best_epoch=%d | best_val=%.6f | bad_epochs=%d",
                                int(_epoch),
                                int(best_epoch),
                                float(best_val),
                                int(bad_epochs),
                            )
                        break

            history["epoch"].append(int(_epoch))
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss) if val_loss is not None else None)

            if logger is not None:
                logger.info(
                    "epoch %d/%d | train=%.6f | val=%s | best=%s | bad=%d | improved=%s | dt=%.2fs",
                    int(_epoch + 1),
                    int(self.cfg.max_epochs),
                    float(train_loss),
                    f"{val_loss:.6f}" if val_loss is not None else "NA",
                    f"{best_val:.6f}" if best_val < float("inf") else "NA",
                    int(bad_epochs),
                    str(improved),
                    float(time.time() - t_epoch0),
                )

        if best_state is not None:
            self.net.load_state_dict(best_state)
        else:
            # No validation split: optionally still save final weights as a checkpoint.
            if ckpt_path:
                final_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                _save_checkpoint(
                    ckpt_path,
                    state_dict=final_state,
                    cfg=self.cfg,
                    meta={
                        "task": "binary",
                        "input_dim": int(X_np.shape[1]),
                        "out_dim": 1,
                        "epoch": int(self.cfg.max_epochs - 1),
                        "best_val_loss": None,
                        "seed": int(self.cfg.seed),
                        "n_train": int(len(tr_idx)),
                        "n_val": int(len(va_idx)),
                        "saved_at": time.time(),
                        "note": "no_validation_split",
                    },
                )

        self.history_ = history
        self.best_epoch_ = int(best_epoch)
        self.best_val_loss_ = (float(best_val) if best_val < float("inf") else None)
        self.stopped_early_ = bool(stopped_early)

        if logger is not None:
            logger.info(
                "done | best_epoch=%s | best_val=%s | stopped_early=%s | total_s=%.2f",
                str(best_epoch if best_epoch >= 0 else "NA"),
                f"{best_val:.6f}" if best_val < float("inf") else "NA",
                str(stopped_early),
                float(time.time() - t_fit0),
            )

        _close_logger(logger)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _require_torch()
        if (not self._is_fitted) or (self.net is None):
            raise RuntimeError("TorchResMLPBinary not fitted")

        X_np = _as_float32(X)
        _assert_finite("X", X_np)

        X_t = torch.from_numpy(X_np)
        ds = TensorDataset(X_t)
        pin = (getattr(self.device, "type", "") == "cuda")

        dl = DataLoader(ds, batch_size=max(256, self.cfg.batch_size), shuffle=False, num_workers=self.cfg.num_workers, pin_memory=pin)

        self.net.eval()
        probs = []
        use_amp = bool(self.cfg.amp) and (getattr(self.device, "type", "") == "cuda")

        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(self.device, non_blocking=True)
                with amp.autocast(device_type=self.device.type, enabled=use_amp):
                    logits = self.net(xb).squeeze(-1)
                    p1 = torch.sigmoid(logits).detach().cpu()
                probs.append(p1)

        p1_all = torch.cat(probs).float().numpy()
        p0_all = 1.0 - p1_all
        return np.stack([p0_all, p1_all], axis=1)


class TorchResMLPMulticlass:
    def __init__(self, cfg: dict, n_classes: int, *, seed: int = 0) -> None:
        _require_torch()
        self.cfg = _cfg_from_dict(cfg, seed=seed)
        self.device = _resolve_device(self.cfg.device)
        self.n_classes = int(n_classes)
        self.net: Optional[ResMLPTabular] = None
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        *,
        group_strata: Optional[np.ndarray] = None,
    ) -> Self:
        _require_torch()
        _set_global_seed(self.cfg.seed)

        X_np = _as_float32(X)
        y_np = _as_int64(y)

        _assert_finite("X", X_np)
        _assert_finite("y", y_np.astype(np.float64, copy=False))

        n = int(len(y_np))
        if n != int(X_np.shape[0]):
            raise ValueError(f"X/y length mismatch: X={X_np.shape[0]} y={n}")

        # split indices (run-grouped + stratified)
        if groups is not None:
            tr_idx, va_idx = _train_val_split_by_groups_stratified(
                y_np, np.asarray(groups), self.cfg.val_frac, self.cfg.seed, group_strata=group_strata
            )
        else:
            rng = np.random.default_rng(self.cfg.seed)
            idx = np.arange(n, dtype=int)
            rng.shuffle(idx)
            n_val = max(1, int(round(self.cfg.val_frac * n)))
            n_val = min(n_val, n - 1)
            va_idx = idx[:n_val]
            tr_idx = idx[n_val:]

        # tensors once
        X_all = torch.from_numpy(X_np)
        y_all = torch.from_numpy(y_np).long()

        ds_all = TensorDataset(X_all, y_all)
        ds_tr = Subset(ds_all, tr_idx.tolist())
        ds_va = Subset(ds_all, va_idx.tolist()) if len(va_idx) > 0 else None

        pin = (getattr(self.device, "type", "") == "cuda")


        dl_tr = DataLoader(
            ds_tr,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=pin,
        )
        dl_va = (
            DataLoader(
                ds_va,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=pin,
            )
            if ds_va is not None
            else None
        )

        self.net = ResMLPTabular(
            input_dim=int(X_np.shape[1]),
            d_model=self.cfg.d_model,
            n_blocks=self.cfg.n_blocks,
            mlp_ratio=self.cfg.mlp_ratio,
            dropout=self.cfg.dropout,
            out_dim=self.n_classes,
            use_grn=self.cfg.use_grn,
            grn_eps=self.cfg.grn_eps,
        ).to(self.device)

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        use_amp = bool(self.cfg.amp) and (getattr(self.device, "type", "") == "cuda")
        scaler = amp.GradScaler(enabled=use_amp)

        log_path = getattr(self.cfg, "log_path", None)
        ckpt_path = getattr(self.cfg, "checkpoint_path", None)
        logger = _make_file_logger(log_path) if log_path else None

        # Per-epoch training curve (JSON-friendly)
        history: dict = {"epoch": [], "train_loss": [], "val_loss": []}

        if logger is not None:
            logger.info(
                "start | task=multiclass | n=%d | n_train=%d | n_val=%d | input_dim=%d | n_classes=%d | device=%s | amp=%s",
                int(n),
                int(len(tr_idx)),
                int(len(va_idx)),
                int(X_np.shape[1]),
                int(self.n_classes),
                str(self.device),
                str(use_amp),
            )
            logger.info("cfg | %s", str(asdict(self.cfg)))
        t_fit0 = time.time()

        best_val = float("inf")
        bad_epochs = 0
        best_state = None
        best_epoch = -1
        stopped_early = False

        for _epoch in range(self.cfg.max_epochs):
            t_epoch0 = time.time()
            self.net.train()

            train_sum = 0.0
            train_n = 0

            for xb, yb in dl_tr:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)

                with amp.autocast(device_type=self.device.type, enabled=use_amp):
                    logits = self.net(xb)
                    loss = F.cross_entropy(logits, yb, label_smoothing=float(self.cfg.label_smoothing))

                bs = int(xb.shape[0])
                train_sum += float(loss.detach().item()) * bs
                train_n += bs

                scaler.scale(loss).backward()
                if float(self.cfg.grad_clip_norm) > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), float(self.cfg.grad_clip_norm))
                scaler.step(opt)
                scaler.update()

            train_loss = train_sum / max(1, train_n)

            val_loss: Optional[float] = None
            improved = False

            if dl_va is not None:
                self.net.eval()
                val_sum = 0.0
                val_n = 0
                with torch.no_grad():
                    for xb, yb in dl_va:
                        xb = xb.to(self.device, non_blocking=True)
                        yb = yb.to(self.device, non_blocking=True)
                        with amp.autocast(device_type=self.device.type, enabled=use_amp):
                            logits = self.net(xb)
                            l = F.cross_entropy(logits, yb)
                        bs = int(xb.shape[0])
                        val_sum += float(l.item()) * bs
                        val_n += bs
                val_loss = val_sum / max(1, val_n)

                if val_loss < best_val - 1e-6:
                    best_val = float(val_loss)
                    best_epoch = int(_epoch)
                    best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                    bad_epochs = 0
                    improved = True

                    if ckpt_path:
                        _save_checkpoint(
                            ckpt_path,
                            state_dict=best_state,
                            cfg=self.cfg,
                            meta={
                                "task": "multiclass",
                                "input_dim": int(X_np.shape[1]),
                                "out_dim": int(self.n_classes),
                                "epoch": int(_epoch),
                                "best_val_loss": float(best_val),
                                "seed": int(self.cfg.seed),
                                "n_train": int(len(tr_idx)),
                                "n_val": int(len(va_idx)),
                                "saved_at": time.time(),
                            },
                        )
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(self.cfg.patience):
                        stopped_early = True
                        if logger is not None:
                            logger.info(
                                "early_stop | epoch=%d | best_epoch=%d | best_val=%.6f | bad_epochs=%d",
                                int(_epoch),
                                int(best_epoch),
                                float(best_val),
                                int(bad_epochs),
                            )
                        break

            history["epoch"].append(int(_epoch))
            history["train_loss"].append(float(train_loss))
            history["val_loss"].append(float(val_loss) if val_loss is not None else None)

            if logger is not None:
                logger.info(
                    "epoch %d/%d | train=%.6f | val=%s | best=%s | bad=%d | improved=%s | dt=%.2fs",
                    int(_epoch + 1),
                    int(self.cfg.max_epochs),
                    float(train_loss),
                    f"{val_loss:.6f}" if val_loss is not None else "NA",
                    f"{best_val:.6f}" if best_val < float("inf") else "NA",
                    int(bad_epochs),
                    str(improved),
                    float(time.time() - t_epoch0),
                )

        if best_state is not None:
            self.net.load_state_dict(best_state)
        else:
            if ckpt_path:
                final_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                _save_checkpoint(
                    ckpt_path,
                    state_dict=final_state,
                    cfg=self.cfg,
                    meta={
                        "task": "multiclass",
                        "input_dim": int(X_np.shape[1]),
                        "out_dim": int(self.n_classes),
                        "epoch": int(self.cfg.max_epochs - 1),
                        "best_val_loss": None,
                        "seed": int(self.cfg.seed),
                        "n_train": int(len(tr_idx)),
                        "n_val": int(len(va_idx)),
                        "saved_at": time.time(),
                        "note": "no_validation_split",
                    },
                )

        self.history_ = history
        self.best_epoch_ = int(best_epoch)
        self.best_val_loss_ = (float(best_val) if best_val < float("inf") else None)
        self.stopped_early_ = bool(stopped_early)

        if logger is not None:
            logger.info(
                "done | best_epoch=%s | best_val=%s | stopped_early=%s | total_s=%.2f",
                str(best_epoch if best_epoch >= 0 else "NA"),
                f"{best_val:.6f}" if best_val < float("inf") else "NA",
                str(stopped_early),
                float(time.time() - t_fit0),
            )

        _close_logger(logger)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _require_torch()
        if (not self._is_fitted) or (self.net is None):
            raise RuntimeError("TorchResMLPMulticlass not fitted")

        X_np = _as_float32(X)
        _assert_finite("X", X_np)

        X_t = torch.from_numpy(X_np)
        ds = TensorDataset(X_t)
        pin = (getattr(self.device, "type", "") == "cuda")

        dl = DataLoader(ds, batch_size=max(256, self.cfg.batch_size), shuffle=False, num_workers=self.cfg.num_workers, pin_memory=pin)

        self.net.eval()
        probs = []
        use_amp = bool(self.cfg.amp) and (getattr(self.device, "type", "") == "cuda")

        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(self.device, non_blocking=True)
                with amp.autocast(device_type=self.device.type, enabled=use_amp):
                    logits = self.net(xb)
                    p = torch.softmax(logits, dim=1).detach().cpu()
                probs.append(p)

        return torch.cat(probs).float().numpy()
# =============================================================================
# Sequence models (GRU / TCN / Transformer)
# =============================================================================

from collections import defaultdict
from typing import Any as _Any, Dict as _Dict, List as _List  # typing helpers


@dataclass
class SeqTrainCfg:
    # sequence construction
    seq_len: int = 8
    add_pad_indicator: bool = True  # append per-timestep feature: 1=PAD, 0=REAL

    # shared model sizing / regularization
    d_model: int = 64
    dropout: float = 0.2

    # GRU / Transformer
    n_layers: int = 1

    # TCN
    n_blocks: int = 3
    kernel_size: int = 3
    use_weight_norm: bool = True
    norm: str = "group"  # "group" | "layer" | "none"

    # Transformer
    n_heads: int = 4
    ff_mult: float = 2.0

    # training
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 80
    patience: int = 10
    grad_clip_norm: float = 1.0
    val_frac: float = 0.2
    device: str = "cuda"
    amp: bool = True
    num_workers: int = 0
    seed: int = 0

    # multiclass-only: apply to TRAIN loss (not internal val loss)
    label_smoothing: float = 0.0

    # optional logging / checkpoints
    log_path: Optional[str] = None
    checkpoint_path: Optional[str] = None


def _cfg_seq_from_dict(d: dict, *, seed: int) -> SeqTrainCfg:
    c = SeqTrainCfg()
    for k, v in d.items():
        if hasattr(c, k):
            cur = getattr(c, k)
            try:
                if isinstance(cur, bool):
                    setattr(c, k, bool(v))
                elif isinstance(cur, int):
                    setattr(c, k, int(v))
                elif isinstance(cur, float):
                    setattr(c, k, float(v))
                else:
                    setattr(c, k, v)
            except Exception:
                setattr(c, k, v)
    c.seed = int(seed)

    # safety clamps
    c.seq_len = int(max(1, c.seq_len))
    c.dropout = float(min(max(c.dropout, 0.0), 0.95))
    c.label_smoothing = float(min(max(c.label_smoothing, 0.0), 0.3))

    # keep heads valid (transformer); if not divisible, fall back to 1 head
    if c.d_model % max(1, int(c.n_heads)) != 0:
        c.n_heads = 1

    return c


def make_left_padded_sequences(
    X: np.ndarray,
    groups: np.ndarray,
    seq_len: int,
    *,
    add_pad_indicator: bool = True,
) -> np.ndarray:
    """Build left-padded, per-group causal sequences.

    For each window t in a run:
      sequence = [t-(L-1), ..., t] (within-run), left-padded with zeros.

    Output shape: (N, L, F') where F' = F + 1 if add_pad_indicator=True.
    The pad indicator is appended as the last feature:
      1.0 for padded timesteps, 0.0 for real timesteps.
    """
    X_np = _as_float32(X)
    n, f = X_np.shape
    if groups is None:
        raise ValueError("Sequence models require `groups` (run_id array) to build per-run causal sequences.")
    g = np.asarray(groups).astype(str).reshape(-1)
    if len(g) != n:
        raise ValueError(f"groups length mismatch: len(groups)={len(g)} vs N={n}")

    f_out = f + (1 if add_pad_indicator else 0)
    out = np.zeros((n, int(seq_len), f_out), dtype=np.float32)

    if add_pad_indicator:
        # default all padded; we overwrite real timesteps with indicator=0.0
        out[:, :, -1] = 1.0

    group_to_rows: _Dict[str, _List[int]] = defaultdict(list)
    for i, gi in enumerate(g.tolist()):
        group_to_rows[gi].append(i)

    L = int(seq_len)
    for rows in group_to_rows.values():
        # rows are in original order because we appended in scan order
        for j, row_i in enumerate(rows):
            j0 = max(0, j - (L - 1))
            src_rows = rows[j0 : j + 1]
            le = len(src_rows)
            dst0 = L - le
            out[row_i, dst0:L, :f] = X_np[src_rows, :]
            if add_pad_indicator:
                out[row_i, dst0:L, -1] = 0.0

    return out


class GRUSeqClassifier(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_layers: int, dropout: float, out_dim: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(int(input_dim), int(d_model))
        self.gru = nn.GRU(
            input_size=int(d_model),
            hidden_size=int(d_model),
            num_layers=int(n_layers),
            dropout=float(dropout) if int(n_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(float(dropout))
        self.ln = nn.LayerNorm(int(d_model))
        self.head = nn.Linear(int(d_model), int(out_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, F)
        h = self.in_proj(x)
        out, h_n = self.gru(h)  # out: (B,L,D); h_n: (n_layers,B,D)
        last = h_n[-1]  # (B,D)
        last = self.drop(self.ln(last))
        return self.head(last)


class _CausalConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int, *, use_weight_norm: bool) -> None:
        super().__init__()
        self.pad = int((kernel_size - 1) * dilation)
        conv = nn.Conv1d(int(c_in), int(c_out), kernel_size=int(kernel_size), dilation=int(dilation))
        self.conv = _apply_weight_norm(conv) if use_weight_norm else conv

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,C,T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, *, use_weight_norm: bool, norm: str) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(channels, channels, kernel_size, dilation, use_weight_norm=use_weight_norm)
        self.conv2 = _CausalConv1d(channels, channels, kernel_size, dilation, use_weight_norm=use_weight_norm)
        self.drop = nn.Dropout(float(dropout))
        self.norm_kind = str(norm).lower().strip()

        if self.norm_kind == "group":
            self.norm1 = nn.GroupNorm(1, channels)
            self.norm2 = nn.GroupNorm(1, channels)
        elif self.norm_kind == "layer":
            # LayerNorm over channels; we apply after transposing to (B,T,C)
            self.norm1 = nn.LayerNorm(channels)
            self.norm2 = nn.LayerNorm(channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def _apply_norm(self, norm_layer: nn.Module, x: Tensor) -> Tensor:
        if isinstance(norm_layer, nn.LayerNorm):
            return norm_layer(x.transpose(1, 2)).transpose(1, 2)
        return norm_layer(x)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,C,T)
        h = self.conv1(x)
        h = self._apply_norm(self.norm1, h)
        h = F.gelu(h)
        h = self.drop(h)

        h = self.conv2(h)
        h = self._apply_norm(self.norm2, h)
        h = F.gelu(h)
        h = self.drop(h)

        return x + h


class TCNSeqClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: int,
        n_blocks: int,
        kernel_size: int,
        dropout: float,
        out_dim: int,
        *,
        use_weight_norm: bool,
        norm: str,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(int(input_dim), int(channels))
        blocks = []
        for b in range(int(n_blocks)):
            dilation = 2**b
            blocks.append(
                _TCNBlock(
                    channels=int(channels),
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(dropout),
                    use_weight_norm=bool(use_weight_norm),
                    norm=str(norm),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.ln_out = nn.LayerNorm(int(channels))
        self.head = nn.Linear(int(channels), int(out_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,L,F) -> (B,C,T)
        h = self.in_proj(x)  # (B,L,C)
        h = h.transpose(1, 2)  # (B,C,L)
        for blk in self.blocks:
            h = blk(h)
        last = h[:, :, -1]  # (B,C)
        last = self.ln_out(last)
        return self.head(last)


class TransformerSeqClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_mult: float,
        dropout: float,
        out_dim: int,
        seq_len: int,
        has_pad_indicator: bool = True,
    ) -> None:
        super().__init__()
        self.has_pad_indicator = bool(has_pad_indicator)
        self.seq_len = int(seq_len)
        self.in_proj = nn.Linear(int(input_dim), int(d_model))
        self.pos = nn.Parameter(torch.zeros(1, int(seq_len), int(d_model)))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=int(max(32, round(int(d_model) * float(ff_mult)))),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        try:
            # Explicitly disable nested tensor fastpath to avoid noisy warnings
            # when norm_first=True (PyTorch will otherwise warn and disable it anyway).
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers), enable_nested_tensor=False)
        except TypeError:  # older PyTorch
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))
        self.ln_out = nn.LayerNorm(int(d_model))
        self.head = nn.Linear(int(d_model), int(out_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B,L,F)
        # optional padding mask inferred from pad-indicator as last feature dim (1=pad)
        pad_mask = None
        if self.has_pad_indicator:
            # if the last feature is a pad indicator it will be exactly 1/0; this is robust enough
            pad_mask = (x[..., -1] > 0.5)

        h = self.in_proj(x) + self.pos[:, : x.shape[1], :]
        h = self.enc(h, src_key_padding_mask=pad_mask)
        last = h[:, -1, :]
        last = self.ln_out(last)
        return self.head(last)


def _train_classifier(
    net: nn.Module,
    dl_tr: DataLoader,
    dl_va: Optional[DataLoader],
    cfg: SeqTrainCfg,
    *,
    task: str,
    logger: Optional[logging.Logger],
    trial: Optional[_Any] = None,
) -> _Dict[str, _Any]:
    """Train a torch classifier with early stopping. Returns history + best metadata."""
    _require_torch()
    assert task in ("binary", "multiclass")

    net = net.to(_resolve_device(cfg.device))
    device = next(net.parameters()).device
    use_amp = bool(cfg.amp) and (device.type == "cuda")
    scaler = amp.GradScaler(enabled=use_amp)

    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    best_state = None
    history: _List[_Dict[str, float]] = []

    for epoch in range(int(cfg.max_epochs)):
        net.train()
        tr_sum = 0.0
        tr_n = 0
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with amp.autocast(device_type=device.type, enabled=use_amp):
                logits = net(xb)
                if task == "binary":
                    logits = logits.squeeze(-1)
                    loss = F.binary_cross_entropy_with_logits(logits, yb)
                else:
                    loss = F.cross_entropy(logits, yb, label_smoothing=float(cfg.label_smoothing))

            scaler.scale(loss).backward()
            if float(cfg.grad_clip_norm) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(cfg.grad_clip_norm))
            scaler.step(opt)
            scaler.update()

            bs = int(xb.shape[0])
            tr_sum += float(loss.item()) * bs
            tr_n += bs

        tr_loss = tr_sum / max(1, tr_n)

        va_loss = float("nan")
        if dl_va is not None:
            net.eval()
            va_sum = 0.0
            va_n = 0
            with torch.no_grad():
                for xb, yb in dl_va:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    with amp.autocast(device_type=device.type, enabled=use_amp):
                        logits = net(xb)
                        if task == "binary":
                            logits = logits.squeeze(-1)
                            l = F.binary_cross_entropy_with_logits(logits, yb)
                        else:
                            # internal val loss: no label smoothing
                            l = F.cross_entropy(logits, yb)
                    bs = int(xb.shape[0])
                    va_sum += float(l.item()) * bs
                    va_n += bs
            va_loss = va_sum / max(1, va_n)

        history.append({"epoch": float(epoch), "train_loss": float(tr_loss), "val_loss": float(va_loss)})

        if logger is not None:
            if dl_va is None:
                logger.info("epoch=%d train_loss=%.6f", epoch + 1, tr_loss)
            else:
                logger.info("epoch=%d train_loss=%.6f val_loss=%.6f", epoch + 1, tr_loss, va_loss)

        # Optuna pruning hook (if provided)
        if trial is not None and dl_va is not None:
            try:
                trial.report(float(va_loss), step=int(epoch))
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
            except AttributeError:
                pass

        if dl_va is None:
            continue

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

            if cfg.checkpoint_path:
                # Save in the shared robust format: {state_dict, cfg, meta}
                # Store epoch/best_val in meta for later inspection.
                _save_checkpoint(
                    str(cfg.checkpoint_path),
                    state_dict=best_state,
                    cfg=cfg,
                    meta={
                        "task": str(task),
                        "epoch": int(epoch + 1),
                        "best_val": float(best_val),
                        "saved_at": float(time.time()),
                    },
                )

        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg.patience):
                break

    if best_state is not None:
        net.load_state_dict(best_state)

    return {
        "history": history,
        "best_val": float(best_val),
        "best_epoch": int(best_epoch) + 1,  # 1-based for humans
        "stopped_early": bool(dl_va is not None and bad_epochs >= int(cfg.patience)),
    }


class _TorchSeqBase:
    """Common bits shared by seq wrappers (not exposed)."""

    def __init__(self, cfg: dict, *, seed: int = 0) -> None:
        _require_torch()
        self.cfg = _cfg_seq_from_dict(cfg, seed=seed)
        self.device = _resolve_device(self.cfg.device)
        self.net: Optional[nn.Module] = None
        self._is_fitted = False
        self.history_: Optional[_List[_Dict[str, float]]] = None

    def _make_logger(self) -> Optional[logging.Logger]:
        if not self.cfg.log_path:
            return None
        return _make_file_logger(self.cfg.log_path)

    def _build_seq(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        return make_left_padded_sequences(
            X,
            groups,
            seq_len=self.cfg.seq_len,
            add_pad_indicator=bool(self.cfg.add_pad_indicator),
        )


class TorchGRUBinary(_TorchSeqBase):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: Optional[np.ndarray] = None,
        group_strata: Optional[np.ndarray] = None,
        trial: Optional[_Any] = None,
    ) -> Self:
        _require_torch()
        _set_global_seed(self.cfg.seed)

        X_np = np.asarray(X)
        y_np = _as_int64(y).reshape(-1)

        if groups is None:
            raise ValueError("TorchGRUBinary.fit requires groups (run_id array).")
        groups_np = np.asarray(groups)

        # internal early-stop split (run-disjoint)
        tr_idx, va_idx = _train_val_split_by_groups_stratified(
            y_np,
            groups_np,
            float(self.cfg.val_frac),
            int(self.cfg.seed),
            group_strata=group_strata,
        )
        if len(va_idx) == 0:
            va_idx = None

        X_seq = self._build_seq(X_np, groups_np)
        X_all = torch.from_numpy(X_seq)
        y_all = torch.from_numpy(y_np).to(dtype=torch.float32)

        X_tr = X_all[tr_idx]
        y_tr = y_all[tr_idx]
        ds_tr = TensorDataset(X_tr, y_tr)

        dl_tr = DataLoader(
            ds_tr,
            batch_size=int(self.cfg.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.num_workers),
            pin_memory=(self.device.type == "cuda"),
        )

        dl_va = None
        if va_idx is not None:
            X_va = X_all[va_idx]
            y_va = y_all[va_idx]
            ds_va = TensorDataset(X_va, y_va)
            dl_va = DataLoader(
                ds_va,
                batch_size=int(self.cfg.batch_size),
                shuffle=False,
                num_workers=int(self.cfg.num_workers),
                pin_memory=(self.device.type == "cuda"),
            )

        self.net = GRUSeqClassifier(
            input_dim=X_all.shape[-1],
            d_model=self.cfg.d_model,
            n_layers=self.cfg.n_layers,
            dropout=self.cfg.dropout,
            out_dim=1,
        ).to(self.device)

        logger = self._make_logger()
        res = _train_classifier(self.net, dl_tr, dl_va, self.cfg, task="binary", logger=logger, trial=trial)
        self.history_ = res["history"]

        if logger is not None:
            _close_logger(logger)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray, *, groups: Optional[np.ndarray] = None) -> np.ndarray:
        _require_torch()
        if not self._is_fitted or self.net is None:
            raise RuntimeError("Model not fitted.")

        if groups is None:
            raise ValueError("TorchGRUBinary.predict_proba requires groups (run_id array).")

        X_seq = self._build_seq(np.asarray(X), np.asarray(groups))
        X_t = torch.from_numpy(X_seq)
        ds = TensorDataset(X_t)
        dl = DataLoader(ds, batch_size=max(256, int(self.cfg.batch_size)), shuffle=False, num_workers=int(self.cfg.num_workers))

        self.net.eval()
        device = next(self.net.parameters()).device
        use_amp = bool(self.cfg.amp) and (device.type == "cuda")

        probs = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(device, non_blocking=True)
                with amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = self.net(xb).squeeze(-1)
                    probs.append(torch.sigmoid(logits).cpu())
        p1 = torch.cat(probs).float().numpy()
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


class TorchGRUMulticlass(_TorchSeqBase):
    def __init__(self, cfg: dict, n_classes: int, *, seed: int = 0) -> None:
        super().__init__(cfg, seed=seed)
        self.n_classes = int(n_classes)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: Optional[np.ndarray] = None,
        group_strata: Optional[np.ndarray] = None,
        trial: Optional[_Any] = None,
    ) -> Self:
        _require_torch()
        _set_global_seed(self.cfg.seed)

        X_np = np.asarray(X)
        y_np = _as_int64(y).reshape(-1)

        if groups is None:
            raise ValueError("TorchGRUMulticlass.fit requires groups (run_id array).")
        groups_np = np.asarray(groups)

        tr_idx, va_idx = _train_val_split_by_groups_stratified(
            y_np,
            groups_np,
            float(self.cfg.val_frac),
            int(self.cfg.seed),
            group_strata=group_strata,
        )
        if len(va_idx) == 0:
            va_idx = None

        X_seq = self._build_seq(X_np, groups_np)
        X_all = torch.from_numpy(X_seq)
        y_all = torch.from_numpy(y_np).to(dtype=torch.long)

        X_tr = X_all[tr_idx]
        y_tr = y_all[tr_idx]
        dl_tr = DataLoader(
            TensorDataset(X_tr, y_tr),
            batch_size=int(self.cfg.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.num_workers),
            pin_memory=(self.device.type == "cuda"),
        )

        dl_va = None
        if va_idx is not None:
            X_va = X_all[va_idx]
            y_va = y_all[va_idx]
            dl_va = DataLoader(
                TensorDataset(X_va, y_va),
                batch_size=int(self.cfg.batch_size),
                shuffle=False,
                num_workers=int(self.cfg.num_workers),
                pin_memory=(self.device.type == "cuda"),
            )

        self.net = GRUSeqClassifier(
            input_dim=X_all.shape[-1],
            d_model=self.cfg.d_model,
            n_layers=self.cfg.n_layers,
            dropout=self.cfg.dropout,
            out_dim=self.n_classes,
        ).to(self.device)

        logger = self._make_logger()
        res = _train_classifier(self.net, dl_tr, dl_va, self.cfg, task="multiclass", logger=logger, trial=trial)
        self.history_ = res["history"]
        if logger is not None:
            _close_logger(logger)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray, *, groups: Optional[np.ndarray] = None) -> np.ndarray:
        _require_torch()
        if not self._is_fitted or self.net is None:
            raise RuntimeError("Model not fitted.")
        if groups is None:
            raise ValueError("TorchGRUMulticlass.predict_proba requires groups (run_id array).")

        X_seq = self._build_seq(np.asarray(X), np.asarray(groups))
        X_t = torch.from_numpy(X_seq)
        dl = DataLoader(TensorDataset(X_t), batch_size=max(256, int(self.cfg.batch_size)), shuffle=False, num_workers=int(self.cfg.num_workers))

        self.net.eval()
        device = next(self.net.parameters()).device
        use_amp = bool(self.cfg.amp) and (device.type == "cuda")

        probs = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(device, non_blocking=True)
                with amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = self.net(xb)
                    probs.append(torch.softmax(logits, dim=1).cpu())
        return torch.cat(probs).float().numpy()


class TorchTCNBinary(_TorchSeqBase):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: Optional[np.ndarray] = None,
        group_strata: Optional[np.ndarray] = None,
        trial: Optional[_Any] = None,
    ) -> Self:
        _require_torch()
        _set_global_seed(self.cfg.seed)

        X_np = np.asarray(X)
        y_np = _as_int64(y).reshape(-1)

        if groups is None:
            raise ValueError("TorchTCNBinary.fit requires groups (run_id array).")
        groups_np = np.asarray(groups)

        tr_idx, va_idx = _train_val_split_by_groups_stratified(
            y_np,
            groups_np,
            float(self.cfg.val_frac),
            int(self.cfg.seed),
            group_strata=group_strata,
        )
        if len(va_idx) == 0:
            va_idx = None

        X_seq = self._build_seq(X_np, groups_np)
        X_all = torch.from_numpy(X_seq)
        y_all = torch.from_numpy(y_np).to(dtype=torch.float32)

        dl_tr = DataLoader(
            TensorDataset(X_all[tr_idx], y_all[tr_idx]),
            batch_size=int(self.cfg.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.num_workers),
            pin_memory=(self.device.type == "cuda"),
        )
        dl_va = None
        if va_idx is not None:
            dl_va = DataLoader(
                TensorDataset(X_all[va_idx], y_all[va_idx]),
                batch_size=int(self.cfg.batch_size),
                shuffle=False,
                num_workers=int(self.cfg.num_workers),
                pin_memory=(self.device.type == "cuda"),
            )

        self.net = TCNSeqClassifier(
            input_dim=X_all.shape[-1],
            channels=self.cfg.d_model,
            n_blocks=self.cfg.n_blocks,
            kernel_size=self.cfg.kernel_size,
            dropout=self.cfg.dropout,
            out_dim=1,
            use_weight_norm=self.cfg.use_weight_norm,
            norm=self.cfg.norm,
        ).to(self.device)

        logger = self._make_logger()
        res = _train_classifier(self.net, dl_tr, dl_va, self.cfg, task="binary", logger=logger, trial=trial)
        self.history_ = res["history"]
        if logger is not None:
            _close_logger(logger)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray, *, groups: Optional[np.ndarray] = None) -> np.ndarray:
        _require_torch()
        if not self._is_fitted or self.net is None:
            raise RuntimeError("Model not fitted.")
        if groups is None:
            raise ValueError("TorchTCNBinary.predict_proba requires groups (run_id array).")

        X_seq = self._build_seq(np.asarray(X), np.asarray(groups))
        X_t = torch.from_numpy(X_seq)
        dl = DataLoader(TensorDataset(X_t), batch_size=max(256, int(self.cfg.batch_size)), shuffle=False, num_workers=int(self.cfg.num_workers))

        self.net.eval()
        device = next(self.net.parameters()).device
        use_amp = bool(self.cfg.amp) and (device.type == "cuda")

        probs = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(device, non_blocking=True)
                with amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = self.net(xb).squeeze(-1)
                    probs.append(torch.sigmoid(logits).cpu())
        p1 = torch.cat(probs).float().numpy()
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


class TorchTCNMulticlass(_TorchSeqBase):
    def __init__(self, cfg: dict, n_classes: int, *, seed: int = 0) -> None:
        super().__init__(cfg, seed=seed)
        self.n_classes = int(n_classes)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: Optional[np.ndarray] = None,
        group_strata: Optional[np.ndarray] = None,
        trial: Optional[_Any] = None,
    ) -> Self:
        _require_torch()
        _set_global_seed(self.cfg.seed)

        X_np = np.asarray(X)
        y_np = _as_int64(y).reshape(-1)

        if groups is None:
            raise ValueError("TorchTCNMulticlass.fit requires groups (run_id array).")
        groups_np = np.asarray(groups)

        tr_idx, va_idx = _train_val_split_by_groups_stratified(
            y_np,
            groups_np,
            float(self.cfg.val_frac),
            int(self.cfg.seed),
            group_strata=group_strata,
        )
        if len(va_idx) == 0:
            va_idx = None

        X_seq = self._build_seq(X_np, groups_np)
        X_all = torch.from_numpy(X_seq)
        y_all = torch.from_numpy(y_np).to(dtype=torch.long)

        dl_tr = DataLoader(
            TensorDataset(X_all[tr_idx], y_all[tr_idx]),
            batch_size=int(self.cfg.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.num_workers),
            pin_memory=(self.device.type == "cuda"),
        )
        dl_va = None
        if va_idx is not None:
            dl_va = DataLoader(
                TensorDataset(X_all[va_idx], y_all[va_idx]),
                batch_size=int(self.cfg.batch_size),
                shuffle=False,
                num_workers=int(self.cfg.num_workers),
                pin_memory=(self.device.type == "cuda"),
            )

        self.net = TCNSeqClassifier(
            input_dim=X_all.shape[-1],
            channels=self.cfg.d_model,
            n_blocks=self.cfg.n_blocks,
            kernel_size=self.cfg.kernel_size,
            dropout=self.cfg.dropout,
            out_dim=self.n_classes,
            use_weight_norm=self.cfg.use_weight_norm,
            norm=self.cfg.norm,
        ).to(self.device)

        logger = self._make_logger()
        res = _train_classifier(self.net, dl_tr, dl_va, self.cfg, task="multiclass", logger=logger, trial=trial)
        self.history_ = res["history"]
        if logger is not None:
            _close_logger(logger)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray, *, groups: Optional[np.ndarray] = None) -> np.ndarray:
        _require_torch()
        if not self._is_fitted or self.net is None:
            raise RuntimeError("Model not fitted.")
        if groups is None:
            raise ValueError("TorchTCNMulticlass.predict_proba requires groups (run_id array).")

        X_seq = self._build_seq(np.asarray(X), np.asarray(groups))
        X_t = torch.from_numpy(X_seq)
        dl = DataLoader(TensorDataset(X_t), batch_size=max(256, int(self.cfg.batch_size)), shuffle=False, num_workers=int(self.cfg.num_workers))

        self.net.eval()
        device = next(self.net.parameters()).device
        use_amp = bool(self.cfg.amp) and (device.type == "cuda")

        probs = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(device, non_blocking=True)
                with amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = self.net(xb)
                    probs.append(torch.softmax(logits, dim=1).cpu())
        return torch.cat(probs).float().numpy()


class TorchTransformerBinary(_TorchSeqBase):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: Optional[np.ndarray] = None,
        group_strata: Optional[np.ndarray] = None,
        trial: Optional[_Any] = None,
    ) -> Self:
        _require_torch()
        _set_global_seed(self.cfg.seed)
        if not bool(self.cfg.add_pad_indicator):
            raise ValueError('Transformer sequence models require add_pad_indicator=True (for causal padding mask).')

        X_np = np.asarray(X)
        y_np = _as_int64(y).reshape(-1)

        if groups is None:
            raise ValueError("TorchTransformerBinary.fit requires groups (run_id array).")
        groups_np = np.asarray(groups)

        tr_idx, va_idx = _train_val_split_by_groups_stratified(
            y_np,
            groups_np,
            float(self.cfg.val_frac),
            int(self.cfg.seed),
            group_strata=group_strata,
        )
        if len(va_idx) == 0:
            va_idx = None

        X_seq = self._build_seq(X_np, groups_np)
        X_all = torch.from_numpy(X_seq)
        y_all = torch.from_numpy(y_np).to(dtype=torch.float32)

        dl_tr = DataLoader(
            TensorDataset(X_all[tr_idx], y_all[tr_idx]),
            batch_size=int(self.cfg.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.num_workers),
            pin_memory=(self.device.type == "cuda"),
        )
        dl_va = None
        if va_idx is not None:
            dl_va = DataLoader(
                TensorDataset(X_all[va_idx], y_all[va_idx]),
                batch_size=int(self.cfg.batch_size),
                shuffle=False,
                num_workers=int(self.cfg.num_workers),
                pin_memory=(self.device.type == "cuda"),
            )

        self.net = TransformerSeqClassifier(
            input_dim=X_all.shape[-1],
            d_model=self.cfg.d_model,
            n_layers=self.cfg.n_layers,
            n_heads=self.cfg.n_heads,
            ff_mult=self.cfg.ff_mult,
            dropout=self.cfg.dropout,
            out_dim=1,
            seq_len=self.cfg.seq_len,
            has_pad_indicator=bool(self.cfg.add_pad_indicator),
        ).to(self.device)

        logger = self._make_logger()
        res = _train_classifier(self.net, dl_tr, dl_va, self.cfg, task="binary", logger=logger, trial=trial)
        self.history_ = res["history"]
        if logger is not None:
            _close_logger(logger)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray, *, groups: Optional[np.ndarray] = None) -> np.ndarray:
        _require_torch()
        if not self._is_fitted or self.net is None:
            raise RuntimeError("Model not fitted.")
        if groups is None:
            raise ValueError("TorchTransformerBinary.predict_proba requires groups (run_id array).")

        X_seq = self._build_seq(np.asarray(X), np.asarray(groups))
        X_t = torch.from_numpy(X_seq)
        dl = DataLoader(TensorDataset(X_t), batch_size=max(256, int(self.cfg.batch_size)), shuffle=False, num_workers=int(self.cfg.num_workers))

        self.net.eval()
        device = next(self.net.parameters()).device
        use_amp = bool(self.cfg.amp) and (device.type == "cuda")

        probs = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(device, non_blocking=True)
                with amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = self.net(xb).squeeze(-1)
                    probs.append(torch.sigmoid(logits).cpu())
        p1 = torch.cat(probs).float().numpy()
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)


class TorchTransformerMulticlass(_TorchSeqBase):
    def __init__(self, cfg: dict, n_classes: int, *, seed: int = 0) -> None:
        super().__init__(cfg, seed=seed)
        self.n_classes = int(n_classes)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: Optional[np.ndarray] = None,
        group_strata: Optional[np.ndarray] = None,
        trial: Optional[_Any] = None,
    ) -> Self:
        _require_torch()
        _set_global_seed(self.cfg.seed)
        if not bool(self.cfg.add_pad_indicator):
            raise ValueError('Transformer sequence models require add_pad_indicator=True (for causal padding mask).')

        X_np = np.asarray(X)
        y_np = _as_int64(y).reshape(-1)

        if groups is None:
            raise ValueError("TorchTransformerMulticlass.fit requires groups (run_id array).")
        groups_np = np.asarray(groups)

        tr_idx, va_idx = _train_val_split_by_groups_stratified(
            y_np,
            groups_np,
            float(self.cfg.val_frac),
            int(self.cfg.seed),
            group_strata=group_strata,
        )
        if len(va_idx) == 0:
            va_idx = None

        X_seq = self._build_seq(X_np, groups_np)
        X_all = torch.from_numpy(X_seq)
        y_all = torch.from_numpy(y_np).to(dtype=torch.long)

        dl_tr = DataLoader(
            TensorDataset(X_all[tr_idx], y_all[tr_idx]),
            batch_size=int(self.cfg.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.num_workers),
            pin_memory=(self.device.type == "cuda"),
        )
        dl_va = None
        if va_idx is not None:
            dl_va = DataLoader(
                TensorDataset(X_all[va_idx], y_all[va_idx]),
                batch_size=int(self.cfg.batch_size),
                shuffle=False,
                num_workers=int(self.cfg.num_workers),
                pin_memory=(self.device.type == "cuda"),
            )

        self.net = TransformerSeqClassifier(
            input_dim=X_all.shape[-1],
            d_model=self.cfg.d_model,
            n_layers=self.cfg.n_layers,
            n_heads=self.cfg.n_heads,
            ff_mult=self.cfg.ff_mult,
            dropout=self.cfg.dropout,
            out_dim=self.n_classes,
            seq_len=self.cfg.seq_len,
            has_pad_indicator=bool(self.cfg.add_pad_indicator),
        ).to(self.device)

        logger = self._make_logger()
        res = _train_classifier(self.net, dl_tr, dl_va, self.cfg, task="multiclass", logger=logger, trial=trial)
        self.history_ = res["history"]
        if logger is not None:
            _close_logger(logger)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray, *, groups: Optional[np.ndarray] = None) -> np.ndarray:
        _require_torch()
        if not self._is_fitted or self.net is None:
            raise RuntimeError("Model not fitted.")
        if groups is None:
            raise ValueError("TorchTransformerMulticlass.predict_proba requires groups (run_id array).")

        X_seq = self._build_seq(np.asarray(X), np.asarray(groups))
        X_t = torch.from_numpy(X_seq)
        dl = DataLoader(TensorDataset(X_t), batch_size=max(256, int(self.cfg.batch_size)), shuffle=False, num_workers=int(self.cfg.num_workers))

        self.net.eval()
        device = next(self.net.parameters()).device
        use_amp = bool(self.cfg.amp) and (device.type == "cuda")

        probs = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(device, non_blocking=True)
                with amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = self.net(xb)
                    probs.append(torch.softmax(logits, dim=1).cpu())
        return torch.cat(probs).float().numpy()