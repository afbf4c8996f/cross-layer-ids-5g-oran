#!/usr/bin/env python3
"""
preprocess_windows.py

Outputs (under out_dir):
  processed/{split_name}/
    - network_W{W}_S{S}_{train|val|test}.parquet
    - radio_W{W}_S{S}_{train|val|test}.parquet
    - preprocess_{stem}.joblib
    - features_{stem}.json
    - summary_{stem}.json (includes log_columns + log_policy)
  processed/processed_index.csv
  processed/stage2_preprocess_config.json

Run example:
  python3 preprocess_windows_v6.py \
    --stage1-out "/path/to/window-output" \
    --out-dir "/path/to/preprocessing-output" \
    --splits "path/to/preprocessing-output/splits" \
    --windows 10:2,5:2 \
    --drop-empty-windows
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

try:
    from joblib import dump  # type: ignore
except Exception as e:
    raise RuntimeError("joblib is required. Install with: python3 -m pip install joblib") from e

try:
    from sklearn.compose import ColumnTransformer  # type: ignore
    from sklearn.impute import SimpleImputer  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer  # type: ignore
except Exception as e:
    raise RuntimeError("scikit-learn is required. Install with: python3 -m pip install scikit-learn") from e


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required for --config. Install with: python3 -m pip install pyyaml")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping/dict at top level.")
    return data


def _deep_get(d: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_windows_spec(spec: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    spec = spec.strip()
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Bad window spec '{part}'. Expected W:S.")
        w_s = part.split(":")
        if len(w_s) != 2:
            raise ValueError(f"Bad window spec '{part}'. Expected W:S.")
        W = int(w_s[0]); S = int(w_s[1])
        if W <= 0 or S <= 0:
            raise ValueError("Window W and stride S must be positive integers.")
        out.append((W, S))
    return out


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---- Column definitions ----
META_COLS_ORDERED = ["run_id", "family", "window_start_s", "window_end_s"]
LABEL_COLS = ["traffic_type_win", "attack_category_win", "attack_type_win"]

LABEL_DERIVED_LEAK = {
    "attack_flow_count",
    "benign_flow_count",
    "attack_flow_frac",
    "window_has_attack_flow",
}

# Always excluded from features
EXCLUDE_FROM_FEATURES = set(META_COLS_ORDERED) | set(LABEL_COLS) | LABEL_DERIVED_LEAK | {"y_bin", "y_cat"}


def _make_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tt = out["traffic_type_win"].astype(str).str.strip().str.lower()
    out["y_bin"] = tt.str.startswith("attack").astype(np.int8)

    ac = out["attack_category_win"].astype("string")
    out["y_cat"] = ac.fillna("Benign").astype("string")
    return out


def _drop_empty_windows(df: pd.DataFrame) -> pd.DataFrame:
    if "n_flows" not in df.columns:
        return df.reset_index(drop=True)
    return df.loc[df["n_flows"].astype(int) > 0].reset_index(drop=True)


def _make_onehot_encoder_dense() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # type: ignore
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)  # type: ignore


def _to_dense(X: Any) -> np.ndarray:
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _make_column_transformer(transformers: List[Tuple[str, Any, List[str]]]) -> ColumnTransformer:
    try:
        return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    except TypeError:
        return ColumnTransformer(transformers=transformers, remainder="drop")


def _ohe_feature_names(ohe: OneHotEncoder, cat_cols: List[str]) -> List[str]:
    try:
        return [str(x) for x in ohe.get_feature_names_out(cat_cols)]
    except AttributeError:
        return [str(x) for x in ohe.get_feature_names(cat_cols)]  # type: ignore


def _select_network_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = df.copy()
    required = {"run_id","window_start_s","window_end_s","traffic_type_win","attack_category_win"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Network windows missing required columns: {sorted(missing)}")

    cat_cols = [c for c in df.columns if c.endswith("_mode")]
    X_cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES and c != "family"]

    cat_cols = [c for c in cat_cols if c in X_cols]
    num_cols = [c for c in X_cols if c not in cat_cols]

    num_cols2: List[str] = []
    for c in num_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols2.append(c)
        else:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() > 0.9:
                df[c] = coerced
                num_cols2.append(c)

    return df, num_cols2, cat_cols


def _select_radio_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    required = {"run_id","window_start_s","window_end_s","traffic_type_win","attack_category_win"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Radio windows missing required columns: {sorted(missing)}")

    drop = set(EXCLUDE_FROM_FEATURES) | {"n_flows", "attack_flow_frac"}
    X_cols = [c for c in df.columns if c not in drop and c != "family"]

    num_cols: List[str] = []
    for c in X_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() > 0.9:
                df[c] = coerced
                num_cols.append(c)

    must = {"radio_window_missing", "radio_missing_frac"}
    if not must.issubset(set(df.columns)):
        raise ValueError("Radio windows must contain radio_window_missing and radio_missing_frac.")

    return df, num_cols


def _assert_unique(names: List[str], context: str) -> None:
    seen = set()
    dups = []
    for n in names:
        if n in seen:
            dups.append(n)
        seen.add(n)
    if dups:
        raise RuntimeError(f"{context}: duplicate feature names detected: {sorted(set(dups))[:20]}")


def _choose_log_cols(train_df: pd.DataFrame, num_cols: List[str]) -> Tuple[List[str], List[str]]:
    plain: List[str] = []
    logc: List[str] = []
    for c in num_cols:
        s = pd.to_numeric(train_df[c], errors="coerce")
        if s.notna().sum() == 0:
            plain.append(c)
            continue
        mn = float(s.min())
        mx = float(s.max())
        if mn >= 0.0 and mx > 10.0:
            logc.append(c)
        else:
            plain.append(c)
    return plain, logc


def _numeric_pipe_plain() -> Pipeline:
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])


def _numeric_pipe_log_safe() -> Pipeline:
    # Clip to nonnegative to guarantee log1p validity in val/test too.
    clip_tf = FunctionTransformer(np.clip, kw_args={"a_min": 0.0, "a_max": None}, validate=False)
    log_tf = FunctionTransformer(np.log1p, validate=False)
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clip0", clip_tf),
        ("log1p", log_tf),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])


def _fit_transform_modality(
    df: pd.DataFrame,
    split: Dict[str, Any],
    modality: str,
    W: int,
    S: int,
    stage2_out_dir: Path,
    drop_empty: bool,
) -> Dict[str, Any]:
    split_name = str(split["split_name"])
    out_split = stage2_out_dir / "processed" / split_name
    _ensure_dir(out_split)

    train_ids = set(map(str, split["train_run_ids"]))
    val_ids = set(map(str, split["val_run_ids"]))
    test_ids = set(map(str, split["test_run_ids"]))

    df = df.copy()
    df["run_id"] = df["run_id"].astype(str)

    train_df = df[df["run_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["run_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["run_id"].isin(test_ids)].reset_index(drop=True)

    if drop_empty:
        train_df = _drop_empty_windows(train_df)
        val_df = _drop_empty_windows(val_df)
        test_df = _drop_empty_windows(test_df)

    train_df = _make_labels(train_df)
    val_df = _make_labels(val_df)
    test_df = _make_labels(test_df)

    log_cols: List[str] = []
    plain_cols: List[str] = []
    cat_cols: List[str] = []

    if modality == "network":
        train_df, num_cols, cat_cols = _select_network_features(train_df)
        val_df, _, _ = _select_network_features(val_df)
        test_df, _, _ = _select_network_features(test_df)

        plain_cols, log_cols = _choose_log_cols(train_df, num_cols)

        transformers: List[Tuple[str, Any, List[str]]] = []
        if plain_cols:
            transformers.append(("num_plain", _numeric_pipe_plain(), plain_cols))
        if log_cols:
            transformers.append(("num_log", _numeric_pipe_log_safe(), log_cols))

        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_onehot_encoder_dense()),
        ])
        if cat_cols:
            transformers.append(("cat", cat_pipe, cat_cols))

        pre = _make_column_transformer(transformers)

        X_train = train_df[plain_cols + log_cols + cat_cols]
        X_val = val_df[plain_cols + log_cols + cat_cols]
        X_test = test_df[plain_cols + log_cols + cat_cols]

    elif modality == "radio":
        train_df, num_cols = _select_radio_features(train_df)
        val_df, _ = _select_radio_features(val_df)
        test_df, _ = _select_radio_features(test_df)

        plain_cols, log_cols = _choose_log_cols(train_df, num_cols)

        transformers = []
        if plain_cols:
            transformers.append(("num_plain", _numeric_pipe_plain(), plain_cols))
        if log_cols:
            transformers.append(("num_log", _numeric_pipe_log_safe(), log_cols))

        pre = _make_column_transformer(transformers)

        X_train = train_df[plain_cols + log_cols]
        X_val = val_df[plain_cols + log_cols]
        X_test = test_df[plain_cols + log_cols]

        cat_cols = []

    else:
        raise ValueError("modality must be 'network' or 'radio'")

    pre.fit(X_train)

    Xt_train = _to_dense(pre.transform(X_train))
    Xt_val = _to_dense(pre.transform(X_val))
    Xt_test = _to_dense(pre.transform(X_test))

    # Feature names (manual)
    feat_names: List[str] = []
    feat_names.extend([str(c) for c in plain_cols])
    feat_names.extend([f"{c}__log1p" for c in log_cols])

    if modality == "network" and cat_cols:
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]  # type: ignore
        feat_names.extend(_ohe_feature_names(ohe, cat_cols))

    _assert_unique(feat_names, f"{modality} W{W} S{S}")

    if Xt_train.shape[1] != len(feat_names):
        raise RuntimeError(
            f"Feature name count mismatch for {modality} W{W} S{S}: "
            f"Xt_train has {Xt_train.shape[1]} cols but feat_names has {len(feat_names)}"
        )

    def pack(orig: pd.DataFrame, Xt: np.ndarray) -> pd.DataFrame:
        meta = orig[META_COLS_ORDERED].copy()
        labels = orig[["y_bin", "y_cat"]].copy()
        feat = pd.DataFrame(Xt, columns=feat_names)
        out = pd.concat([meta.reset_index(drop=True), labels.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)
        if out.columns.duplicated().any():
            dups = out.columns[out.columns.duplicated()].tolist()
            raise RuntimeError(f"{modality} W{W} S{S}: duplicated output columns: {sorted(set(dups))[:20]}")
        return out

    out_train = pack(train_df, Xt_train)
    out_val = pack(val_df, Xt_val)
    out_test = pack(test_df, Xt_test)

    stem = f"{modality}_W{W}_S{S}"
    out_train_path = out_split / f"{stem}_train.parquet"
    out_val_path = out_split / f"{stem}_val.parquet"
    out_test_path = out_split / f"{stem}_test.parquet"

    out_train.to_parquet(out_train_path, index=False)
    out_val.to_parquet(out_val_path, index=False)
    out_test.to_parquet(out_test_path, index=False)

    art_path = out_split / f"preprocess_{stem}.joblib"
    dump(pre, art_path)

    names_path = out_split / f"features_{stem}.json"
    names_path.write_text(json.dumps({"feature_names": feat_names}, indent=2), encoding="utf-8")

    summ = {
        "created_utc": _utc_now_iso(),
        "split_name": split_name,
        "modality": modality,
        "W": W, "S": S,
        "drop_empty_windows": bool(drop_empty),
        "n_train": int(out_train.shape[0]),
        "n_val": int(out_val.shape[0]),
        "n_test": int(out_test.shape[0]),
        "y_bin_rate_train": float(out_train["y_bin"].mean()) if out_train.shape[0] else float("nan"),
        "y_bin_rate_val": float(out_val["y_bin"].mean()) if out_val.shape[0] else float("nan"),
        "y_bin_rate_test": float(out_test["y_bin"].mean()) if out_test.shape[0] else float("nan"),
        "n_features": int(len(feat_names)),
        "log_columns": log_cols,
        "log_policy": "clip_to_[0,+inf] then log1p (to avoid invalid log1p in val/test)",
        "artifacts": {
            "preprocess_joblib": str(art_path),
            "feature_names_json": str(names_path),
        },
        "outputs": {
            "train": str(out_train_path),
            "val": str(out_val_path),
            "test": str(out_test_path),
        },
    }
    summ_path = out_split / f"summary_{stem}.json"
    summ_path.write_text(json.dumps(summ, indent=2), encoding="utf-8")

    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML config (optional)")
    ap.add_argument("--stage1-out", type=str, default=None, help="Stage-1 output dir containing windows/ (overrides config)")
    ap.add_argument("--out-dir", type=str, default=None, help="Stage-2 output dir containing splits/ and where processed/ will be written (overrides config)")
    ap.add_argument("--splits", type=str, default=None, help="Comma-separated split JSON files (paths) or a directory; default: out_dir/splits")
    ap.add_argument("--windows", type=str, default=None, help="Window specs '10:2,5:2' (overrides config)")
    ap.add_argument("--modalities", type=str, default=None, help="Comma-separated: network,radio (default both)")
    ap.add_argument("--drop-empty-windows", action="store_true", help="Drop windows with n_flows==0 (recommended)")
    ap.add_argument("--keep-empty-windows", action="store_true", help="Keep empty windows")
    args = ap.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = _read_yaml(Path(args.config).expanduser())

    stage1_out = Path(args.stage1_out or _deep_get(cfg, ["paths", "stage1_out_dir"], _deep_get(cfg, ["paths", "out_dir"], "./window-output"))).expanduser()
    out_dir = Path(args.out_dir or _deep_get(cfg, ["paths", "out_dir"], str(stage1_out))).expanduser()

    windows_arg = args.windows or _deep_get(cfg, ["windows_spec"], None)
    if windows_arg is None:
        win_list = _deep_get(cfg, ["windows"], [[10, 2], [5, 2]])
        windows = [(int(w), int(s)) for (w, s) in win_list]
    else:
        windows = _parse_windows_spec(str(windows_arg))

    mods_arg = args.modalities or _deep_get(cfg, ["preprocess", "modalities"], "network,radio")
    modalities = [m.strip().lower() for m in str(mods_arg).split(",") if m.strip()]
    for m in modalities:
        if m not in {"network", "radio"}:
            raise ValueError(f"Unknown modality: {m}")

    drop_empty = True
    if args.keep_empty_windows:
        drop_empty = False
    if args.drop_empty_windows:
        drop_empty = True

    splits_arg = args.splits or _deep_get(cfg, ["paths", "splits"], None)
    if splits_arg is None:
        splits_dir = out_dir / "splits"
        split_files = sorted(splits_dir.glob("*.json"))
    else:
        p = Path(str(splits_arg)).expanduser()
        if p.is_dir():
            split_files = sorted(p.glob("*.json"))
        else:
            split_files = [Path(x.strip()).expanduser() for x in str(splits_arg).split(",") if x.strip()]

    if not split_files:
        raise FileNotFoundError("No split JSON files found. Run make_run_splits.py first.")

    _ensure_dir(out_dir / "processed")

    all_summaries: List[Dict[str, Any]] = []

    for (W, S) in windows:
        net_path = stage1_out / "windows" / f"network_windows_W{W}_S{S}.parquet"
        rad_path = stage1_out / "windows" / f"radio_windows_W{W}_S{S}.parquet"
        if "network" in modalities and not net_path.exists():
            raise FileNotFoundError(f"Missing: {net_path}")
        if "radio" in modalities and not rad_path.exists():
            raise FileNotFoundError(f"Missing: {rad_path}")

        net_df = pd.read_parquet(net_path) if "network" in modalities else None
        rad_df = pd.read_parquet(rad_path) if "radio" in modalities else None

        for split_file in split_files:
            split = json.loads(split_file.read_text(encoding="utf-8"))
            if "split_name" not in split:
                split["split_name"] = split_file.stem

            if "network" in modalities:
                summ = _fit_transform_modality(net_df, split, modality="network", W=W, S=S, stage2_out_dir=out_dir, drop_empty=drop_empty)
                all_summaries.append(summ)
            if "radio" in modalities:
                summ = _fit_transform_modality(rad_df, split, modality="radio", W=W, S=S, stage2_out_dir=out_dir, drop_empty=drop_empty)
                all_summaries.append(summ)

    summ_path = out_dir / "processed" / "processed_index.csv"
    pd.DataFrame(all_summaries).to_csv(summ_path, index=False)
    print(f"Wrote: {summ_path} (n={len(all_summaries)})")

    snap = {
        "created_utc": _utc_now_iso(),
        "stage1_out_dir": str(stage1_out),
        "out_dir": str(out_dir),
        "splits": [str(p) for p in split_files],
        "windows": [{"W": W, "S": S} for (W, S) in windows],
        "modalities": modalities,
        "drop_empty_windows": drop_empty,
    }
    snap_path = out_dir / "processed" / "stage2_preprocess_config.json"
    snap_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    print(f"Wrote: {snap_path}")


if __name__ == "__main__":
    main()
