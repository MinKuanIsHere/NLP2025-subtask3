#!/usr/bin/env python3
import argparse
import json
import pickle
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
for path in (PROJECT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

try:
    from postpreprocess.va_regression.dataset_builder import build_features
    from postpreprocess.va_regression.model import VARegressor
except ModuleNotFoundError:
    from va_regression.dataset_builder import build_features
    from va_regression.model import VARegressor


DOMAIN_FILES = {
    "lap": {
        "train": "subtask3/data/zho_laptop_train_alltasks.jsonl",
        "dev": "subtask3/data/zho_laptop_dev_task3.jsonl",
    },
    "res": {
        "train": "subtask3/data/zho_restaurant_train_alltasks.jsonl",
        "dev": "subtask3/data/zho_restaurant_dev_task3.jsonl",
    },
}


def features_to_arrays(features):
    X = []
    y_val = []
    y_aro = []
    for feat in features:
        X.append(
            [
                feat.get("neighbor_val_mean") or 0.0,
                feat.get("neighbor_val_std") or 0.0,
                feat.get("neighbor_aro_mean") or 0.0,
                feat.get("neighbor_aro_std") or 0.0,
            ]
        )
        y_val.append(feat["target_val"])
        y_aro.append(feat["target_aro"])
    return np.array(X, dtype=np.float32), np.array(y_val), np.array(y_aro)


def dump_features(features, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(features, f)


def train_and_eval(domain: str, top_k: int, output_dir: Path, feature_dir: Path):
    files = DOMAIN_FILES[domain]
    train_features = build_features(Path(files["train"]), retriever=None, top_k=top_k)
    feature_path = feature_dir / f"{domain}_train.pkl"
    dump_features(train_features, feature_path)

    X_train, y_val_train, y_aro_train = features_to_arrays(train_features)
    if X_train.size == 0:
        raise RuntimeError(f"No training samples generated for domain {domain}")

    X_fit, X_hold, y_val_fit, y_val_hold = train_test_split(
        X_train, y_val_train, test_size=0.2, random_state=42
    )
    _, _, y_aro_fit, y_aro_hold = train_test_split(
        X_train, y_aro_train, test_size=0.2, random_state=42
    )

    val_model = GradientBoostingRegressor()
    aro_model = GradientBoostingRegressor()
    val_model.fit(X_fit, y_val_fit)
    aro_model.fit(X_fit, y_aro_fit)

    if X_hold.size:
        val_hold_pred = val_model.predict(X_hold)
        aro_hold_pred = aro_model.predict(X_hold)
        val_hold_rmse = float(np.sqrt(np.mean((val_hold_pred - y_val_hold) ** 2)))
        aro_hold_rmse = float(np.sqrt(np.mean((aro_hold_pred - y_aro_hold) ** 2)))
    else:
        val_hold_rmse = float("nan")
        aro_hold_rmse = float("nan")

    # retrain using full dataset before saving
    val_model.fit(X_train, y_val_train)
    aro_model.fit(X_train, y_aro_train)

    model = VARegressor(valence_model=val_model, arousal_model=aro_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{domain}_va.pkl"
    model.save(model_path)

    metrics = {
        "domain": domain,
        "val_rmse_holdout": val_hold_rmse,
        "aro_rmse_holdout": aro_hold_rmse,
        "train_samples": int(len(train_features)),
        "holdout_samples": int(len(y_val_hold)),
        "feature_cache": str(feature_path),
    }
    metrics_path = output_dir / f"{domain}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[train_va] Saved model to {model_path}")
    print(f"[train_va] Metrics saved to {metrics_path}: {metrics}")


def main():
    parser = argparse.ArgumentParser(description="Train VA regressor per domain.")
    parser.add_argument("--domain", choices=DOMAIN_FILES.keys(), required=True)
    parser.add_argument("--top_k", type=int, default=3, help="Neighbor count for feature builder.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("postpreprocess/models"),
        help="Directory to store trained models/metrics.",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("postpreprocess/models/features"),
        help="Directory to store cached training features.",
    )
    args = parser.parse_args()

    train_and_eval(args.domain, args.top_k, args.output_dir, args.feature_dir)


if __name__ == "__main__":
    main()
