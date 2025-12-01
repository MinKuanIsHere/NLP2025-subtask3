import argparse
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from dataset_builder import build_features
from model import VARegressor


def main():
    parser = argparse.ArgumentParser(description="Train VA regression model")
    parser.add_argument("--data", required=True, help="Path to training predictions JSONL with gold VA")
    parser.add_argument("--output", required=True, help="Output model path (.pkl)")
    parser.add_argument("--rag_store", type=str, default=None)
    parser.add_argument("--rag_top_k", type=int, default=3)
    args = parser.parse_args()

    retriever = None
    if args.rag_store:
        try:
            from src.rag.retriever import RetrievalService
        except ModuleNotFoundError:
            raise RuntimeError("RetrievalService not found; ensure src/rag is on PYTHONPATH")
        retriever = RetrievalService(args.rag_store)

    features = build_features(Path(args.data), retriever, top_k=args.rag_top_k)
    X = []
    y_val = []
    y_aro = []
    for feat in features:
        X.append([
            feat.get("neighbor_val_mean") or 0.0,
            feat.get("neighbor_val_std") or 0.0,
            feat.get("neighbor_aro_mean") or 0.0,
            feat.get("neighbor_aro_std") or 0.0,
        ])
        y_val.append(feat["target_val"])
        y_aro.append(feat["target_aro"])
    X = np.array(X)
    val_model = GradientBoostingRegressor()
    aro_model = GradientBoostingRegressor()
    val_model.fit(X, y_val)
    aro_model.fit(X, y_aro)
    va_model = VARegressor(valence_model=val_model, arousal_model=aro_model)
    va_model.save(args.output)


if __name__ == "__main__":
    main()
