#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
for path in (PROJECT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

try:
    from postpreprocess.va_regression.model import VARegressor
except ModuleNotFoundError:
    from va_regression.model import VARegressor

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "clean"
ENRICHED_DIR = BASE_DIR / "enriched"
DOMAIN_MODEL_DEFAULTS = {
    "lap": BASE_DIR / "models" / "lap_va.pkl",
    "res": BASE_DIR / "models" / "res_va.pkl",
}


def load_records(path: Path):
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def build_feature_vector(neighbor_stats):
    return np.array(
        [
            neighbor_stats.get("valence_mean") or 0.0,
            neighbor_stats.get("valence_std") or 0.0,
            neighbor_stats.get("arousal_mean") or 0.0,
            neighbor_stats.get("arousal_std") or 0.0,
        ],
        dtype=np.float32,
    ).reshape(1, -1)


def enrich_file(
    domain: str,
    input_path: Path,
    output_path: Path,
    model_path: Path,
):
    if domain not in DOMAIN_MODEL_DEFAULTS:
        raise ValueError(f"Unsupported domain {domain}")
    model = VARegressor.load(model_path)

    records = load_records(input_path)
    predictions = 0
    for record in records:
        text = record.get("Text", "")
        quadruplets = record.get("Quadruplet") or []
        feature_vec = build_feature_vector({})
        val = float(model.valence_model.predict(feature_vec)[0])
        aro = float(model.arousal_model.predict(feature_vec)[0])
        for quad in quadruplets:
            quad["VA"] = f"{val:.2f}#{aro:.2f}"
            predictions += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f_out:
        for record in records:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[enrich_va] {input_path.name}: updated {predictions} quadruplets -> {output_path}")


def resolve_input(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    candidate = DATA_DIR / path_str
    return candidate if candidate.exists() else path


def main():
    parser = argparse.ArgumentParser(description="Enrich cleaned predictions with VA regression outputs.")
    parser.add_argument("--domain", choices=DOMAIN_MODEL_DEFAULTS.keys(), required=True)
    parser.add_argument("--input", required=True, help="Cleaned JSONL file to enrich (absolute or relative to clean/).")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for enriched JSONL (default: postpreprocess/enriched/<input_name>).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to VA model (.pkl). Defaults to postpreprocess/models/{domain}_va.pkl.",
    )
    args = parser.parse_args()

    input_path = resolve_input(args.input)
    model_path = args.model or DOMAIN_MODEL_DEFAULTS[args.domain]
    output_path = Path(args.output) if args.output else ENRICHED_DIR / input_path.name

    enrich_file(
        domain=args.domain,
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
    )


if __name__ == "__main__":
    main()
