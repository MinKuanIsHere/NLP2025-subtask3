import argparse
import json
from pathlib import Path
import numpy as np

from model import VARegressor


def main():
    parser = argparse.ArgumentParser(description="Apply VA regressor to predictions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = VARegressor.load(args.model)
    with open(args.input, encoding="utf-8") as f_in, open(args.output, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            record = json.loads(line)
            quadruplets = record.get("Quadruplet", [])
            for quad in quadruplets:
                features = np.array([[0, 0, 0, 0]])
                val = model.valence_model.predict(features)[0]
                aro = model.arousal_model.predict(features)[0]
                quad["VA"] = f"{val:.2f}#{aro:.2f}"
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
