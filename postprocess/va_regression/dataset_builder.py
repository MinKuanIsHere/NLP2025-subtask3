import json
from pathlib import Path
from typing import List, Dict

import numpy as np


def build_features(predictions_path: Path, retriever, top_k: int = 3) -> List[Dict]:
    features = []
    with predictions_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("Text", "")
            quadruplets = record.get("Quadruplet", [])
            retrieval = retriever.retrieve(text=text, top_k=top_k, exclude_id=record.get("ID")) if retriever else None
            neighbor_stats = retrieval["aggregate_va"] if retrieval else {}
            for quad in quadruplets:
                valence = float(quad.get("VA", "0#0").split("#")[0])
                arousal = float(quad.get("VA", "0#0").split("#")[1])
                aspect = quad.get("Aspect", "")
                opinion = quad.get("Opinion", "")
                features.append(
                    {
                        "aspect": aspect,
                        "opinion": opinion,
                        "text": text,
                        "neighbor_val_mean": neighbor_stats.get("valence_mean"),
                        "neighbor_val_std": neighbor_stats.get("valence_std"),
                        "neighbor_aro_mean": neighbor_stats.get("arousal_mean"),
                        "neighbor_aro_std": neighbor_stats.get("arousal_std"),
                        "target_val": valence,
                        "target_aro": arousal,
                    }
                )
    return features
