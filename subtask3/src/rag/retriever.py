import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RetrievalService:
    def __init__(
        self,
        store_dir: str,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    ):
        store_dir = Path(store_dir)
        self.index = faiss.read_index(str(store_dir / "index.faiss"))
        self.metadata = self._load_metadata(store_dir / "metadata.jsonl")
        self.model = SentenceTransformer(embedding_model)

    @staticmethod
    def _load_metadata(path: Path) -> List[Dict]:
        items = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        return items

    def retrieve(self, text: str, top_k: int = 5, exclude_id: str = None) -> Dict[str, object]:
        embedding = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(embedding)
        scores, indices = self.index.search(embedding, top_k)

        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            record = dict(self.metadata[idx])
            if exclude_id and record.get("id") == exclude_id:
                continue
            record["score"] = float(score)
            hits.append(record)

        aggregate = self._aggregate_va(hits)
        return {"neighbors": hits, "aggregate_va": aggregate}

    @staticmethod
    def _aggregate_va(hits: List[Dict]) -> Dict[str, float]:
        valences = [h["valence_mean"] for h in hits if h.get("valence_mean") is not None]
        arousals = [h["arousal_mean"] for h in hits if h.get("arousal_mean") is not None]
        agg = {}
        if valences:
            arr = np.array(valences, dtype=np.float32)
            agg["valence_mean"] = float(arr.mean())
            agg["valence_std"] = float(arr.std())
        if arousals:
            arr = np.array(arousals, dtype=np.float32)
            agg["arousal_mean"] = float(arr.mean())
            agg["arousal_std"] = float(arr.std())
        return agg
