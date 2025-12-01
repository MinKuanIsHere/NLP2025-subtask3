import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


def load_data(jsonl_path: Path) -> List[dict]:
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def compute_va_stats(entries: List[Dict]) -> Tuple[float, float, float, float]:
    valences = []
    arousals = []
    for item in entries:
        va = item.get("VA")
        if not va:
            continue
        parts = str(va).split("#")
        if len(parts) >= 2:
            try:
                valences.append(float(parts[0]))
                arousals.append(float(parts[1]))
            except ValueError:
                continue
    if not valences or not arousals:
        return None, None, None, None
    v_arr = np.array(valences, dtype=np.float32)
    a_arr = np.array(arousals, dtype=np.float32)
    return (
        float(v_arr.mean()),
        float(v_arr.std()),
        float(a_arr.mean()),
        float(a_arr.std()),
    )


def prepare_metadata(records: List[dict], domain: str) -> Tuple[List[str], List[dict]]:
    texts = []
    metadata = []
    for record in records:
        text = record.get("Text", "").strip()
        quadruplets = record.get("Quadruplet") or record.get("Triplet") or []
        val_mean, val_std, aro_mean, aro_std = compute_va_stats(quadruplets)
        meta = {
            "id": record.get("ID"),
            "text": text,
            "domain": domain,
            "quadruplets": quadruplets,
            "valence_mean": val_mean,
            "valence_std": val_std,
            "arousal_mean": aro_mean,
            "arousal_std": aro_std,
        }
        metadata.append(meta)
        texts.append(text)
    return texts, metadata


def build_index(texts: List[str], model_name: str) -> faiss.Index:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["lap", "res"], help="Dataset domain")
    parser.add_argument("--data", required=True, help="Path to train JSONL")
    parser.add_argument("--out", required=True, help="Output directory for index + metadata")
    parser.add_argument("--embedding", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                        help="SentenceTransformer model name")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_records = load_data(data_path)
    texts, metadata = prepare_metadata(raw_records, args.domain)
    index = build_index(texts, args.embedding)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    (out_dir / "metadata.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in metadata),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
