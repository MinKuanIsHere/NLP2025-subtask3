import argparse
import json
from pathlib import Path
from collections import Counter

from retriever import RetrievalService


def load_metadata(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(domain_dir: Path, sample_size: int, top_k: int):
    metadata = load_metadata(domain_dir / "metadata.jsonl")
    service = RetrievalService(str(domain_dir))

    total = 0
    success = 0
    examples = []

    for record in metadata[:sample_size]:
        text = record.get("text", "").strip()
        if not text:
            continue
        result = service.retrieve(text=text, top_k=top_k, exclude_id=record.get("id"))
        target_categories = {
            (quad.get("Aspect"), quad.get("Category"))
            for quad in record.get("quadruplets", [])
            if quad.get("Category")
        }
        neighbor_categories = {
            (quad.get("Aspect"), quad.get("Category"))
            for hit in result["neighbors"]
            for quad in hit.get("quadruplets", [])
            if quad.get("Category")
        }
        matched = bool(target_categories & neighbor_categories)
        total += 1
        if matched:
            success += 1
        examples.append(
            {
                "query_id": record.get("id"),
                "text": text,
                "matched": matched,
                "neighbor_sample": result["neighbors"][:3],
            }
        )

    return {
        "queries": total,
        "hit_rate": success / total if total else 0.0,
        "examples": examples,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick retrieval quality check")
    parser.add_argument("--domain", required=True, help="Path to rag/stores/<domain>")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    report = evaluate(Path(args.domain), args.samples, args.top_k)
    print(f"Evaluated {report['queries']} queries; hit rate {report['hit_rate']:.2f}")
    for example in report["examples"][:5]:
        print("---")
        print(example["text"])
        print("Matched:", example["matched"])
        for neighbor in example["neighbor_sample"]:
            print("  Neighbor score", f"{neighbor['score']:.3f}", neighbor.get("quadruplets", [])[:1])
