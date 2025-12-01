import argparse
import json
from pathlib import Path

from client import OpenRouterClient
from prompts import build_prompt

try:
    from src.rag.retriever import RetrievalService
    from src.rag.context_formatter import format_neighbors_to_tokens
except ModuleNotFoundError:
    RetrievalService = None


def load_dataset(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description="LLM inference via OpenRouter")
    parser.add_argument("--data", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--domain", choices=["lap", "res"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--rag_store", type=str, default=None)
    parser.add_argument("--rag_top_k", type=int, default=3)
    parser.add_argument("--rag_examples", type=int, default=2)
    parser.add_argument("--rag_max_tokens", type=int, default=128)
    args = parser.parse_args()

    client = OpenRouterClient()
    retriever = None
    if args.use_rag:
        if RetrievalService is None:
            raise RuntimeError("Retrieval modules unavailable")
        if args.rag_store is None:
            raise ValueError("--rag_store required when --use_rag is set")
        retriever = RetrievalService(args.rag_store)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for record in load_dataset(Path(args.data)):
            text = record.get("Text", "")
            snippets = []
            if retriever is not None:
                result = retriever.retrieve(text=text, top_k=args.rag_top_k, exclude_id=record.get("ID"))
                tokens = format_neighbors_to_tokens(result["neighbors"], args.rag_examples, args.rag_max_tokens)
                if tokens:
                    snippets.append(" ".join(tokens))
            messages = build_prompt(text, domain=args.domain, retrieval_snippets=snippets)
            response = client.chat_completion(
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            out_record = {
                "ID": record.get("ID"),
                "Text": text,
                "Quadruplet": response.get("content"),
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
