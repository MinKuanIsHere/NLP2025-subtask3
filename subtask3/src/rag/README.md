# RAG Retrieval Layer

This module builds and serves vector stores for the laptop and restaurant datasets so both BERT-based and LLM-based pipelines can retrieve labeled exemplars.

## Components
- `build_store.py`: command-line tool that ingests JSONL data, encodes sentences with SentenceTransformers, and writes FAISS indexes + metadata.
- `retriever.py`: provides a Python API `retrieve(text, domain, top_k, use_gpu)` returning exemplars, including aspect/category/opinion/VA.
- `config.py`: shared defaults (embedding model, index paths).

## Workflow
1. Build per-domain stores (each under its own directory):
   ```bash
   python src/rag/build_store.py --domain lap --data subtask3/data/zho_laptop_train_alltasks.jsonl --out rag/stores/lap
   python src/rag/build_store.py --domain res --data subtask3/data/zho_restaurant_train_alltasks.jsonl --out rag/stores/res
   ```
   Each directory contains:
   - `index.faiss`: FAISS inner-product index over normalized embeddings.
   - `metadata.jsonl`: aligned metadata with text, quadruplets, and VA stats for every training sample.

2. Use the retriever inside training/inference:
   ```python
   from src.rag.retriever import RetrievalService
   service = RetrievalService("rag/stores/lap")
   result = service.retrieve(text="螢幕很讚", top_k=5)
   print(result["neighbors"][0]["quadruplets"])
   ```
3. Each result returns the original text, quadruplet metadata, and precomputed VA statistics (mean/std across neighbors).
4. (Optional) Evaluate retrieval quality:
   ```bash
   python src/rag/eval_retrieval.py --domain rag/stores/lap --samples 50 --top_k 5
   python src/rag/eval_retrieval.py --domain rag/stores/res --samples 50 --top_k 5
   ```
   The script reports a simple hit rate (neighbor shares a matching aspect/category) and prints example neighbors for inspection.

Detailed implementation will be fleshed out during Phase 1.
