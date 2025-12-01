# NLP2025 Subtask 3 – RAG & LLM Pipeline

This repository hosts the NYCU NLP 2025 final project (subtask 3) with the following key features:

- Classic DimABSA training/inference scripts (BERT-based) with optional Retrieval-Augmented Generation (RAG) support.
- LLM baselines via OpenRouter (GPT-5.1, Gemini 3 Pro Preview, Llama 3.3 70B Instruct).
- Retrieval layer built with FAISS and SentenceTransformers to provide exemplar context.
- VA regression module for predicting Valence/Arousal separately.
- Automation scripts for sampling data, launching experiments, and validating outputs.

## Requirements
- Docker with NVIDIA runtime; see `subtask3/Dockerfile` and `subtask3/docker-compose.yml`.
- `.env` file in `subtask3/.env` defining `OPENROUTER_API_KEY` (and optional `OPENROUTER_SITE_URL`/`OPENROUTER_SITE_NAME`).
- GPU(s) with CUDA 12 support.

## Quick Start
1. Fill `subtask3/.env` with valid OpenRouter credentials.
2. Build and start the container:
   ```bash
   cd subtask3
   docker compose build
   docker compose up -d
   ```
3. Access the container:
   ```bash
   docker exec -it nlp_dimabsa_subtask3 /bin/bash
   ```
4. Run individual experiments or use automation scripts (see below).

## Scripts
- `scripts/sample_and_validate.sh <jsonl> [N] [output]`: take a sample of N entries and validate JSON.
- `scripts/run_all_experiments.sh [bert|llm|all]`: orchestrate all BERT and/or LLM experiments (includes RAG variants) inside the container.

## RAG
- Vector stores reside in `subtask3/rag/stores/<domain>`; build them with `python src/rag/build_store.py ...` (already prepared).
- Retrieval quality can be checked via `python src/rag/eval_retrieval.py --domain rag/stores/lap --samples 50 --top_k 5`.

## LLM Integration
- Located in `src/llm_clients/`.
- Use `python src/llm_clients/runner.py --data ... --output ... --domain lap --model openai/gpt-5.1 [--use_rag ...]`.

## VA Regression
- Located in `src/va_regression/`.
- Train: `python src/va_regression/train.py --data subtask3/data/zho_laptop_train_alltasks.jsonl --output va_models/lap_va.pkl --rag_store rag/stores/lap`.
- Predict: `python src/va_regression/predict.py --model va_models/lap_va.pkl --input <pred.jsonl> --output <pred_with_va.jsonl>`.

## Experiment Tracking
- Use `src/experiments/results.md` to log benchmark scores for each configuration.
