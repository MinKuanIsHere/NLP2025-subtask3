# LLM Clients (Phase 3)

This module integrates OpenRouter-accessible LLMs (GPT‑5.1, Gemini 3 Pro Preview, Llama‑3.3‑70B Instruct) into the ABSA pipeline.

## Structure
- `client.py`: thin wrapper around the OpenRouter REST API.
- `prompts.py`: prompt templates for laptop/restaurant domains with/without retrieval context.
- `runner.py`: script to iterate over JSONL datasets, call the LLM, enforce JSON schema, and write prediction files.

## Configuration
- Requires `OPENROUTER_API_KEY` environment variable.
- Optional: `OPENROUTER_SITE_URL`, `OPENROUTER_SITE_NAME` for ranking metadata.
- Models to support:
  - `openai/gpt-5.1`
  - `google/gemini-3-pro-preview`
  - `meta-llama/llama-3.3-70b-instruct`

The runner will later accept flags for `--model`, `--use_rag`, `--rag_store` to reuse the RetrievalService.
