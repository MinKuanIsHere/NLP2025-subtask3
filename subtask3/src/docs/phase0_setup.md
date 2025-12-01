# Phase 0 – Environment & Config Decisions

## 1. Tech Stack
| Layer | Decision | Notes |
|-------|----------|-------|
| Python runtime | 3.10 (conda env inside container) | Matches FAISS/cu12 wheels and keeps compatibility with Torch 2.5.1 |
| Core DL libs | PyTorch 2.5.1 + CUDA 12.1, Transformers (main branch) | Already baked into container via `requirements.txt` |
| Retrieval | FAISS 1.8 (cpu + cu12) | Installed globally in container; GPU or CPU index selectable at runtime |
| Embedding model | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (default) | Balanced accuracy vs. latency across laptop/restaurant corpora; allow override through config |
| Tokenizer for retrieval text | HuggingFace AutoTokenizer | Share with BERT so token alignment is consistent |
| LLM access | OpenRouter JS/Python SDK (REST) | Use HTTPS client with API key via env var `OPENROUTER_API_KEY` |
| Config/CLI | YAML files parsed by `omegaconf` | Enables hierarchical overrides for experiments |
| Logging | Python `logging` + JSONL experiment summary | Standardize metrics per run |

## 2. Experiment Config Schema
Each experiment will load a YAML file under `src/configs/`. Schema (expressed in YAML) and field meanings:

```yaml
experiment:
  name: "lap_multilingual_rag"
  task: 3                  # 2 or 3
  seed: 42

model:
  type: "bert"            # bert | bert_rag | gpt_openrouter
  checkpoint: "bert-base-multilingual-cased"
  prompt_template: "optimized"   # original | medium | optimized | rag
  max_seq_len: 512

retrieval:
  enabled: true
  store_path: "rag/stores/lap.faiss"
  embedding_model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  top_k: 5
  use_gpu: true

llm:
  provider: "openrouter"
  model: "openai/gpt-5.1"          # also support google/gemini-3-pro-preview, meta-llama/llama-3.3-70b-instruct
  temperature: 0.0
  max_tokens: 1024
  json_schema: "configs/schema_quadruplet.json"

training:
  epochs: 3
  batch_size: 4
  learning_rate: 1e-3
  tuning_bert_rate: 1e-5
  warmup_ratio: 0.1

va_regression:
  enabled: true
  model_type: "xgboost"
  features: ["opinion_embedding", "retrieved_va_mean", "retrieved_va_std"]

paths:
  data_dir: "subtask3/data"
  output_dir: "subtask3/tasks"
  log_dir: "subtask3/log"

## 3. Experiment Coverage
- `experiment.name` should encode domain + model, e.g. `lap_bert-base-multilingual-cased_rag`.
- Domains: laptop, restaurant (both Chinese datasets).
- BERT family: `bert-base-multilingual-cased`, `bert-base-chinese`, `chinese-roberta-wwm-ext-large`.
- LLMs via OpenRouter: `openai/gpt-5.1`, `google/gemini-3-pro-preview`, `meta-llama/llama-3.3-70b-instruct`.
```

### Validation Rules
- `retrieval.enabled` must be true when `model.type` is `bert_rag` or `gpt_openrouter` with RAG.
- `llm` block only required for `model.type == gpt_openrouter`.
- `va_regression.enabled` dictates whether the pipeline calls the separate regressor after extracting Aspect/Category/Opinion.

### Next Steps
- Store configs in `src/configs/*.yaml` following this schema.
- Implement a loader utility (`src/utils/config.py`) that validates fields and injects defaults.
