# RAG Augmentation Project

This directory hosts new experiments that extend the existing DimABSA system with Retrieval-Augmented Generation (RAG) pipelines and large language models. The goal is to compare and combine classic fine-tuned BERT encoders with frontier LLMs (GPT-5, Gemini) for Tasks 2/3 across laptop and restaurant domains.

## Objectives
- Build a reusable retrieval layer over the multilingual training corpora, exposing per-domain vector stores with labeled exemplars.
- Implement modular inference runners:
  - Baseline encoders (`bert-base-multilingual-cased`, `bert-base-chinese`, `chinese-roberta-wwm-ext-large`).
  - RAG-enhanced versions of these encoders (context-augmented MRC inputs).
  - LLM-based extractors via OpenRouter (GPT-5, Gemini) with and without retrieval context.
- Separate valence/arousal regression from aspect/category/opinion extraction by training dedicated regressors that leverage retrieved statistics.
- Provide reproducible evaluation scripts and documentation for future iterations.

## Layout
```
src/
├── README.md              # This overview
├── plan.md                # Detailed execution plan & milestones
├── rag/                   # Retrieval utilities and vector store builders (future)
├── pipelines/             # Inference/training scripts for BERT + RAG (future)
├── llm_clients/           # OpenRouter/GPT/Gemini integration (future)
└── va_regression/         # Independent VA regression models (future)
```

Each subdirectory will include its own module-level README once implementation begins.

## Usage (future work)
- Build vector stores: `python -m rag.build_store --domain lap`.
- Run BERT + RAG inference: `python -m pipelines.bert_rag_infer --config configs/lap_multilingual.yaml`.
- Query GPT-5 via OpenRouter: `python -m llm_clients.run_openrouter --model openai/gpt-5 --with-rag`.

Detailed steps, configs, and experiment tracking live in `plan.md` until code is implemented.
