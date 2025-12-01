# RAG Enhancement Plan

This plan tracks the work needed to deliver Retrieval-Augmented Generation (RAG) support and LLM-based baselines for the DimABSA project. Tasks are grouped into phases but can run in parallel when resources allow.

## Phase 0 – Scaffolding (current)
- [x] Create `src/` workspace with documentation.
- [ ] Decide tech stack for new modules (Python 3.10+, FAISS, sentence-transformers, OpenRouter SDK).
- [ ] Define configuration schema (YAML/JSON) for experiments.

## Phase 1 – Retrieval Layer
- [ ] **Embedding selection**: evaluate multilingual SentenceTransformers models (e.g., `paraphrase-multilingual-mpnet-base-v2`) on laptop/restaurant snippets.
- [ ] **Vector store builder**: parse existing train JSONL files.
  - [ ] Encode sentences and quadruplets; store embeddings + metadata.
  - [ ] Build FAISS indexes per domain (`rag/stores/{domain}.faiss`).
- [ ] **Retriever API**: expose `retrieve(sentence, domain, k)` returning exemplars + VA stats via a simple Python module.
- [ ] **Quality checks**: measure recall@k vs. gold triples to ensure neighbors are meaningful.

## Phase 2 – BERT Pipelines with RAG
- [ ] **DataProcess extensions**: add hooks to prepend retrieved exemplars to query templates (token budget aware).
- [ ] **Configurable prompts**: allow switching between original, medium, optimized, and RAG-augmented templates via env vars or config files.
- [ ] **Training scripts**: new modules under `pipelines/` to fine-tune `bert-base-multilingual-cased`, `bert-base-chinese`, and `chinese-roberta-wwm-ext-large` using retrieval context.
- [ ] **Evaluation**: reuse `/subtask3/evaluation_script` to compare with baseline F1.

## Phase 3 – LLM Baselines via OpenRouter
- [ ] **Client abstraction** (`llm_clients/openrouter.py`): wrap API key handling, retries, and structured output parsing.
- [ ] **Prompt templates**: domain-specific instructions listing entity/attribute labels.
  - [ ] Variants with/without retrieved exemplars in the message.
- [ ] **Batch runner**: script to iterate through dev/test JSONL, call GPT-5 or Gemini, enforce JSON schema, and write predictions to `tasks/subtask_3/`.
- [ ] **Cost/latency logging**: store stats for each run to support trade-off analysis.

## Phase 4 – Valence/Arousal Regression
- [ ] **Feature extraction**: combine opinion span embeddings, retrieved VA values, and lexical cues.
- [ ] **Modeling**: prototype Gradient Boosting, lightweight transformer, or k-NN smoothing.
- [ ] **Integration**: pipeline writes Aspect/Category/Opinion via extractor, then calls VA regressor to fill `VA` fields.
- [ ] **Metrics**: compute RMSE/Pearson and compare with current CLS-based regressors.

## Phase 5 – Experiment Tracking & Documentation
- [ ] Maintain config + results table (model, domain, RAG flag, F1, VA metrics, cost).
- [ ] Update README with implementation details per module.
- [ ] Prepare final report summarizing gains, failure cases, and next steps.

## Milestones
| Milestone | Target | Exit Criteria |
|-----------|--------|----------------|
| Retrieval layer ready | Week 1 | Domain indexes built, retriever API returns >70% relevant neighbors on spot-checks. |
| BERT+RAG baseline | Week 2 | Training scripts runnable, F1 >= baseline ± delta logged. |
| LLM baselines | Week 3 | GPT-5/Gemini runs produce valid JSON for both domains. |
| VA regressor | Week 3 | Separate model achieves lower RMSE vs. current approach. |
| Report | Week 4 | README updated, experiment table completed. |

## Dependencies & Risks
- **API keys**: OpenRouter/GPT access must be secured; throttle limits could slow experiments.
- **Compute**: FAISS building and BERT fine-tuning require GPU time; schedule accordingly.
- **Prompt length**: RAG context must stay within token budgets, especially for Gemini/GPT models.
- **Data privacy**: confirm the dataset terms permit sending text to third-party APIs.

Track progress by converting unchecked tasks into issues or TODOs once coding begins.
