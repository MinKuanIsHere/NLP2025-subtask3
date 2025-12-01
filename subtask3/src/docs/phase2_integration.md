# Phase 2 – BERT + RAG Integration Notes

## Integration Points Identified
1. **DataProcess.py**
   - Functions `make_QA`, `make_inference_QA`, and `train_data_process` construct query templates by concatenating `[CLS] query tokens [SEP]` with the tokenized review text (`word_list`).
   - Hook: before building `word_list`, append retrieved exemplars (formatted text such as `Retrieved Example #1: ... Quadruplet: ...`) so the MRC queries include contextual evidence.
   - Need ability to truncation so combined length stays within 512 tokens.

2. **run_task2&3_trainer_multilingual.py**
   - Prior to dataset preparation, load `RetrievalService` when `args.use_rag` is true.
   - Pass retriever into `dataset_process`/`train_data_process` so each sample fetches top-k neighbors.
   - For inference/evaluate paths, ensure retrieval is also applied to compose queries.

3. **Configuration/CLI**
   - Add CLI flags: `--use_rag`, `--rag_store_path`, `--rag_top_k`, `--rag_template` (controls formatting), `--rag_max_tokens`.
   - Expose these in YAML configs under `model`/`retrieval` sections.

## Template Strategy
- For each retrieved neighbor, build a summary string containing Aspect, Category, Opinion, VA.
- Example snippet appended to `word_list`:
  ```
  [SEP] RETRIEVED [NUM] TEXT : ... ; QUAD : Aspect=<...> Category=<...> Opinion=<...>
  ```
- Limit to `rag_top_k` neighbors and optionally only include top categories.

## Data Flow
1. `train_data_process` receives `retriever` and current record ID/text.
2. It calls `retriever.retrieve(text, top_k=ragtK, exclude_id=ID)`.
3. Format retrieved info using helper (new module `rag/utils.py`).
4. Append formatted tokens to `word_list` before query building.
5. Downstream logic (padding, masks, loss) remains unchanged because sequences already handle arbitrary length.

## Additional Considerations
- Cache retrieval results while building dataset to avoid redundant calls when multiple passes (train/dev splits).
- Provide deterministic behavior by storing retrieved metadata alongside dataset so training/inference uses same context.
- Logging: for early experiments, dump a few samples showing appended RAG context to verify formatting.
