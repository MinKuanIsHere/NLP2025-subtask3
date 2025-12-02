# Post-Processing Workflow Report

This report summarizes the methodology used to clean historical prediction JSONLs, retrain/post-apply Valence/Arousal regressors, and leverage the vendored utilities under `postpreprocess/va_regression`. All commands assume repository root (`~/NLP2025-subtask3`) and run inside the Docker container unless noted.

## 1. Data Cleaning for Legacy Predictions

### Objective
Remove noisy tokens (`[UNK]`), trim whitespace, and drop entries with empty `Quadruplet` arrays in archived predictions under `postpreprocess/data/`.

### Implementation
- Module: `postpreprocess/cleaning/json_cleaner.py` – provides helpers to strip `[UNK]` and filter quadruplets.
- CLI: `postpreprocess/scripts/clean_json.py` – iterates through `postpreprocess/data/*.jsonl`, writes cleaned files to `postpreprocess/clean/`, and appends per-file stats to `postpreprocess/clean/cleanup_stats.csv`.

```bash
python postpreprocess/scripts/clean_json.py
```

### Outputs
- Cleaned JSONLs (same filenames) in `postpreprocess/clean/`.
- Cleanup statistics (kept/dropped counts per file) in `postpreprocess/clean/cleanup_stats.csv`.

## 2. VA Regression Training

### Objective
Train lightweight regressors per domain (Laptop/Restaurant) using labeled data from `subtask3/data/zho_{domain}_train_alltasks.jsonl`.

### Implementation
- Reuses the vendored feature builder from `postpreprocess/va_regression/dataset_builder.py` to aggregate VA targets and optional neighbor stats.
- Script: `postpreprocess/models/train_va.py`
  - Loads training JSONLs, builds features, caches them under `postpreprocess/models/features/{domain}_train.pkl`.
  - Performs an 80/20 holdout split to estimate RMSE (records stored in `postpreprocess/models/{domain}_metrics.json`).
  - Fits Gradient Boosting regressors (from scikit-learn) and saves bundled models as `postpreprocess/models/{domain}_va.pkl` (leveraging `postpreprocess/va_regression/model.VARegressor`).

```bash
python postpreprocess/models/train_va.py --domain lap
python postpreprocess/models/train_va.py --domain res
```

### Outputs
- Models: `postpreprocess/models/lap_va.pkl`, `postpreprocess/models/res_va.pkl`.
- Metrics: `postpreprocess/models/{domain}_metrics.json` (includes holdout RMSE, sample counts, feature cache path).
- Feature caches for reproducibility/debugging: `postpreprocess/models/features/{domain}_train.pkl`.

## 3. Enrichment of Historical Predictions

### Objective
Apply the trained VA models to the cleaned legacy predictions to refresh `VA` fields before benchmark submission or analysis.

### Implementation
- Script: `postpreprocess/scripts/enrich_va.py`
  - Loads the relevant model (`postpreprocess/models/{domain}_va.pkl` by default).
  - For each record, generates the simple four-feature vector `[neighbor_val_mean, val_std, aro_mean, aro_std]` (currently zeros unless a custom retriever is plugged in), and predicts new `VA`.
  - Writes enriched files to `postpreprocess/enriched/<original_filename>`.

```bash
python postpreprocess/scripts/enrich_va.py \
  --domain lap \
  --input postpreprocess/clean/opt_query_bert_chinese_lap_full_pred.jsonl
```
(Repeat for other domain/file combinations; add `--rag_store` if neighbor statistics should be used.)

### Outputs
- Enriched JSONLs with updated `VA` per quadruplet under `postpreprocess/enriched/`.
- Validation performed via existing helper:
  ```bash
  bash scripts/sample_and_validate.sh postpreprocess/enriched/<file>.jsonl 5 /tmp/check.jsonl
  ```

## 4. Vendored Utilities

- `postpreprocess/va_regression/dataset_builder.py`: extracts features/targets from labeled training data (adapted from the main `src/` tree).
- `postpreprocess/va_regression/model.py`: model serialization/deserialization for saving `VARegressor` bundles.
- Retrieval hooks are currently disabled; if RAG-based stats are needed in the future, a local retriever module can be added alongside these utilities.

## 5. Next Steps / Reporting

- Perform consistency checks and VA distribution comparisons; recommended to document findings in `postpreprocess/reports/` (e.g., `qa_log.md`, `*_va_comparison.md`).
- Decide whether enriched outputs replace originals when submitting to benchmarks, and update `README` or pipeline docs accordingly.
