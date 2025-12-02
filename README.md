# PostPreprocess Toolkit

PostPreprocess is a self-contained toolkit for cleaning legacy aspect-based sentiment predictions and refreshing every quadruplet with up-to-date Valence/Arousal (VA) scores. The pipeline takes noisy JSONL exports, strips hallucinated tokens, removes empty quadruplets, and applies Gradient Boosting regressors that were retrained on authoritative domain datasets. The resulting enriched JSONLs become a consistent handoff artifact for downstream evaluation or redistribution.

## Project Layout
```
postprocess/
├── data/                # Place raw prediction JSONLs here before running the toolkit
├── clean/               # Cleaner outputs + cleanup_stats.csv (auto-generated)
├── enriched/            # Final VA-enriched JSONLs (auto-generated)
├── cleaning/            # Core Python modules for text/quadruplet normalization
├── models/              # VA regressor code, metrics, and serialized *.pkl weights
├── reports/             # Optional QA notes or distribution comparisons
├── scripts/             # CLI entrypoints (clean_json.py, enrich_va.py, run_enrich.sh, etc.)
├── va_regression/       # Feature builder + model wrapper reused across commands
├── Dockerfile           # Runtime image with all Python dependencies
├── docker-compose.yml   # GPU-ready service definition for batch processing
└── plan.md / README.md  # Docs and roadmap
```

## Methodology & Purpose
- **Cleaning (`cleaning/json_cleaner.py`)**: Each record is parsed, `[UNK]` placeholders are removed, whitespace is compacted, and quadruplets without either aspect or opinion text are dropped. The cleaner keeps the original metadata but guarantees every surviving quadruplet has meaningful `Aspect`/`Opinion` content. Because the step is deterministic, `clean/cleanup_stats.csv` becomes a reproducible audit trail (total vs. kept vs. dropped counts per file).
- **VA Regression (`models/train_va.py`)**: Domain-specific datasets (laptop/restaurant) feed a feature builder (`va_regression/dataset_builder.py`) that iterates over every quadruplet. For each instance it records (a) optional retrieval aggregates—neighbor valence/arousal mean and standard deviation computed from the top-`k` similar texts, and (b) the gold VA targets parsed from the original `VA` field. Those four numeric features (`neighbor_val_mean`, `neighbor_val_std`, `neighbor_aro_mean`, `neighbor_aro_std`) become the regressors’ inputs, while `target_val` and `target_aro` serve as prediction targets. Two Gradient Boosting models are trained per domain (one for valence, one for arousal), evaluated with an 80/20 holdout split, and finally refit on all training samples before being serialized via `VARegressor`.
- **Enrichment (`scripts/enrich_va.py`)**: Cleaned records are reopened and, for each quadruplet, a compact feature vector is built (currently the neighbor stats default to zeros, but the same scaffold can ingest richer aggregates). The trained `VARegressor` predicts valence and arousal separately, then formats the result as `"{val:.2f}#{aro:.2f}"`. This consistent representation overwrites any noisy legacy VA strings, ensuring downstream consumers receive harmonized, noise-free quadruplets while preserving the original aspect/opinion/category text.

The objective is to standardize historical predictions so that analysts and downstream models can compare runs fairly, regardless of original model quirks or tokenization errors.

## Preparing Data & Environment
1. **Populate `data/`** – copy every raw prediction JSONL (e.g., `*_lap_full_pred.jsonl`, `subtask_3-*.jsonl`) into `postpreprocess/data/`. The pipeline only reads from this folder; file names are preserved downstream.
2. **Start the Docker workspace** – the included compose stack bundles all dependencies and maps this directory to `/workspace`.
   ```bash
   cd postpreprocess
   docker compose build             # optional after code changes
   docker compose up -d             # launch container + GPU runtime
   docker exec -it nlp_dimabsa_subtask3 bash   # drop into /workspace shell
   ```
   Stop the environment with `docker compose down` when finished.
3. **Optional local run** – if Docker is unavailable, ensure Python 3.10+ plus `numpy`, `scikit-learn`, and `joblib` are installed, then run the same scripts directly from the host.

## Running the Pipeline

### Quick Start (Docker container)
1. `cd postpreprocess && docker compose up -d`
2. Copy raw prediction JSONLs into `postpreprocess/data/` on the host (they appear inside `/workspace/data/`).
3. `docker exec -it nlp_dimabsa_subtask3 bash` to enter the container.
4. Inside `/workspace`, run `bash scripts/run_enrich.sh`.
5. Collect cleaned files from `/workspace/clean/` and VA-enriched outputs from `/workspace/enriched/`.

### 1. Clean raw files
```bash
python scripts/clean_json.py \
  --input postpreprocess/data \
  --output-dir postpreprocess/clean \
  --stats postpreprocess/clean/cleanup_stats.csv
```
Outputs cleaned JSONLs in `clean/` and logs per-file totals to `cleanup_stats.csv`.

### 2. (Optional) Retrain VA models
```bash
python models/train_va.py --domain lap
python models/train_va.py --domain res
```
Trained weights are stored under `models/{lap,res}_va.pkl`, with metrics/feature caches saved next to them.

### 3. Enrich cleaned predictions
```bash
python scripts/enrich_va.py \
  --domain lap \
  --input postpreprocess/clean/opt_query_bert_chinese_lap_full_pred.jsonl \
  --output postpreprocess/enriched/opt_query_bert_chinese_lap_full_pred.jsonl
```
Domains choose between the `lap` and `res` regressors; multiple files can be processed independently.

### 4. End-to-end batch helper
```bash
bash scripts/run_enrich.sh
```
This convenience script scans `data/`, cleans everything, and enriches each cleaned file into `enriched/` automatically. Use it when distributing the toolkit or when the Docker container is the primary execution target.

## Outputs
- `clean/<original_name>.jsonl` – sanitized records ready for reuse.
- `clean/cleanup_stats.csv` – append-only log capturing total/kept/dropped counts per run.
- `enriched/<original_name>.jsonl` – final assets with refreshed `VA` values.
- `models/*.pkl` and `models/*_metrics.json` – reproducible VA regressors plus holdout metrics.
- `reports/*` – (optional) manually curated QA or comparison notes.

These artifacts can be zipped along with the README to share a complete, reproducible post-processing package with collaborators or evaluation organizers.
