# Valence/Arousal Regression (Phase 4)

Goal: predict VA scores separately from the main extractor by using retrieved exemplars and span embeddings.

## Planned Components
- `dataset_builder.py`: collates training samples (aspect/opinion spans, retrieved VA stats).
- `model.py`: simple regressors (baseline: GradientBoostingRegressor via scikit-learn, optional PyTorch MLP).
- `train.py`: trains VA models per domain and saves `.pkl` files.
- `predict.py`: loads trained regressor and fills VA field after aspect/category/opinion extraction.

## Feature Ideas
1. **Span embeddings**: average of token embeddings for the opinion span from the encoder.
2. **Retrieved stats**: mean/std valence & arousal from RetrievalService neighbors.
3. **Lexical cues**: sentiment lexicon scores or opinion text embeddings.

Implementation will start with span embedding + retrieved stats fed into GradientBoostingRegressor.

## Usage
1. Extract features and train models:
   ```bash
   docker exec -it nlp_dimabsa_subtask3 /bin/bash -lc \
     "python src/va_regression/train.py --data subtask3/data/zho_laptop_train_alltasks.jsonl \
        --output va_models/lap_va.pkl --rag_store rag/stores/lap"
   ```
2. Apply the trained regressor to predictions (after BERT/LLM extraction):
   ```bash
   docker exec -it nlp_dimabsa_subtask3 /bin/bash -lc \
     "python src/va_regression/predict.py --model va_models/lap_va.pkl \
        --input subtask3/tasks/subtask_3/medium_query_bert_multilingual_lap_full_pred.jsonl \
        --output subtask3/tasks/subtask_3/medium_query_bert_multilingual_lap_full_pred_va.jsonl"
   ```
3. Later, integrate `predict.py` call into the main pipeline so `Quadruplet` entries use regressed VA values before evaluation.
