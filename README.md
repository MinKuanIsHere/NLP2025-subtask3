# NLP2025-subtask3

NYCU NLP 2025 final project subtask 3

## Overview

- **Models**: `bert-base-multilingual-cased`, `bert-base-chinese`, & `hfl/chinese-roberta-wwm-ext-large`
- **Dataset**: Laptop & Restaurant in Chinese (zho) for Task 3
- **Task**: Dimensional Aspect-Based Sentiment Analysis (DimABSA) - Quadruplet Extraction

---

## Quick Start

### 1. Docker Environment Setup

```bash
# Clone repository
git clone https://github.com/MinKuanIsHere/NLP2025-subtask3.git
cd NLP2025-subtask3/subtask3

# Build and start Docker container
docker compose build --no-cache && docker compose --compatibility up -d

# Enter the container
docker exec -it nlp_dimabsa_subtask3 /bin/bash

# Verify GPU access (inside container)
nvidia-smi
```

### 2. Training Scripts

Inside the container, navigate to the training directory:

```bash
cd /workspace/starter_kit/task2task3
```

We provide **4 training scripts** that cover all combinations of query templates and data sizes:

#### Script 1: Original Query Templates - Quick Test (50 samples)

```bash
./train_original_query_quick.sh
```

- Uses **original simple query templates**
- Trains on **50 samples** per combination (quick validation)
- Configuration: `epoch_num=1`, `batch_size=1`, `max_train_samples=50`

**Output Files:**
- `tasks/subtask_3/original_query_bert_multilingual_lap_quick_pred.jsonl`
- `tasks/subtask_3/original_query_bert_multilingual_res_quick_pred.jsonl`
- `tasks/subtask_3/original_query_bert_chinese_lap_quick_pred.jsonl`
- `tasks/subtask_3/original_query_bert_chinese_res_quick_pred.jsonl`

#### Script 2: Original Query Templates - Full Data

```bash
./train_original_query_full.sh
```

- Uses **original simple query templates**
- Trains on **full dataset** (all training samples)
- Configuration: `epoch_num=1`, `batch_size=1`

**Output Files:**
- `tasks/subtask_3/original_query_bert_multilingual_lap_full_pred.jsonl`
- `tasks/subtask_3/original_query_bert_multilingual_res_full_pred.jsonl`
- `tasks/subtask_3/original_query_bert_chinese_lap_full_pred.jsonl`
- `tasks/subtask_3/original_query_bert_chinese_res_full_pred.jsonl`

#### Script 3: Optimized Query Templates - Quick Test (50 samples) ⭐ Recommended

```bash
./train_opt_query_quick.sh
```

- Uses **enhanced query templates** with detailed task descriptions
- Trains on **50 samples** per combination (quick validation)
- Configuration: `epoch_num=1`, `batch_size=1`, `max_train_samples=50`

**Output Files:**
- `tasks/subtask_3/opt_query_bert_multilingual_lap_quick_pred.jsonl`
- `tasks/subtask_3/opt_query_bert_multilingual_res_quick_pred.jsonl`
- `tasks/subtask_3/opt_query_bert_chinese_lap_quick_pred.jsonl`
- `tasks/subtask_3/opt_query_bert_chinese_res_quick_pred.jsonl`

#### Script 4: Optimized Query Templates - Full Data ⭐ Recommended

```bash
./train_opt_query_full.sh
```

- Uses **enhanced query templates** with detailed task descriptions
- Trains on **full dataset** (all training samples)
- Configuration: `epoch_num=1`, `batch_size=1`

**Output Files:**
- `tasks/subtask_3/opt_query_bert_multilingual_lap_full_pred.jsonl`
- `tasks/subtask_3/opt_query_bert_multilingual_res_full_pred.jsonl`
- `tasks/subtask_3/opt_query_bert_chinese_lap_full_pred.jsonl`
- `tasks/subtask_3/opt_query_bert_chinese_res_full_pred.jsonl`

**All scripts train 4 combinations:**
1. BERT-multilingual + Laptop
2. BERT-multilingual + Restaurant  
3. BERT-chinese + Laptop
4. BERT-chinese + Restaurant

---

## Query Template Switching Mechanism

The project supports **two versions** of query templates:

1. **Original Templates**: Simple, concise queries (default in original codebase)
2. **Optimized Templates**: Enhanced queries with detailed task descriptions and examples

### How It Works

The template selection is controlled by the environment variable `USE_OPTIMIZED_QUERY`:
- `USE_OPTIMIZED_QUERY=1` (or unset): Uses optimized templates
- `USE_OPTIMIZED_QUERY=0`: Uses original templates

The training scripts automatically set this variable:
- `train_opt_query_quick.sh` → Sets `USE_OPTIMIZED_QUERY=1`
- `train_opt_query_full.sh` → Sets `USE_OPTIMIZED_QUERY=1`
- `train_original_query_quick.sh` → Sets `USE_OPTIMIZED_QUERY=0`
- `train_original_query_full.sh` → Sets `USE_OPTIMIZED_QUERY=0`

### Manual Control

You can also manually control the template version:

```bash
# Use optimized templates
export USE_OPTIMIZED_QUERY=1
python run_task2\&3_trainer_multilingual.py ...

# Use original templates
export USE_OPTIMIZED_QUERY=0
python run_task2\&3_trainer_multilingual.py ...
```

---

## Optimized Query Templates

The **optimized query templates** use enhanced queries with detailed task descriptions and examples:

### Template Features

- **Forward Aspect**: Includes "ASPECT SPAN" definition and span extraction task description
- **Forward Opinion**: Includes "OPINION SPAN" definition with evaluative stance explanation
- **Backward Opinion**: Includes "OPINION SPANS" with sentiment/evaluation identification
- **Backward Aspect**: Includes "ASPECT SPAN" with entity linking description
- **Category**: Domain-specific examples (Laptop: `DISPLAY#QUALITY`, Restaurant: `FOOD#QUALITY` or `SERVICE#GENERAL`)
- **Valence**: Includes "VALENCE SCORE" with Pleasure/Displeasure dimension explanation (1-9 scale)
- **Arousal**: Includes "AROUSAL SCORE" with Activation/Deactivation dimension explanation (1-9 scale)

### Original Query Templates

The **original query templates** are simple and concise:

- **Forward Aspect**: `["[CLS]", "what", "aspects", "?", "[SEP]"]`
- **Forward Opinion**: `["[CLS]", "what", "opinion", "given", "the", "aspect", "?", "[SEP]"]`
- **Backward Opinion**: `["[CLS]", "what", "opinions", "?", "[SEP]"]`
- **Backward Aspect**: `["[CLS]", "what", "aspect", "does", "the", "opinion", "describe", "?", "[SEP]"]`
- **Category**: `["[CLS]", "what", "category", "given", "the", "aspect", "and", "the", "opinion", "?", "[SEP]"]`
- **Valence**: `["[CLS]", "what", "valence", "given", "the", "aspect", "and", "the", "opinion", "?", "[SEP]"]`
- **Arousal**: `["[CLS]", "what", "arousal", "given", "the", "aspect", "and", "the", "opinion", "?", "[SEP]"]`

### Token Length Considerations

- **Original templates**: ~5-10 tokens per query
- **Optimized templates**: ~30-45 tokens per query (adds ~30-40 tokens)
- Maximum token length is monitored during training (displayed as `Max length of training tokens`)
- BERT model limit: **512 tokens** (all current configurations are within safe range)

---

## BERT Large (RoBERTa) Training

To train with **`hfl/chinese-roberta-wwm-ext-large`** (1024 hidden dimensions), you need to add the `--hidden_size 1024` argument:

### Quick Test (50 samples) - Optimized Query Templates

```bash
cd /workspace/starter_kit/task2task3

# Laptop domain
export USE_OPTIMIZED_QUERY=1
CUDA_VISIBLE_DEVICES=0 python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain lap \
  --language zho \
  --train_data zho_laptop_train_alltasks.jsonl \
  --infer_data zho_laptop_dev_task3.jsonl \
  --bert_model_type hfl/chinese-roberta-wwm-ext-large \
  --hidden_size 1024 \
  --mode train \
  --gpu True \
  --epoch_num 1 \
  --batch_size 1 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --model_name opt_query_roberta_large_lap_quick \
  --prediction_name opt_query_roberta_large_lap_quick_pred.jsonl \
  --max_train_samples 50

# Restaurant domain
CUDA_VISIBLE_DEVICES=0 python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain res \
  --language zho \
  --train_data zho_restaurant_train_alltasks.jsonl \
  --infer_data zho_restaurant_dev_task3.jsonl \
  --bert_model_type hfl/chinese-roberta-wwm-ext-large \
  --hidden_size 1024 \
  --mode train \
  --gpu True \
  --epoch_num 1 \
  --batch_size 1 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --model_name opt_query_roberta_large_res_quick \
  --prediction_name opt_query_roberta_large_res_quick_pred.jsonl \
  --max_train_samples 50
```

### Full Data - Optimized Query Templates

```bash
cd /workspace/starter_kit/task2task3

# Laptop domain
export USE_OPTIMIZED_QUERY=1
CUDA_VISIBLE_DEVICES=0 python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain lap \
  --language zho \
  --train_data zho_laptop_train_alltasks.jsonl \
  --infer_data zho_laptop_dev_task3.jsonl \
  --bert_model_type hfl/chinese-roberta-wwm-ext-large \
  --hidden_size 1024 \
  --mode train \
  --gpu True \
  --epoch_num 20 \
  --batch_size 4 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --model_name opt_query_roberta_large_lap_full \
  --prediction_name opt_query_roberta_large_lap_full_pred.jsonl

# Restaurant domain
CUDA_VISIBLE_DEVICES=0 python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain res \
  --language zho \
  --train_data zho_restaurant_train_alltasks.jsonl \
  --infer_data zho_restaurant_dev_task3.jsonl \
  --bert_model_type hfl/chinese-roberta-wwm-ext-large \
  --hidden_size 1024 \
  --mode train \
  --gpu True \
  --epoch_num 20 \
  --batch_size 4 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --model_name opt_query_roberta_large_res_full \
  --prediction_name opt_query_roberta_large_res_full_pred.jsonl
```

### Quick Test (50 samples) - Original Query Templates

```bash
cd /workspace/starter_kit/task2task3

# Laptop domain
export USE_OPTIMIZED_QUERY=0
CUDA_VISIBLE_DEVICES=0 python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain lap \
  --language zho \
  --train_data zho_laptop_train_alltasks.jsonl \
  --infer_data zho_laptop_dev_task3.jsonl \
  --bert_model_type hfl/chinese-roberta-wwm-ext-large \
  --hidden_size 1024 \
  --mode train \
  --gpu True \
  --epoch_num 1 \
  --batch_size 1 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --model_name original_query_roberta_large_lap_quick \
  --prediction_name original_query_roberta_large_lap_quick_pred.jsonl \
  --max_train_samples 50

# Restaurant domain
CUDA_VISIBLE_DEVICES=0 python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain res \
  --language zho \
  --train_data zho_restaurant_train_alltasks.jsonl \
  --infer_data zho_restaurant_dev_task3.jsonl \
  --bert_model_type hfl/chinese-roberta-wwm-ext-large \
  --hidden_size 1024 \
  --mode train \
  --gpu True \
  --epoch_num 1 \
  --batch_size 1 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --model_name original_query_roberta_large_res_quick \
  --prediction_name original_query_roberta_large_res_quick_pred.jsonl \
  --max_train_samples 50
```

### Full Data - Original Query Templates

```bash
cd /workspace/starter_kit/task2task3

# Laptop domain
export USE_OPTIMIZED_QUERY=0
CUDA_VISIBLE_DEVICES=0 python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain lap \
  --language zho \
  --train_data zho_laptop_train_alltasks.jsonl \
  --infer_data zho_laptop_dev_task3.jsonl \
  --bert_model_type hfl/chinese-roberta-wwm-ext-large \
  --hidden_size 1024 \
  --mode train \
  --gpu True \
  --epoch_num 20 \
  --batch_size 4 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --model_name original_query_roberta_large_lap_full \
  --prediction_name original_query_roberta_large_lap_full_pred.jsonl

# Restaurant domain
CUDA_VISIBLE_DEVICES=0 python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain res \
  --language zho \
  --train_data zho_restaurant_train_alltasks.jsonl \
  --infer_data zho_restaurant_dev_task3.jsonl \
  --bert_model_type hfl/chinese-roberta-wwm-ext-large \
  --hidden_size 1024 \
  --mode train \
  --gpu True \
  --epoch_num 20 \
  --batch_size 4 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --model_name original_query_roberta_large_res_full \
  --prediction_name original_query_roberta_large_res_full_pred.jsonl
```

**Note:** The `--hidden_size 1024` argument is required for RoBERTa-large models, as they use 1024-dimensional hidden states instead of the default 768.

---

## Manual Training Examples

### BERT-multilingual + Laptop

```bash
cd /workspace/starter_kit/task2task3

CUDA_VISIBLE_DEVICES=0 python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain lap \
  --language zho \
  --train_data zho_laptop_train_alltasks.jsonl \
  --infer_data zho_laptop_dev_task3.jsonl \
  --bert_model_type bert-base-multilingual-cased \
  --mode train \
  --gpu True \
  --epoch_num 1 \
  --batch_size 1 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --model_name zho_lap_subtask3 \
  --prediction_name zho_lap_subtask3_dev.jsonl
```

### BERT-chinese + Restaurant

```bash
cd /workspace/starter_kit/task2task3

CUDA_VISIBLE_DEVICES=0 python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain res \
  --language zho \
  --train_data zho_restaurant_train_alltasks.jsonl \
  --infer_data zho_restaurant_dev_task3.jsonl \
  --bert_model_type bert-base-chinese \
  --mode train \
  --gpu True \
  --epoch_num 1 \
  --batch_size 1 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --model_name res_zho_subtask3_bertzh \
  --prediction_name res_zho_subtask3_bertzh_dev.jsonl
```

---

## Output Files

### Prediction Files
- Location: `tasks/subtask_3/{prediction_name}.jsonl`
- Format: JSONL (one JSON object per line)
- Structure:
```json
{
  "ID": "6724235:S003",
  "Quadruplet": [
    {
      "Aspect": "規格",
      "Category": "LAPTOP#DESIGN_FEATURES",
      "Opinion": "不錯",
      "VA": "6.00#5.00"
    }
  ]
}
```

### Model Checkpoints
- Location: `model/{model_name}.pth`
- Contains: Model weights and optimizer state

### Training Logs
- Location: `log/{model_name}.log`
- Contains: Training losses, evaluation metrics, and token length statistics

---

## Inference Mode

To run inference with a trained model:

```bash
cd /workspace/starter_kit/task2task3

CUDA_VISIBLE_DEVICES=0 python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain lap \
  --language zho \
  --train_data zho_laptop_train_alltasks.jsonl \
  --infer_data zho_laptop_dev_task3.jsonl \
  --bert_model_type bert-base-multilingual-cased \
  --mode inference \
  --gpu True \
  --inference_beta 0.9 \
  --model_name zho_lap_subtask3 \
  --reload True
```

---

## Evaluation

Run the official evaluation script:

```bash
python evaluation_script/metrics_subtask_1_2_3.py \
  -t 3 \
  -p tasks/subtask_3/pred_zho_laptop.jsonl \
  -g evaluation_script/sample\ data/subtask_3/zho/gold_zho_laptop.jsonl
```

---

## Key CLI Arguments

- `--task <int>`: Task type (2 or 3)
- `--domain <str>`: Dataset domain (`res` | `lap` | `hot` | `fin`)
- `--language <str>`: Dataset language (`eng` | `zho` | `jpn`)
- `--bert_model_type <str>`: Pretrained BERT model name
- `--mode <str>`: Operation mode (`train` | `evaluate` | `inference`)
- `--epoch_num <int>`: Number of training epochs (default: 3)
- `--batch_size <int>`: Training batch size (default: 4)
- `--max_train_samples <int>`: Limit training samples (for quick testing)
- `--model_name <str>`: Controls checkpoint and log file names
- `--prediction_name <str>`: Custom prediction output filename
- `--hidden_size <int>`: Hidden dimension size (required for RoBERTa-large: `1024`, default for BERT-base: `768`)

---

## PyTorch 2.6 Upgrade

This project uses **PyTorch 2.6.0** with **torchvision 0.21.0** (installed from CUDA 12.1 wheel index). This upgrade:
- Satisfies upstream security requirements ([CVE-2025-32434](https://nvd.nist.gov/vuln/detail/CVE-2025-32434))
- Enables `hfl/chinese-roberta-wwm-ext-large` (1024-dim; scripts auto-pass `--hidden_size 1024`)
- Pins compatible torchvision version to prevent runtime errors

After rebuilding the Docker image with `docker compose build --no-cache`, all models—including RoBERTa-large—run by default.

---

## Notes

1. **Large Models**: Large models run by default. To skip them, set `ENABLE_LARGE_MODELS=0` before running scripts.

2. **GPU Selection**: Adjust `CUDA_VISIBLE_DEVICES` to control which GPU is used.

3. **Quick Testing**: Using `--max_train_samples 50` will result in F1=0.0, which is normal due to insufficient samples.

4. **Token Length**: Monitor `Max length of training tokens` output during training to ensure it stays below 512.

5. **Domain-Specific Templates**: Category templates automatically adapt based on `--domain`:
   - Laptop: Uses `DISPLAY#QUALITY` example
   - Restaurant: Uses `FOOD#QUALITY` or `SERVICE#GENERAL` examples
