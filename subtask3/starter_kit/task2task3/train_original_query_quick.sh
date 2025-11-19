#!/usr/bin/env bash
set -euo pipefail

# 設置使用原始版本的 query template
export USE_OPTIMIZED_QUERY=0

echo "=========================================="
echo "Training with Original Query Templates (Quick Test - 50 samples)"
echo "=========================================="

# 1. BERT-multilingual + Laptop
echo ""
echo ">>> [1/4] BERT-multilingual + Laptop"
echo "=========================================="
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
  --model_name original_query_bert_multilingual_lap_quick \
  --prediction_name original_query_bert_multilingual_lap_quick_pred.jsonl \
  --max_train_samples 50

# 2. BERT-multilingual + Restaurant
echo ""
echo ">>> [2/4] BERT-multilingual + Restaurant"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain res \
  --language zho \
  --train_data zho_restaurant_train_alltasks.jsonl \
  --infer_data zho_restaurant_dev_task3.jsonl \
  --bert_model_type bert-base-multilingual-cased \
  --mode train \
  --gpu True \
  --epoch_num 1 \
  --batch_size 1 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --model_name original_query_bert_multilingual_res_quick \
  --prediction_name original_query_bert_multilingual_res_quick_pred.jsonl \
  --max_train_samples 50

# 3. BERT-chinese + Laptop
echo ""
echo ">>> [3/4] BERT-chinese + Laptop"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain lap \
  --language zho \
  --train_data zho_laptop_train_alltasks.jsonl \
  --infer_data zho_laptop_dev_task3.jsonl \
  --bert_model_type bert-base-chinese \
  --mode train \
  --gpu True \
  --epoch_num 1 \
  --batch_size 1 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --model_name original_query_bert_chinese_lap_quick \
  --prediction_name original_query_bert_chinese_lap_quick_pred.jsonl \
  --max_train_samples 50

# 4. BERT-chinese + Restaurant
echo ""
echo ">>> [4/4] BERT-chinese + Restaurant"
echo "=========================================="
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
  --model_name original_query_bert_chinese_res_quick \
  --prediction_name original_query_bert_chinese_res_quick_pred.jsonl \
  --max_train_samples 50

echo ""
echo "=========================================="
echo "All training completed!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - tasks/subtask_3/original_query_bert_multilingual_lap_quick_pred.jsonl"
echo "  - tasks/subtask_3/original_query_bert_multilingual_res_quick_pred.jsonl"
echo "  - tasks/subtask_3/original_query_bert_chinese_lap_quick_pred.jsonl"
echo "  - tasks/subtask_3/original_query_bert_chinese_res_quick_pred.jsonl"
echo ""

