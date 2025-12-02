#!/bin/bash

# ============================================
# 中等長度 Query Template 完整訓練腳本
# ============================================
# 用途：使用全部樣本訓練中等長度模板（無 max_train_samples 限制）
# 使用方法：./train_medium_query_full.sh

echo "=========================================="
echo "中等長度 Query Template 完整訓練"
echo "=========================================="
echo ""

# 設置環境變數：使用中等長度模板
export USE_OPTIMIZED_QUERY=medium

# 1. BERT-multilingual + Laptop
echo ""
echo ">>> [1/4] BERT-multilingual + Laptop (中等長度模板 - 完整數據集)"
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
  --model_name medium_query_bert_multilingual_lap_full \
  --prediction_name medium_query_bert_multilingual_lap_full_pred.jsonl

# 2. BERT-multilingual + Restaurant
echo ""
echo ">>> [2/4] BERT-multilingual + Restaurant (中等長度模板 - 完整數據集)"
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
  --model_name medium_query_bert_multilingual_res_full \
  --prediction_name medium_query_bert_multilingual_res_full_pred.jsonl

# 3. BERT-chinese + Laptop
echo ""
echo ">>> [3/4] BERT-chinese + Laptop (中等長度模板 - 完整數據集)"
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
  --model_name medium_query_bert_chinese_lap_full \
  --prediction_name medium_query_bert_chinese_lap_full_pred.jsonl

# 4. BERT-chinese + Restaurant
echo ""
echo ">>> [4/4] BERT-chinese + Restaurant (中等長度模板 - 完整數據集)"
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
  --model_name medium_query_bert_chinese_res_full \
  --prediction_name medium_query_bert_chinese_res_full_pred.jsonl

echo ""
echo "=========================================="
echo "中等長度模板完整訓練完成！"
echo "=========================================="
echo "輸出文件位置：subtask3/tasks/subtask_3/medium_query_*_full_pred.jsonl"
echo "日誌文件位置：subtask3/log/medium_query_*_full.log"
echo "模型檢查點：subtask3/model/medium_query_*_full.pth"
echo ""
echo "下一步："
echo "1. 檢查輸出 JSONL 文件是否正常生成"
echo "2. 上傳結果到 benchmark 評分"
echo "3. 對比原始、中等、完整優化三種版本的表現"

