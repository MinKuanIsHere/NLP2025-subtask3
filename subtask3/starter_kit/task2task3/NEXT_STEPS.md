# 下一步行動指南

## 🎯 目標
測試中等長度模板的效果，並與原始版本和完整優化版本進行對比。

## 📋 步驟 1：快速測試中等長度模板（推薦先做）

### 1.1 執行快速測試腳本

```bash
cd /home/minkuanchen/nycu/NLP2025-subtask3/subtask3/starter_kit/task2task3
./train_medium_query_quick.sh
```

**預期時間**：約 10-20 分鐘（取決於 GPU）

**輸出文件**：
- `subtask3/tasks/subtask_3/medium_query_*_quick_pred.jsonl` (4 個文件)
- `subtask3/log/medium_query_*.log` (4 個日誌文件)

### 1.2 檢查輸出

```bash
# 檢查文件是否生成
ls -lh subtask3/tasks/subtask_3/medium_query_*_quick_pred.jsonl

# 查看其中一個文件的內容（檢查格式是否正確）
head -5 subtask3/tasks/subtask_3/medium_query_bert_multilingual_lap_quick_pred.jsonl

# 檢查是否有錯誤
grep -i "error\|exception\|traceback" subtask3/log/medium_query_*.log
```

### 1.3 驗證模板是否正確使用

```bash
# 檢查日誌中是否有模板長度信息
grep -i "max length\|template\|token" subtask3/log/medium_query_bert_multilingual_lap_quick.log | head -10
```

**預期**：應該看到中等長度模板的 token 長度（約 75 tokens 總長度）

## 📊 步驟 2：對比三種版本的表現

### 2.1 整理現有結果

你已經有：
- ✅ **原始版本**：`train_original_query_quick.sh` 的結果
- ✅ **完整優化版本**：`train_opt_query_quick.sh` 的結果
- 🔄 **中等長度版本**：`train_medium_query_quick.sh` 的結果（待測試）

### 2.2 上傳到 Benchmark 評分

將中等長度模板的結果上傳到 benchmark，獲得 F1 分數：

| 版本 | laptop (multilingual) | restaurant (multilingual) | laptop (chinese) | restaurant (chinese) |
|------|----------------------|--------------------------|------------------|---------------------|
| 原始 | 0.2207 | 0.4348 | 0.3539 | 0.5481 |
| 完整優化 | 0.22 | 0.43 | 0.34 | 0.54 |
| **中等長度** | **待測試** | **待測試** | **待測試** | **待測試** |

### 2.3 分析結果

比較三種版本的表現：
- 如果中等長度 > 完整優化：說明模板長度確實是問題
- 如果中等長度 ≈ 原始：可能需要更多訓練輪數
- 如果中等長度 > 原始：說明優化有效，但需要平衡長度

## 🚀 步驟 3：根據結果決定下一步

### 情況 A：中等長度模板表現良好（F1 提升 > 2%）

**下一步**：進行完整數據集訓練

```bash
# 創建完整訓練腳本（需要我幫你創建嗎？）
# 或者手動執行：
export USE_OPTIMIZED_QUERY=medium
python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain lap \
  --language zho \
  --train_data zho_laptop_train_alltasks.jsonl \
  --infer_data zho_laptop_dev_task3.jsonl \
  --bert_model_type bert-base-multilingual-cased \
  --mode train \
  --gpu True \
  --epoch_num 3 \
  --batch_size 1 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --model_name medium_query_bert_multilingual_lap_full \
  --prediction_name medium_query_bert_multilingual_lap_full_pred.jsonl
```

### 情況 B：中等長度模板表現一般（F1 提升 < 2%）

**下一步**：增加訓練輪數

```bash
# 使用 3-5 epochs 重新訓練
export USE_OPTIMIZED_QUERY=medium
python run_task2\&3_trainer_multilingual.py \
  --epoch_num 3 \  # 或 5
  # ... 其他參數相同
```

### 情況 C：中等長度模板表現不佳（F1 下降）

**下一步**：
1. 檢查是否有錯誤（查看日誌）
2. 嘗試調整學習率
3. 考慮使用 BERT Large 模型

## 🔍 步驟 4：深入分析（可選）

### 4.1 檢查 token 長度

```bash
# 在 Python 中檢查實際使用的模板長度
python3 << 'EOF'
import os
os.environ['USE_OPTIMIZED_QUERY'] = 'medium'
from DataProcess import (
    forward_aspect_query_template,
    forward_opinion_query_template,
    backward_opinion_query_template,
    backward_aspect_query_template
)

print("中等長度模板實際長度：")
print(f"forward_aspect: {len(forward_aspect_query_template)} tokens")
print(f"forward_opinion: {len(forward_opinion_query_template)} tokens")
print(f"backward_opinion: {len(backward_opinion_query_template)} tokens")
print(f"backward_aspect: {len(backward_aspect_query_template)} tokens")
EOF
```

### 4.2 檢查預測質量

```bash
# 檢查預測結果中的 "null" 數量
grep -o '"Aspect": "null"' subtask3/tasks/subtask_3/medium_query_*_quick_pred.jsonl | wc -l

# 檢查空的 Quadruplet 數量
grep -o '"Quadruplet": \[\]' subtask3/tasks/subtask_3/medium_query_*_quick_pred.jsonl | wc -l
```

## 📝 快速檢查清單

- [ ] 執行 `./train_medium_query_quick.sh`
- [ ] 確認輸出文件正常生成
- [ ] 檢查日誌文件無錯誤
- [ ] 上傳結果到 benchmark 評分
- [ ] 對比三種版本的 F1 分數
- [ ] 根據結果決定下一步行動

## 💡 提示

1. **先做快速測試**：用 50 個樣本快速驗證，避免浪費時間
2. **記錄結果**：將 F1 分數記錄在 `query_optimization_analysis.md` 中
3. **逐步優化**：不要一次改變太多參數，這樣才能知道哪個改進有效
4. **保存模型**：如果結果好，記得保存模型檢查點

## ❓ 需要幫助？

如果遇到問題：
1. 檢查日誌文件中的錯誤信息
2. 確認環境變數 `USE_OPTIMIZED_QUERY=medium` 是否正確設置
3. 確認代碼已正確更新（`DataProcess.py` 中的中等長度模板）

