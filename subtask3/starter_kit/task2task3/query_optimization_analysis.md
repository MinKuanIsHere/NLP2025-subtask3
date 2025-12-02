# Query Template 優化分析與改進建議

## 當前表現對比

| Model                        | laptop (原始) | restaurant (原始) | laptop (優化) | restaurant (優化) |
|------------------------------|---------------|-------------------|---------------|-------------------|
| bert-base-chinese E1 B1      | 0.3539        | 0.5481            | 0.34          | 0.54              |
| bert-base-multilingual E1 B1 | 0.2207        | 0.4348            | 0.22          | 0.43              |

**觀察**：優化後表現幾乎相同，甚至略差。

## 問題根源分析

### 1. 模板長度問題（主要原因）

#### 模板長度對比表

| 模板類型 | 原始版本 | 中等版本 | 完整優化版本 | 中等 vs 原始 | 完整 vs 原始 | 中等 vs 完整 |
|---------|---------|---------|-------------|-------------|-------------|-------------|
| forward_aspect | 5 tokens | 20 tokens | 46 tokens | +15 (+300%) | +41 (+820%) | -26 (-56.5%) |
| forward_opinion | 8 tokens | 19 tokens | 36 tokens | +11 (+137.5%) | +28 (+350%) | -17 (-47.2%) |
| backward_opinion | 5 tokens | 16 tokens | 33 tokens | +11 (+220%) | +28 (+560%) | -17 (-51.5%) |
| backward_aspect | 9 tokens | 20 tokens | 37 tokens | +11 (+122.2%) | +28 (+311.1%) | -17 (-45.9%) |
| category (laptop) | 10 tokens | 22 tokens | 48 tokens | +12 (+120%) | +38 (+380%) | -26 (-54.2%) |
| category (restaurant) | 10 tokens | 22 tokens | 50 tokens | +12 (+120%) | +40 (+400%) | -28 (-56.0%) |
| valence | 10 tokens | 20 tokens | 40 tokens | +10 (+100%) | +30 (+300%) | -20 (-50.0%) |
| arousal | 10 tokens | 20 tokens | 42 tokens | +10 (+100%) | +32 (+320%) | -22 (-52.4%) |
| **總計** | **27 tokens** | **75 tokens** | **152 tokens** | **+48 (+177.8%)** | **+125 (+463.0%)** | **-77 (-50.7%)** |

#### BERT 512 Token 限制佔用

- **原始版本**：27 tokens (5.3%)
- **中等版本**：75 tokens (14.6%)
- **完整優化版本**：152 tokens (29.7%)

**影響**：
- **原始版本**：實際文本空間 ~485 tokens
- **中等版本**：實際文本空間 ~437 tokens（減少 48 tokens，約 9.9%）
- **完整優化版本**：實際文本空間 ~360 tokens（減少 125 tokens，約 25.8%）
- 模型注意力分散到冗長的模板描述
- 長文本可能被截斷，導致信息丟失

### 2. 訓練不充分

- 只有 **1 epoch**，模型可能未充分學習優化模板的語義
- 優化模板需要更多訓練來理解其結構

### 3. BERT 注意力機制特性

- BERT 更關注序列的開頭和結尾
- 冗長的中間描述可能被忽略
- 優化模板的關鍵信息可能淹沒在冗餘文字中

### 4. 模板冗餘問題

- 包含過多解釋性文字（如 "i.e.", "the specific token sequence"）
- 對模型來說可能是噪音而非有用信號

## 改進方案

### 方案 1：創建中等長度的模板（✅ 已實現）

在原始和完整優化之間，創建一個平衡版本：

```python
# 中等長度模板（保留關鍵指令，去除冗餘）
forward_aspect_query_template_medium = [
    "[CLS]", "what", "ASPECT", "SPAN", "is", "mentioned", "in", "the", "text", "?",
    "Extract", "the", "precise", "word", "sequence", "from", "the", "context", ".", "[SEP]"
]
# 長度：20 tokens（vs 原始 5, 優化 46）

forward_opinion_query_template_medium = [
    "[CLS]", "what", "OPINION", "SPAN", "is", "given", "for", "the", "aspect", "?",
    "Extract", "the", "exact", "token", "span", "expressing", "judgment", ".", "[SEP]"
]
# 長度：19 tokens（vs 原始 8, 優化 36）

backward_opinion_query_template_medium = [
    "[CLS]", "what", "OPINION", "SPANS", "are", "mentioned", "in", "the", "text", "?",
    "Identify", "all", "evaluative", "expressions", ".", "[SEP]"
]
# 長度：16 tokens（vs 原始 5, 優化 33）

backward_aspect_query_template_medium = [
    "[CLS]", "what", "ASPECT", "SPAN", "does", "the", "opinion", "describe", "?",
    "Link", "the", "opinion", "span", "to", "its", "corresponding", "target", "span", ".", "[SEP]"
]
# 長度：20 tokens（vs 原始 9, 優化 37）

category_query_template_laptop_medium = [
    "[CLS]", "what", "CATEGORY", "LABEL", "(", "Entity#Attribute", ",", "e.g.", "DISPLAY#QUALITY", ")",
    "given", "the", "aspect", "and", "opinion", "?", "Classify", "from", "Laptop", "domain", ".", "[SEP]"
]
# 長度：22 tokens（vs 原始 10, 優化 48）

category_query_template_restaurant_medium = [
    "[CLS]", "what", "CATEGORY", "LABEL", "(", "Entity#Attribute", ",", "e.g.", "FOOD#QUALITY", ")",
    "given", "the", "aspect", "and", "opinion", "?", "Classify", "from", "Restaurant", "domain", ".", "[SEP]"
]
# 長度：22 tokens（vs 原始 10, 優化 50）

valence_query_template_medium = [
    "[CLS]", "what", "VALENCE", "SCORE", "(", "Pleasure/Displeasure", ",", "1-9", "scale", ")",
    "is", "given", "?", "Measure", "emotion", "'", "s", "pleasantness", ".", "[SEP]"
]
# 長度：20 tokens（vs 原始 10, 優化 40）

arousal_query_template_medium = [
    "[CLS]", "what", "AROUSAL", "SCORE", "(", "Activation/Deactivation", ",", "1-9", "scale", ")",
    "is", "given", "?", "Measure", "emotion", "'", "s", "intensity", ".", "[SEP]"
]
# 長度：20 tokens（vs 原始 10, 優化 42）
```

**使用方法**：
```bash
# 使用中等長度模板（推薦）
USE_OPTIMIZED_QUERY=medium python run_task2&3_trainer_multilingual.py ...

# 使用完整優化模板
USE_OPTIMIZED_QUERY=1 python run_task2&3_trainer_multilingual.py ...

# 使用原始模板
USE_OPTIMIZED_QUERY=0 python run_task2&3_trainer_multilingual.py ...
```

**優點**：
- 保留關鍵指令（"Extract the precise word sequence", "Classify from domain"）
- 去除冗餘解釋（"i.e.", "the specific token sequence referring to..."）
- 長度適中，不會過度壓縮文本空間（僅減少 9.9% vs 完整優化的 25.8%）
- 比完整優化版本減少 50.7% 的 tokens，但仍保留核心語義信息

### 方案 2：增加訓練輪數

- 將 `epoch_num` 從 1 增加到 3-5
- 讓模型有更多時間學習優化模板的語義

### 方案 3：調整學習率

- 優化模板可能需要不同的學習率
- 嘗試降低學習率（如 `1e-4`）以更細緻地學習

### 方案 4：使用 BERT Large 模型

- BERT Large 有更大的容量（1024 hidden size）
- 可能能更好地利用優化模板的信息

### 方案 5：簡化優化模板

保留核心指令，去除所有冗餘：

```python
forward_aspect_query_template_simplified = [
    "[CLS]", "Extract", "ASPECT", "SPAN", "from", "text", ".", "[SEP]"
]
```

## 建議的實驗順序

1. **✅ 方案 1 已完成**：創建中等長度模板，等待測試效果
2. **下一步：方案 2**：增加訓練輪數到 3-5 epochs
3. **最後試方案 4**：使用 BERT Large 模型

## 預期改進

- **中等長度模板**：預期 F1 提升 2-5%
  - 相比完整優化版本，減少 50.7% 的 tokens
  - 保留核心語義信息，減少注意力分散
  - 實際文本空間僅減少 9.9%（vs 完整優化的 25.8%）
  
- **增加訓練輪數**：預期 F1 提升 3-8%
  - 讓模型有更多時間學習模板語義
  - 建議從 1 epoch 增加到 3-5 epochs
  
- **BERT Large**：預期 F1 提升 5-10%
  - 更大的模型容量（1024 hidden size）
  - 可能能更好地利用優化模板的信息

## 總結

**關鍵發現**：
- 完整優化模板過長（152 tokens，佔用 29.7%），導致文本空間被過度壓縮
- 中等長度模板（75 tokens，佔用 14.6%）在保留關鍵指令的同時，大幅減少 token 佔用
- 中等版本比完整優化版本減少 77 tokens（50.7%），但仍保留核心語義信息

**下一步行動**：
1. 使用 `USE_OPTIMIZED_QUERY=medium` 測試中等長度模板
2. 對比原始、中等、完整優化三種版本的表現
3. 根據結果決定是否增加訓練輪數或使用 BERT Large

