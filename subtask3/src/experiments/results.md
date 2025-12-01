# Experiment Tracking

填寫 benchmark 分數與提交狀態。每次跑完實驗（BERT/LLM ± RAG）後，記錄日期、設定、benchmark F1、筆記。

| Model & Setting | Domain | Epoch | Batch | RAG | Benchmark F1 | Notes |
|-----------------|--------|-------|-------|-----|--------------|-------|
| bert-base-multilingual-cased | laptop | 1 | 1 | No |  |  |
| bert-base-multilingual-cased | laptop | 1 | 2 | No |  |  |
| bert-base-multilingual-cased | laptop | 1 | 1 | Yes |  |  |
| bert-base-multilingual-cased | laptop | 1 | 2 | Yes |  |  |
| bert-base-multilingual-cased | restaurant | 1 | 1 | No |  |  |
| bert-base-multilingual-cased | restaurant | 1 | 2 | No |  |  |
| bert-base-multilingual-cased | restaurant | 1 | 1 | Yes |  |  |
| bert-base-multilingual-cased | restaurant | 1 | 2 | Yes |  |  |
| bert-base-chinese | laptop | 1 | 1 | No |  |  |
| bert-base-chinese | laptop | 1 | 2 | No |  |  |
| bert-base-chinese | laptop | 1 | 1 | Yes |  |  |
| bert-base-chinese | laptop | 1 | 2 | Yes |  |  |
| bert-base-chinese | restaurant | 1 | 1 | No |  |  |
| bert-base-chinese | restaurant | 1 | 2 | No |  |  |
| bert-base-chinese | restaurant | 1 | 1 | Yes |  |  |
| bert-base-chinese | restaurant | 1 | 2 | Yes |  |  |
| chinese-roberta-wwm-ext-large | laptop | 1 | 1 | No |  |  |
| chinese-roberta-wwm-ext-large | laptop | 1 | 2 | No |  |  |
| chinese-roberta-wwm-ext-large | laptop | 1 | 1 | Yes |  |  |
| chinese-roberta-wwm-ext-large | laptop | 1 | 2 | Yes |  |  |
| chinese-roberta-wwm-ext-large | restaurant | 1 | 1 | No |  |  |
| chinese-roberta-wwm-ext-large | restaurant | 1 | 2 | No |  |  |
| chinese-roberta-wwm-ext-large | restaurant | 1 | 1 | Yes |  |  |
| chinese-roberta-wwm-ext-large | restaurant | 1 | 2 | Yes |  |  |
| openai/gpt-5.1 | laptop | - | - | No |  |  |
| openai/gpt-5.1 | laptop | - | - | Yes |  |  |
| openai/gpt-5.1 | restaurant | - | - | No |  |  |
| openai/gpt-5.1 | restaurant | - | - | Yes |  |  |
| google/gemini-3-pro-preview | laptop | - | - | No |  |  |
| google/gemini-3-pro-preview | laptop | - | - | Yes |  |  |
| google/gemini-3-pro-preview | restaurant | - | - | No |  |  |
| google/gemini-3-pro-preview | restaurant | - | - | Yes |  |  |
| meta-llama/llama-3.3-70b-instruct | laptop | - | - | No |  |  |
| meta-llama/llama-3.3-70b-instruct | laptop | - | - | Yes |  |  |
| meta-llama/llama-3.3-70b-instruct | restaurant | - | - | No |  |  |
| meta-llama/llama-3.3-70b-instruct | restaurant | - | - | Yes |  |  |

此外請於跑分時保留輸出 JSONL 路徑與 benchmark submission ID，方便回溯。後續也可把 VA regressor 的 RMSE/皮爾森結果補在同檔案或另開章節。
