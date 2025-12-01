#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-all}
CONTAINER=${CONTAINER:-nlp_dimabsa_subtask3}
WORKDIR=/workspace
TRAINER_DIR=$WORKDIR/starter_kit/task2task3
TASKS_DIR=$WORKDIR/tasks/subtask_3
DATA_DIR=$WORKDIR/subtask3/data
RAG_STORE_ROOT=$WORKDIR/rag/stores

declare -A TRAIN_FILE=(
  ["lap"]="zho_laptop_train_alltasks.jsonl"
  ["res"]="zho_restaurant_train_alltasks.jsonl"
)
declare -A INFER_FILE=(
  ["lap"]="zho_laptop_dev_task3.jsonl"
  ["res"]="zho_restaurant_dev_task3.jsonl"
)
declare -A PRED_FILE=(
  ["lap"]="pred_zho_laptop.jsonl"
  ["res"]="pred_zho_restaurant.jsonl"
)

readarray -t BERT_MODELS <<'EOF'
bert-base-multilingual-cased|bert_multi
bert-base-chinese|bert_chinese
hfl/chinese-roberta-wwm-ext-large|chinese_roberta_large
EOF
BATCH_SIZES=("1" "2")
DOMAINS=("lap" "res")
RAG_FLAGS=("no" "yes")

readarray -t LLM_MODELS <<'EOF'
openai/gpt-5.1|gpt51
google/gemini-3-pro-preview|gemini3pro
meta-llama/llama-3.3-70b-instruct|llama33
EOF

run_in_container() {
  docker exec "$CONTAINER" /bin/bash -lc "$1"
}

run_bert() {
  for model_entry in "${BERT_MODELS[@]}"; do
    IFS='|' read -r model_name model_tag <<<"$model_entry"
    for domain in "${DOMAINS[@]}"; do
      for batch in "${BATCH_SIZES[@]}"; do
        for rag in "${RAG_FLAGS[@]}"; do
          run_tag="${model_tag}_${domain}_b${batch}_e1_${rag}"
          echo "===== BERT RUN: $run_tag ====="
          cmd="cd $TRAINER_DIR && python run_task2\\&3_trainer_multilingual.py \
            --task 3 --domain ${domain} --language zho \
            --train_data ${TRAIN_FILE[$domain]} \
            --infer_data ${INFER_FILE[$domain]} \
            --bert_model_type ${model_name} \
            --mode train \
            --epoch_num 1 \
            --batch_size ${batch} \
            --learning_rate 1e-3 \
            --tuning_bert_rate 1e-5 \
            --model_name ${run_tag}"
          if [[ "$rag" == "yes" ]]; then
            cmd+=" --use_rag --rag_store_root $RAG_STORE_ROOT"
          fi
          run_in_container "$cmd"
          copy_cmd="cp $TASKS_DIR/${PRED_FILE[$domain]} $TASKS_DIR/${run_tag}.jsonl"
          run_in_container "$copy_cmd"
        done
      done
    done
  done
}

run_llm() {
  for model_entry in "${LLM_MODELS[@]}"; do
    IFS='|' read -r model_name model_tag <<<"$model_entry"
    for domain in "${DOMAINS[@]}"; do
      for rag in "${RAG_FLAGS[@]}"; do
        run_tag="llm_${model_tag}_${domain}_${rag}"
        echo "===== LLM RUN: $run_tag ====="
        cmd="cd $WORKDIR; python src/llm_clients/runner.py \
          --data subtask3/data/${INFER_FILE[$domain]} \
          --output tasks/subtask_3/${run_tag}.jsonl \
          --domain ${domain} \
          --model ${model_name}"
        if [[ "$rag" == "yes" ]]; then
          cmd+=" --use_rag --rag_store $RAG_STORE_ROOT/${domain}"
        fi
        run_in_container "$cmd"
      done
    done
  done
}

case "$MODE" in
  bert) run_bert ;;
  llm) run_llm ;;
  all) run_bert; run_llm ;;
  *) echo "Usage: $0 [all|bert|llm]" >&2; exit 1 ;;
esac
