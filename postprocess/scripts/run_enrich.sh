#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
DATA_DIR="$PROJECT_ROOT/data"
CLEAN_DIR="$PROJECT_ROOT/clean"
ENRICHED_DIR="$PROJECT_ROOT/enriched"
PYTHON_BIN=${PYTHON_BIN:-python3}

mkdir -p "$CLEAN_DIR" "$ENRICHED_DIR"

echo "[run_enrich] Cleaning raw JSONLs under $DATA_DIR ..."
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/clean_json.py" \
  --input "$DATA_DIR" \
  --output-dir "$CLEAN_DIR" \
  --stats "$CLEAN_DIR/cleanup_stats.csv"

for input in "$CLEAN_DIR"/*.jsonl; do
  [[ -e "$input" ]] || continue
  file=$(basename "$input")
  if [[ "$file" == *"lap"* ]]; then
    domain="lap"
  else
    domain="res"
  fi
  output="$ENRICHED_DIR/$file"
  echo "[run_enrich] Domain=$domain Input=$file -> $output"
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/enrich_va.py" \
    --domain "$domain" \
    --input "$input" \
    --output "$output"
done

echo "[run_enrich] Enriched files available in $ENRICHED_DIR"
