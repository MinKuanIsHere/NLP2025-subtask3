#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${1:-subtask3/data/zho_laptop_train_alltasks.jsonl}
SAMPLE_COUNT=${2:-20}
OUTPUT_PATH=${3:-/tmp/sample.jsonl}

if [[ ! -f "$DATA_PATH" ]]; then
  echo "[ERROR] Data file not found: $DATA_PATH" >&2
  exit 1
fi

head -n "$SAMPLE_COUNT" "$DATA_PATH" > "$OUTPUT_PATH"

python3 - "$OUTPUT_PATH" <<'PY'
import json, sys, pathlib
path = pathlib.Path(sys.argv[1])
errors = []
count = 0
with path.open(encoding='utf-8') as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
            count += 1
        except Exception as exc:
            errors.append((line_no, str(exc)))

if errors:
    for lineno, err in errors:
        print(f"[INVALID] line {lineno}: {err}")
    sys.exit(1)
print(f"Validated {count} JSONL entries from {path}")
PY

echo "Sample saved to $OUTPUT_PATH"
