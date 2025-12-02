#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
for path in (PROJECT_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

try:
    from postpreprocess.cleaning.json_cleaner import load_and_clean_jsonl
except ModuleNotFoundError:
    from cleaning.json_cleaner import load_and_clean_jsonl


def iter_input_files(input_path: Path):
    if input_path.is_file():
        yield input_path
        return
    for path in sorted(input_path.glob("*.jsonl")):
        if path.is_file():
            yield path


def main():
    parser = argparse.ArgumentParser(description="Clean legacy prediction JSONLs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("postpreprocess/data"),
        help="Path to raw JSONL file or directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("postpreprocess/clean"),
        help="Directory to write cleaned JSONLs.",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("postpreprocess/clean/cleanup_stats.csv"),
        help="CSV file to append per-file stats.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)

    stats_rows = []
    for json_path in iter_input_files(args.input):
        cleaned = load_and_clean_jsonl(json_path)
        total = sum(1 for _ in open(json_path, encoding="utf-8") if _.strip())
        kept = len(cleaned)
        dropped = total - kept
        output_path = args.output_dir / json_path.name
        with output_path.open("w", encoding="utf-8") as f_out:
            for record in cleaned:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        stats_rows.append(
            {
                "file": json_path.name,
                "input_path": str(json_path),
                "output_path": str(output_path),
                "total_records": total,
                "kept_records": kept,
                "dropped_records": dropped,
            }
        )
        print(f"[clean] {json_path.name}: kept {kept}/{total}")

    write_header = not args.stats.exists()
    with args.stats.open("a", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "file",
                "input_path",
                "output_path",
                "total_records",
                "kept_records",
                "dropped_records",
            ],
        )
        if write_header:
            writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
