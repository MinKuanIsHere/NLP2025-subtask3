import json
from pathlib import Path
from typing import Dict, List, Optional


UNK_TOKEN = "[UNK]"


def _strip_unk(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    cleaned = text.replace(UNK_TOKEN, "").strip()
    return " ".join(cleaned.split()) if cleaned else ""


def _clean_quadruplet(entry: Dict[str, str]) -> Optional[Dict[str, str]]:
    aspect = _strip_unk(entry.get("Aspect"))
    opinion = _strip_unk(entry.get("Opinion"))
    category = entry.get("Category")
    va = entry.get("VA") or ""

    if not aspect and not opinion:
        return None

    cleaned = {
        "Aspect": aspect or "",
        "Opinion": opinion or "",
    }
    if category:
        cleaned["Category"] = category.strip()
    if va:
        cleaned["VA"] = va.strip()
    return cleaned


def clean_record(record: Dict[str, object]) -> Optional[Dict[str, object]]:
    text = _strip_unk(record.get("Text"))
    quads: List[Dict[str, str]] = record.get("Quadruplet") or []

    cleaned_quads: List[Dict[str, str]] = []
    for quad in quads:
        cleaned = _clean_quadruplet(quad or {})
        if cleaned and any(cleaned.values()):
            cleaned_quads.append(cleaned)

    if not cleaned_quads:
        return None

    cleaned_record = dict(record)
    if text is not None:
        cleaned_record["Text"] = text
    cleaned_record["Quadruplet"] = cleaned_quads
    return cleaned_record


def load_and_clean_jsonl(path: Path) -> List[Dict[str, object]]:
    cleaned: List[Dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            cleaned_record = clean_record(record)
            if cleaned_record:
                cleaned.append(cleaned_record)
    return cleaned
