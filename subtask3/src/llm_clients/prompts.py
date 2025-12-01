from typing import Dict, List

LAPTOP_LABELS = {
    "entities": [
        "LAPTOP", "DISPLAY", "KEYBOARD", "MOUSE", "MOTHERBOARD", "CPU", "FANS_COOLING",
        "PORTS", "MEMORY", "POWER_SUPPLY", "OPTICAL_DRIVES", "BATTERY", "GRAPHICS",
        "HARD_DISK", "MULTIMEDIA_DEVICES", "HARDWARE", "SOFTWARE", "OS", "WARRANTY",
        "SHIPPING", "SUPPORT", "COMPANY", "OUT_OF_SCOPE"
    ],
    "attributes": [
        "GENERAL", "PRICE", "QUALITY", "DESIGN_FEATURES", "OPERATION_PERFORMANCE",
        "USABILITY", "PORTABILITY", "CONNECTIVITY", "MISCELLANEOUS"
    ],
}

RESTAURANT_LABELS = {
    "entities": ["RESTAURANT", "FOOD", "DRINKS", "AMBIENCE", "SERVICE", "LOCATION"],
    "attributes": ["GENERAL", "PRICES", "QUALITY", "STYLE_OPTIONS", "MISCELLANEOUS"],
}


def build_prompt(text: str, domain: str, retrieval_snippets: List[str] = None) -> List[Dict]:
    labels = LAPTOP_LABELS if domain == "lap" else RESTAURANT_LABELS
    label_text = (
        f"Entities: {', '.join(labels['entities'])}. "
        f"Attributes: {', '.join(labels['attributes'])}."
    )
    retrieval_block = ""
    if retrieval_snippets:
        retrieval_block = "\nRetrieved exemplars:\n" + "\n".join(retrieval_snippets)
    prompt = (
        "You are an ABSA system. "
        + label_text
        + "\nText: "
        + text
        + retrieval_block
        + "\nReturn JSON array with items {Aspect, Category, Opinion, VA}."
    )
    return [{"role": "user", "content": prompt}]
