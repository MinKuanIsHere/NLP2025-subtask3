from typing import List, Dict


def format_neighbors_to_tokens(neighbors: List[Dict], max_examples: int, max_tokens: int) -> List[str]:
    tokens: List[str] = []
    example_count = 0
    for idx, neighbor in enumerate(neighbors):
        if example_count >= max_examples:
            break
        snippet_parts = []
        quads = neighbor.get("quadruplets") or []
        if quads:
            first = quads[0]
            snippet_parts.extend(
                [
                    f"asp:{first.get(Aspect, )}",
                    f"cat:{first.get(Category, )}",
                    f"opi:{first.get(Opinion, )}",
                ]
            )
        if neighbor.get("text"):
            snippet_parts.append(neighbor["text"])
        snippet_text = " ".join(snippet_parts).strip().lower()
        if not snippet_text:
            continue
        snippet_tokens = snippet_text.split()
        tokens.extend(["[SEP]", "retrieved", str(idx + 1), ":"])
        tokens.extend(snippet_tokens)
        example_count += 1
        if len(tokens) >= max_tokens:
            return tokens[:max_tokens]
    return tokens
