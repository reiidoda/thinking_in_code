from __future__ import annotations

import re
from collections.abc import Iterable


def sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if s.strip()]


def token_set(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[a-zA-Z0-9]{3,}", text)}


def snippet_overlap(text: str, snippet: str) -> float:
    """Compute Jaccard overlap between text and snippet token sets."""
    return jaccard(token_set(text), token_set(snippet))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def sentence_overlap(sentence: str, corpus: list[str]) -> float:
    stoks = token_set(sentence)
    best = 0.0
    for chunk in corpus:
        ctoks = token_set(chunk)
        best = max(best, jaccard(stoks, ctoks))
    return best
