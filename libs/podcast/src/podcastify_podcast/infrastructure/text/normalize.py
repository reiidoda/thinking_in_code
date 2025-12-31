from __future__ import annotations

import re
from collections import Counter
from typing import List

from podcastify_podcast.domain.models import PageText

def fix_hyphenation(text: str) -> str:
    # Remove line-break hyphenation (e.g., "tech-\nniques" -> "techniques")
    return re.sub(r"-\s*\n", "", text)

def normalize_whitespace(text: str) -> str:
    # Collapse multiple spaces/newlines into single spaces while preserving paragraphs
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"[ ]+\n", "\n", text)
    text = re.sub(r"\n[ ]+", "\n", text)
    return text.strip()

def clean_page_text(raw: str) -> str:
    text = fix_hyphenation(raw)
    text = text.replace("\u00ad", "")  # soft hyphen
    text = normalize_whitespace(text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        # Drop table-like lines (common in PDFs with columns)
        if "|" in stripped or "____" in stripped:
            continue
        if re.search(r"\s{3,}\S+\s{3,}\S+", stripped):
            continue
        lines.append(stripped)
    # Preserve paragraph breaks
    deduped: list[str] = []
    for ln in lines:
        if ln == "" and (not deduped or deduped[-1] == ""):
            continue
        deduped.append(ln)
    return "\n".join(deduped)

def drop_repeated_headers_footers(pages: List[PageText], *, min_repeats: int = 2) -> List[PageText]:
    """Remove page headers/footers that repeat across pages (simple heuristic)."""
    if len(pages) < min_repeats:
        return pages

    header_counts: Counter[str] = Counter()
    footer_counts: Counter[str] = Counter()
    split_lines: list[list[str]] = []

    for p in pages:
        lines = [ln.strip() for ln in p.text.splitlines() if ln.strip()]
        split_lines.append(lines)
        if lines:
            header_counts[lines[0]] += 1
            footer_counts[lines[-1]] += 1

    header_lines = {line for line, count in header_counts.items() if count >= min_repeats and len(line) <= 120}
    footer_lines = {line for line, count in footer_counts.items() if count >= min_repeats and len(line) <= 120}

    cleaned_pages: list[PageText] = []
    for p, lines in zip(pages, split_lines):
        filtered: list[str] = []
        for idx, line in enumerate(lines):
            is_header = idx == 0 and line in header_lines
            is_footer = idx == len(lines) - 1 and line in footer_lines
            looks_like_page_num = bool(re.fullmatch(r"(page\s+)?\d{1,4}(/\\d{1,4})?", line, flags=re.I))
            if is_header or is_footer or looks_like_page_num:
                continue
            filtered.append(line)
        cleaned_pages.append(PageText(page_number=p.page_number, text="\n".join(filtered), source=p.source))

    return cleaned_pages
