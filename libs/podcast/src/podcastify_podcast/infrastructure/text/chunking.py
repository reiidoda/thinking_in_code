from __future__ import annotations

import re

from podcastify_contracts.podcast_job import Citation
from podcastify_podcast.application.ports import Chunker
from podcastify_podcast.domain.models import Chunk, PageText


class SentenceChunker(Chunker):
    """Sentence-aware chunker that preserves citation metadata."""

    def __init__(self, max_chars: int = 1400, min_chars: int = 600) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars

    def _split_sentences(self, text: str) -> list[str]:
        # Lightweight sentence splitter to avoid external deps.
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if s.strip()]

    def chunk(self, pages: list[PageText]) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer: list[str] = []
        citations: list[Citation] = []
        current_len = 0

        for page in pages:
            sentences = self._split_sentences(page.text)
            for sentence in sentences:
                if not sentence:
                    continue

                if current_len + len(sentence) > self.max_chars and current_len >= self.min_chars:
                    chunks.append(Chunk(text=" ".join(buffer), citations=citations))
                    buffer, citations, current_len = [], [], 0

                buffer.append(sentence)
                citations.append(
                    Citation(
                        source=page.source or "pdf",
                        page=page.page_number,
                        snippet=sentence[:200],
                    )
                )
                current_len += len(sentence) + 1

        if buffer:
            chunks.append(Chunk(text=" ".join(buffer), citations=citations))

        return chunks
