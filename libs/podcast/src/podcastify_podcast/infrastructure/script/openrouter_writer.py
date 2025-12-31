from __future__ import annotations

import json
from typing import List

from podcastify_contracts.podcast_job import EpisodeSegment, Citation
from podcastify_podcast.application.ports import ScriptWriter
from podcastify_podcast.domain.models import Chunk
from podcastify_podcast.infrastructure.llm.openrouter import OpenRouterTextGenerator
from podcastify_podcast.infrastructure.script.script_writer import _load_prompt, sentence_split
from podcastify_podcast.infrastructure.logging import get_logger

log = get_logger(__name__)


class OpenRouterScriptWriter(ScriptWriter):
    """Script writer backed by OpenRouter (e.g., minimax/minimax-m2.1)."""

    def __init__(self, *, generator: OpenRouterTextGenerator) -> None:
        self.generator = generator

    def write_script(
        self,
        *,
        chunks: List[Chunk],
        minutes: float,
        language: str,
        style: str,
    ) -> List[EpisodeSegment]:
        system = _load_prompt("script_system.md")
        user = _load_prompt("script_user.md")

        context_blocks: list[str] = []
        for idx, chunk in enumerate(chunks[:8], start=1):
            context_blocks.append(f"[Context {idx} p.{chunk.citations[0].page if chunk.citations else '?'}]\n{chunk.text}")
        context = "\n\n".join(context_blocks)

        prompt = f"""{system}

{user}
Language: {language}
Style: {style}
Target minutes: {minutes}

Format the podcast as a natural conversation with two hosts:
- Host 1 (Male): curious, enthusiastic interviewer
- Host 2 (Female): knowledgeable expert

Return JSON with top-level key 'segments': list of objects with fields title, speaker, text.
Structure: Hook, 3-6 body segments, Recap, Takeaway. Enforce citations with [p.#] where possible.

REFERENCE CONTEXT:
{context}
"""
        raw = self.generator.generate(prompt=prompt)
        try:
            data = json.loads(self._extract_json(raw))
            segs = data.get("segments", data)
            if not isinstance(segs, list):
                raise ValueError("segments field is not a list")
            out: list[EpisodeSegment] = []
            for item in segs:
                out.append(
                    EpisodeSegment(
                        title=item.get("title") or "Segment",
                        speaker=item.get("speaker") or "Host",
                        text=item.get("text") or "",
                        citations=[],
                    )
                )
            return out if out else self._fallback(chunks)
        except Exception:
            log.warning("OpenRouter script parse failed; falling back to single segment.")
            return self._fallback(chunks)

    def _extract_json(self, raw: str) -> str:
        import re
        match = re.search(r"\{.*\}|\[.*\]", raw, re.S)
        return match.group(0) if match else raw

    def _fallback(self, chunks: List[Chunk]) -> List[EpisodeSegment]:
        text = " ".join(c.text for c in chunks[:3]) or "Podcast episode."
        return [EpisodeSegment(title="Episode", speaker="Host", text=text, citations=[])]
