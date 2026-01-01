from __future__ import annotations

from dataclasses import dataclass, field

from podcastify_contracts.podcast_job import Citation


@dataclass(frozen=True)
class Segment:
    title: str
    speaker: str
    text: str
    citations: list[Citation] = field(default_factory=list)


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    source: str | None = None


@dataclass(frozen=True)
class Chunk:
    text: str
    citations: list[Citation] = field(default_factory=list)


@dataclass(frozen=True)
class Episode:
    title: str
    segments: list[Segment]
    language: str = "en"
    minutes: int = 8
    summary: str | None = None
