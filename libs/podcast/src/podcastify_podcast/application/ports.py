from __future__ import annotations

from pathlib import Path
from typing import Protocol

from podcastify_contracts.podcast_job import EpisodeSegment

from podcastify_podcast.domain.models import Chunk, PageText


class PdfExtractor(Protocol):
    def extract(self, pdf_bytes: bytes) -> list[PageText]: ...


class Chunker(Protocol):
    def chunk(self, pages: list[PageText]) -> list[Chunk]:
        """Return list of chunk objects with citations."""


class ScriptWriter(Protocol):
    def write_script(self, *, chunks: list[Chunk], minutes: int, language: str, style: str) -> list[EpisodeSegment]: ...


class EmbeddingGenerator(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    def index(self, *, job_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> str: ...
    def query(
        self, *, job_id: str, embedding: list[float], top_k: int = 4, min_score: float = 0.0, focus_pages: list[int] | None = None
    ) -> list[tuple[Chunk, float]]: ...


class TtsSynthesizer(Protocol):
    def synthesize(self, *, segments: list[EpisodeSegment], out_dir: str) -> list[str]:
        """Return list of audio segment file paths."""


class AudioAssembler(Protocol):
    def assemble(
        self,
        *,
        audio_segments: list[str],
        out_path: str,
        segment_word_counts: list[int] | None = None,
        segment_pause_multipliers: list[float] | None = None,
    ) -> str: ...


class JobStore(Protocol):
    def write_artifact(self, *, job_id: str, name: str, content: bytes) -> str: ...
    def job_dir(self, job_id: str) -> Path: ...
