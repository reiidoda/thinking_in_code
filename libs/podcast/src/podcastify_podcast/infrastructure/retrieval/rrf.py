from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi  # type: ignore
from podcastify_podcast.infrastructure.text.overlap import token_set

from podcastify_contracts.podcast_job import Citation
from podcastify_podcast.application.ports import EmbeddingGenerator, VectorStore
from podcastify_podcast.domain.models import Chunk


class RrfRetriever:
    """Reciprocal-rank fusion over dense (vector store) and sparse (BM25) results."""

    def __init__(self, *, embedder: EmbeddingGenerator, vector_store: VectorStore, base_dir: str = "data/processed"):
        self.embedder = embedder
        self.vector_store = vector_store
        self.base_dir = Path(base_dir)

    def _load_chunks(self, job_id: str) -> List[Chunk]:
        path = self.base_dir / job_id / "chunks.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks: list[Chunk] = []
        for item in data:
            chunks.append(
                Chunk(
                    text=item.get("text", ""),
                    citations=[Citation.model_validate(c) for c in item.get("citations", [])],
                )
            )
        return chunks

    def _bm25_scores(self, query: str, chunks: List[Chunk], top_k: int) -> List[Tuple[Chunk, float]]:
        if not chunks:
            return []
        corpus = [c.text.split() for c in chunks]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query.split())
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def run(self, *, job_id: str, query: str, top_k: int = 4, min_score: float = 0.0, focus_pages: list[int] | None = None) -> List[Tuple[Chunk, float]]:
        embedding = self.embedder.embed([query])[0]
        dense_results = self.vector_store.query(job_id=job_id, embedding=embedding, top_k=top_k, min_score=min_score, focus_pages=focus_pages)

        chunks = self._load_chunks(job_id)
        if focus_pages:
            chunks = [
                c for c in chunks if any((cit.page in focus_pages) for cit in c.citations if cit.page is not None)
            ]
        # Math bias: prioritize chunks with math tokens when query mentions math/science
        if _looks_math(query):
            chunks.sort(key=lambda c: _math_token_count(c.text), reverse=True)
        sparse_results = self._bm25_scores(query, chunks, top_k)

        # RRF fusion
        fused: dict[str, Tuple[Chunk, float]] = {}
        def add_results(results: List[Tuple[Chunk, float]], weight: float, rank_offset: float = 60.0):
            for rank, (chunk, score) in enumerate(results, start=1):
                key = chunk.text[:80]
                fused.setdefault(key, (chunk, 0.0))
                fused[key] = (chunk, fused[key][1] + weight * (1.0 / (rank_offset + rank)))

        add_results(dense_results, weight=1.0)
        add_results(sparse_results, weight=0.8)

        fused_list = list(fused.values())
        # Small bonus for citation-rich chunks
        fused_list.sort(key=lambda x: (len([c for c in x[0].citations if c.page]), x[1]), reverse=True)
        return fused_list[:top_k]


def _looks_math(text: str) -> bool:
    return any(tok in text.lower() for tok in ["math", "probab", "bayes", "integral", "derivative", "sigma", "lambda", "alpha", "beta"])


def _math_token_count(text: str) -> int:
    import re
    tokens = re.findall(r"[A-Za-z]*[αβγλμσππ∑∫∂]|[0-9]+[./][0-9]+|e\\^|\\^\\d+", text)
    return len(tokens)
