from __future__ import annotations

import json
import math
from pathlib import Path

from podcastify_contracts.podcast_job import Citation

from podcastify_podcast.application.ports import VectorStore
from podcastify_podcast.domain.models import Chunk


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class LocalVectorStore(VectorStore):
    """Simple JSON-backed vector store per job."""

    def __init__(self, base_dir: str = "data/processed") -> None:
        self.base_dir = Path(base_dir)

    def _path(self, job_id: str) -> Path:
        return self.base_dir / job_id / "vector_store.json"

    def index(self, *, job_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> str:
        if len(chunks) != len(embeddings):
            raise ValueError("Embeddings and chunks must have same length.")
        path = self._path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: list[dict] = []
        for chunk, emb in zip(chunks, embeddings, strict=False):
            data.append(
                {
                    "text": chunk.text,
                    "citations": [c.model_dump() for c in chunk.citations],
                    "embedding": emb,
                }
            )

        path.write_text(json.dumps(data), encoding="utf-8")
        return str(path)

    def query(
        self, *, job_id: str, embedding: list[float], top_k: int = 4, min_score: float = 0.0, focus_pages: list[int] | None = None
    ) -> list[tuple[Chunk, float]]:
        path = self._path(job_id)
        if not path.exists():
            raise FileNotFoundError(f"Vector store not found for job {job_id}")

        items = json.loads(path.read_text(encoding="utf-8"))
        scored: list[tuple[Chunk, float]] = []
        for item in items:
            chunk = Chunk(
                text=item.get("text", ""),
                citations=[Citation.model_validate(i) for i in item.get("citations", [])],
            )
            score = _cosine_similarity(embedding, item.get("embedding", []))
            scored.append((chunk, score))

        def page_match(chunk: Chunk) -> bool:
            if not focus_pages:
                return True
            pages = {c.page for c in chunk.citations if c.page}
            return bool(pages & set(focus_pages))

        scored = [(chunk, score) for chunk, score in scored if score >= min_score and page_match(chunk)]
        # Simple rerank: prioritize chunks with citations/pages and higher score
        def math_bonus(chunk: Chunk) -> int:
            return _math_token_count(chunk.text)

        scored.sort(key=lambda tup: (len([c for c in tup[0].citations if c.page]), math_bonus(tup[0]), tup[1]), reverse=True)
        return scored[:top_k]


def _math_token_count(text: str) -> int:
    import re
    tokens = re.findall(r"[A-Za-z]*[αβγλμσπ∑∫∂]|[0-9]+[./][0-9]+|e\\^|\\^\\d+", text)
    return len(tokens)
