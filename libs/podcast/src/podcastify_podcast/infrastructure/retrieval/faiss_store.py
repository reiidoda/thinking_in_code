from __future__ import annotations

import json
from pathlib import Path

from podcastify_contracts.podcast_job import Citation

from podcastify_podcast.application.ports import VectorStore
from podcastify_podcast.domain.models import Chunk


class FaissVectorStore(VectorStore):
    """FAISS-backed vector store with cosine scoring and math-aware rerank."""

    def __init__(self, base_dir: str = "data/processed") -> None:
        self.base_dir = Path(base_dir)

    def _job_dir(self, job_id: str) -> Path:
        return self.base_dir / job_id

    def _index_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "faiss.index"

    def _meta_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "faiss_meta.json"

    def index(self, *, job_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> str:
        if len(chunks) != len(embeddings):
            raise ValueError("Embeddings and chunks must have same length.")

        try:
            import faiss  # type: ignore
            import numpy as np
        except Exception as e:  # pragma: no cover
            raise RuntimeError("faiss is required for the FAISS vector store. Install with `pip install faiss-cpu`.") from e

        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        arr = np.array(embeddings, dtype="float32")
        if arr.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        arr = arr / norms
        dim = arr.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(arr)
        faiss.write_index(index, str(self._index_path(job_id)))

        meta: list[dict] = []
        for chunk in chunks:
            meta.append(
                {
                    "text": chunk.text,
                    "citations": [c.model_dump() for c in chunk.citations],
                }
            )
        self._meta_path(job_id).write_text(json.dumps(meta), encoding="utf-8")
        return str(self._index_path(job_id))

    def query(
        self, *, job_id: str, embedding: list[float], top_k: int = 4, min_score: float = 0.0, focus_pages: list[int] | None = None
    ) -> list[tuple[Chunk, float]]:
        try:
            import faiss  # type: ignore
            import numpy as np
        except Exception as e:  # pragma: no cover
            raise RuntimeError("faiss is required for the FAISS vector store. Install with `pip install faiss-cpu`.") from e

        index_path = self._index_path(job_id)
        meta_path = self._meta_path(job_id)
        if not index_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"FAISS index not found for job {job_id}")

        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta:
            return []

        vec = np.array([embedding], dtype="float32")
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
        scores, idxs = index.search(vec, min(top_k * 4, len(meta)))
        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], idxs[0], strict=False):
            if idx < 0 or idx >= len(meta):
                continue
            item = meta[idx]
            chunk = Chunk(
                text=item.get("text", ""),
                citations=[Citation.model_validate(c) for c in item.get("citations", [])],
            )
            results.append((chunk, float(score)))

        def page_match(chunk: Chunk) -> bool:
            if not focus_pages:
                return True
            pages = {c.page for c in chunk.citations if c.page}
            return bool(pages & set(focus_pages))

        def rerank_key(item: tuple[Chunk, float]) -> tuple:
            chunk, score = item
            math_bonus = _math_token_count(chunk.text)
            cite_pages = len([c for c in chunk.citations if c.page])
            return (score, cite_pages, math_bonus)

        filtered = [(c, s) for c, s in results if s >= min_score and page_match(c)]
        filtered.sort(key=rerank_key, reverse=True)
        return filtered[:top_k]


def _math_token_count(text: str) -> int:
    import re

    tokens = re.findall(r"[A-Za-z]*[αβγλμσπ∑∫∂]|[0-9]+[./][0-9]+|e\\^|\\^\\d+|\\d+[A-Za-z]*", text)
    return len(tokens)
