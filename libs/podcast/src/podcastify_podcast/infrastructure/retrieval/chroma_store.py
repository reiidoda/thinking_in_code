from __future__ import annotations

import json
from pathlib import Path

from podcastify_contracts.podcast_job import Citation

from podcastify_podcast.application.ports import VectorStore
from podcastify_podcast.domain.models import Chunk


class ChromaVectorStore(VectorStore):
    """Chroma-backed vector store, persisted per job."""

    def __init__(self, base_dir: str = "data/processed", collection_prefix: str = "job") -> None:
        self.base_dir = Path(base_dir)
        self.collection_prefix = collection_prefix

    def _client_and_collection(self, job_id: str):
        try:
            import chromadb  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("chromadb is required for this vector store. Install with `pip install chromadb`.") from e

        persist_dir = self.base_dir / job_id / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir))
        coll = client.get_or_create_collection(name=f"{self.collection_prefix}_{job_id}")
        return client, coll

    def index(self, *, job_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> str:
        if len(chunks) != len(embeddings):
            raise ValueError("Embeddings and chunks must have same length.")
        _, coll = self._client_and_collection(job_id)
        ids = [f"{job_id}_{i}" for i in range(len(chunks))]
        documents = [c.text for c in chunks]
        metadata = [
            {
                "citations": json.dumps([c.model_dump() for c in chunk.citations]),
            }
            for chunk in chunks
        ]
        coll.upsert(ids=ids, documents=documents, metadatas=metadata, embeddings=embeddings)
        return str(self.base_dir / job_id / "chroma")

    def query(
        self, *, job_id: str, embedding: list[float], top_k: int = 4, min_score: float = 0.0, focus_pages: list[int] | None = None
    ) -> list[tuple[Chunk, float]]:
        _, coll = self._client_and_collection(job_id)
        res = coll.query(query_embeddings=[embedding], n_results=top_k)

        results: list[tuple[Chunk, float]] = []
        docs = res.get("documents", [[]])[0] if res else []
        metas = res.get("metadatas", [[]])[0] if res else []
        distances = res.get("distances", [[]])[0] if res else []

        for doc, meta, dist in zip(docs, metas, distances, strict=False):
            # Chroma returns distance; convert to similarity (1 - distance) if possible.
            score = 1 - float(dist) if dist is not None else 0.0
            if score < min_score:
                continue
            citations_raw = meta.get("citations") if isinstance(meta, dict) else None
            citations: list[Citation] = []
            if citations_raw:
                try:
                    citations = [Citation.model_validate(c) for c in json.loads(citations_raw)]
                except Exception:
                    citations = []
            results.append((Chunk(text=doc, citations=citations), score))

        def page_match(chunk: Chunk) -> bool:
            if not focus_pages:
                return True
            pages = {c.page for c in chunk.citations if c.page}
            return bool(pages & set(focus_pages))

        # Rerank: citations and score
        results = [r for r in results if r[1] >= min_score and page_match(r[0])]
        results.sort(
            key=lambda tup: (len([c for c in tup[0].citations if c.page]), _math_token_count(tup[0].text), tup[1]),
            reverse=True,
        )
        return results[:top_k]


def _math_token_count(text: str) -> int:
    import re
    tokens = re.findall(r"[A-Za-z]*[αβγλμσπ∑∫∂]|[0-9]+[./][0-9]+|e\\^|\\^\\d+", text)
    return len(tokens)
