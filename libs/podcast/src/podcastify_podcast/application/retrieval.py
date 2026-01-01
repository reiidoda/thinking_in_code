from __future__ import annotations

from podcastify_contracts.retrieval import RetrievalResult, RetrievedChunk

from podcastify_podcast.application.ports import EmbeddingGenerator, VectorStore


class RetrieveChunks:
    """Retrieve top-k chunks for a job using a vector store."""

    def __init__(self, *, embedder: EmbeddingGenerator, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def run(self, *, job_id: str, query: str, top_k: int = 4, min_score: float = 0.0, focus_pages: list[int] | None = None) -> RetrievalResult:
        embedding = self.embedder.embed([query])[0]
        # bias math queries toward math-heavy chunks if supported by vector store
        results = self.vector_store.query(job_id=job_id, embedding=embedding, top_k=top_k, min_score=min_score, focus_pages=focus_pages)
        return RetrievalResult(
            job_id=job_id,
            query=query,
            results=[
                RetrievedChunk(text=chunk.text, citations=chunk.citations, score=score)
                for chunk, score in results
            ],
        )
