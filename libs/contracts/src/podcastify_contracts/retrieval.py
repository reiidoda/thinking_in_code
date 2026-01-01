from __future__ import annotations

from pydantic import BaseModel, Field

from podcastify_contracts.podcast_job import Citation


class RetrievalQuery(BaseModel):
    query: str = Field(..., description="User question or topic for retrieval.")
    top_k: int = Field(default=4, ge=1, le=10)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    focus_pages: list[int] | None = Field(default=None, description="Restrict results to these page numbers if provided.")


class RetrievedChunk(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    score: float = Field(..., description="Cosine similarity score.")


class RetrievalResult(BaseModel):
    job_id: str
    query: str
    results: list[RetrievedChunk] = Field(default_factory=list)
