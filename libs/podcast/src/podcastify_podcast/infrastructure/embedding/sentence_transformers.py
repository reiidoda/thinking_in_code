from __future__ import annotations

from podcastify_podcast.application.ports import EmbeddingGenerator


class SentenceTransformersEmbedder(EmbeddingGenerator):
    """Local embeddings via sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for local embeddings. Install with `pip install sentence-transformers`."
            ) from e

        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # sentence-transformers returns numpy.ndarray; convert to list for serialization
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [vec.tolist() for vec in embeddings]
