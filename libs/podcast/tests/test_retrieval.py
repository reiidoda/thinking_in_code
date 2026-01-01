from podcastify_contracts.podcast_job import Citation
from podcastify_podcast.application.retrieval import RetrieveChunks
from podcastify_podcast.domain.models import Chunk
from podcastify_podcast.infrastructure.retrieval.chroma_store import ChromaVectorStore
from podcastify_podcast.infrastructure.retrieval.local_store import LocalVectorStore


class DummyEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, texts):
        return self.vectors[: len(texts)]


def test_local_vector_store_index_and_query(tmp_path):
    store = LocalVectorStore(base_dir=tmp_path)
    chunks = [
        Chunk(text="cats and dogs", citations=[Citation(source="pdf", page=1, snippet="cats")]),
        Chunk(text="quantum physics", citations=[Citation(source="pdf", page=2, snippet="quantum")]),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    store.index(job_id="job1", chunks=chunks, embeddings=embeddings)

    results = store.query(job_id="job1", embedding=[1.0, 0.0], top_k=1, min_score=0.1, focus_pages=[1])
    assert len(results) == 1
    top_chunk, score = results[0]
    assert "cats" in top_chunk.citations[0].snippet
    assert score > 0.99


def test_retrieve_chunks_use_case(tmp_path):
    store = LocalVectorStore(base_dir=tmp_path)
    chunks = [Chunk(text="renewable energy is growing", citations=[])]
    store.index(job_id="job2", chunks=chunks, embeddings=[[0.5, 0.5]])

    embedder = DummyEmbedder([[0.5, 0.5]])
    use_case = RetrieveChunks(embedder=embedder, vector_store=store)
    res = use_case.run(job_id="job2", query="energy", top_k=1, min_score=0.0, focus_pages=None)

    assert res.results[0].text.startswith("renewable")
    assert res.results[0].score > 0.0


def test_chroma_vector_store_if_available(tmp_path):
    try:
        import chromadb  # noqa: F401
    except Exception:
        return  # skip if chroma not installed

    store = ChromaVectorStore(base_dir=tmp_path)
    chunks = [Chunk(text="ai safety", citations=[Citation(source="pdf", page=1, snippet="ai safety")])]
    store.index(job_id="job3", chunks=chunks, embeddings=[[1.0, 0.0]])
    results = store.query(job_id="job3", embedding=[1.0, 0.0], top_k=1, min_score=0.0)
    assert len(results) == 1
    assert results[0][0].citations[0].source == "pdf"
