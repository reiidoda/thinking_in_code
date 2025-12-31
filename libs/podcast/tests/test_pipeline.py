from __future__ import annotations

from pathlib import Path

import pytest

from podcastify_contracts.errors import PipelineError
from podcastify_contracts.podcast_job import Citation, EpisodeSegment, JobStatus, PodcastJobRequest
from podcastify_podcast.application.use_cases import GeneratePodcastFromPdf
from podcastify_podcast.domain.models import Chunk, PageText
from podcastify_podcast.infrastructure.audio.pacing import segment_pause_multiplier, segment_rate_multiplier
from podcastify_podcast.infrastructure.storage.fs_store import FileSystemJobStore


class DummyPdfExtractor:
    def extract(self, pdf_bytes: bytes):
        return [
            PageText(page_number=1, text="Energy usage grows with demand.", source="paper.pdf"),
            PageText(page_number=2, text="Efficiency gains reduce cost over time.", source="paper.pdf"),
        ]


class DummyChunker:
    def chunk(self, pages):
        return [
            Chunk(
                text="Energy usage grows with demand and efficiency gains reduce cost.",
                citations=[Citation(source="paper.pdf", page=1, snippet="Energy usage grows")],
            )
        ]


class DummyScriptWriter:
    def write_script(self, *, chunks, minutes, language, style):
        return [
            EpisodeSegment(
                title="Hook",
                speaker="Host",
                text="Think of energy like a morning routine. Demand rises fast, and efficiency trims the waste.",
                citations=chunks[0].citations,
            ),
            EpisodeSegment(
                title="Recap",
                speaker="Host",
                text="The routine repeats: demand rises, efficiency counters it.",
                citations=chunks[0].citations,
            ),
            EpisodeSegment(
                title="Takeaway",
                speaker="Host",
                text="Small daily gains compound into big savings.",
                citations=chunks[0].citations,
            ),
        ]


class FailingScriptWriter:
    def write_script(self, *, chunks, minutes, language, style):
        raise RuntimeError("LLM failed")


class FailingPdfExtractor:
    def extract(self, pdf_bytes: bytes):
        raise RuntimeError("broken extractor")


def test_pipeline_success_without_audio(tmp_path):
    job_store = FileSystemJobStore(base_dir=str(tmp_path))
    pipeline = GeneratePodcastFromPdf(
        pdf_extractor=DummyPdfExtractor(),
        chunker=DummyChunker(),
        script_writer=DummyScriptWriter(),
        job_store=job_store,
    )

    request = PodcastJobRequest(input_filename="paper.pdf", language="en", style="everyday", target_minutes=6)
    result = pipeline.run(job_id="job-1", request=request, pdf_bytes=b"fake")

    assert result.status == JobStatus.SUCCEEDED
    assert result.title == "Hook"

    job_dir = Path(tmp_path) / "job-1"
    assert (job_dir / "script.md").exists()
    assert (job_dir / "transcript.txt").exists()
    assert (job_dir / "quality.json").exists()


def test_pipeline_uses_fallback_on_script_error(tmp_path):
    job_store = FileSystemJobStore(base_dir=str(tmp_path))
    pipeline = GeneratePodcastFromPdf(
        pdf_extractor=DummyPdfExtractor(),
        chunker=DummyChunker(),
        script_writer=FailingScriptWriter(),
        job_store=job_store,
    )

    request = PodcastJobRequest(input_filename="paper.pdf", language="en", style="everyday", target_minutes=6)
    result = pipeline.run(job_id="job-2", request=request, pdf_bytes=b"fake")

    titles = [seg.title for seg in result.segments]
    assert "Hook" in titles
    assert "Recap" in titles
    assert "Takeaway" in titles


def test_pipeline_raises_pipeline_error_on_extract_failure(tmp_path):
    job_store = FileSystemJobStore(base_dir=str(tmp_path))
    pipeline = GeneratePodcastFromPdf(
        pdf_extractor=FailingPdfExtractor(),
        chunker=DummyChunker(),
        script_writer=DummyScriptWriter(),
        job_store=job_store,
    )

    request = PodcastJobRequest(input_filename="paper.pdf", language="en", style="everyday", target_minutes=6)
    with pytest.raises(PipelineError):
        pipeline.run(job_id="job-3", request=request, pdf_bytes=b"fake")


def test_pipeline_applies_voice_profiles(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_PROFILE_MAP", "hook=VoiceA,default=VoiceB")
    job_store = FileSystemJobStore(base_dir=str(tmp_path))
    pipeline = GeneratePodcastFromPdf(
        pdf_extractor=DummyPdfExtractor(),
        chunker=DummyChunker(),
        script_writer=DummyScriptWriter(),
        job_store=job_store,
    )

    request = PodcastJobRequest(input_filename="paper.pdf", language="en", style="everyday", target_minutes=6)
    result = pipeline.run(job_id="job-4", request=request, pdf_bytes=b"fake")

    assert result.segments[0].voice == "VoiceA"
    assert result.segments[-1].voice == "VoiceB"


def test_pacing_multipliers_with_dense_text():
    simple = "Coffee cools down on the table." * 4
    dense = "N(t)=N0*e^{rt} and P(H|D)=P(D|H)P(H)/P(D), with sigma/mu trends." * 2

    rate_simple = segment_rate_multiplier(simple)
    pause_simple = segment_pause_multiplier(simple)
    rate_dense = segment_rate_multiplier(dense)
    pause_dense = segment_pause_multiplier(dense)

    assert 0.72 <= rate_simple <= 1.0
    assert 0.85 <= pause_simple <= 2.2
    assert 0.72 <= rate_dense <= 1.0
    assert 0.85 <= pause_dense <= 2.2
    assert rate_dense <= rate_simple
    assert pause_dense >= pause_simple
