from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from podcastify_contracts.errors import PipelineError
from podcastify_contracts.podcast_job import (
    EpisodeArtifact,
    EpisodeSegment,
    JobStatus,
    PodcastJobRequest,
    PodcastJobResult,
)

from podcastify_podcast.application.ports import (
    AudioAssembler,
    Chunker,
    EmbeddingGenerator,
    JobStore,
    PdfExtractor,
    ScriptWriter,
    TtsSynthesizer,
    VectorStore,
)
from podcastify_podcast.domain.models import Chunk
from podcastify_podcast.infrastructure.audio.pacing import segment_pause_multiplier_for
from podcastify_podcast.infrastructure.logging import get_logger

log = get_logger(__name__)


class GeneratePodcastFromPdf:
    """Orchestrates the pipeline using injected adapters.

    Big-tech style: business flow lives here, not inside FastAPI or worker glue.
    """

    def __init__(
        self,
        *,
        pdf_extractor: PdfExtractor,
        chunker: Chunker,
        script_writer: ScriptWriter,
        job_store: JobStore,
        embedder: EmbeddingGenerator | None = None,
        vector_store: VectorStore | None = None,
        tts: TtsSynthesizer | None = None,
        audio: AudioAssembler | None = None,
    ) -> None:
        self.pdf_extractor = pdf_extractor
        self.chunker = chunker
        self.script_writer = script_writer
        self.job_store = job_store
        self.embedder = embedder
        self.vector_store = vector_store
        self.tts = tts
        self.audio = audio

    def run(self, *, job_id: str, request: PodcastJobRequest, pdf_bytes: bytes) -> PodcastJobResult:
        started = time.monotonic()
        timings: dict[str, float] = {}
        try:
            t0 = time.monotonic()
            pages = self.pdf_extractor.extract(pdf_bytes)
            timings["extract"] = time.monotonic() - t0

            t0 = time.monotonic()
            chunks = self.chunker.chunk(pages)
            timings["chunk"] = time.monotonic() - t0

            # allow finer-grained duration control (minutes + seconds)
            effective_minutes = float(request.target_minutes or 0) + float(getattr(request, "target_seconds", 0) or 0) / 60.0
            if effective_minutes <= 0:
                effective_minutes = 3.0

            t0 = time.monotonic()
            try:
                segments = self.script_writer.write_script(
                    chunks=chunks,
                    minutes=effective_minutes,
                    language=request.language,
                    style=request.style,
                )
                timings["script"] = time.monotonic() - t0
            except Exception as e:
                log.error("Script generation failed, using fallback script: %s", e)
                segments = self._fallback_script(chunks, style=request.style)
                timings["script_fallback"] = time.monotonic() - t0

            segments = self._apply_voice_profiles(segments)

            # Optional embeddings/vector store for retrieval
            if self.embedder and self.vector_store:
                try:
                    t0 = time.monotonic()
                    embeddings = self._load_cached_embeddings(job_id, chunks)
                    self.vector_store.index(job_id=job_id, chunks=chunks, embeddings=embeddings)
                    timings["embed_index"] = time.monotonic() - t0
                except Exception as e:
                    log.warning("Embedding/vector store failed (continuing without retrieval): %s", e)

            artifacts: list[EpisodeArtifact] = []
            job_dir = self.job_store.job_dir(job_id)

            # Save extraction + chunks for traceability
            self.job_store.write_artifact(job_id=job_id, name="extracted.txt", content="\n\n".join(p.text for p in pages).encode("utf-8"))
            self.job_store.write_artifact(job_id=job_id, name="chunks.txt", content=self._chunks_debug(chunks).encode("utf-8"))
            self.job_store.write_artifact(job_id=job_id, name="chunks.json", content=self._chunks_json(chunks).encode("utf-8"))

            t0 = time.monotonic()
            # Save script
            script_md = self._to_markdown(segments)
            script_path = self.job_store.write_artifact(
                job_id=job_id, name="script.md", content=script_md.encode("utf-8")
            )
            artifacts.append(EpisodeArtifact(kind="script_markdown", path=script_path))

            transcript_txt = self._transcript_text(segments)
            transcript_path = self.job_store.write_artifact(
                job_id=job_id, name="transcript.txt", content=transcript_txt.encode("utf-8")
            )
            artifacts.append(EpisodeArtifact(kind="transcript_txt", path=transcript_path))

            srt = self._to_srt(segments)
            srt_path = self.job_store.write_artifact(job_id=job_id, name="transcript.srt", content=srt.encode("utf-8"))
            artifacts.append(EpisodeArtifact(kind="transcript_srt", path=srt_path))
            timings["script_artifacts"] = time.monotonic() - t0

            audio_quality_data: dict | None = None
            audio_error: str | None = None
            # Optional audio
            if self.tts and self.audio:
                try:
                    t0 = time.monotonic()
                    seg_paths = self.tts.synthesize(segments=segments, out_dir=str(job_dir))
                    word_counts = [len(s.text.split()) for s in segments]
                    pause_multipliers = [segment_pause_multiplier_for(s.title, s.text) for s in segments]
                    final_path = self.audio.assemble(
                        audio_segments=seg_paths,
                        out_path=str(job_dir / "episode.mp3"),
                        segment_word_counts=word_counts,
                        segment_pause_multipliers=pause_multipliers,
                    )
                    artifacts.append(EpisodeArtifact(kind="audio_mp3", path=final_path))
                    meta_path = getattr(self.audio, "last_metadata_path", None)
                    if meta_path:
                        artifacts.append(EpisodeArtifact(kind="audio_metadata", path=meta_path))
                    try:
                        from podcastify_podcast.infrastructure.audio.audio_quality import (
                            AudioQualityChecker,
                        )
                    except Exception:
                        AudioQualityChecker = None  # type: ignore
                    if AudioQualityChecker:
                        quality_checker = AudioQualityChecker()
                        audio_quality_path = quality_checker.write_report(final_path, str(job_dir / "audio_quality.json"))
                        artifacts.append(EpisodeArtifact(kind="audio_quality", path=audio_quality_path))
                        try:
                            audio_quality_data = json.loads(Path(audio_quality_path).read_text(encoding="utf-8"))
                        except Exception:
                            audio_quality_data = None
                    timings["audio"] = time.monotonic() - t0
                except Exception as e:
                    audio_error = str(e)
                    log.error("Audio/TTS failed, continuing without audio: %s", e)

            # Quality artifacts (includes audio if available)
            t0 = time.monotonic()
            self._write_quality_artifacts(job_id=job_id, segments=segments, chunks=chunks, audio_quality=audio_quality_data)
            timings["quality_artifacts"] = time.monotonic() - t0

            total = time.monotonic() - started
            timings["total"] = total
            try:
                timing_path = self.job_store.write_artifact(
                    job_id=job_id, name="job_metrics.json", content=json.dumps(timings, indent=2).encode("utf-8")
                )
                artifacts.append(EpisodeArtifact(kind="job_metrics", path=timing_path))
            except Exception:
                pass

            title = segments[0].title if segments else "Podcast Episode"
            return PodcastJobResult(
                job_id=job_id,
                status=JobStatus.SUCCEEDED,
                title=title,
                segments=segments,
                artifacts=artifacts,
                error=audio_error,
            )
        except Exception as e:
            raise PipelineError(str(e)) from e

    def _to_markdown(self, segments) -> str:
        lines = ["# Podcast Script", ""]
        for s in segments:
            lines.append(f"## {s.title}")
            lines.append(f"**{s.speaker}:** {s.text}")
            lines.append("")
        return "\n".join(lines)

    def _chunks_debug(self, chunks: list[Chunk]) -> str:
        lines: list[str] = []
        for idx, ch in enumerate(chunks, start=1):
            lines.append(f"Chunk {idx}:")
            lines.append(ch.text)
            if ch.citations:
                lines.append("Citations:")
                for c in ch.citations:
                    lines.append(f"- {c.source or 'source'} p.{c.page or '?'}: {c.snippet or ''}")
            lines.append("")
        return "\n".join(lines)

    def _chunks_json(self, chunks: list[Chunk]) -> str:
        import json
        data = []
        for ch in chunks:
            data.append(
                {
                    "text": ch.text,
                    "citations": [c.model_dump() for c in ch.citations],
                }
            )
        return json.dumps(data, indent=2)

    def _apply_voice_profiles(self, segments: list[EpisodeSegment]) -> list[EpisodeSegment]:
        mapping = self._voice_profile_map()
        if not mapping:
            return segments
        for seg in segments:
            if seg.voice:
                continue
            voice = self._voice_for_segment(seg, mapping)
            if voice:
                seg.voice = voice
        return segments

    def _voice_profile_map(self) -> dict[str, str]:
        raw = os.getenv("VOICE_PROFILE_MAP", "").strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return {str(k).lower(): str(v) for k, v in data.items()}
        except Exception:
            pass
        mapping: dict[str, str] = {}
        for chunk in raw.split(","):
            if not chunk.strip():
                continue
            if "=" in chunk:
                key, val = chunk.split("=", 1)
            elif ":" in chunk:
                key, val = chunk.split(":", 1)
            else:
                continue
            mapping[key.strip().lower()] = val.strip()
        return mapping

    def _voice_for_segment(self, seg: EpisodeSegment, mapping: dict[str, str]) -> str | None:
        title = (seg.title or "").lower()
        for key, voice in mapping.items():
            if key in {"default", "speaker"}:
                continue
            if key and key in title:
                return voice
        speaker_key = (seg.speaker or "").lower()
        if speaker_key and speaker_key in mapping:
            return mapping[speaker_key]
        return mapping.get("default")

    def _write_quality_artifacts(self, *, job_id: str, segments, chunks: list[Chunk], audio_quality: dict | None) -> None:
        quality = self._quality_report(segments, chunks, audio_quality=audio_quality)
        self.job_store.write_artifact(job_id=job_id, name="quality.json", content=quality.encode("utf-8"))
        quality_log = self._quality_log(segments, chunks)
        self.job_store.write_artifact(job_id=job_id, name="quality.log", content=quality_log.encode("utf-8"))

    def _quality_report(self, segments, chunks: list[Chunk], *, audio_quality: dict | None = None) -> str:
        chunk_tokens = [set(self._tokens(c.text)) for c in chunks]
        report = {
            "total_segments": len(segments),
            "segments": [],
        }
        confidences: list[float] = []
        for seg in segments:
            seg_tokens = set(self._tokens(seg.text))
            overlap = 0.0
            if chunk_tokens and seg_tokens:
                overlap = max(self._jaccard(seg_tokens, ct) for ct in chunk_tokens)
            paragraphs = [p.strip() for p in seg.text.split("\n\n") if p.strip()]
            length_words = len(seg.text.split())
            math_density = self._math_density(seg.text)
            citations_per_para = len(seg.citations) / max(1, len(paragraphs))
            length_score = self._length_score(length_words, target_min=60, target_max=260)
            retention = self._retention_score(length_words, len(paragraphs))
            confidence, confidence_hmean = self._evidence_confidence(
                overlap=overlap,
                citations_per_para=citations_per_para,
                math_density=math_density,
                length_score=length_score,
            )
            confidences.append(confidence)
            report["segments"].append(
                {
                    "title": seg.title,
                    "citations": len(seg.citations),
                    "paragraphs": len(paragraphs),
                    "citations_per_paragraph": round(citations_per_para, 4),
                    "overlap": round(overlap, 4),
                    "math_density": round(math_density, 4),
                    "length_words": length_words,
                    "length_score": round(length_score, 4),
                    "retention_score": round(retention, 4),
                    "confidence": round(confidence, 4),
                    "confidence_hmean": round(confidence_hmean, 4),
                }
            )

        if confidences:
            report["confidence_min"] = round(min(confidences), 4)
            report["confidence_avg"] = round(sum(confidences) / len(confidences), 4)
            report["confidence_geo"] = round(math.exp(sum(math.log(max(c, 1e-6)) for c in confidences) / len(confidences)), 4)
            report["retention_avg"] = round(
                sum(s.get("retention_score", 0.0) for s in report["segments"]) / max(1, len(report["segments"])), 4
            )

        if audio_quality:
            report["audio_quality"] = audio_quality

        return json.dumps(report, indent=2)

    def _tokens(self, text: str) -> list[str]:
        import re
        return re.findall(r"[a-zA-Z0-9]{3,}", text.lower())

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def _quality_log(self, segments, chunks: list[Chunk]) -> str:
        from podcastify_podcast.infrastructure.text.overlap import sentence_split

        chunk_tokens = [set(self._tokens(c.text)) for c in chunks]
        lines: list[str] = []
        for seg in segments:
            lines.append(f"[SEGMENT] {seg.title}")
            for sent in sentence_split(seg.text):
                stoks = set(self._tokens(sent))
                overlap = max((self._jaccard(stoks, ct) for ct in chunk_tokens), default=0.0)
                lines.append(f"  sentence_overlap={overlap:.4f} | {sent[:200]}")
            lines.append("")
        return "\n".join(lines)

    def _math_density(self, text: str) -> float:
        tokens = self._tokens(text)
        if not tokens:
            return 0.0
        mathish = sum(1 for t in tokens if any(ch.isdigit() for ch in t) or any(op in t for op in ("+", "-", "*", "/", "^", "=")))
        return mathish / max(len(tokens), 1)

    def _length_score(self, length: int, *, target_min: int, target_max: int) -> float:
        # Gaussian penalty outside the mid-band to discourage rambles or thin sections.
        mid = 0.5 * (target_min + target_max)
        sigma = max(1.0, (target_max - target_min) / 4)
        return math.exp(-((length - mid) ** 2) / (2 * sigma**2))

    def _retention_score(self, length: int, paragraphs: int) -> float:
        """Heuristic retention: spacing effect (more than 1 para), and sweet-spot duration."""
        para_bonus = 1 - math.exp(-0.8 * max(paragraphs - 1, 0))
        # Ebbinghaus-inspired: too short forgets, too long fatigues
        sweet_min, sweet_max = 110, 190
        if length < sweet_min:
            length_factor = length / sweet_min
        elif length > sweet_max:
            length_factor = sweet_max / max(length, 1)
        else:
            length_factor = 1.0
        return max(0.0, min(1.0, 0.6 * length_factor + 0.4 * para_bonus))

    def _evidence_confidence(self, *, overlap: float, citations_per_para: float, math_density: float, length_score: float) -> tuple[float, float]:
        # Blend a logistic regressor with a harmonic mean to penalize any weak pillar.
        overlap_c = min(max(overlap, 0.0), 1.0)
        citations_c = min(citations_per_para / 2.0, 1.2)  # cap two strong cites per paragraph
        math_c = min(math_density * 1.5, 1.0)
        length_c = min(max(length_score, 0.0), 1.0)
        eps = 1e-6
        vals = [overlap_c, citations_c, math_c, length_c]
        hmean = len(vals) / sum(1.0 / (v + eps) for v in vals)

        log_citations = math.log1p(citations_per_para)
        linear = 3.6 * overlap_c + 1.6 * log_citations + 1.1 * length_c + 0.8 * math_c - 1.25
        logistic = 1 / (1 + math.exp(-linear))
        blended = 0.6 * logistic + 0.4 * hmean
        return (max(0.0, min(1.0, blended)), max(0.0, min(1.0, hmean)))

    def _load_cached_embeddings(self, job_id: str, chunks: list[Chunk]) -> list[list[float]]:
        import json
        from pathlib import Path

        cache_path = Path(self.job_store.job_dir(job_id)) / "embeddings.json"
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(data, list) and len(data) == len(chunks):
                    return data
            except Exception:
                pass

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts) if self.embedder else []
        cache_path.write_text(json.dumps(embeddings), encoding="utf-8")
        return embeddings

    def _transcript_text(self, segments) -> str:
        lines: list[str] = []
        for s in segments:
            lines.append(f"{s.title} - {s.speaker}: {s.text}")
        return "\n\n".join(lines)

    def _to_srt(self, segments) -> str:
        """Rudimentary SRT with estimated timings based on word count."""
        entries: list[str] = []
        start = 0.0
        words_per_second = 2.6  # ~156 wpm
        for idx, s in enumerate(segments, start=1):
            duration = max(4.0, len(s.text.split()) / words_per_second)
            end = start + duration
            entries.append(
                f"{idx}\n{self._fmt_ts(start)} --> {self._fmt_ts(end)}\n{s.speaker}: {s.text}\n"
            )
            start = end + 0.5
        return "\n".join(entries)

    def _fmt_ts(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        millis = int((secs - int(secs)) * 1000)
        return f"{hours:02}:{minutes:02}:{int(secs):02},{millis:03}"

    def _fallback_script(self, chunks: list[Chunk], style: str) -> list[EpisodeSegment]:
        """Create a deterministic, citation-carrying script without LLM if generation fails."""
        if not chunks:
            return [
                EpisodeSegment(
                    title="Hook",
                    speaker="Host",
                    text=f"A quick explainer in the style of {style}. Source context missing, so this is a generic intro.",
                    citations=[],
                )
            ]

        body = chunks[:4]
        segments: list[EpisodeSegment] = []
        segments.append(
            EpisodeSegment(
                title="Hook",
                speaker="Host",
                text=f"{body[0].text[:400]}",
                citations=body[0].citations[:3],
            )
        )
        for idx, chunk in enumerate(body[1:], start=1):
            segments.append(
                EpisodeSegment(
                    title=f"Segment {idx}",
                    speaker="Host",
                    text=chunk.text[:600],
                    citations=chunk.citations[:5],
                )
            )
        tail = chunks[-1]
        segments.append(
            EpisodeSegment(
                title="Recap",
                speaker="Host",
                text=tail.text[:400],
                citations=tail.citations[:3],
            )
        )
        segments.append(
            EpisodeSegment(
                title="Takeaway",
                speaker="Host",
                text="Key takeaway: the research above in plain language. [Fallback script used due to LLM error.]",
                citations=tail.citations[:2],
            )
        )
        return segments
