from __future__ import annotations

import json
from pathlib import Path

from podcastify_podcast.application.ports import AudioAssembler


class PydubAudioAssembler(AudioAssembler):
    """Concatenate audio segments, normalize loudness, and export a single file."""

    def __init__(
        self,
        target_format: str = "mp3",
        target_dbfs: float = -16.0,
        intro_path: str | None = None,
        outro_path: str | None = None,
        metadata_path: str | None = None,
        segment_silence_ms: int = 300,
        fade_ms: int = 200,
        silence_per_word_ms: float = 3.0,
        min_segment_silence_ms: int = 200,
        max_segment_silence_ms: int = 2000,
    ) -> None:
        self.target_format = target_format
        self.target_dbfs = target_dbfs
        self.intro_path = intro_path
        self.outro_path = outro_path
        self.metadata_path = metadata_path
        self.last_metadata_path: str | None = None
        self.segment_silence_ms = segment_silence_ms
        self.fade_ms = fade_ms
        self.silence_per_word_ms = silence_per_word_ms
        self.min_segment_silence_ms = min_segment_silence_ms
        self.max_segment_silence_ms = max_segment_silence_ms

    def assemble(
        self,
        *,
        audio_segments: list[str],
        out_path: str,
        segment_word_counts: list[int] | None = None,
        segment_pause_multipliers: list[float] | None = None,
    ) -> str:
        try:
            from pydub import AudioSegment
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pydub is required for audio assembly. Install with `pip install pydub`.") from e

        if not audio_segments:
            raise ValueError("No audio segments provided.")
        word_counts = segment_word_counts or []
        pause_multipliers = segment_pause_multipliers or []

        combined = AudioSegment.silent(duration=300)  # short lead-in

        if self.intro_path and Path(self.intro_path).exists():
            intro = AudioSegment.from_file(self.intro_path)
            if self.fade_ms:
                intro = intro.fade_in(self.fade_ms)
            combined += intro

        gaps_applied: list[int] = []
        for idx, segment_path in enumerate(audio_segments):
            seg = AudioSegment.from_file(segment_path)
            combined += seg
            if idx < len(audio_segments) - 1:
                pause_mult = pause_multipliers[idx] if idx < len(pause_multipliers) else None
                gap_ms = self._gap_for_word_count(word_counts[idx] if idx < len(word_counts) else None, pause_mult)
                gaps_applied.append(gap_ms)
                combined += AudioSegment.silent(duration=max(0, gap_ms))

        if self.outro_path and Path(self.outro_path).exists():
            outro = AudioSegment.from_file(self.outro_path)
            if self.fade_ms:
                outro = outro.fade_out(self.fade_ms)
            combined += outro

        # Simple loudness normalization (approximate)
        if combined.dBFS != float("-inf"):
            gain = self.target_dbfs - combined.dBFS
            combined = combined.apply_gain(gain)
        # Soft limiting
        if combined.max_dBFS > -1.0:
            combined = combined.limit(-1.0)

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.export(out, format=self.target_format)

        # Write metadata (durations, target loudness, files used)
        try:
            meta_path = Path(self.metadata_path) if self.metadata_path else out.parent / "audio_metadata.json"
            metadata = {
                "target_dbfs": self.target_dbfs,
                "target_format": self.target_format,
                "intro_path": self.intro_path,
                "outro_path": self.outro_path,
                "segment_silence_ms": self.segment_silence_ms,
                "silence_per_word_ms": self.silence_per_word_ms,
                "min_segment_silence_ms": self.min_segment_silence_ms,
                "max_segment_silence_ms": self.max_segment_silence_ms,
                "fade_ms": self.fade_ms,
                "segment_files": audio_segments,
                "segment_word_counts": word_counts,
                "segment_pause_multipliers": pause_multipliers[: len(audio_segments)] if pause_multipliers else [],
                "applied_gaps_ms": gaps_applied,
                "duration_seconds": round(combined.duration_seconds, 2),
            }
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self.last_metadata_path = str(meta_path)
        except Exception:
            self.last_metadata_path = None

        return str(out)

    def _gap_for_word_count(self, word_count: int | None, pause_multiplier: float | None = None) -> int:
        if word_count is None or word_count <= 0:
            scaled = max(0, self.segment_silence_ms)
        else:
            scaled = int(self.segment_silence_ms + word_count * self.silence_per_word_ms)
        if pause_multiplier is not None:
            scaled = int(round(scaled * pause_multiplier))
        return int(min(self.max_segment_silence_ms, max(self.min_segment_silence_ms, scaled)))
