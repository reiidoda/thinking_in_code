from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List
import math
import re

from podcastify_contracts.podcast_job import EpisodeSegment
from podcastify_podcast.application.ports import TtsSynthesizer
from podcastify_podcast.infrastructure.audio.pacing import segment_pause_multiplier_for


class PiperSynthesizer(TtsSynthesizer):
    """Piper CLI synthesizer (requires piper binary and model)."""

    def __init__(
        self,
        model_path: str,
        speaker: int | None = None,
        speaker_map: dict[str, int] | None = None,
        sentence_silence: float = 0.2,
        math_silence_gain: float = 0.8,
        math_silence_ceiling: float = 0.8,
    ) -> None:
        self.model_path = model_path
        self.speaker = speaker
        self.speaker_map = speaker_map or {}
        self.sentence_silence = sentence_silence
        self.math_silence_gain = math_silence_gain
        self.math_silence_ceiling = math_silence_ceiling

    def synthesize(self, *, segments: List[EpisodeSegment], out_dir: str) -> list[str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for idx, seg in enumerate(segments, start=1):
            path = out / f"segment-{idx:02}.wav"
            silence = self._adaptive_silence(self.sentence_silence, seg.title, seg.text)
            speaker = self._speaker_for(seg)
            cmd = [
                "piper",
                "--model",
                self.model_path,
                "--output_file",
                str(path),
                "--sentence_silence",
                f"{silence:.3f}",
            ]
            if speaker is not None:
                cmd += ["--speaker", str(speaker)]
            try:
                subprocess.run(cmd, input=seg.text.encode("utf-8"), check=True)
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"Piper synthesis failed: {e}") from e
            paths.append(str(path))
        return paths

    def _adaptive_silence(self, base: float, title: str | None, text: str) -> float:
        """Increase inter-sentence pause when equations are dense."""
        pause_mult = segment_pause_multiplier_for(title, text)
        density = self._math_density(text)
        formula_hits = len(re.findall(r"[=+\\-*/^]", text))
        load = density + 0.35 * math.log1p(formula_hits)
        factor = 1 + self.math_silence_gain * load
        boosted = base * pause_mult * factor
        ceiling = base * pause_mult * (1 + self.math_silence_ceiling)
        return min(ceiling, boosted)

    def _speaker_for(self, seg: EpisodeSegment) -> int | None:
        if seg.voice:
            mapped = self.speaker_map.get(seg.voice)
            if mapped is not None:
                return mapped
            if str(seg.voice).isdigit():
                return int(seg.voice)
        return self.speaker

    def _math_density(self, text: str) -> float:
        tokens = re.findall(r"[A-Za-z0-9\\+\\-\\=\\*\\^\\/\\(\\)]+", text)
        if not tokens:
            return 0.0
        math_tokens = sum(1 for t in tokens if re.search(r"[0-9\\=\\+\\-\\*\\/\\^]", t))
        return math_tokens / max(len(tokens), 1)
