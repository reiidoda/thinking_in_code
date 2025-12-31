from __future__ import annotations

from pathlib import Path
from typing import List
import math
import re

from podcastify_contracts.podcast_job import EpisodeSegment
from podcastify_podcast.application.ports import TtsSynthesizer
from podcastify_podcast.infrastructure.audio.pacing import segment_rate_multiplier_for


class Pyttsx3Synthesizer(TtsSynthesizer):
    """Offline TTS using pyttsx3 (sapi5/nsss/espeak). Supports per-segment voice overrides."""

    def __init__(
        self,
        voice: str | None = None,
        rate: int | None = None,
        voice_map: dict[str, str] | None = None,
        math_rate_floor: int | None = None,
        math_rate_ceiling: int | None = None,
        math_rate_beta: float = 1.15,
    ) -> None:
        self.voice = voice
        self.rate = rate
        self.voice_map = voice_map or {}
        self.math_rate_floor = math_rate_floor
        self.math_rate_ceiling = math_rate_ceiling
        self.math_rate_beta = math_rate_beta

    def synthesize(self, *, segments: List[EpisodeSegment], out_dir: str) -> list[str]:
        try:
            import pyttsx3
        except Exception as e:  # pragma: no cover
            raise RuntimeError("pyttsx3 is required for TTS. Install with `pip install pyttsx3`.") from e

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        engine = pyttsx3.init()
        if self.voice:
            engine.setProperty("voice", self.voice)
        if self.rate:
            engine.setProperty("rate", self.rate)
        base_rate = self.rate or engine.getProperty("rate") or 180

        paths: list[str] = []
        for idx, seg in enumerate(segments, start=1):
            seg_voice = None
            if seg.voice:
                seg_voice = self.voice_map.get(seg.voice) or seg.voice
            if not seg_voice and seg.speaker:
                seg_voice = self.voice_map.get(seg.speaker)
            if seg_voice:
                engine.setProperty("voice", seg_voice)
            seg_rate = self._adaptive_rate(base_rate, seg.title, seg.text)
            engine.setProperty("rate", seg_rate)
            path = out / f"segment-{idx:02}.wav"
            engine.save_to_file(seg.text, str(path))
            engine.runAndWait()
            paths.append(str(path))

        return paths

    def _adaptive_rate(self, base_rate: int, title: str | None, text: str) -> int:
        """Rate control uses cognitive load and math density to improve clarity."""
        mult = segment_rate_multiplier_for(title, text)
        density = self._math_density(text)
        formula_hits = len(re.findall(r"[=+\\-*/^]", text))
        load = density + 0.35 * math.log1p(formula_hits)
        effective = math.sqrt(max(load, 0.0))
        damp = 1 / (1 + self.math_rate_beta * effective)
        target = int(base_rate * mult * damp)
        floor = self.math_rate_floor or int(base_rate * 0.7)
        ceil = self.math_rate_ceiling or int(base_rate * 1.05)
        return max(floor, min(ceil, target))

    def _math_density(self, text: str) -> float:
        tokens = re.findall(r"[A-Za-z0-9\\+\\-\\=\\*\\^\\/\\(\\)]+", text)
        if not tokens:
            return 0.0
        math_tokens = sum(1 for t in tokens if re.search(r"[0-9\\=\\+\\-\\*\\/\\^]", t))
        return math_tokens / max(len(tokens), 1)
