from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from podcastify_contracts.podcast_job import EpisodeSegment

from podcastify_podcast.application.ports import TtsSynthesizer
from podcastify_podcast.infrastructure.audio.pacing import segment_rate_multiplier_for


class CoquiSynthesizer(TtsSynthesizer):
    """Coqui TTS via CLI (`tts` command). Requires `pip install TTS`."""

    def __init__(
        self,
        model_path: str,
        speaker_idx: int | None = None,
        speed: float | None = None,
        speaker_map: dict[str, int] | None = None,
    ) -> None:
        self.model_path = model_path
        self.speaker_idx = speaker_idx
        self.speed = speed
        self.speaker_map = speaker_map or {}

    def synthesize(self, *, segments: list[EpisodeSegment], out_dir: str) -> list[str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        for idx, seg in enumerate(segments, start=1):
            path = out / f"segment-{idx:02}.wav"
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
                tf.write(seg.text)
                tmp_txt = tf.name

            speed = (self.speed if self.speed is not None else 1.0) * segment_rate_multiplier_for(seg.title, seg.text)
            cmd = [
                "tts",
                "--text_file",
                tmp_txt,
                "--model_path",
                self.model_path,
                "--out_path",
                str(path),
            ]
            speaker_idx = self._speaker_for(seg)
            if speaker_idx is not None:
                cmd += ["--speaker_idx", str(speaker_idx)]
            cmd += ["--speed", f"{speed:.2f}"]

            try:
                subprocess.run(cmd, check=True)
            except Exception as e:  # pragma: no cover
                raise RuntimeError(f"Coqui synthesis failed: {e}") from e
            finally:
                try:
                    Path(tmp_txt).unlink(missing_ok=True)
                except Exception:
                    pass

            paths.append(str(path))

        return paths

    def _speaker_for(self, seg: EpisodeSegment) -> int | None:
        if seg.voice:
            mapped = self.speaker_map.get(seg.voice)
            if mapped is not None:
                return mapped
            if str(seg.voice).isdigit():
                return int(seg.voice)
        return self.speaker_idx
