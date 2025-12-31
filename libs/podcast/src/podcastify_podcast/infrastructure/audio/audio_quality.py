from __future__ import annotations

import json
from pathlib import Path

from pydub import AudioSegment
import pyloudnorm as pyln  # type: ignore


class AudioQualityChecker:
    """Compute simple audio quality metrics."""

    def analyze(self, path: str) -> dict:
        audio = AudioSegment.from_file(path)
        duration = round(audio.duration_seconds, 2)
        loudness = round(audio.dBFS, 2) if audio.dBFS != float("-inf") else -120.0
        peak = round(audio.max_dBFS, 2) if hasattr(audio, "max_dBFS") else loudness
        clipping = peak > -0.1
        # LUFS estimate
        meter = pyln.Meter(audio.frame_rate)
        samples = audio.get_array_of_samples()
        lufs = round(meter.integrated_loudness(pyln.util.to_float32(samples)), 2)

        return {
            "duration_seconds": duration,
            "loudness_dbfs": loudness,
            "peak_dbfs": peak,
            "clipping": clipping,
            "lufs": lufs,
            "channels": audio.channels,
            "frame_rate": audio.frame_rate,
            "sample_width": audio.sample_width,
        }

    def write_report(self, path: str, out_path: str) -> str:
        report = self.analyze(path)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return str(out)
