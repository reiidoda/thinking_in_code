from __future__ import annotations

import os
import time
from pathlib import Path
import httpx

from podcastify_contracts.podcast_job import EpisodeSegment
from podcastify_podcast.application.ports import TtsSynthesizer
from podcastify_podcast.infrastructure.logging import get_logger
from podcastify_podcast.infrastructure.audio.pacing import segment_rate_multiplier_for

log = get_logger(__name__)


class MinimaxSynthesizer(TtsSynthesizer):
    """
    Cloud TTS using Minimax Speech async API.

    Endpoints:
    - POST https://api.minimax.io/v1/t2a_async_v2
    - GET  https://api.minimax.io/v1/query/t2a_async_query_v2?task_id=...
    - GET  https://api.minimax.io/v1/files/retrieve_content?file_id=...
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "speech-2.6-hd",
        voice_male: str = "English_Explanatory_Man",
        voice_female: str = "English_captivating_female1",
        timeout_s: float = 30.0,
        poll_interval_s: float = 2.0,
        max_attempts: int = 60,
        speed: float = 1.0,
        speed_floor: float = 0.85,
        speed_ceiling: float = 1.15,
    ) -> None:
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        self.model = model
        self.voice_male = voice_male
        self.voice_female = voice_female
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.max_attempts = max_attempts
        self.speed = speed
        self.speed_floor = speed_floor
        self.speed_ceiling = speed_ceiling
        if not self.api_key:
            raise RuntimeError("MinimaxSynthesizer requires MINIMAX_API_KEY")

        self.base_url = "https://api.minimax.io/v1/t2a_async_v2"
        self.query_url = "https://api.minimax.io/v1/query/t2a_async_query_v2"
        self.file_url = "https://api.minimax.io/v1/files/retrieve_content"

    def synthesize(self, *, segments: list[EpisodeSegment], out_dir: str) -> list[str]:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        for idx, seg in enumerate(segments, start=1):
            voice = self._voice_for(seg)
            target = Path(out_dir) / f"segment-{idx:02}.mp3"
            log.info("minimax.tts segment=%s speaker=%s voice=%s", idx, seg.speaker, voice)
            try:
                speed = self._adaptive_speed(seg.title, seg.text)
                file_id = self._submit_and_poll(seg.text, voice, speed)
                self._download(file_id, target)
                paths.append(str(target))
            except Exception as e:
                raise RuntimeError(f"Minimax TTS failed for segment {idx}: {e}") from e
        return paths

    def _voice_for(self, seg: EpisodeSegment) -> str:
        # Use explicit voice if provided; otherwise derive from speaker name.
        if seg.voice:
            return seg.voice
        name = (seg.speaker or "").lower()
        if any(k in name for k in ["host 2", "female", "she", "her"]):
            return self.voice_female
        return self.voice_male

    def _submit_and_poll(self, text: str, voice: str, speed: float) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "text": text,
            "voice_setting": {
                "voice_id": voice,
                "speed": speed,
                "vol": 1.0,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
            },
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(self.base_url, headers=headers, json=payload)
            r.raise_for_status()
            task_id = r.json().get("task_id")
        if not task_id:
            raise RuntimeError("No task_id returned from Minimax.")

        # Poll for completion
        for _ in range(self.max_attempts):
            with httpx.Client(timeout=self.timeout_s) as client:
                qr = client.get(f"{self.query_url}?task_id={task_id}", headers=headers)
                qr.raise_for_status()
                data = qr.json()
                status = data.get("status")
                if status == "Success":
                    file_id = data.get("file_id")
                    if not file_id:
                        raise RuntimeError("No file_id returned after success.")
                    return file_id
                if status == "Failed":
                    raise RuntimeError(f"Minimax task failed: {data.get('error')}")
            time.sleep(self.poll_interval_s)
        raise RuntimeError("Minimax TTS polling timed out.")

    def _adaptive_speed(self, title: str | None, text: str) -> float:
        mult = segment_rate_multiplier_for(title, text)
        target = self.speed * mult
        return max(self.speed_floor, min(self.speed_ceiling, target))

    def _download(self, file_id: str, target: Path) -> None:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.get(f"{self.file_url}?file_id={file_id}", headers=headers)
            r.raise_for_status()
            target.write_bytes(r.content)
