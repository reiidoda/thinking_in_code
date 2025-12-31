from __future__ import annotations

import os
import time
from pathlib import Path
from podcastify_podcast.application.ports import JobStore

class FileSystemJobStore(JobStore):
    def __init__(self, base_dir: str = "data/processed") -> None:
        self.base_dir = Path(base_dir)
        self.write_retries = int(os.getenv("ARTIFACT_WRITE_RETRIES", "3"))
        self.write_backoff_s = float(os.getenv("ARTIFACT_WRITE_BACKOFF_S", "0.2"))

    def job_dir(self, job_id: str) -> Path:
        d = self.base_dir / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_artifact(self, *, job_id: str, name: str, content: bytes) -> str:
        path = self.job_dir(job_id) / name
        self._write_bytes_with_retry(path, content)
        return str(path)

    def _write_bytes_with_retry(self, path: Path, content: bytes) -> None:
        last_exc: Exception | None = None
        for attempt in range(max(1, self.write_retries)):
            try:
                path.write_bytes(content)
                return
            except Exception as exc:  # pragma: no cover - defensive
                last_exc = exc
                time.sleep(self.write_backoff_s * (2**attempt))
        if last_exc:
            raise last_exc
