from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _score_job(priority: float | None, meta: dict[str, Any] | None) -> float:
    if priority is not None:
        return float(priority)
    meta = meta or {}
    num_pdfs = int(meta.get("num_pdfs") or 1)
    minutes = float(meta.get("minutes") or 0)
    base = 100.0 + min(max(num_pdfs, 1), 10) * 5.0
    if minutes:
        base += max(0.0, 20.0 - minutes)
    return base + (time.time() % 1) * 0.01


def enqueue_dir(queue_dir: Path, job_id: str, meta: dict[str, Any] | None = None) -> str:
    queue_dir.mkdir(parents=True, exist_ok=True)
    payload = {"job_id": job_id, "meta": meta or {}}
    path = queue_dir / f"{job_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def enqueue_file(queue_file: Path, job_id: str, meta: dict[str, Any] | None = None) -> str:
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"job_id": job_id, "meta": meta or {}}
    with queue_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload))
        f.write("\n")
    return str(queue_file)


def enqueue_redis(
    redis_url: str,
    key: str,
    *,
    job_id: str,
    meta: dict[str, Any] | None = None,
    priority: float | None = None,
) -> float:
    try:
        import redis  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("redis is required for QUEUE_MODE=redis. Install with `pip install redis`.") from e

    score = _score_job(priority, meta)
    payload = json.dumps({"job_id": job_id, "meta": meta or {}})
    client = redis.Redis.from_url(redis_url)
    client.zadd(key, {payload: score})
    return score
