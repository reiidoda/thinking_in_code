from __future__ import annotations

import json
import time
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


def enqueue(
    redis_url: str,
    key: str,
    *,
    job_id: str,
    priority: float | None = None,
    meta: dict[str, Any] | None = None,
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


def dequeue(redis_url: str, key: str) -> dict[str, Any] | None:
    try:
        import redis  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("redis is required for QUEUE_MODE=redis. Install with `pip install redis`.") from e

    client = redis.Redis.from_url(redis_url)
    items = client.zpopmax(key, count=1)
    if not items:
        return None
    raw, score = items[0]
    payload = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    try:
        data = json.loads(payload)
    except Exception:
        data = {"job_id": payload}
    data["score"] = float(score)
    return data


def queue_depth(redis_url: str, key: str) -> int:
    try:
        import redis  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("redis is required for QUEUE_MODE=redis. Install with `pip install redis`.") from e

    client = redis.Redis.from_url(redis_url)
    return int(client.zcard(key))
