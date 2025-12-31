from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def enqueue(queue_dir: Path, job_id: str, meta: dict[str, Any] | None = None) -> str:
    queue_dir.mkdir(parents=True, exist_ok=True)
    payload = {"job_id": job_id, "meta": meta or {}}
    path = queue_dir / f"{job_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def dequeue(queue_dir: Path) -> dict[str, Any] | None:
    if not queue_dir.exists():
        return None
    files = sorted(queue_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {"job_id": path.stem}
        try:
            path.unlink()
        except Exception:
            pass
        return payload
    return None


def queue_depth(queue_dir: Path) -> int:
    if not queue_dir.exists():
        return 0
    return len(list(queue_dir.glob("*.json")))
