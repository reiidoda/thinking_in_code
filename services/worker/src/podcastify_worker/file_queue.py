from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def enqueue(queue_file: Path, job_id: str, meta: dict[str, Any] | None = None) -> str:
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"job_id": job_id, "meta": meta or {}}
    with queue_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload))
        f.write("\n")
    return str(queue_file)


def _offset_path(queue_file: Path) -> Path:
    return queue_file.with_suffix(queue_file.suffix + ".offset")


def dequeue(queue_file: Path) -> dict[str, Any] | None:
    if not queue_file.exists():
        return None
    lines = queue_file.read_text(encoding="utf-8").splitlines()
    offset_path = _offset_path(queue_file)
    try:
        offset = int(offset_path.read_text(encoding="utf-8"))
    except Exception:
        offset = 0
    if offset >= len(lines):
        return None
    line = lines[offset]
    offset_path.write_text(str(offset + 1), encoding="utf-8")
    try:
        return json.loads(line)
    except Exception:
        return {"job_id": line.strip()}


def queue_depth(queue_file: Path) -> int:
    if not queue_file.exists():
        return 0
    lines = queue_file.read_text(encoding="utf-8").splitlines()
    offset_path = _offset_path(queue_file)
    try:
        offset = int(offset_path.read_text(encoding="utf-8"))
    except Exception:
        offset = 0
    return max(0, len(lines) - offset)
