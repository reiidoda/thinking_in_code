from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field, ValidationError
from podcastify_contracts.podcast_job import JobStatus, PodcastJobRequest

from podcastify_api.queue import enqueue_dir, enqueue_file, enqueue_redis

app = FastAPI(title="Thinking in Code API")
log = logging.getLogger("podcastify_api")

def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path)


_load_env()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
STATIC_DIR = Path(__file__).resolve().parents[2] / "static"
ASSETS_DIR = Path(__file__).resolve().parents[4] / "assets"
JOBS_DIR = DATA_DIR / "jobs"
PROCESSED_DIR = DATA_DIR / "processed"
FEEDBACK_DIR = DATA_DIR / "feedback"
QUEUE_DIR = Path(os.getenv("QUEUE_DIR", str(DATA_DIR / "queue")))
QUEUE_FILE = Path(os.getenv("QUEUE_FILE", str(QUEUE_DIR / "queue.jsonl")))
QUEUE_MODE = os.getenv("QUEUE_MODE", "dir").lower()
QUEUE_REDIS_URL = os.getenv("QUEUE_REDIS_URL", "redis://localhost:6379/0")
QUEUE_REDIS_KEY = os.getenv("QUEUE_REDIS_KEY", "podcastify:jobs")
API_KEY = os.getenv("API_KEY", "")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


def _ensure_data_dirs() -> None:
    for path in [DATA_DIR, JOBS_DIR, PROCESSED_DIR, FEEDBACK_DIR, QUEUE_DIR, DATA_DIR / "research"]:
        path.mkdir(parents=True, exist_ok=True)


_ensure_data_dirs()


def _init_logging() -> None:
    fmt = os.getenv("LOG_FORMAT", "plain").lower()
    if fmt == "json":
        logging.basicConfig(level=logging.INFO, format='{"level":"%(levelname)s","message":"%(message)s"}')
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


_init_logging()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _write_status(
    job_id: str,
    *,
    status: JobStatus,
    stage: str | None = None,
    detail: str | None = None,
    error: str | None = None,
    result_path: str | None = None,
) -> None:
    payload = {
        "job_id": job_id,
        "status": status.value,
        "stage": stage,
        "detail": detail,
        "error": error,
        "result_path": result_path,
        "updated_at": _now_iso(),
    }
    path = _status_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _require_api_key(request: Request) -> None:
    if not API_KEY:
        return
    header = request.headers.get("x-api-key") or request.headers.get("authorization")
    token = header.replace("Bearer ", "").strip() if header else None
    if not token:
        token = request.query_params.get("api_key")
    if not token:
        raise HTTPException(status_code=401, detail="Missing API key")
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _enqueue_job(job_id: str, *, meta: dict | None = None) -> None:
    if QUEUE_MODE == "redis":
        enqueue_redis(QUEUE_REDIS_URL, QUEUE_REDIS_KEY, job_id=job_id, meta=meta)
    elif QUEUE_MODE == "file":
        enqueue_file(QUEUE_FILE, job_id, meta=meta)
    else:
        enqueue_dir(QUEUE_DIR, job_id, meta=meta)


def _queue_depth() -> int:
    if QUEUE_MODE == "redis":
        try:
            import redis  # type: ignore
        except Exception:
            return 0
        client = redis.Redis.from_url(QUEUE_REDIS_URL)
        return int(client.zcard(QUEUE_REDIS_KEY))
    if QUEUE_MODE == "file":
        if not QUEUE_FILE.exists():
            return 0
        lines = QUEUE_FILE.read_text(encoding="utf-8").splitlines()
        offset_path = QUEUE_FILE.with_suffix(QUEUE_FILE.suffix + ".offset")
        try:
            offset = int(offset_path.read_text(encoding="utf-8"))
        except Exception:
            offset = 0
        return max(0, len(lines) - offset)
    if not QUEUE_DIR.exists():
        return 0
    return len(list(QUEUE_DIR.glob("*.json")))


def _job_status_counts() -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for status_path in JOBS_DIR.glob("*/status.json"):
        data = _read_json(status_path) or {}
        status = data.get("status", "unknown")
        counts[status] += 1
    return dict(counts)


def _stage_metrics() -> dict[str, dict[str, float]]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for metrics_path in PROCESSED_DIR.glob("*/job_metrics.json"):
        data = _read_json(metrics_path)
        if not data:
            continue
        for key, value in data.items():
            if isinstance(value, (int, float)):
                totals[key] += float(value)
                counts[key] += 1
    averages = {k: round(totals[k] / counts[k], 4) for k in totals if counts[k] > 0}
    return {"average_seconds": averages, "samples": dict(counts)}


def _audio_metrics() -> dict[str, Any]:
    total = 0
    with_audio = 0
    for result_path in PROCESSED_DIR.glob("*/result.json"):
        data = _read_json(result_path)
        if not data:
            continue
        total += 1
        artifacts = data.get("artifacts") or []
        has_audio = False
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            if artifact.get("kind") == "audio_mp3":
                has_audio = True
                break
            path = artifact.get("path")
            if isinstance(path, str) and path.endswith(".mp3"):
                has_audio = True
                break
        if has_audio:
            with_audio += 1
    rate = (with_audio / total) if total else 0.0
    return {"total_results": total, "with_audio": with_audio, "audio_success_rate": round(rate, 4)}


class FeedbackPayload(BaseModel):
    message: str = Field(..., min_length=4, max_length=4000)
    name: str | None = None
    email: str | None = None
    job_id: str | None = None
    context: dict[str, Any] | None = None


def _safe_artifact_path(job_id: str, artifact_name: str) -> Path:
    job_dir = (PROCESSED_DIR / job_id).resolve()
    candidate = (job_dir / artifact_name).resolve()
    if job_dir not in candidate.parents and candidate != job_dir:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return candidate


def _job_artifacts(job_id: str) -> list[dict[str, Any]]:
    result_path = PROCESSED_DIR / job_id / "result.json"
    result = _read_json(result_path)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    artifacts: list[dict[str, Any]] = []
    for artifact in result.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str):
            continue
        name = Path(raw_path).name
        try:
            path = _safe_artifact_path(job_id, name)
        except HTTPException:
            continue
        artifacts.append(
            {
                "name": name,
                "kind": artifact.get("kind"),
                "path": str(path),
                "download_url": f"/v1/jobs/{job_id}/artifacts/{name}",
                "size_bytes": path.stat().st_size,
            }
        )
    return artifacts


class FeedbackPayload(BaseModel):
    message: str = Field(..., min_length=4, max_length=4000)
    name: str | None = Field(default=None, max_length=120)
    email: str | None = Field(default=None, max_length=200)
    job_id: str | None = Field(default=None, max_length=80)
    context: dict[str, Any] | None = None


def _safe_artifact_path(job_id: str, artifact_name: str) -> Path:
    job_dir = (PROCESSED_DIR / job_id).resolve()
    candidate = (job_dir / artifact_name).resolve()
    if job_dir not in candidate.parents and candidate != job_dir:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return candidate


def _job_artifacts(job_id: str) -> list[dict[str, Any]]:
    result_path = PROCESSED_DIR / job_id / "result.json"
    result = _read_json(result_path)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    artifacts: list[dict[str, Any]] = []
    for artifact in result.get("artifacts", []):
        if not isinstance(artifact, dict):
            continue
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str):
            continue
        name = Path(raw_path).name
        try:
            path = _safe_artifact_path(job_id, name)
        except HTTPException:
            continue
        artifacts.append(
            {
                "name": name,
                "kind": artifact.get("kind"),
                "path": str(path),
                "download_url": f"/v1/jobs/{job_id}/artifacts/{name}",
                "size_bytes": path.stat().st_size,
            }
        )
    return artifacts


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_model=None)
def root() -> Response:
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"status": "ok"})


@app.post("/v1/jobs", dependencies=[Depends(_require_api_key)])
async def create_job(
    file: UploadFile = File(...),
    language: str = Form("en"),
    style: str = Form("everyday"),
    target_minutes: int = Form(8),
    target_seconds: int = Form(0),
    job_id: str | None = Form(None),
) -> JSONResponse:
    job_id = job_id or f"job-{uuid.uuid4().hex[:8]}"
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(file.filename or "upload.pdf").name
    if not filename:
        filename = "upload.pdf"
    input_path = job_dir / filename
    content = await file.read()
    input_path.write_bytes(content)

    try:
        req = PodcastJobRequest(
            input_filename=filename,
            language=language,
            style=style,
            target_minutes=target_minutes,
            target_seconds=target_seconds,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    (job_dir / "request.json").write_text(req.model_dump_json(indent=2), encoding="utf-8")

    _write_status(job_id, status=JobStatus.QUEUED, stage="queued")
    _enqueue_job(job_id, meta={"num_pdfs": 1, "minutes": target_minutes})

    return JSONResponse(
        {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "request": req.model_dump(),
        }
    )


@app.get("/v1/jobs/{job_id}/status", dependencies=[Depends(_require_api_key)])
def job_status(job_id: str) -> JSONResponse:
    status = _read_json(_status_path(job_id))
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(status)


@app.get("/v1/jobs/{job_id}/result", dependencies=[Depends(_require_api_key)])
def job_result(job_id: str) -> JSONResponse:
    result_path = PROCESSED_DIR / job_id / "result.json"
    result = _read_json(result_path)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return JSONResponse(result)


@app.get("/v1/jobs/{job_id}/artifacts", dependencies=[Depends(_require_api_key)])
def job_artifacts(job_id: str) -> JSONResponse:
    artifacts = _job_artifacts(job_id)
    return JSONResponse({"job_id": job_id, "artifacts": artifacts})


@app.get("/v1/jobs/{job_id}/artifacts/{artifact_name}", dependencies=[Depends(_require_api_key)])
def job_artifact_download(job_id: str, artifact_name: str) -> FileResponse:
    path = _safe_artifact_path(job_id, artifact_name)
    return FileResponse(path, filename=artifact_name)


@app.get("/v1/jobs/{job_id}/progress", dependencies=[Depends(_require_api_key)])
async def job_progress(job_id: str) -> StreamingResponse:
    async def event_stream():
        last_payload = None
        while True:
            status = _read_json(_status_path(job_id))
            if not status:
                payload = json.dumps({"error": "Job not found"})
                yield f"event: error\ndata: {payload}\n\n"
                break
            payload = json.dumps(status)
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if status.get("status") in {JobStatus.SUCCEEDED.value, JobStatus.FAILED.value}:
                break
            await asyncio.sleep(1.0)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/v1/metrics/summary", dependencies=[Depends(_require_api_key)])
def metrics_summary() -> JSONResponse:
    payload = {
        "queue_mode": QUEUE_MODE,
        "queue_depth": _queue_depth(),
        "job_counts": _job_status_counts(),
        "stage_metrics": _stage_metrics(),
        "audio_metrics": _audio_metrics(),
        "generated_at": _now_iso(),
    }
    return JSONResponse(payload)


@app.post("/v1/feedback", dependencies=[Depends(_require_api_key)])
def submit_feedback(payload: FeedbackPayload, request: Request) -> JSONResponse:
    feedback_id = f"feedback-{uuid.uuid4().hex[:8]}"
    record = {
        "feedback_id": feedback_id,
        "message": payload.message,
        "name": payload.name,
        "email": payload.email,
        "job_id": payload.job_id,
        "context": payload.context,
        "user_agent": request.headers.get("user-agent"),
        "received_at": _now_iso(),
    }
    path = FEEDBACK_DIR / f"{feedback_id}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return JSONResponse({"feedback_id": feedback_id, "status": "received"})
