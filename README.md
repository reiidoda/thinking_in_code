# <img src="assets/thinking_in_code_icon.png" alt="Thinking in Code" width="28" style="vertical-align:middle; border-radius:6px;" /> Thinking in Code

<img src="assets/thinking_in_code.png" alt="Thinking in Code" width="100%" />

Thinking in Code is a local-first system that turns research PDFs into a cited, listener-ready podcast script with optional narration and audio artifacts. The API registers uploads and creates a job. A worker processes the job through extraction, chunking, script generation, and optional TTS/audio assembly. All outputs are written to `data/processed/<job_id>/` and progress is tracked in `data/jobs/<job_id>/status.json` (streamed via SSE).

## Highlights
- Evidence-aware script generation with per-segment citations.
- Optional retrieval (local vectors, Chroma, or FAISS) for grounding.
- Pluggable LLM and TTS providers (Ollama, OpenRouter, pyttsx3, Piper, Coqui, Minimax).
- Audio assembly with loudness normalization and QA metadata.
- Web Studio for uploads, status tracking, artifact downloads, and feedback.

## How it works
1) Extract text from PDFs with page-level citation metadata.
2) Normalize and chunk text while preserving citations.
3) Generate a structured episode script with guardrails and evidence checks.
4) Optionally embed/chunk for retrieval.
5) Optionally synthesize audio and assemble the final mix.

## Outputs
Common artifacts written under `data/processed/<job_id>/`:
- `script.md` (narration script)
- `transcript.txt` and `transcript.srt`
- `quality.json` / `quality.log` (evidence + retention checks)
- `job_metrics.json` (stage timings)
- `episode.mp3` (when audio is enabled)
- `audio_metadata.json` / `audio_quality.json` (when audio is enabled)

## Architecture
- API service (`services/api`): upload, job creation, SSE progress, artifact access.
- Worker service (`services/worker`): pipeline execution and artifact generation.
- Core engine (`libs/podcast`): clean-architecture pipeline + adapters.
- Shared contracts (`libs/contracts`): stable schemas across services.
- Prompts (`prompts/`): LLM prompt templates.
- Data (`data/`): runtime storage (jobs, processed outputs, research).

## Repository structure

| Path                         | Purpose                                      |
| ---------------------------- | -------------------------------------------- |
| `services/api`               | FastAPI service (job submit/status, SSE)     |
| `services/worker`            | Worker pipeline (PDF -> script -> TTS -> audio) |
| `libs/contracts`             | Shared Pydantic models                       |
| `libs/podcast`               | Core engine (domain/application/infrastructure) |
| `prompts/`                   | Prompt templates                             |
| `scripts/`                   | Utilities (e.g., Redis enqueue helper)       |
| `data/`                      | Runtime: jobs, processed outputs, research   |
| `Dockerfile.api`, `Dockerfile.worker` | Container builds                    |
| `docker-compose.yaml`        | Local stack (API + worker + Redis + Ollama)  |

## Web Studio
The Studio UI is served at `/` from the API service. It supports upload, job status, artifact downloads, and feedback submission. If `API_KEY` is set, enter it in the Studio panel to authorize requests.

## API surface (quick reference)
- `GET /health`
- `POST /v1/jobs` (multipart: `file`, `language`, `style`, `target_minutes`, `target_seconds`)
- `GET /v1/jobs/{job_id}/status`
- `GET /v1/jobs/{job_id}/progress` (SSE)
- `GET /v1/jobs/{job_id}/result`
- `GET /v1/jobs/{job_id}/artifacts` (list artifacts + download URLs)
- `GET /v1/jobs/{job_id}/artifacts/{artifact_name}` (download)
- `GET /v1/metrics/summary` (queue depth, stage timings, audio success)
- `POST /v1/feedback`

## Configuration
Copy the example env files and keep queue settings aligned between API and worker:
```
cp services/api/.env.example services/api/.env
cp services/worker/.env.example services/worker/.env
```

Key settings:
- Queue: `QUEUE_MODE` (`dir`, `file`, `redis`), `QUEUE_REDIS_URL`, `QUEUE_REDIS_KEY`
- LLM: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_NUM_PREDICT`, `MAX_CONTEXT_CHARS`
- Script provider (optional): `SCRIPT_PROVIDER=openrouter`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`
- TTS/audio: `ENABLE_TTS`, `ENABLE_AUDIO`, `TTS_PROVIDER`, `AUDIO_FORMAT`, `AUDIO_TARGET_DBFS`
- Narration personalization: `VOICE_PROFILE_MAP`, `SEGMENT_EMPHASIS_MAP`, `PACING_PRESET` (or `PACING_MODE`)
- Voice mapping: `TTS_VOICE_MAP`, `PIPER_SPEAKER_MAP`, `COQUI_SPEAKER_MAP`
- Metrics: `METRICS_ENABLED`, `METRICS_PORT`

Optional retrieval extras:
- Install with `pip install -e "libs/podcast[retrieval]"` to enable Chroma and sentence-transformers.
- Optional lock files: `libs/podcast/requirements.optional.lock` (Python 3.11) and `libs/podcast/requirements.optional.py312.lock` (Python 3.12).

Local-only data:
- Keep runtime outputs under `data/` out of version control (the repo `.gitignore` already covers them).

## Build and run

### Local (venv)
1) Prereqs: Python 3.11+, Ollama running, FFmpeg, optional tesseract.
2) Install:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -U pip
   pip install -e libs/contracts -e libs/podcast -e services/api -e services/worker
   ```
3) Run API:
   ```bash
   uvicorn podcastify_api.main:app --reload --port 8000
   ```
4) Run worker:
   ```bash
   python -m podcastify_worker.main
   ```
5) Submit a job:
   ```bash
   curl -F "file=@/path/to/paper.pdf" -F "language=en" -F "style=everyday" -F "target_minutes=8" http://localhost:8000/v1/jobs
   ```
6) Stream progress:
   ```bash
   curl -N http://localhost:8000/v1/jobs/<job_id>/progress
   ```

### Docker Compose (API + worker + Redis + Ollama)
```bash
docker-compose up --build -d
docker exec -it ollama ollama pull llama3:instruct
# Optional fallback/smaller: docker exec -it ollama ollama pull llama3:8b
# Optional embeddings: docker exec -it ollama ollama pull nomic-embed-text
```
- API: http://localhost:8000 (set `API_KEY` to require a key)
- Worker: pulls jobs from Redis sorted set
- Outputs: `data/processed/<job_id>/` (script, quality, metrics, audio if enabled)

## Testing
```bash
PYTHONPATH=libs/podcast/src:libs/contracts/src pytest -q
```
