from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from podcastify_contracts.podcast_job import JobStatus, PodcastJobRequest, PodcastJobResult
from podcastify_podcast.application.use_cases import GeneratePodcastFromPdf
from podcastify_podcast.infrastructure.audio.pydub_assembler import PydubAudioAssembler
from podcastify_podcast.infrastructure.embedding.sentence_transformers import (
    SentenceTransformersEmbedder,
)
from podcastify_podcast.infrastructure.llm.ollama import (
    OllamaEmbeddingGenerator,
    OllamaTextGenerator,
)
from podcastify_podcast.infrastructure.llm.openrouter import OpenRouterTextGenerator
from podcastify_podcast.infrastructure.logging import (
    clear_correlation_id,
    get_logger,
    maybe_init_tracing,
    set_correlation_id,
    setup_logging,
)
from podcastify_podcast.infrastructure.metrics import (
    job_failed,
    job_started,
    job_succeeded,
    maybe_start_server,
    observe_stage,
    set_queue_depth,
)
from podcastify_podcast.infrastructure.pdf.extractors import (
    PdfPlumberExtractor,
    TopicCorpusExtractor,
)
from podcastify_podcast.infrastructure.retrieval.chroma_store import ChromaVectorStore
from podcastify_podcast.infrastructure.retrieval.faiss_store import FaissVectorStore
from podcastify_podcast.infrastructure.retrieval.local_store import LocalVectorStore
from podcastify_podcast.infrastructure.script.openrouter_writer import OpenRouterScriptWriter
from podcastify_podcast.infrastructure.script.script_writer import OllamaScriptWriter
from podcastify_podcast.infrastructure.storage.fs_store import FileSystemJobStore
from podcastify_podcast.infrastructure.text.chunking import SentenceChunker
from podcastify_podcast.infrastructure.tts.coqui_adapter import CoquiSynthesizer
from podcastify_podcast.infrastructure.tts.minimax_adapter import MinimaxSynthesizer
from podcastify_podcast.infrastructure.tts.piper_adapter import PiperSynthesizer
from podcastify_podcast.infrastructure.tts.pyttsx3_adapter import Pyttsx3Synthesizer

from podcastify_worker.dir_queue import dequeue as dir_dequeue
from podcastify_worker.dir_queue import queue_depth as dir_queue_depth
from podcastify_worker.file_queue import dequeue as file_dequeue
from podcastify_worker.file_queue import queue_depth as file_queue_depth
from podcastify_worker.redis_queue import dequeue as redis_dequeue
from podcastify_worker.redis_queue import queue_depth as redis_queue_depth

log = get_logger("podcastify_worker")

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
JOBS_DIR = DATA_DIR / "jobs"
PROCESSED_DIR = DATA_DIR / "processed"
QUEUE_DIR = Path(os.getenv("QUEUE_DIR", str(DATA_DIR / "queue")))
QUEUE_FILE = Path(os.getenv("QUEUE_FILE", str(QUEUE_DIR / "queue.jsonl")))
QUEUE_MODE = os.getenv("QUEUE_MODE", "dir").lower()
QUEUE_REDIS_URL = os.getenv("QUEUE_REDIS_URL", "redis://localhost:6379/0")
QUEUE_REDIS_KEY = os.getenv("QUEUE_REDIS_KEY", "podcastify:jobs")
POLL_INTERVAL_S = float(os.getenv("QUEUE_POLL_INTERVAL_S", "2.0"))
WRITE_RETRIES = int(os.getenv("WRITE_RETRIES", "4"))
WRITE_BACKOFF_S = float(os.getenv("WRITE_BACKOFF_S", "0.2"))


def _ensure_data_dirs() -> None:
    for path in [DATA_DIR, JOBS_DIR, PROCESSED_DIR, QUEUE_DIR, DATA_DIR / "research"]:
        path.mkdir(parents=True, exist_ok=True)


_ensure_data_dirs()


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _parse_map_env(name: str) -> dict[str, str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k).lower(): str(v) for k, v in data.items()}
    except Exception:
        pass
    mapping: dict[str, str] = {}
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        if "=" in chunk:
            key, val = chunk.split("=", 1)
        elif ":" in chunk:
            key, val = chunk.split(":", 1)
        else:
            continue
        mapping[key.strip().lower()] = val.strip()
    return mapping


def _parse_int_map_env(name: str) -> dict[str, int]:
    raw_map = _parse_map_env(name)
    mapping: dict[str, int] = {}
    for key, value in raw_map.items():
        try:
            mapping[key] = int(value)
        except ValueError:
            continue
    return mapping


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
    path = JOBS_DIR / job_id / "status.json"
    _write_json_with_retry(path, payload, raise_on_fail=False)


def _write_json_with_retry(path: Path, payload: dict, *, raise_on_fail: bool) -> None:
    last_exc: Exception | None = None
    for attempt in range(max(1, WRITE_RETRIES)):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(WRITE_BACKOFF_S * (2**attempt))
    if last_exc:
        if raise_on_fail:
            raise last_exc
        log.error("Failed to write %s: %s", path, last_exc)


def _load_request(job_id: str) -> PodcastJobRequest:
    job_dir = JOBS_DIR / job_id
    req_path = job_dir / "request.json"
    if req_path.exists():
        data = json.loads(req_path.read_text(encoding="utf-8"))
        return PodcastJobRequest.model_validate(data)

    pdfs = sorted(job_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No request.json or PDF found for job {job_id}")
    return PodcastJobRequest(input_filename=pdfs[0].name)


def _load_pdf_bytes(job_id: str, request: PodcastJobRequest) -> tuple[bytes | None, str | None]:
    job_dir = JOBS_DIR / job_id
    pdf_path = job_dir / request.input_filename
    if not pdf_path.exists():
        pdfs = sorted(job_dir.glob("*.pdf"))
        if pdfs:
            pdf_path = pdfs[0]
        else:
            return None, None
    return pdf_path.read_bytes(), pdf_path.name


def _build_script_writer() -> object:
    provider = os.getenv("SCRIPT_PROVIDER", "ollama").lower()
    if provider == "openrouter":
        generator = OpenRouterTextGenerator(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            model=os.getenv("OPENROUTER_MODEL", "minimax/minimax-m2.1"),
            timeout_s=float(os.getenv("OPENROUTER_TIMEOUT_S", "40")),
        )
        return OpenRouterScriptWriter(generator=generator)

    generator = OllamaTextGenerator(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL", "llama3:instruct"),
        timeout_s=float(os.getenv("OLLAMA_TIMEOUT_S", "600")),
        num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "480")),
        temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.4")),
        top_p=float(os.getenv("OLLAMA_TOP_P", "0.9")),
        fallback_model=os.getenv("OLLAMA_FALLBACK_MODEL"),
        fallback_num_predict=int(os.getenv("OLLAMA_FALLBACK_NUM_PREDICT", "0") or 0) or None,
    )
    section_pass = _env_bool("SCRIPT_SECTION_PASS", False)
    return OllamaScriptWriter(generator=generator, section_pass=section_pass)


def _build_embeddings() -> tuple[object | None, object | None]:
    provider = os.getenv("EMBEDDING_PROVIDER", "").lower()
    if provider in {"", "none"}:
        return None, None

    if provider == "ollama":
        embedder = OllamaEmbeddingGenerator(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            timeout_s=float(os.getenv("EMBEDDING_TIMEOUT_S", "60")),
        )
    else:
        embedder = SentenceTransformersEmbedder(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            device=os.getenv("EMBEDDING_DEVICE"),
        )

    store_provider = os.getenv("VECTOR_STORE_PROVIDER", "local").lower()
    if store_provider == "chroma":
        store = ChromaVectorStore(base_dir=str(PROCESSED_DIR))
    elif store_provider == "faiss":
        store = FaissVectorStore(base_dir=str(PROCESSED_DIR))
    else:
        store = LocalVectorStore(base_dir=str(PROCESSED_DIR))

    return embedder, store


def _build_tts_and_audio() -> tuple[object | None, object | None]:
    enable_tts = _env_bool("ENABLE_TTS", False)
    enable_audio = _env_bool("ENABLE_AUDIO", False)
    if not (enable_tts and enable_audio):
        return None, None

    voice_map = _parse_map_env("TTS_VOICE_MAP")
    piper_map = _parse_int_map_env("PIPER_SPEAKER_MAP")
    coqui_map = _parse_int_map_env("COQUI_SPEAKER_MAP")

    provider = os.getenv("TTS_PROVIDER", "pyttsx3").lower()
    if provider == "piper":
        tts = PiperSynthesizer(
            model_path=os.getenv("PIPER_MODEL_PATH", ""),
            speaker=int(os.getenv("PIPER_SPEAKER", "0")) if os.getenv("PIPER_SPEAKER") else None,
            speaker_map=piper_map,
            sentence_silence=float(os.getenv("PIPER_SENTENCE_SILENCE", "0.2")),
        )
    elif provider == "coqui":
        tts = CoquiSynthesizer(
            model_path=os.getenv("COQUI_MODEL_PATH", ""),
            speaker_idx=int(os.getenv("COQUI_SPEAKER_IDX", "0")) if os.getenv("COQUI_SPEAKER_IDX") else None,
            speed=float(os.getenv("COQUI_SPEED", "1.0")) if os.getenv("COQUI_SPEED") else None,
            speaker_map=coqui_map,
        )
    elif provider == "minimax":
        tts = MinimaxSynthesizer(
            api_key=os.getenv("MINIMAX_API_KEY"),
            model=os.getenv("MINIMAX_MODEL", "speech-2.6-hd"),
            voice_male=os.getenv("MINIMAX_VOICE_MALE", "English_Explanatory_Man"),
            voice_female=os.getenv("MINIMAX_VOICE_FEMALE", "English_captivating_female1"),
            timeout_s=float(os.getenv("MINIMAX_TIMEOUT_S", "30")),
            poll_interval_s=float(os.getenv("MINIMAX_POLL_INTERVAL_S", "2")),
            max_attempts=int(os.getenv("MINIMAX_MAX_ATTEMPTS", "60")),
            speed=float(os.getenv("MINIMAX_SPEED", "1.0")),
            speed_floor=float(os.getenv("MINIMAX_SPEED_FLOOR", "0.85")),
            speed_ceiling=float(os.getenv("MINIMAX_SPEED_CEILING", "1.15")),
        )
    else:
        tts = Pyttsx3Synthesizer(
            voice=os.getenv("TTS_VOICE"),
            rate=int(os.getenv("TTS_RATE", "0") or 0) or None,
            voice_map=voice_map,
        )

    audio = PydubAudioAssembler(
        target_format=os.getenv("AUDIO_FORMAT", "mp3"),
        target_dbfs=float(os.getenv("AUDIO_TARGET_DBFS", "-16")),
        intro_path=os.getenv("AUDIO_INTRO_PATH"),
        outro_path=os.getenv("AUDIO_OUTRO_PATH"),
        segment_silence_ms=int(os.getenv("SEGMENT_SILENCE_MS", "300")),
        silence_per_word_ms=float(os.getenv("SILENCE_PER_WORD_MS", "3")),
        min_segment_silence_ms=int(os.getenv("MIN_SEGMENT_SILENCE_MS", "200")),
        max_segment_silence_ms=int(os.getenv("MAX_SEGMENT_SILENCE_MS", "2000")),
    )
    return tts, audio


def _build_pdf_extractor(source_name: str | None) -> object:
    topic_dir = os.getenv("TOPIC_RESEARCH_DIR")
    if topic_dir and _env_bool("USE_TOPIC_CORPUS", False):
        return TopicCorpusExtractor(
            base_dir=topic_dir,
            top_k_files=int(os.getenv("TOPIC_RESEARCH_TOP_K_FILES", "3")),
            min_chars=int(os.getenv("TOPIC_RESEARCH_MIN_CHARS", "200")),
        )
    return PdfPlumberExtractor(source_name=source_name)


def _build_pipeline(source_name: str | None) -> GeneratePodcastFromPdf:
    chunker = SentenceChunker(
        max_chars=int(os.getenv("CHUNK_MAX_CHARS", "1400")),
        min_chars=int(os.getenv("CHUNK_MIN_CHARS", "600")),
    )
    script_writer = _build_script_writer()
    job_store = FileSystemJobStore(base_dir=str(PROCESSED_DIR))
    embedder, vector_store = _build_embeddings()
    tts, audio = _build_tts_and_audio()

    return GeneratePodcastFromPdf(
        pdf_extractor=_build_pdf_extractor(source_name),
        chunker=chunker,
        script_writer=script_writer,
        job_store=job_store,
        embedder=embedder,
        vector_store=vector_store,
        tts=tts,
        audio=audio,
    )


def _write_result(job_id: str, result: PodcastJobResult) -> str:
    path = PROCESSED_DIR / job_id / "result.json"
    _write_json_with_retry(path, result.model_dump(), raise_on_fail=True)
    return str(path)


def _process_job(job_id: str) -> None:
    set_correlation_id(job_id)
    _write_status(job_id, status=JobStatus.RUNNING, stage="start")
    job_started()
    start = time.monotonic()

    try:
        request = _load_request(job_id)
        pdf_bytes, source_name = _load_pdf_bytes(job_id, request)
        if pdf_bytes is None:
            topic_dir = os.getenv("TOPIC_RESEARCH_DIR")
            if not topic_dir:
                raise FileNotFoundError("No PDF found and TOPIC_RESEARCH_DIR not set")
            extractor = TopicCorpusExtractor(
                base_dir=topic_dir,
                top_k_files=int(os.getenv("TOPIC_RESEARCH_TOP_K_FILES", "3")),
                min_chars=int(os.getenv("TOPIC_RESEARCH_MIN_CHARS", "200")),
            )
            pipeline = _build_pipeline(None)
            pipeline.pdf_extractor = extractor
            result = pipeline.run(job_id=job_id, request=request, pdf_bytes=b"")
        else:
            pipeline = _build_pipeline(source_name)
            result = pipeline.run(job_id=job_id, request=request, pdf_bytes=pdf_bytes)

        result_path = _write_result(job_id, result)
        _write_status(job_id, status=JobStatus.SUCCEEDED, stage="done", result_path=result_path)
        job_succeeded()
    except Exception as exc:
        log.exception("Job failed: %s", exc)
        _write_status(job_id, status=JobStatus.FAILED, stage="error", error=str(exc))
        job_failed()
    finally:
        observe_stage("total", time.monotonic() - start)
        clear_correlation_id()


def _next_job() -> str | None:
    if QUEUE_MODE == "redis":
        payload = redis_dequeue(QUEUE_REDIS_URL, QUEUE_REDIS_KEY)
        if payload:
            return payload.get("job_id")
        return None
    if QUEUE_MODE == "file":
        payload = file_dequeue(QUEUE_FILE)
        if payload:
            return payload.get("job_id")
        return None
    payload = dir_dequeue(QUEUE_DIR)
    if payload:
        return payload.get("job_id")
    return None


def _update_queue_metrics() -> None:
    try:
        if QUEUE_MODE == "redis":
            depth = redis_queue_depth(QUEUE_REDIS_URL, QUEUE_REDIS_KEY)
        elif QUEUE_MODE == "file":
            depth = file_queue_depth(QUEUE_FILE)
        else:
            depth = dir_queue_depth(QUEUE_DIR)
        set_queue_depth(depth)
    except Exception:
        pass


def main() -> None:
    setup_logging(fmt=os.getenv("LOG_FORMAT", "plain"))
    maybe_init_tracing("podcastify-worker")
    if _env_bool("METRICS_ENABLED", False) or _env_bool("PROMETHEUS_ENABLED", False):
        maybe_start_server(int(os.getenv("METRICS_PORT", "9000")))

    log.info("worker.start queue_mode=%s", QUEUE_MODE)
    while True:
        _update_queue_metrics()
        job_id = _next_job()
        if not job_id:
            time.sleep(POLL_INTERVAL_S)
            continue
        _process_job(job_id)


if __name__ == "__main__":
    main()
