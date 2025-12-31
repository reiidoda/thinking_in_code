from __future__ import annotations

import os
from typing import Optional

try:  # Optional dependency
    from prometheus_client import Counter, Histogram, Gauge, start_http_server  # type: ignore
except Exception:  # pragma: no cover
    class _Noop:
        def __init__(self, *args, **kwargs): ...
        def labels(self, *args, **kwargs): return self
        def inc(self, *args, **kwargs): ...
        def observe(self, *args, **kwargs): ...
        def set(self, *args, **kwargs): ...
    Counter = Histogram = Gauge = _Noop  # type: ignore
    def start_http_server(*args, **kwargs): ...  # type: ignore

_JOB_STARTED = Counter("podcastify_job_started_total", "Jobs started")
_JOB_SUCCEEDED = Counter("podcastify_job_succeeded_total", "Jobs succeeded")
_JOB_FAILED = Counter("podcastify_job_failed_total", "Jobs failed")
_STAGE_SEC = Histogram(
    "podcastify_stage_seconds",
    "Stage durations in seconds",
    ["stage"],
    buckets=(0.1, 0.5, 1, 2, 4, 8, 16, 32, 60, 120, 300, 600),
)
_QUEUE_DEPTH = Gauge("podcastify_queue_depth", "Queue depth")

_server_started = False

def maybe_start_server(port: int) -> None:
    global _server_started
    if _server_started:
        return
    try:
        start_http_server(port)
        _server_started = True
    except Exception:
        pass

def job_started() -> None:
    _JOB_STARTED.inc()

def job_succeeded() -> None:
    _JOB_SUCCEEDED.inc()

def job_failed() -> None:
    _JOB_FAILED.inc()

def observe_stage(stage: str, duration_sec: float) -> None:
    try:
        _STAGE_SEC.labels(stage=stage).observe(max(0.0, duration_sec))
    except Exception:
        pass

def set_queue_depth(depth: int) -> None:
    try:
        _QUEUE_DEPTH.set(depth)
    except Exception:
        pass

def metrics_enabled() -> bool:
    return os.getenv("PROMETHEUS_ENABLED", "").lower() in {"1", "true", "yes"} or os.getenv("METRICS_ENABLED", "").lower() in {"1", "true", "yes"}
