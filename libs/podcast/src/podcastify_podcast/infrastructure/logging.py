from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar

from rich.logging import RichHandler

_corr_id: ContextVar[str | None] = ContextVar("corr_id", default=None)

class CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.corr_id = _corr_id.get() or "-"
        return True

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - simple formatter
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
            "corr_id": getattr(record, "corr_id", None),
            "name": record.name,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)

def setup_logging(level: int = logging.INFO, *, fmt: str = "plain") -> None:
    if fmt == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handlers = [handler]
        format_str = "%(message)s"
    else:
        handlers = [RichHandler(rich_tracebacks=True)]
        format_str = "%(levelname)s [%(corr_id)s] %(message)s"

    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt="[%X]",
        handlers=handlers,
    )
    logging.getLogger().addFilter(CorrelationFilter())

def set_correlation_id(corr_id: str | None) -> None:
    _corr_id.set(corr_id)

def get_correlation_id() -> str | None:
    return _corr_id.get()

def clear_correlation_id() -> None:
    _corr_id.set(None)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def maybe_init_tracing(service_name: str) -> None:  # pragma: no cover - optional
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(provider)
