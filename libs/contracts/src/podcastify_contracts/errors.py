from __future__ import annotations

class PodcastifyError(Exception):
    """Base exception for domain-safe errors."""

class ConfigurationError(PodcastifyError):
    pass

class PipelineError(PodcastifyError):
    pass
