from __future__ import annotations

import os

import httpx
from podcastify_podcast.infrastructure.logging import get_logger

log = get_logger(__name__)


class OpenRouterTextGenerator:
    """Simple client for OpenRouter chat completions."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "minimax/minimax-m2.1",
        timeout_s: float = 40.0,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OpenRouterTextGenerator requires OPENROUTER_API_KEY")
        self.model = model
        self.timeout_s = timeout_s
        self.base_url = base_url

    def generate(self, *, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        log.info("openrouter.generate model=%s", self.model)
        with httpx.Client(timeout=self.timeout_s) as client:
            r = client.post(self.base_url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned from OpenRouter")
        content = choices[0].get("message", {}).get("content", "")
        return str(content).strip()
