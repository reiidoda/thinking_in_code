from __future__ import annotations

import httpx
from podcastify_podcast.infrastructure.logging import get_logger

log = get_logger(__name__)

class OllamaTextGenerator:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float = 600.0,
        num_predict: int | None = 480,
        temperature: float | None = 0.4,
        top_p: float | None = 0.9,
        fallback_model: str | None = None,
        fallback_num_predict: int | None = None,
        options: dict | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.fallback_model = fallback_model
        self.fallback_num_predict = fallback_num_predict
        default_opts = {
            "num_predict": num_predict,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": 1.1,
        }
        # Remove None values so we do not override server defaults unintentionally.
        self.options = {k: v for k, v in default_opts.items() if v is not None}
        if options:
            self.options.update(options)

    def generate(self, *, prompt: str) -> str:
        def _call(model: str, opts: dict | None):
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
            }
            if opts:
                payload["options"] = opts
            log.info("ollama.generate model=%s", model)
            timeout = httpx.Timeout(self.timeout_s, read=self.timeout_s)
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
            return str(data.get("response", "")).strip()

        try:
            return _call(self.model, self.options)
        except Exception as e:
            if self.fallback_model:
                log.warning("Primary model failed (%s); trying fallback=%s", e, self.fallback_model)
                fb_opts = dict(self.options or {})
                if self.fallback_num_predict is not None:
                    fb_opts["num_predict"] = self.fallback_num_predict
                return _call(self.fallback_model, fb_opts)
            raise


class OllamaEmbeddingGenerator:
    def __init__(self, *, base_url: str, model: str, timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def embed(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/api/embeddings"
        embeddings: list[list[float]] = []
        with httpx.Client(timeout=self.timeout_s) as client:
            for text in texts:
                payload = {"model": self.model, "prompt": text}
                log.info("ollama.embed model=%s", self.model)
                r = client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
                emb = data.get("embedding")
                if not isinstance(emb, list):
                    raise RuntimeError("Invalid embedding response from Ollama.")
                embeddings.append([float(x) for x in emb])
        return embeddings
