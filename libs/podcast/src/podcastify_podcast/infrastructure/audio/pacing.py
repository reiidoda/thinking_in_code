from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from functools import lru_cache


def segment_rate_multiplier(text: str) -> float:
    return _pacing_profile(text)[0]


def segment_pause_multiplier(text: str) -> float:
    return _pacing_profile(text)[1]


def segment_rate_multiplier_for(title: str | None, text: str) -> float:
    rate, _ = _pacing_profile(text)
    emphasis = _segment_emphasis(title)
    adjusted = rate * (1.0 - 0.18 * (emphasis - 1.0))
    return _clamp(adjusted, 0.65, 1.25)


def segment_pause_multiplier_for(title: str | None, text: str) -> float:
    _, pause = _pacing_profile(text)
    emphasis = _segment_emphasis(title)
    adjusted = pause * (1.0 + 0.30 * (emphasis - 1.0))
    return _clamp(adjusted, 0.55, 2.5)


def _pacing_profile(text: str) -> tuple[float, float]:
    tokens = _tokens(text)
    word_count = len(tokens)
    sentences = _sentences(text)
    sentence_count = max(len(sentences), 1)
    avg_sentence_len = word_count / sentence_count

    math_density = _math_density(tokens)
    numeric_density = _numeric_density(tokens)
    lexical_diversity = _lexical_diversity(tokens)
    entropy = _entropy(tokens)

    # Cognitive load uses a logistic map of multiple signals.
    z = (
        1.2 * math_density
        + 0.8 * numeric_density
        + 0.6 * min(avg_sentence_len / 24.0, 2.0)
        + 0.7 * (1.0 - lexical_diversity)
        + 0.5 * entropy
    )
    load = _sigmoid(z - 1.1)

    # Engagement is modeled with a Gaussian centered at moderate load (Yerkes-Dodson).
    engagement = math.exp(-((load - 0.55) ** 2) / (2 * 0.18**2))

    mode = (os.getenv("PACING_PRESET") or os.getenv("PACING_MODE", "slow_authoritative")).lower()
    if mode == "energetic_academic":
        # Faster baseline for energy, but stronger slow-down on dense segments.
        rate_multiplier = _clamp(1.12 - 0.65 * load + 0.10 * engagement, 0.80, 1.18)
        pause_multiplier = _clamp(0.65 + 1.30 * load + 0.15 * (1.0 - engagement), 0.55, 2.00)
    elif mode in {"slow_authoritative", "authoritative_slow", "authoritative", "slow"}:
        # Slower baseline with deliberate pauses for clarity and authority.
        rate_multiplier = _clamp(0.98 - 0.45 * load + 0.05 * engagement, 0.72, 1.00)
        pause_multiplier = _clamp(1.05 + 1.25 * load + 0.20 * (1.0 - engagement), 0.85, 2.20)
    elif mode == "energetic":
        rate_multiplier = _clamp(1.15 - 0.45 * load + 0.06 * engagement, 0.85, 1.25)
        pause_multiplier = _clamp(0.60 + 0.90 * load + 0.10 * (1.0 - engagement), 0.50, 1.60)
    elif mode == "academic":
        rate_multiplier = _clamp(1.00 - 0.55 * load + 0.04 * engagement, 0.75, 1.05)
        pause_multiplier = _clamp(0.90 + 1.10 * load + 0.15 * (1.0 - engagement), 0.75, 2.10)
    else:
        rate_multiplier = _clamp(1.03 - 0.35 * load + 0.08 * (1.0 - engagement), 0.82, 1.08)
        pause_multiplier = _clamp(0.90 + 0.90 * load + 0.20 * (1.0 - engagement), 0.70, 1.70)
    return rate_multiplier, pause_multiplier


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _lexical_diversity(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / max(len(tokens), 1)


def _numeric_density(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    numeric = sum(1 for t in tokens if any(ch.isdigit() for ch in t))
    return numeric / max(len(tokens), 1)


def _math_density(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    mathish = sum(1 for t in tokens if any(ch.isdigit() for ch in t) or any(op in t for op in ("+", "-", "*", "/", "^", "=")))
    return mathish / max(len(tokens), 1)


def _entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    entropy = -sum((c / total) * math.log((c / total) + 1e-9) for c in counts.values())
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    return entropy / max_entropy


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@lru_cache(maxsize=1)
def _emphasis_map() -> dict[str, float]:
    raw = os.getenv("SEGMENT_EMPHASIS_MAP", "").strip()
    if not raw:
        return {
            "hook": 0.95,
            "intro": 0.95,
            "recap": 1.05,
            "takeaway": 1.1,
            "closing": 1.1,
            "default": 1.0,
        }
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k).lower(): float(v) for k, v in data.items()}
    except Exception:
        pass
    mapping: dict[str, float] = {}
    for chunk in raw.split(","):
        if not chunk.strip():
            continue
        if "=" in chunk:
            key, val = chunk.split("=", 1)
        elif ":" in chunk:
            key, val = chunk.split(":", 1)
        else:
            continue
        key = key.strip().lower()
        try:
            mapping[key] = float(val.strip())
        except ValueError:
            continue
    if "default" not in mapping:
        mapping["default"] = 1.0
    return mapping


def _segment_emphasis(title: str | None) -> float:
    mapping = _emphasis_map()
    if not title:
        return mapping.get("default", 1.0)
    lowered = title.lower()
    for key, value in mapping.items():
        if key == "default":
            continue
        if key in lowered:
            return value
    return mapping.get("default", 1.0)
