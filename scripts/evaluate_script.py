#!/usr/bin/env python
"""
Lightweight evaluator for generated scripts.
Scores structure, citation presence, and length bands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from podcastify_contracts.podcast_job import EpisodeSegment


def score_segments(segments: List[EpisodeSegment], chunks: list[str] | None = None) -> dict:
    scores = {
        "has_hook": False,
        "has_recap": False,
        "has_takeaway": False,
        "body_segments": 0,
        "segments_with_citations": 0,
        "length_ok": 0,
        "sentences_with_overlap": 0,
        "total_sentences": 0,
    }
    corpus_tokens = [set(c.lower().split()) for c in chunks] if chunks else []

    for idx, seg in enumerate(segments):
        title_lower = seg.title.lower()
        if idx == 0 and "hook" in title_lower:
            scores["has_hook"] = True
        if "recap" in title_lower:
            scores["has_recap"] = True
        if "takeaway" in title_lower:
            scores["has_takeaway"] = True
        if 80 <= len(seg.text.split()) <= 220:
            scores["length_ok"] += 1
        if seg.citations:
            scores["segments_with_citations"] += 1
        # overlap
        sentences = seg.text.split(".")
        for sent in sentences:
            if not sent.strip():
                continue
            scores["total_sentences"] += 1
            if corpus_tokens:
                stoks = set(sent.lower().split())
                if any(stoks & ct for ct in corpus_tokens):
                    scores["sentences_with_overlap"] += 1
    if len(segments) >= 3:
        scores["body_segments"] = max(0, len(segments) - 3)  # excluding hook/recap/takeaway
    return scores


def load_segments(path: Path) -> List[EpisodeSegment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = data.get("segments", data)
    return [EpisodeSegment.model_validate(s) for s in segs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("script_json", type=Path, help="Path to script JSON (segments array).")
    ap.add_argument("--chunks", type=Path, help="Optional path to chunks.json for overlap scoring.")
    ap.add_argument("--audio-quality", type=Path, help="Optional path to audio_quality.json to include in report.")
    args = ap.parse_args()

    segments = load_segments(args.script_json)
    chunks = None
    if args.chunks and args.chunks.exists():
        chunks = [item["text"] for item in json.loads(args.chunks.read_text(encoding="utf-8"))]
    scores = score_segments(segments, chunks)
    if args.audio_quality and args.audio_quality.exists():
        audio = json.loads(args.audio_quality.read_text(encoding="utf-8"))
        scores["audio_quality"] = audio
    print(json.dumps({"segments": len(segments), **scores}, indent=2))


if __name__ == "__main__":
    main()
