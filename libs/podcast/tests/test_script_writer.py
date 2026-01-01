import json

from podcastify_podcast.domain.models import Chunk
from podcastify_podcast.infrastructure.script.script_writer import OllamaScriptWriter


class DummyGenerator:
    def __init__(self, payload):
        self.payload = payload

    def generate(self, *, prompt: str) -> str:  # pragma: no cover - simple stub
        return self.payload


def test_script_writer_normalizes_structure():
    # Missing takeaway + recap, too few segments
    payload = json.dumps(
        {
            "segments": [
                {"title": "Segment A", "speaker": "Alice", "text": "short text one " * 10},
                {"title": "Segment B", "speaker": "Bob", "text": "short text two " * 10},
            ]
        }
    )
    writer = OllamaScriptWriter(generator=DummyGenerator(payload))
    chunks = [Chunk(text="chunk one"), Chunk(text="chunk two")]

    segments = writer.write_script(chunks=chunks, minutes=8, language="en", style="everyday")

    titles = [s.title.lower() for s in segments]
    assert "hook" in titles[0]
    assert any("recap" in t for t in titles)
    assert any("takeaway" in t for t in titles)
    # At least 3 body segments between hook and final two
    body = segments[1:-2]
    assert 3 <= len(body) <= 6


def test_script_writer_trims_long_segments():
    long_text = "word " * 400
    payload = json.dumps({"segments": [{"title": "Hook", "speaker": "Host", "text": long_text}]})
    writer = OllamaScriptWriter(generator=DummyGenerator(payload))
    segments = writer.write_script(chunks=[], minutes=8, language="en", style="everyday")
    assert len(segments[0].text.split()) <= 250  # allow evidence/fact-note padding
