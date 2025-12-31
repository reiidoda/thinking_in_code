from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from podcastify_podcast.infrastructure.audio.pydub_assembler import PydubAudioAssembler


class DummyAudioSegment:
    def __init__(self, duration_ms: int = 0, dBFS: float = -20.0, max_dBFS: float = -2.0) -> None:
        self.duration_ms = duration_ms
        self.dBFS = dBFS
        self.max_dBFS = max_dBFS

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0

    def __add__(self, other: "DummyAudioSegment") -> "DummyAudioSegment":
        return DummyAudioSegment(self.duration_ms + other.duration_ms, self.dBFS, self.max_dBFS)

    def __iadd__(self, other: "DummyAudioSegment") -> "DummyAudioSegment":
        self.duration_ms += other.duration_ms
        return self

    @classmethod
    def silent(cls, duration: int) -> "DummyAudioSegment":
        return cls(duration_ms=duration)

    @classmethod
    def from_file(cls, path: str) -> "DummyAudioSegment":
        return cls(duration_ms=1000)

    def fade_in(self, ms: int) -> "DummyAudioSegment":
        return self

    def fade_out(self, ms: int) -> "DummyAudioSegment":
        return self

    def apply_gain(self, gain: float) -> "DummyAudioSegment":
        self.dBFS += gain
        return self

    def limit(self, threshold: float) -> "DummyAudioSegment":
        self.max_dBFS = min(self.max_dBFS, threshold)
        return self

    def export(self, out: str, format: str) -> None:
        Path(out).write_bytes(b"fake-audio")


def install_dummy_pydub(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("pydub")
    module.AudioSegment = DummyAudioSegment
    monkeypatch.setitem(sys.modules, "pydub", module)


def test_assemble_writes_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_dummy_pydub(monkeypatch)
    seg1 = tmp_path / "seg1.wav"
    seg2 = tmp_path / "seg2.wav"
    seg1.write_bytes(b"x")
    seg2.write_bytes(b"x")
    out_path = tmp_path / "out.mp3"
    meta_path = tmp_path / "audio_metadata.json"

    assembler = PydubAudioAssembler(
        target_format="mp3",
        metadata_path=str(meta_path),
        segment_silence_ms=400,
        silence_per_word_ms=2.0,
        min_segment_silence_ms=100,
        max_segment_silence_ms=2000,
    )

    result = assembler.assemble(
        audio_segments=[str(seg1), str(seg2)],
        out_path=str(out_path),
        segment_word_counts=[100, 50],
        segment_pause_multipliers=[1.2, 1.0],
    )

    assert result == str(out_path)
    assert out_path.exists()
    assert meta_path.exists()

    data = json.loads(meta_path.read_text())
    assert data["segment_files"] == [str(seg1), str(seg2)]
    assert data["segment_word_counts"] == [100, 50]
    assert data["applied_gaps_ms"] == [720]
    assert data["segment_pause_multipliers"] == [1.2, 1.0]


def test_assemble_requires_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_dummy_pydub(monkeypatch)
    assembler = PydubAudioAssembler()
    with pytest.raises(ValueError):
        assembler.assemble(audio_segments=[], out_path=str(tmp_path / "out.mp3"))
