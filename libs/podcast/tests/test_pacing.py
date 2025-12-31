from podcastify_podcast.infrastructure.audio import pacing


def test_emphasis_map_adjusts_rate_and_pause(monkeypatch):
    pacing._emphasis_map.cache_clear()
    monkeypatch.setenv("SEGMENT_EMPHASIS_MAP", "hook=1.2,default=1.0")
    pacing._emphasis_map.cache_clear()

    text = "E=mc^2 explains energy; 3.14 and 2.71 appear in formulas."
    base_rate = pacing.segment_rate_multiplier(text)
    base_pause = pacing.segment_pause_multiplier(text)

    emphasized_rate = pacing.segment_rate_multiplier_for("Hook", text)
    emphasized_pause = pacing.segment_pause_multiplier_for("Hook", text)

    assert emphasized_rate < base_rate
    assert emphasized_pause > base_pause

    pacing._emphasis_map.cache_clear()
