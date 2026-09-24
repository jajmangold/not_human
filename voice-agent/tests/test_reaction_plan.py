"""Unit tests for reaction_plan (wav_duration/wav_prosody/reaction_plan).

Ported verbatim from musetalk-volta's proven `/ws/live/` demo -- these
tests exist to catch any accidental divergence from that demo's tuned
behavior, not to re-derive the tuning itself.
"""

from __future__ import annotations

import io
import math
import struct
import wave

from reaction_plan import reaction_plan, wav_duration, wav_prosody


def _make_wav(samples: list[int], *, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


def _tone(duration_s: float, *, amplitude: int = 12000, sample_rate: int = 24000) -> list[int]:
    n = int(sample_rate * duration_s)
    return [int(amplitude * math.sin(2 * math.pi * 220 * i / sample_rate)) for i in range(n)]


def _silence(duration_s: float, *, sample_rate: int = 24000) -> list[int]:
    return [0] * int(sample_rate * duration_s)


def test_wav_duration_matches_real_length():
    wav_bytes = _make_wav(_tone(0.5))
    assert wav_duration(wav_bytes) == 0.5


def test_wav_prosody_of_silence_is_all_pause():
    wav_bytes = _make_wav(_silence(0.5))
    prosody = wav_prosody(wav_bytes)
    assert prosody["energy"] == 0.0
    assert prosody["pause_ratio"] == 1.0
    assert prosody["onset"] == 0.0


def test_wav_prosody_of_loud_tone_has_high_energy_and_low_pause():
    wav_bytes = _make_wav(_tone(0.5))
    prosody = wav_prosody(wav_bytes)
    assert prosody["energy"] > 0.5
    assert prosody["pause_ratio"] < 0.2


def test_wav_prosody_malformed_audio_returns_safe_defaults():
    # sample width/channel mismatch (stereo) must not raise -- degrade to
    # the same safe defaults as silence rather than crashing the pipeline
    # over one bad TTS response.
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(struct.pack("<4h", 100, 100, 200, 200))
    prosody = wav_prosody(buf.getvalue())
    assert prosody == {"energy": 0.0, "pause_ratio": 1.0, "onset": 0.0}


def test_reaction_plan_neutral_stays_neutral_regardless_of_prosody():
    sampled = {"reaction": "neutral", "strength": 0.0}
    prosody = {"energy": 1.0, "pause_ratio": 0.0, "onset": 1.0}
    plan = reaction_plan(sampled, prosody)
    assert plan["reaction"] == "neutral"
    assert plan["strength"] == 0.0


def test_reaction_plan_caps_strength_well_below_full_intensity():
    # Even maximal classifier confidence + maximal acoustic energy must not
    # authorize a full-strength (1.0) expression -- this is the "semantics
    # bias, they don't authorize a full pose" damping the plan documents.
    sampled = {"reaction": "amusement", "strength": 0.60}
    prosody = {"energy": 1.0, "pause_ratio": 0.0, "onset": 1.0}
    plan = reaction_plan(sampled, prosody)
    assert plan["reaction"] == "amusement"
    assert 0.0 < plan["strength"] <= 0.45


def test_reaction_plan_below_threshold_collapses_to_neutral():
    sampled = {"reaction": "interest", "strength": 0.05}
    prosody = {"energy": 0.1, "pause_ratio": 0.9, "onset": 0.1}
    plan = reaction_plan(sampled, prosody)
    assert plan["reaction"] == "neutral"
    assert plan["strength"] == 0.0


def test_reaction_plan_marks_repeat_continuation_when_repeated():
    sampled = {"reaction": "amusement", "strength": 0.5}
    prosody = {"energy": 0.5, "pause_ratio": 0.3, "onset": 0.3}
    plan = reaction_plan(sampled, prosody, repeated=True)
    assert plan.get("repeat_continuation") is True
    assert plan["start_ratio"] == 0.18


def test_reaction_plan_not_repeated_has_zero_start_ratio():
    sampled = {"reaction": "amusement", "strength": 0.5}
    prosody = {"energy": 0.5, "pause_ratio": 0.3, "onset": 0.3}
    plan = reaction_plan(sampled, prosody, repeated=False)
    assert "repeat_continuation" not in plan
    assert plan["start_ratio"] == 0.0


def test_reaction_plan_higher_energy_yields_higher_strength():
    sampled = {"reaction": "surprise", "strength": 0.4}
    quiet = reaction_plan(sampled, {"energy": 0.1, "pause_ratio": 0.5, "onset": 0.1})
    loud = reaction_plan(sampled, {"energy": 0.9, "pause_ratio": 0.1, "onset": 0.9})
    assert loud["strength"] > quiet["strength"]
