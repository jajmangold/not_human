"""Fuse a classified reaction with fast acoustic (prosody) cues into one
bounded, damped trajectory ready for a MuseTalk job payload.

Ported verbatim from musetalk-volta's web/app.py (its `/ws/live/` demo --
the "previous demo site" this LiveKit agent was compared against,
internal issue #45). This is real, already-tuned production logic
(safety-tuned control ranges per musetalk-volta#28/#30/#23) covering
things not obvious from first principles: damping strength so semantics
alone can't authorize a full-strength pose, not collapsing a *repeated*
reaction back to neutral between phrases, and lightly biasing attack/hold
timing off phrase energy/pauses. Reimplementing this from scratch would
either miss those tuned behaviors or silently redo the tuning work; this
is the same function, not a reinterpretation of it.
"""

from __future__ import annotations

import array
import io
import wave


def wav_duration(payload: bytes) -> float:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def wav_prosody(payload: bytes) -> dict[str, float]:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        samples = array.array("h", wav.readframes(wav.getnframes()))
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1 or not samples:
            return {"energy": 0.0, "pause_ratio": 1.0, "onset": 0.0}
    window = 320
    rms_values = []
    for start in range(0, len(samples), window):
        block = samples[start:start + window]
        if block:
            rms_values.append((sum(value * value for value in block) / len(block)) ** 0.5 / 32768)
    peak = max(rms_values, default=0.0)
    active = [value for value in rms_values if value > max(0.006, peak * 0.12)]
    return {
        "energy": min(1.0, (sum(active) / max(1, len(active))) * 7.0),
        "pause_ratio": 1.0 - len(active) / max(1, len(rms_values)),
        "onset": min(1.0, peak * 5.0),
    }


def reaction_plan(
    sampled: dict[str, object], prosody: dict[str, float], *, repeated: bool = False
) -> dict[str, object]:
    """Fuse phrase semantics and fast acoustic cues into one bounded trajectory."""
    plan = dict(sampled)
    reaction = str(plan.get("reaction", "neutral"))
    semantic = max(0.0, min(0.60, float(plan.get("strength", 0.0))))
    acoustic_gain = 0.68 + 0.22 * prosody["energy"] + 0.10 * prosody["onset"]
    # Semantics bias a slow conversational state; they do not authorize a
    # deliberately posed, full-strength expression anchor.
    strength = min(0.45, semantic * acoustic_gain * 0.82)
    if repeated and reaction != "neutral":
        # Repeated phrase labels represent a sustained conversational state,
        # not a new gesture and not a reason to collapse back toward neutral.
        plan["repeat_continuation"] = True
    primitives = plan.get("primitives", {})
    if isinstance(primitives, dict):
        mixtures = sorted(
            (
                {"reaction": str(name), "weight": min(0.16, max(0.0, float(value)) * 0.18)}
                for name, value in primitives.items()
                if name != reaction and float(value) >= 0.10
            ),
            key=lambda item: item["weight"],
            reverse=True,
        )
        plan["primitive_mix"] = mixtures[:2]
    plan.update(
        {
            "reaction": reaction,
            "strength": strength,
            "start_ratio": 0.18 if repeated and reaction != "neutral" else 0.0,
            "end_ratio": 0.18,
            "attack_ms": max(170, int(float(plan.get("attack_ms", 220)) * (1.0 - 0.10 * prosody["onset"]))),
            "hold_ms": max(140, int(float(plan.get("hold_ms", 320)) * (1.0 - 0.25 * prosody["pause_ratio"]))),
            "decay_ms": max(650, int(float(plan.get("decay_ms", 700)))),
            "clock": "audio",
            "semantic_sample": "phrase",
            "prosody_sample_hz": 50,
        }
    )
    if reaction == "neutral" or strength < 0.08:
        plan["reaction"], plan["strength"] = "neutral", 0.0
    return plan
