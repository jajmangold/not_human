"""Bounded respiratory and conversational upper-body motion planning."""

from __future__ import annotations

import random


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def plan_motion(
    seed: int,
    duration: float,
    prosody: dict[str, float] | None = None,
    controls: dict[str, float] | None = None,
) -> dict[str, object]:
    """Return one subtle plan expressed in SMPL-X ownership terms.

    This planner deliberately contains no sway oscillator. Respiration is a
    variable-duration inhale/exhale state machine in the client; conversational
    pose is a critically damped response to one phrase-level target.
    """

    rng = random.Random(int(seed))
    prosody, controls = prosody or {}, controls or {}
    energy = _clamp(prosody.get("energy", 0.0), 0.0, 1.0)
    pause_ratio = _clamp(prosody.get("pause_ratio", 0.0), 0.0, 1.0)
    lean = _clamp(controls.get("lean", 0.0), 0.0, 0.45)
    recoil = _clamp(controls.get("recoil", 0.0), 0.0, 0.45)
    nod = _clamp(controls.get("nod", 0.0), 0.0, 0.45)
    tilt = _clamp(controls.get("tilt", 0.0), -0.45, 0.45)
    gaze_x = _clamp(controls.get("gaze_x", 0.0), -0.45, 0.45)

    rate_bpm = _clamp(13.2 + 2.0 * energy - 1.0 * pause_ratio + rng.uniform(-0.8, 0.8), 11.0, 17.5)
    inhale_ratio = _clamp(0.37 + rng.uniform(-0.025, 0.025), 0.33, 0.42)
    respiratory_gain = _clamp(0.56 + 0.18 * energy + rng.uniform(-0.05, 0.05), 0.48, 0.78)

    return {
        "clock": "audio",
        "duration": round(max(0.0, float(duration)), 4),
        "respiration": {
            "rate_bpm": round(rate_bpm, 3),
            "inhale_ratio": round(inhale_ratio, 4),
            "gain": round(respiratory_gain, 4),
            "speech_phase": "exhale",
        },
        "pose": {
            # Fractions are intentionally tiny and later scaled by the fitted
            # shoulder span/torso height, not by the full image dimensions.
            "sternum_y": round(_clamp(0.014 * lean - 0.011 * recoil, -0.004, 0.004), 5),
            "shoulder_lift": round(_clamp(0.007 * energy + rng.uniform(-0.001, 0.001), 0.0, 0.004), 5),
            "torso_yaw": round(_clamp(0.024 * gaze_x, -0.006, 0.006), 5),
            "head_pitch": 0.0,
            "head_roll": round(_clamp(0.024 * tilt, -0.007, 0.007), 5),
            # A nod is an impulse executed by the audio-clocked client, not a
            # phrase-long head pose and not a second semantic canvas transform.
            "nod_impulse": round(_clamp(0.026 * nod, 0.0, 0.008), 5),
        },
        "spring": {"omega": 3.2, "critical_damping": 1.0},
        "ownership": {
            "torso_shoulders_neck_gross_head": "upper-body",
            "upper_face_residual": "advanced-live-portrait",
            "mouth_jaw_articulation": "musetalk",
        },
    }
