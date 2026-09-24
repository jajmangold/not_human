"""Pure numeric analysis for ALP controllability / cross-coupling (#30).

Deliberately has no rendering or extraction dependency: it consumes already-
measured AvatarFrame-shaped arrays (blendshapes + head_rotation) and answers
"how much did channel X move, and is that more than noise" questions. Keeping
this pure/testable is what lets scripts/measure_controllability.py be a thin,
mostly-untested I/O shell around a well-tested core.

Everything here operates in *measured canonical channel space*
(BLENDSHAPE_NAMES + head_rotation), never in ALP slider units -- the point of
#30 is to describe how ALP inputs move that canonical space, not to assume a
mapping in advance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avatar.controls.schema import BLENDSHAPE_NAMES

MOUTH_CHANNELS = tuple(name for name in BLENDSHAPE_NAMES if name.startswith(("mouth", "jaw")))
BROW_CHANNELS = tuple(name for name in BLENDSHAPE_NAMES if name.startswith("brow"))
EYE_CHANNELS = tuple(name for name in BLENDSHAPE_NAMES if name.startswith(("eye", "cheek")))

# Controls whose intended effect is the eye/brow region, per the live-safe
# ownership mapping already documented in web/app.py's /api/motion/controls.
EYE_REGION_CONTROLS = ("pupil_x", "pupil_y", "blink", "wink", "eyebrow")

# Which blendshape *regions* count as cross-coupling for a given control's own
# safe envelope, i.e. NOT that control's own intended target. This is our own
# judgment call from the sliders' documented semantics (web/app.py's control
# descriptions), not an independently verified ground truth -- eyebrow's own
# job is brow motion, so brow response isn't leakage for it (only mouth is);
# aaa/eee/woo/smile's job is mouth shape, so mouth response isn't leakage for
# them (only brow is); blink and wink close the eyelid, not the brow, so both
# mouth and brow response are genuine cross-coupling for them; the three
# rotate_* controls have no blendshape "home" at all, so any blendshape
# movement is potentially unintended coupling for them.
CROSS_COUPLING_REGIONS: dict[str, tuple[str, ...]] = {
    "rotate_pitch": ("mouth", "brow"), "rotate_yaw": ("mouth", "brow"), "rotate_roll": ("mouth", "brow"),
    "blink": ("mouth", "brow"), "wink": ("mouth", "brow"),
    "eyebrow": ("mouth",), "pupil_x": ("mouth", "brow"), "pupil_y": ("mouth", "brow"),
    "aaa": ("brow",), "eee": ("brow",), "woo": ("brow",), "smile": ("brow",),
}


def top_response_channels(deltas: np.ndarray, count: int = 5) -> list[tuple[str, float]]:
    """The ``count`` canonical channels that moved most, signed, largest first.

    Purely empirical and home-region-agnostic -- this is the "controllability"
    half of #30 (did the control move *something*, and how much), independent
    of the "cross-coupling" judgment calls in CROSS_COUPLING_REGIONS.
    """
    order = np.argsort(-np.abs(deltas))[:count]
    return [(BLENDSHAPE_NAMES[int(index)], float(deltas[index])) for index in order]


_INDEX = {name: index for index, name in enumerate(BLENDSHAPE_NAMES)}


def _indices(names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([_INDEX[name] for name in names], dtype=np.int64)


MOUTH_INDICES = _indices(MOUTH_CHANNELS)
BROW_INDICES = _indices(BROW_CHANNELS)


@dataclass(frozen=True)
class NoiseFloor:
    """Per-channel measurement noise, estimated from repeated neutral renders."""

    blendshape_std: np.ndarray  # (52,)
    rotation_std: np.ndarray  # (3,)
    sample_count: int

    def __post_init__(self) -> None:
        if self.blendshape_std.shape != (len(BLENDSHAPE_NAMES),):
            raise ValueError("blendshape_std must have shape (52,)")
        if self.rotation_std.shape != (3,):
            raise ValueError("rotation_std must have shape (3,)")
        if self.sample_count < 2:
            raise ValueError("noise floor needs at least 2 repeated neutral samples")


def noise_floor(neutral_blendshapes: np.ndarray, neutral_rotation: np.ndarray) -> NoiseFloor:
    """Estimate per-channel noise from N>=2 repeated neutral (all-zero) renders."""
    blendshapes = np.asarray(neutral_blendshapes, dtype=np.float64)
    rotation = np.asarray(neutral_rotation, dtype=np.float64)
    if blendshapes.ndim != 2 or blendshapes.shape[1] != len(BLENDSHAPE_NAMES):
        raise ValueError(f"neutral_blendshapes must have shape (N, 52), got {blendshapes.shape}")
    if rotation.ndim != 2 or rotation.shape[1] != 3:
        raise ValueError(f"neutral_rotation must have shape (N, 3), got {rotation.shape}")
    if blendshapes.shape[0] != rotation.shape[0]:
        raise ValueError("neutral_blendshapes and neutral_rotation must have the same sample count")
    if blendshapes.shape[0] < 2:
        raise ValueError("noise floor needs at least 2 repeated neutral samples")
    return NoiseFloor(
        blendshape_std=np.std(blendshapes, axis=0, ddof=1).astype(np.float32),
        rotation_std=np.std(rotation, axis=0, ddof=1).astype(np.float32),
        sample_count=int(blendshapes.shape[0]),
    )


def channel_deltas(baseline_blendshapes: np.ndarray, sample_blendshapes: np.ndarray) -> np.ndarray:
    """Signed per-channel change from a neutral baseline to a sample, in [-1, 1]."""
    baseline = np.asarray(baseline_blendshapes, dtype=np.float64)
    sample = np.asarray(sample_blendshapes, dtype=np.float64)
    if baseline.shape != (len(BLENDSHAPE_NAMES),) or sample.shape != (len(BLENDSHAPE_NAMES),):
        raise ValueError("both arrays must have shape (52,)")
    return (sample - baseline).astype(np.float32)


def region_leakage(
    deltas: np.ndarray,
    noise: NoiseFloor,
    region_indices: np.ndarray,
    threshold_multiplier: float = 3.0,
    absolute_floor: float = 0.02,
) -> dict[str, object]:
    """How much a set of "not this control's region" channels moved.

    A channel counts as leaking if its delta exceeds ``threshold_multiplier``
    times its own measured noise floor -- a channel with high natural
    render-to-render noise needs a proportionally larger signal before it's
    called real. ``absolute_floor`` additionally guards the case measured
    here in practice: the resident ALP + MediaPipe pipeline given identical
    (portrait, controls) input is exactly deterministic (repeated-neutral-
    render std is 0.0 to the last bit, confirmed empirically), so a purely
    multiplicative threshold would call *any* nonzero delta "leakage." The
    floor is a blendshape-score magnitude judged negligible on this ARKit-
    style [0, 1] scale, not a statistical estimate.
    """
    region_deltas = deltas[region_indices]
    region_noise = np.maximum(noise.blendshape_std[region_indices], 1e-6)
    threshold = np.maximum(threshold_multiplier * region_noise, absolute_floor)
    exceeds = np.abs(region_deltas) > threshold
    return {
        "max_abs_delta": float(np.max(np.abs(region_deltas))) if region_deltas.size else 0.0,
        "mean_abs_delta": float(np.mean(np.abs(region_deltas))) if region_deltas.size else 0.0,
        "leaking_channel_count": int(np.sum(exceeds)),
        "leaking_channels": sorted(
            BLENDSHAPE_NAMES[int(region_indices[i])]
            for i in np.nonzero(exceeds)[0]
        ),
    }


def eye_to_mouth_leakage(
    control_name: str,
    deltas_by_point: dict[str, np.ndarray],
    noise: NoiseFloor,
    threshold_multiplier: float = 3.0,
) -> dict[str, object] | None:
    """Mouth-region response to an eye-region control's sweep, or None if n/a."""
    if control_name not in EYE_REGION_CONTROLS:
        return None
    worst = max(
        (region_leakage(deltas, noise, MOUTH_INDICES, threshold_multiplier) for deltas in deltas_by_point.values()),
        key=lambda entry: entry["max_abs_delta"],
        default=None,
    )
    return worst


def blink_to_brow_leakage(
    deltas_by_point: dict[str, np.ndarray],
    noise: NoiseFloor,
    threshold_multiplier: float = 3.0,
) -> dict[str, object]:
    """Brow-region response to the blink control's sweep."""
    return max(
        (region_leakage(deltas, noise, BROW_INDICES, threshold_multiplier) for deltas in deltas_by_point.values()),
        key=lambda entry: entry["max_abs_delta"],
        default={"max_abs_delta": 0.0, "mean_abs_delta": 0.0, "leaking_channel_count": 0, "leaking_channels": []},
    )


def safe_envelope(
    point_values: list[float],
    point_leaking_counts: list[int],
    default_value: float = 0.0,
    max_leaking_channels: int = 0,
) -> tuple[float, float]:
    """The widest span around ``default_value`` where leakage stays bounded.

    Walks outward from the default point in each direction over the sampled
    points (sorted by value) and stops at the first point whose leaking
    channel count exceeds ``max_leaking_channels``. This is deliberately
    coarse: with only the six named grid points per control, the envelope
    can only be as precise as the nearest sampled point on each side, not a
    continuously interpolated boundary. Treat it as a conservative starting
    range for #31's adapter, not a tight bound -- denser sampling would
    narrow it.
    """
    if len(point_values) != len(point_leaking_counts):
        raise ValueError("point_values and point_leaking_counts must be the same length")
    order = sorted(range(len(point_values)), key=lambda i: point_values[i])
    values = [point_values[i] for i in order]
    leaking = [point_leaking_counts[i] for i in order]
    default_index = min(range(len(values)), key=lambda i: abs(values[i] - default_value))

    low = values[default_index]
    for i in range(default_index, -1, -1):
        if leaking[i] > max_leaking_channels:
            break
        low = values[i]

    high = values[default_index]
    for i in range(default_index, len(values)):
        if leaking[i] > max_leaking_channels:
            break
        high = values[i]

    return (low, high)
