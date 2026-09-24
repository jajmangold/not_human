"""Deterministic ALP calibration control grids."""

from __future__ import annotations

import random

import numpy as np


CONTROL_RANGES: dict[str, tuple[float, float]] = {
    "rotate_pitch": (-20.0, 20.0), "rotate_yaw": (-20.0, 20.0),
    "rotate_roll": (-20.0, 20.0), "blink": (-20.0, 20.0),
    "eyebrow": (-40.0, 20.0), "wink": (0.0, 25.0),
    "pupil_x": (-20.0, 20.0), "pupil_y": (-20.0, 20.0),
    "aaa": (-30.0, 120.0), "eee": (-20.0, 20.0),
    "woo": (-20.0, 20.0), "smile": (-2.0, 2.0),
}


def control_grid(points: int, names: list[str] | tuple[str, ...] | None = None) -> list[dict[str, float]]:
    if points < 2:
        raise ValueError("points must be at least 2")
    selected = list(names) if names is not None else list(CONTROL_RANGES)
    unknown = [name for name in selected if name not in CONTROL_RANGES]
    if unknown:
        raise ValueError(f"unknown controls: {', '.join(unknown)}")
    grid: list[dict[str, float]] = []
    for name in selected:
        low, high = CONTROL_RANGES[name]
        for value in np.linspace(low, high, points):
            controls = {key: 0.0 for key in CONTROL_RANGES}
            controls[name] = round(float(value), 6)
            grid.append(controls)
    return grid


def random_controls(count: int, seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    return [
        {name: round(rng.uniform(low, high), 6) for name, (low, high) in CONTROL_RANGES.items()}
        for _ in range(max(0, count))
    ]


def neutral_controls() -> dict[str, float]:
    """All twelve controls at zero -- the no-input baseline sample."""
    return {name: 0.0 for name in CONTROL_RANGES}


def named_control_points(name: str) -> dict[str, float]:
    """Six labeled points for one control: min, q1, default, mid, q3, max.

    ``mid`` is the arithmetic midpoint of the control's own range, ``default``
    is 0.0 when the range spans zero (the slider's actual UI default)
    otherwise the endpoint closest to zero -- these two coincide for the
    symmetric ranges (e.g. rotate_yaw) and differ for the asymmetric ones
    (e.g. eyebrow, wink, aaa), matching the six-point vocabulary used by the
    prior manual slider-matrix QA run.
    """
    if name not in CONTROL_RANGES:
        raise ValueError(f"unknown control: {name}")
    low, high = CONTROL_RANGES[name]
    mid = (low + high) / 2.0
    default = 0.0 if low <= 0.0 <= high else min(low, high, key=abs)
    return {
        "min": low,
        "q1": (low + mid) / 2.0,
        "default": default,
        "mid": mid,
        "q3": (mid + high) / 2.0,
        "max": high,
    }


def named_grid(names: list[str] | tuple[str, ...] | None = None) -> list[dict[str, object]]:
    """One-control-at-a-time samples at each of its six named points.

    Each entry is ``{"control": name, "point": point_name, "value": value,
    "requested": <full 12-key control dict>}`` so a report can group results
    by control and by point without re-deriving which point a sample came
    from.
    """
    selected = list(names) if names is not None else list(CONTROL_RANGES)
    unknown = [name for name in selected if name not in CONTROL_RANGES]
    if unknown:
        raise ValueError(f"unknown controls: {', '.join(unknown)}")
    grid: list[dict[str, object]] = []
    for name in selected:
        for point, value in named_control_points(name).items():
            requested = neutral_controls()
            requested[name] = round(float(value), 6)
            grid.append({"control": name, "point": point, "value": requested[name], "requested": requested})
    return grid


def bounded_combined_controls(count: int, seed: int, max_active: int = 3, fraction: float = 0.5) -> list[dict[str, float]]:
    """Deterministic samples with 2..max_active controls simultaneously active.

    Each active control is drawn from the ``fraction`` of its range centered
    on its own midpoint, not its extremes -- "bounded" combined poses, held
    out for later adapter evaluation rather than mixed into the single-axis
    sweep used for per-control controllability/leakage measurement.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if not 2 <= max_active <= len(CONTROL_RANGES):
        raise ValueError(f"max_active must be between 2 and {len(CONTROL_RANGES)}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    rng = random.Random(seed)
    names = list(CONTROL_RANGES)
    samples: list[dict[str, float]] = []
    for _ in range(count):
        active = rng.sample(names, rng.randint(2, max_active))
        controls = neutral_controls()
        for name in active:
            low, high = CONTROL_RANGES[name]
            mid = (low + high) / 2.0
            span = (high - low) * fraction / 2.0
            controls[name] = round(rng.uniform(max(low, mid - span), min(high, mid + span)), 6)
        samples.append(controls)
    return samples
