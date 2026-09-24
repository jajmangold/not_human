#!/usr/bin/env python3
"""Quantify natural-motion distributions from MediaPipe metric JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


CHANNELS = (
    "centerX", "centerY", "roll", "yaw", "pitch", "gazeX", "gazeY",
    "leftEyeOpen", "rightEyeOpen", "leftBrow", "rightBrow", "smile", "smirk", "mouth",
)
NEUTRAL_CHANNELS = ("roll", "yaw", "gazeX", "gazeY", "leftBrow", "rightBrow", "smile", "smirk")


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def _derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.empty(0, dtype=np.float64)
    dt = np.maximum(np.diff(times), 1e-3)
    return np.diff(values) / dt


def analyze_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    usable = []
    for index, row in enumerate(rows):
        metrics = row.get("metrics", row)
        if not isinstance(metrics, dict):
            continue
        time_value = row.get("time", row.get("sampleTime", index / 12.5))
        if all(name in metrics for name in CHANNELS):
            usable.append((float(time_value), metrics, str(row.get("state", "unknown"))))
    if len(usable) < 8:
        raise ValueError("at least eight complete landmark observations are required")
    usable.sort(key=lambda item: item[0])
    times = np.array([item[0] for item in usable], dtype=np.float64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("observation times must be strictly increasing")
    matrix = np.array([[float(item[1][name]) for name in CHANNELS] for item in usable])
    medians = np.median(matrix, axis=0)
    centered = matrix - medians
    q25, q75 = np.percentile(matrix, (25, 75), axis=0)
    scales = np.maximum(q75 - q25, 1e-4)

    channel_report: dict[str, object] = {}
    for column, name in enumerate(CHANNELS):
        values = centered[:, column]
        velocity = _derivative(values, times)
        acceleration = _derivative(velocity, times[1:])
        jerk = _derivative(acceleration, times[2:])
        channel_report[name] = {
            "median": float(medians[column]),
            "absolute_deviation_p50": float(np.percentile(np.abs(values), 50)),
            "absolute_deviation_p75": float(np.percentile(np.abs(values), 75)),
            "absolute_deviation_p90": float(np.percentile(np.abs(values), 90)),
            "velocity_rms": _rms(velocity),
            "acceleration_rms": _rms(acceleration),
            "jerk_rms": _rms(jerk),
        }

    neutral_columns = [CHANNELS.index(name) for name in NEUTRAL_CHANNELS]
    normalized = np.abs(centered[:, neutral_columns]) / scales[neutral_columns]
    near_neutral = np.all(normalized <= 0.75, axis=1)
    correlation = np.corrcoef(matrix, rowvar=False)
    pairs = []
    for left in range(len(CHANNELS)):
        for right in range(left + 1, len(CHANNELS)):
            value = float(correlation[left, right])
            if np.isfinite(value):
                pairs.append((abs(value), CHANNELS[left], CHANNELS[right], value))
    strongest = sorted(pairs, reverse=True)[:12]
    states = Counter(item[2] for item in usable)
    return {
        "observations": len(usable),
        "duration_seconds": float(times[-1] - times[0]),
        "sample_rate_hz": float((len(times) - 1) / max(1e-6, times[-1] - times[0])),
        "near_neutral_fraction": float(np.mean(near_neutral)),
        "states": dict(states),
        "channels": channel_report,
        "strongest_correlations": [
            {"first": first, "second": second, "correlation": value}
            for _, first, second, value in strongest
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, help="MediaPipe observations, one JSON object per line")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    report = json.dumps(analyze_rows(rows), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(report + "\n")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
