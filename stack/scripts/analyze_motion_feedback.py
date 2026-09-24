#!/usr/bin/env python3
"""Measure whether a generated blink moves the eyes without leaking into the mouth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _region_mean(frame: np.ndarray, bounds: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bounds
    region = frame[y1:y2, x1:x2]
    return float(region.mean()) if region.size else 0.0


def analyze_frames(frames: list[np.ndarray], bbox: list[float], fps: float) -> dict[str, float | int | bool]:
    if len(frames) < 3:
        raise ValueError("at least three frames are required")
    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise ValueError("all frames must have equal dimensions")
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("face bbox is invalid")
    face_height = y2 - y1
    split = min(y2, y1 + int(round(face_height * 0.64)))
    upper = (x1, y1, x2, split)
    lower = (x1, split, x2, y2)

    neutral = frames[0].astype(np.float32)
    differences = [np.abs(frame.astype(np.float32) - neutral).mean(axis=2) / 255.0 for frame in frames]
    upper_curve = np.array([_region_mean(diff, upper) for diff in differences])
    lower_curve = np.array([_region_mean(diff, lower) for diff in differences])
    peak_index = int(np.argmax(upper_curve))
    peak_upper = float(upper_curve[peak_index])
    peak_lower = float(lower_curve[peak_index])
    leak_ratio = peak_lower / max(peak_upper, 1e-6)

    threshold = peak_upper * 0.10
    active = np.flatnonzero(upper_curve >= threshold)
    duration_ms = float((active[-1] - active[0] + 1) * 1000.0 / fps) if active.size else 0.0
    closure_ms = float((peak_index - active[0]) * 1000.0 / fps) if active.size else 0.0
    reopening_ms = float((active[-1] - peak_index) * 1000.0 / fps) if active.size else 0.0
    velocity = np.diff(upper_curve)
    acceleration = np.diff(velocity)
    jerk = np.diff(acceleration)

    return {
        "frames": len(frames),
        "fps": float(fps),
        "peak_frame": peak_index,
        "peak_upper_motion": peak_upper,
        "peak_lower_motion": peak_lower,
        "lower_face_leak_ratio": float(leak_ratio),
        "blink_duration_ms": duration_ms,
        "closure_ms": closure_ms,
        "reopening_ms": reopening_ms,
        "reopening_slower_than_closure": reopening_ms > closure_ms,
        "trajectory_jerk_rms": float(np.sqrt(np.mean(jerk**2))) if jerk.size else 0.0,
        "passes_mouth_isolation": peak_upper >= 0.005 and leak_ratio <= 0.18,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", type=Path, help="directory containing ordered PNG/JPEG frames")
    parser.add_argument("--bbox", type=Path, help="JSON [x1,y1,x2,y2]; defaults to frames/face_bbox.json")
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--check", action="store_true", help="exit nonzero if the mouth-isolation gate fails")
    args = parser.parse_args()
    bbox_path = args.bbox or args.frames / "face_bbox.json"
    bbox = json.loads(bbox_path.read_text())
    paths = sorted(path for path in args.frames.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    frames = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    if not frames:
        raise SystemExit("no readable frames found")
    report = analyze_frames(frames, bbox, args.fps)
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(args.check and not report["passes_mouth_isolation"])


if __name__ == "__main__":
    raise SystemExit(main())
