"""BodyControlV1 adapter: deterministic extraction of 33 MediaPipe body landmarks.

This module implements the BodyControlV1 contract for the nothuman avatar
system. It extracts body control data with:

- Pinned 33 MediaPipe Pose landmark indices (deterministic ordering)
- Per-landmark visibility and confidence values
- Normalized joint positions (canonical body frame)
- Validity state and reasoning
- Timestamps in nanoseconds

The implementation is deterministic: identical inputs produce bit-identical
outputs. License-neutral per ADR-0003: no SMPL-X/AMASS dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

SCHEMA = "BodyControlV1"
SCHEMA_VERSION = 1

# The 33 pinned MediaPipe Pose landmark indices, in canonical order.
# Source: MediaPipe Pose landmark topology (stable across MediaPipe versions).
PINNED_LANDMARKS: tuple[str, ...] = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

# Canonical body frame: origin at pelvis (midpoint of left/right hip),
# +Y up (head direction), +X right (subject's right), +Z toward camera.
# All positions are normalized to [-1, 1] in each axis after filtering.
CANONICAL_BODY_VERSION = "canonical-body-v1"


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _finite_nonnegative(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _visibility(value: float, name: str) -> float:
    number = _finite_number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


@dataclass(frozen=True)
class Landmark:
    """A single MediaPipe body landmark in the canonical body frame.

    Attributes:
        x: Normalized x position in [-1, 1].
        y: Normalized y position in [-1, 1].
        z: Normalized z position in [-1, 1].
        visibility: Detection confidence in [0, 1].
    """

    x: float
    y: float
    z: float
    visibility: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_number(self.x, "x"))
        object.__setattr__(self, "y", _finite_number(self.y, "y"))
        object.__setattr__(self, "z", _finite_number(self.z, "z"))
        object.__setattr__(self, "visibility", _visibility(self.visibility, "visibility"))


@dataclass(frozen=True)
class BodyControlV1:
    """Deterministic body control frame with pinned 33 MediaPipe landmarks.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds (non-negative integer).
        sequence: Sequence number (non-negative integer).
        landmarks: Mapping of landmark name to Landmark (canonical order).
        valid: Whether this frame is valid.
        reason: Reason string if invalid.
    """

    timestamp_ns: int
    sequence: int
    landmarks: Mapping[str, Landmark] = field(default_factory=dict)
    valid: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns",
            _finite_nonnegative(self.timestamp_ns, "timestamp_ns"),
        )
        object.__setattr__(self, "sequence", _finite_nonnegative(self.sequence, "sequence"))

        # Validate and pin landmarks to canonical categories
        lm: dict[str, Landmark] = {}
        for key, value in self.landmarks.items():
            name = str(key)
            if name not in PINNED_LANDMARKS:
                raise ValueError(f"landmarks[{name}]: not a pinned landmark")
            if not isinstance(value, Landmark):
                raise ValueError(f"landmarks[{name}]: expected Landmark instance")
            lm[name] = value
        # Sort by canonical order
        canonical = {name: lm[name] for name in PINNED_LANDMARKS if name in lm}
        object.__setattr__(self, "landmarks", canonical)

        if not isinstance(self.valid, bool) or not isinstance(self.reason, str):
            raise ValueError("validity fields have invalid types")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "canonical_body_version": CANONICAL_BODY_VERSION,
            "timestamp_ns": self.timestamp_ns,
            "sequence": self.sequence,
            "landmarks": {
                name: {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for name, lm in self.landmarks.items()
            },
            "validity": {"valid": self.valid, "reason": self.reason},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BodyControlV1":
        if value.get("schema") != SCHEMA or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported BodyControlV1 schema")
        validity = value.get("validity") or {}
        raw_landmarks = value.get("landmarks", {})
        landmarks: dict[str, Landmark] = {}
        for name, raw in raw_landmarks.items():
            landmarks[str(name)] = Landmark(
                x=raw["x"],
                y=raw["y"],
                z=raw["z"],
                visibility=raw.get("visibility", 1.0),
            )
        return cls(
            timestamp_ns=value["timestamp_ns"],
            sequence=value["sequence"],
            landmarks=landmarks,
            valid=validity.get("valid", True),
            reason=validity.get("reason", ""),
        )

    @classmethod
    def from_json(cls, value: str) -> "BodyControlV1":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("BodyControlV1 JSON must be an object")
        return cls.from_dict(parsed)
