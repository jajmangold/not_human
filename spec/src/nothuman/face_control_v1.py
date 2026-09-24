"""FaceControlV1 adapter: deterministic extraction of 52 pinned MediaPipe blendshape categories.

This module implements the FaceControlV1 contract for the nothuman avatar system.
It extracts face control data with:
- Pinned 52 blendshape categories (deterministic ordering)
- Per-category confidence values
- Transform parameters
- Validity state and reasoning
- Timestamps in nanoseconds

The implementation is deterministic: identical inputs produce bit-identical outputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

SCHEMA = "FaceControlV1"
SCHEMA_VERSION = 1

# The 52 pinned MediaPipe FaceLandmark blendshape categories, in canonical order.
PINNED_CATEGORIES: tuple[str, ...] = (
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterLeft",
    "browOuterRight",
    "cheekPuff",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawOpen",
    "mouthClose",
    "mouthFrown",
    "mouthPucker",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthFunnel",
    "mouthPurse",
    "mouthLeft",
    "mouthRight",
    "noseSneerLeft",
    "noseSneerRight",
    "tongueUp",
    "tongueDown",
    "cheekSquishLeft",
    "cheekSquishRight",
    "earLeft",
    "earRight",
    "eyebrowDownLeft",
    "eyebrowDownRight",
    "eyebrowUpLeft",
    "eyebrowUpRight",
    "jawLeft",
    "jawRight",
    "mouthShrug",
    "mouthPress",
)


def _finite_nonnegative(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _confidence(value: float, name: str) -> float:
    number = _finite_number(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number

@dataclass(frozen=True)
class FaceControlV1:
    """Deterministic face control frame with pinned 52 blendshape categories.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds (non-negative integer).
        sequence: Sequence number (non-negative integer).
        blendshapes: Mapping of category name to blendshape value (0.0 to 1.0).
        confidence: Mapping of category name to confidence value (0.0 to 1.0).
        transforms: Optional transform parameters (position, rotation).
        valid: Whether this frame is valid.
        reason: Reason string if invalid.
    """

    timestamp_ns: int
    sequence: int
    blendshapes: Mapping[str, float] = field(default_factory=dict)
    confidence: Mapping[str, float] = field(default_factory=dict)
    transforms: Mapping[str, float] = field(default_factory=dict)
    valid: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "timestamp_ns", _finite_nonnegative(self.timestamp_ns, "timestamp_ns")
        )
        object.__setattr__(self, "sequence", _finite_nonnegative(self.sequence, "sequence"))

        # Validate and pin blendshapes to canonical categories
        shapes: dict[str, float] = {}
        for key, value in self.blendshapes.items():
            name = str(key)
            if name not in PINNED_CATEGORIES:
                raise ValueError(f"blendshapes[{name}]: not a pinned category")
            number = _finite_number(value, f"blendshapes[{name}]")
            if not 0.0 <= number <= 1.0:
                raise ValueError(f"blendshapes[{name}] must be between 0 and 1")
            shapes[name] = number
        # Sort by canonical order
        canonical = {name: shapes[name] for name in PINNED_CATEGORIES if name in shapes}
        object.__setattr__(self, "blendshapes", canonical)

        # Validate and pin confidence to canonical categories
        conf: dict[str, float] = {}
        for key, value in self.confidence.items():
            name = str(key)
            if name not in PINNED_CATEGORIES:
                raise ValueError(f"confidence[{name}]: not a pinned category")
            conf[name] = _confidence(value, f"confidence[{name}]")
        canonical_conf = {name: conf[name] for name in PINNED_CATEGORIES if name in conf}
        object.__setattr__(self, "confidence", canonical_conf)

        # Validate transforms
        transforms: dict[str, float] = {}
        for key, value in self.transforms.items():
            name = str(key)
            transforms[name] = _finite_number(value, f"transforms[{name}]")
        object.__setattr__(self, "transforms", dict(sorted(transforms.items())))

        if not isinstance(self.valid, bool) or not isinstance(self.reason, str):
            raise ValueError("validity fields have invalid types")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "timestamp_ns": self.timestamp_ns,
            "sequence": self.sequence,
            "blendshapes": dict(self.blendshapes),
            "confidence": dict(self.confidence),
            "transforms": dict(self.transforms),
            "validity": {"valid": self.valid, "reason": self.reason},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FaceControlV1":
        if value.get("schema") != SCHEMA or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported FaceControlV1 schema")
        validity = value.get("validity") or {}
        return cls(
            timestamp_ns=value["timestamp_ns"],
            sequence=value["sequence"],
            blendshapes=value.get("blendshapes", {}),
            confidence=value.get("confidence", {}),
            transforms=value.get("transforms", {}),
            valid=validity.get("valid", True),
            reason=validity.get("reason", ""),
        )

    @classmethod
    def from_json(cls, value: str) -> "FaceControlV1":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("FaceControlV1 JSON must be an object")
        return cls.from_dict(parsed)
