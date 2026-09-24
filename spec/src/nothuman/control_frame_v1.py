"""Dependency-free ControlFrameV1 contract and deterministic JSON debug view."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

SCHEMA = "ControlFrameV1"
SCHEMA_VERSION = 1
SCHEMA_REVISION = "v1.0.0"

_MAX_AUTHORITY = 1.0


def _finite_nonnegative(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _numbers(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    return tuple(_finite_number(v, name) for v in values)


def _authority(values: Mapping[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in sorted(values.items()):
        number = _finite_number(value, f"authority[{key}]")
        if not 0.0 <= number <= _MAX_AUTHORITY:
            raise ValueError(f"authority[{key}] must be between 0 and 1")
        result[str(key)] = number
    return result


@dataclass(frozen=True)
class ControlFrameV1:
    timestamp_ns: int
    sequence: int
    root_position: tuple[float, ...] = ()
    root_rotation6d: tuple[float, ...] = ()
    face_blendshapes: Mapping[str, float] = field(default_factory=dict)
    authority: Mapping[str, float] = field(default_factory=dict)
    validity: bool = True
    validity_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "timestamp_ns", _finite_nonnegative(self.timestamp_ns, "timestamp_ns")
        )
        object.__setattr__(self, "sequence", _finite_nonnegative(self.sequence, "sequence"))
        if len(self.root_position) not in (0, 3):
            raise ValueError("root_position must contain exactly 3 values")
        if len(self.root_rotation6d) not in (0, 6):
            raise ValueError("root_rotation6d must contain exactly 6 values")
        object.__setattr__(self, "root_position", _numbers(self.root_position, "root_position"))
        object.__setattr__(
            self, "root_rotation6d", _numbers(self.root_rotation6d, "root_rotation6d")
        )
        shapes = {
            str(k): _finite_number(v, f"face_blendshapes[{k}]")
            for k, v in self.face_blendshapes.items()
        }
        object.__setattr__(self, "face_blendshapes", dict(sorted(shapes.items())))
        object.__setattr__(self, "authority", _authority(self.authority))
        if not isinstance(self.validity, bool) or not isinstance(self.validity_reason, str):
            raise ValueError("validity fields have invalid types")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "schema_revision": SCHEMA_REVISION,
            "revision": SCHEMA_REVISION,
            "provenance": {
                "model": "none",
                "data": "synthetic-fixture",
                "license": "none",
            },
            "timestamp_ns": self.timestamp_ns,
            "sequence": self.sequence,
            "root": {
                "position": list(self.root_position),
                "rotation6d": list(self.root_rotation6d),
            },
            "face": {"blendshapes": dict(self.face_blendshapes)},
            "authority": dict(self.authority),
            "validity": {"valid": self.validity, "reason": self.validity_reason},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ControlFrameV1":
        if value.get("schema") != SCHEMA or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported ControlFrameV1 schema")
        parsed = dict(value)
        for key in ("schema_revision", "revision", "provenance"):
            parsed.pop(key, None)
        value = parsed
        root = value.get("root") or {}
        face = value.get("face") or {}
        validity = value.get("validity") or {}
        return cls(
            timestamp_ns=value["timestamp_ns"],
            sequence=value["sequence"],
            root_position=tuple(root.get("position", ())),
            root_rotation6d=tuple(root.get("rotation6d", ())),
            face_blendshapes=face.get("blendshapes", {}),
            authority=value.get("authority", {}),
            validity=validity.get("valid", True),
            validity_reason=validity.get("reason", ""),
        )

    @classmethod
    def from_json(cls, value: str) -> "ControlFrameV1":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("ControlFrameV1 JSON must be an object")
        return cls.from_dict(parsed)
