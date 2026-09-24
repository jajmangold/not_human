"""PCIe x1 placement and transfer-budget contract (ADR-0002).

Encodes the ADR-0002 decision as frozen contract constants and a small
admissibility check:

- Renderer tensors are GPU-resident and never cross a process or host
  boundary (0 bytes per frame across the PCIe x1 link).
- Only control vectors or crops may cross the boundary, and the total
  cross-boundary bytes in a frame must fit the pinned per-frame transfer
  budget.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

# --- Frozen link constants (ADR-0002 evidence) -------------------------------
# Measured hardware link: PCIe x1, Gen 4 -> 2.5 Gb/s = 312.5 MB/s effective.
LINK_NAME = "PCIe x1 Gen 4"
LINK_THROUGHPUT_BYTES_PER_SEC = 312_500_000  # 2.5 Gb/s
# Frame period at 30 fps.
FRAME_PERIOD_NS = 33_333_333
# Pinned per-frame transfer budget: 8 MiB (~80.5% of the measured link).
TRANSFER_BUDGET_BYTES_PER_FRAME = 8 * 1024 * 1024


class PayloadKind(str, Enum):
    """Kinds of payload that are allowed to cross a process/host boundary."""

    CONTROL_VECTOR = "control_vector"
    CROP = "crop"
    RENDERER_TENSOR = "renderer_tensor"

PAYLOAD_KIND_VALUES: tuple[str, ...] = (
    PayloadKind.CONTROL_VECTOR.value,
    PayloadKind.CROP.value,
    PayloadKind.RENDERER_TENSOR.value,
)
PAYLOAD_KINDS = frozenset(PAYLOAD_KIND_VALUES)
PAYLOAD_KIND_CONTROL_VECTOR = PayloadKind.CONTROL_VECTOR
PAYLOAD_KIND_CROP = PayloadKind.CROP
PAYLOAD_KIND_RENDERER_TENSOR = PayloadKind.RENDERER_TENSOR


@dataclass(frozen=True)
class PayloadTransfer:
    """One cross-boundary payload within a single frame."""

    kind: PayloadKind
    name: str
    bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PayloadKind):
            raise ValueError("kind must be a PayloadKind")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 0:
            raise ValueError("bytes must be a non-negative integer")


@dataclass(frozen=True)
class PlacementRecord:
    """A per-frame placement decision and its cross-boundary payload set."""

    frame_period_ns: int
    budget_bytes_per_frame: int
    payloads: tuple[PayloadTransfer, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_period_ns, bool)
            or not isinstance(self.frame_period_ns, int)
            or self.frame_period_ns <= 0
        ):
            raise ValueError("frame_period_ns must be a positive integer")
        if (
            isinstance(self.budget_bytes_per_frame, bool)
            or not isinstance(self.budget_bytes_per_frame, int)
            or self.budget_bytes_per_frame < 0
        ):
            raise ValueError("budget_bytes_per_frame must be a non-negative integer")
        if not isinstance(self.payloads, tuple):
            raise ValueError("payloads must be a tuple of PayloadTransfer")
        for payload in self.payloads:
            if not isinstance(payload, PayloadTransfer):
                raise ValueError("payloads must contain only PayloadTransfer items")

    @property
    def transfer_bytes(self) -> int:
        """Total bytes that cross the boundary this frame (tensors excluded)."""
        return sum(
            p.bytes for p in self.payloads if p.kind != PayloadKind.RENDERER_TENSOR
        )

    def check(self) -> "PlacementCheck":
        """Evaluate the ADR-0002 boundary for this frame."""
        violations: list[str] = []
        for payload in self.payloads:
            if payload.kind == PayloadKind.RENDERER_TENSOR:
                violations.append(
                    f"renderer_tensor '{payload.name}' must stay GPU-resident"
                )
        if self.transfer_bytes > self.budget_bytes_per_frame:
            violations.append(
                f"transfer budget exceeded: {self.transfer_bytes} > "
                f"{self.budget_bytes_per_frame}"
            )
        return PlacementCheck(
            ok=not violations,
            transfer_bytes=self.transfer_bytes,
            budget_bytes_per_frame=self.budget_bytes_per_frame,
            violations=tuple(violations),
        )
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "PlacementRecordV1",
            "frame_period_ns": self.frame_period_ns,
            "budget_bytes_per_frame": self.budget_bytes_per_frame,
            "payloads": [
                {"kind": p.kind.value, "name": p.name, "bytes": p.bytes}
                for p in self.payloads
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PlacementRecord":
        if value.get("schema") != "PlacementRecordV1":
            raise ValueError("unsupported PlacementRecordV1 schema")
        payloads = tuple(
            PayloadTransfer(
                kind=PayloadKind(str(item["kind"])),
                name=str(item["name"]),
                bytes=int(item["bytes"]),
            )
            for item in value.get("payloads", [])
        )
        return cls(
            frame_period_ns=int(value["frame_period_ns"]),
            budget_bytes_per_frame=int(value["budget_bytes_per_frame"]),
            payloads=payloads,
        )

    @classmethod
    def from_json(cls, value: str) -> "PlacementRecord":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("PlacementRecordV1 JSON must be an object")
        return cls.from_dict(parsed)


@dataclass(frozen=True)
class PlacementCheck:
    """Result of evaluating a frame against the ADR-0002 boundary."""

    ok: bool
    transfer_bytes: int
    budget_bytes_per_frame: int
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "transfer_bytes": self.transfer_bytes,
            "budget_bytes_per_frame": self.budget_bytes_per_frame,
            "violations": list(self.violations),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def transfer_budget_ok(payload_bytes: int) -> bool:
    """True if a single cross-boundary payload fits the per-frame budget."""
    if isinstance(payload_bytes, bool) or not isinstance(payload_bytes, int) or payload_bytes < 0:
        raise ValueError("payload_bytes must be a non-negative integer")
    return payload_bytes <= TRANSFER_BUDGET_BYTES_PER_FRAME


def default_frame(*payloads: PayloadTransfer) -> PlacementRecord:
    """Build a frame with the pinned ADR-0002 budget constants."""
    return PlacementRecord(
        frame_period_ns=FRAME_PERIOD_NS,
        budget_bytes_per_frame=TRANSFER_BUDGET_BYTES_PER_FRAME,
        payloads=tuple(payloads),
    )


def _budget_is_sound() -> bool:
    """The pinned budget must fit within the measured link with headroom."""
    return (
        FRAME_PERIOD_NS > 0
        and TRANSFER_BUDGET_BYTES_PER_FRAME > 0
        and (TRANSFER_BUDGET_BYTES_PER_FRAME / FRAME_PERIOD_NS * 1e9)
        <= LINK_THROUGHPUT_BYTES_PER_SEC
    )


assert _budget_is_sound(), "ADR-0002 budget must fit within the measured link"


