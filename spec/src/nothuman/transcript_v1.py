"""TranscriptV1 adapter: deterministic speech-to-text segment normalization and provenance.

Named after what downstream consumers read (a transcript segment), not the
engine that produces it -- matching FaceControlV1/BodyControlV1's
convention, not KokoroV1's. Runtime model selection is deferred to a later
work item, same as every other adapter in this package: this module
normalizes and validates ASR output into a stable contract, it does not
call a speech-recognition model.

The field shape (timestamp bounds, is_final, no_speech_prob) matches what a
real streaming ASR backend actually emits -- CrispASR (a whisper.cpp fork)
specifically, evaluated end-to-end in docs/lab-notebook/01-hearing-stack.md and running
live as this fleet's musetalk-volta-crispasr-stt-1 service -- rather than an
invented shape. `no_speech_prob` in particular is the exact signal CrispASR's
own decode loop uses internally to decide whether a single-timestamp-ending
chunk is genuine silence or content it should keep decoding (see that
session doc's fix #1); it is not exposed by the OpenAI-compatible
`/v1/audio/transcriptions` endpoint this fleet's deployment currently uses,
but is included here as an optional field since a richer/native endpoint
can supply it, and #94 (the listening-face fast reaction path this adapter
feeds) explicitly needs to distinguish real silence from a dropped partial.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

SCHEMA = "TranscriptV1"
SCHEMA_VERSION = 1


class TranscriptV1Error(Exception):
    """Base error for TranscriptV1 adapter failures."""


class TranscriptV1ValidationError(TranscriptV1Error):
    """Raised when a segment violates the stable TranscriptV1 contract."""


class TranscriptV1ReplayError(TranscriptV1Error):
    """Raised when a segment cannot be replayed deterministically."""


class TranscriptV1Cancellation(TranscriptV1Error):
    """Raised when a stream is cancelled before completion."""


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TranscriptV1ValidationError(f"{name} must be a non-negative integer")
    return value


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise TranscriptV1ValidationError("text must be a string")
    return value


def _unit_prob_or_none(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TranscriptV1ValidationError(f"{name} must be a finite number or null")
    if not 0.0 <= value <= 1.0:
        raise TranscriptV1ValidationError(f"{name} must be in [0.0, 1.0]")
    return float(value)


@dataclass(frozen=True)
class TranscriptProvenance:
    """Deterministic provenance for a transcript segment."""

    model: str
    model_sha256: str
    data: str
    data_sha256: str
    revision: str
    fixture: str

    def to_dict(self) -> dict[str, str]:
        return {
            "model": self.model,
            "model_sha256": self.model_sha256,
            "data": self.data,
            "data_sha256": self.data_sha256,
            "revision": self.revision,
            "fixture": self.fixture,
        }


@dataclass(frozen=True)
class TranscriptV1Metrics:
    segments: int = 0
    finals: int = 0
    dropped: int = 0
    cancelled: int = 0
    replayed: int = 0


@dataclass(frozen=True)
class TranscriptV1Segment:
    """One deterministic ASR transcript segment with provenance.

    Attributes:
        timestamp_ns: Segment start timestamp in nanoseconds.
        sequence: Monotonic sequence number.
        text: Decoded text for this segment (may be empty for a genuine
            silence/no-speech segment; a real backend still emits a
            zero-length segment rather than nothing, so consumers can tell
            "no speech happened" apart from "the event never arrived").
        start_ns: Start of the audio span this segment covers, relative to
            the containing utterance.
        end_ns: End of the audio span this segment covers. Must be
            >= start_ns.
        is_final: False for a streaming partial that may still change,
            True once this segment's text is committed.
        no_speech_prob: Optional [0.0, 1.0] no-speech confidence from the
            backend, when available (see module docstring). None when the
            backend does not expose it.
        provenance: Model/data provenance record.
    """

    timestamp_ns: int
    sequence: int
    text: str = ""
    start_ns: int = 0
    end_ns: int = 0
    is_final: bool = False
    no_speech_prob: float | None = None
    provenance: TranscriptProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", _nonneg_int(self.timestamp_ns, "timestamp_ns"))
        object.__setattr__(self, "sequence", _nonneg_int(self.sequence, "sequence"))
        object.__setattr__(self, "text", _text(self.text))
        object.__setattr__(self, "start_ns", _nonneg_int(self.start_ns, "start_ns"))
        object.__setattr__(self, "end_ns", _nonneg_int(self.end_ns, "end_ns"))
        if self.end_ns < self.start_ns:
            raise TranscriptV1ValidationError("end_ns must be >= start_ns")
        if not isinstance(self.is_final, bool):
            raise TranscriptV1ValidationError("is_final must be a boolean")
        object.__setattr__(
            self, "no_speech_prob", _unit_prob_or_none(self.no_speech_prob, "no_speech_prob")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "timestamp_ns": self.timestamp_ns,
            "sequence": self.sequence,
            "text": self.text,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "is_final": self.is_final,
            "no_speech_prob": self.no_speech_prob,
            "provenance": self.provenance.to_dict() if self.provenance is not None else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TranscriptV1Segment":
        if value.get("schema") != SCHEMA or value.get("schema_version") != SCHEMA_VERSION:
            raise TranscriptV1ValidationError("unsupported TranscriptV1 schema")
        provenance = value.get("provenance")
        return cls(
            timestamp_ns=value["timestamp_ns"],
            sequence=value["sequence"],
            text=value.get("text", ""),
            start_ns=value.get("start_ns", 0),
            end_ns=value.get("end_ns", 0),
            is_final=value.get("is_final", False),
            no_speech_prob=value.get("no_speech_prob"),
            provenance=TranscriptProvenance(**provenance) if provenance is not None else None,
        )

    @classmethod
    def from_json(cls, value: str) -> "TranscriptV1Segment":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise TranscriptV1ValidationError("TranscriptV1 JSON must be an object")
        return cls.from_dict(parsed)


class TranscriptV1Adapter:
    """Stable, cancellable, deterministic ASR transcript segment adapter.

    Normalizes raw segment fields, records metrics, and supports
    cancellation plus deterministic replay -- mirrors KokoroV1Adapter's
    shape for the analogous input-side contract.
    """

    def __init__(self, provenance: TranscriptProvenance | None = None) -> None:
        self.provenance = provenance
        self._metrics = TranscriptV1Metrics()
        self._cancelled = False
        self._replay: list[TranscriptV1Segment] = []

    @property
    def metrics(self) -> TranscriptV1Metrics:
        return self._metrics

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _inc(self, **kwargs: int) -> None:
        m = self._metrics
        self._metrics = TranscriptV1Metrics(
            segments=m.segments + kwargs.get("segments", 0),
            finals=m.finals + kwargs.get("finals", 0),
            dropped=m.dropped + kwargs.get("dropped", 0),
            cancelled=m.cancelled + kwargs.get("cancelled", 0),
            replayed=m.replayed + kwargs.get("replayed", 0),
        )

    def cancel(self) -> None:
        if not self._cancelled:
            self._cancelled = True
            self._inc(cancelled=1)

    def is_cancelled(self) -> bool:
        return self._cancelled

    def _check_cancelled(self) -> None:
        if self._cancelled:
            raise TranscriptV1Cancellation("adapter cancelled")

    def push(
        self,
        timestamp_ns: int,
        sequence: int,
        text: str = "",
        start_ns: int = 0,
        end_ns: int = 0,
        is_final: bool = False,
        no_speech_prob: float | None = None,
        provenance: TranscriptProvenance | None = None,
    ) -> TranscriptV1Segment:
        """Normalize and store one segment; raises when cancelled."""
        self._check_cancelled()
        segment = TranscriptV1Segment(
            timestamp_ns=timestamp_ns,
            sequence=sequence,
            text=text,
            start_ns=start_ns,
            end_ns=end_ns,
            is_final=is_final,
            no_speech_prob=no_speech_prob,
            provenance=provenance if provenance is not None else self.provenance,
        )
        self._replay.append(segment)
        self._inc(segments=1, finals=1 if is_final else 0)
        return segment

    def pop(self) -> TranscriptV1Segment:
        if not self._replay:
            raise TranscriptV1ReplayError("no segments to replay")
        segment = self._replay.pop(0)
        self._inc(replayed=1)
        return segment

    def replay(self) -> list[TranscriptV1Segment]:
        return list(self._replay)

    def metrics_dict(self) -> dict[str, int]:
        return {
            "segments": self._metrics.segments,
            "finals": self._metrics.finals,
            "dropped": self._metrics.dropped,
            "cancelled": self._metrics.cancelled,
            "replayed": self._metrics.replayed,
        }


def segment_fingerprint(segment: TranscriptV1Segment) -> str:
    """Deterministic SHA-256 fingerprint of a segment payload plus provenance."""
    payload = json.dumps(
        {
            "timestamp_ns": segment.timestamp_ns,
            "sequence": segment.sequence,
            "text": segment.text,
            "start_ns": segment.start_ns,
            "end_ns": segment.end_ns,
            "is_final": segment.is_final,
            "no_speech_prob": segment.no_speech_prob,
            "provenance": segment.provenance.to_dict() if segment.provenance is not None else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
