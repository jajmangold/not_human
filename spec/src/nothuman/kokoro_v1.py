"""KokoroV1 adapter: deterministic 24 kHz PCM chunk normalization and provenance."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SCHEMA = "KokoroV1"
SCHEMA_VERSION = 1
SAMPLE_RATE_HZ = 24_000
SAMPLES_PER_MS = SAMPLE_RATE_HZ // 1000
SAMPLE_WIDTH = 2
CHANNELS = 1


class KokoroV1Error(Exception):
    """Base error for KokoroV1 adapter failures."""


class KokoroV1ValidationError(KokoroV1Error):
    """Raised when a chunk violates the stable KokoroV1 contract."""


class KokoroV1ReplayError(KokoroV1Error):
    """Raised when a chunk cannot be replayed deterministically."""


class KokoroV1Cancellation(KokoroV1Error):
    """Raised when a chunk is cancelled before completion."""


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise KokoroV1ValidationError(f"{name} must be a non-negative integer")
    return value


def _pcm_sample(value: Any, index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise KokoroV1ValidationError(f"samples[{index}] must be a finite number")
    if not -32768 <= value <= 32767:
        raise KokoroV1ValidationError(f"samples[{index}] must be a 16-bit signed PCM sample")
    return int(value)


def _pcm_bytes(samples: Sequence[int]) -> bytes:
    return b"".join(struct.pack("<h", s) for s in samples)


def normalize_pcm(samples: Sequence[int] | bytes | bytearray) -> bytes:
    """Normalize any valid sample input to deterministic little-endian int16 bytes."""
    if isinstance(samples, (bytes, bytearray)):
        raw = bytes(samples)
        if len(raw) % SAMPLE_WIDTH != 0:
            raise KokoroV1ValidationError("pcm_bytes must be little-endian int16")
        return raw
    return _pcm_bytes(tuple(_pcm_sample(s, i) for i, s in enumerate(samples)))


@dataclass(frozen=True)
class KokoroProvenance:
    """Deterministic provenance for a Kokoro chunk."""

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
class KokoroV1Metrics:
    chunks: int = 0
    samples: int = 0
    dropped: int = 0
    cancelled: int = 0
    replayed: int = 0

@dataclass(frozen=True)
class KokoroV1Chunk:
    """One deterministic 24 kHz mono PCM chunk with provenance.

    Attributes:
        timestamp_ns: Start timestamp in nanoseconds.
        sequence: Monotonic sequence number.
        samples: Signed 16-bit PCM samples at SAMPLE_RATE_HZ.
        provenance: Model/data provenance record.
    """

    timestamp_ns: int
    sequence: int
    samples: tuple[int, ...] = ()
    provenance: KokoroProvenance | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_ns", _nonneg_int(self.timestamp_ns, "timestamp_ns"))
        object.__setattr__(self, "sequence", _nonneg_int(self.sequence, "sequence"))
        if isinstance(self.samples, (bytes, bytearray)):
            raw = bytes(self.samples)
            if len(raw) % SAMPLE_WIDTH != 0:
                raise KokoroV1ValidationError("pcm_bytes must be little-endian int16")
            samples = tuple(struct.unpack_from("<h", raw, i) for i in range(0, len(raw), SAMPLE_WIDTH))
        else:
            samples = tuple(_pcm_sample(s, i) for i, s in enumerate(self.samples))
        object.__setattr__(self, "samples", samples)

    def to_bytes(self) -> bytes:
        return _pcm_bytes(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "timestamp_ns": self.timestamp_ns,
            "sequence": self.sequence,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "channels": CHANNELS,
            "sample_width": SAMPLE_WIDTH,
            "samples": list(self.samples),
            "provenance": self.provenance.to_dict() if self.provenance is not None else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "KokoroV1Chunk":
        if value.get("schema") != SCHEMA or value.get("schema_version") != SCHEMA_VERSION:
            raise KokoroV1ValidationError("unsupported KokoroV1 schema")
        provenance = value.get("provenance")
        return cls(
            timestamp_ns=value["timestamp_ns"],
            sequence=value["sequence"],
            samples=tuple(value.get("samples", ())),
            provenance=KokoroProvenance(**provenance) if provenance is not None else None,
        )

    @classmethod
    def from_json(cls, value: str) -> "KokoroV1Chunk":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise KokoroV1ValidationError("KokoroV1 JSON must be an object")
        return cls.from_dict(parsed)


class KokoroV1Adapter:
    """Stable, cancellable, deterministic Kokoro PCM chunk adapter.

    The adapter normalizes raw signed samples into 24 kHz mono int16 chunks,
    records metrics, and supports cancellation plus deterministic replay.
    """

    def __init__(self, provenance: KokoroProvenance | None = None) -> None:
        self.provenance = provenance
        self._metrics = KokoroV1Metrics()
        self._cancelled = False
        self._replay: list[KokoroV1Chunk] = []

    @property
    def metrics(self) -> KokoroV1Metrics:
        return self._metrics

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _inc(self, **kwargs: int) -> None:
        m = self._metrics
        self._metrics = KokoroV1Metrics(
            chunks=m.chunks + kwargs.get("chunks", 0),
            samples=m.samples + kwargs.get("samples", 0),
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
            raise KokoroV1Cancellation("adapter cancelled")

    def push(
        self,
        timestamp_ns: int,
        sequence: int,
        samples: Sequence[int] | bytes | bytearray,
        provenance: KokoroProvenance | None = None,
    ) -> KokoroV1Chunk:
        """Normalize and store one chunk; raises when cancelled."""
        self._check_cancelled()
        if isinstance(samples, (bytes, bytearray)):
            raw = bytes(samples)
            if len(raw) % SAMPLE_WIDTH != 0:
                raise KokoroV1ValidationError("pcm_bytes must be little-endian int16")
            pcm = tuple(struct.unpack_from("<h", raw, i) for i in range(0, len(raw), SAMPLE_WIDTH))
        else:
            pcm = tuple(_pcm_sample(s, i) for i, s in enumerate(samples))
        chunk = KokoroV1Chunk(
            timestamp_ns=timestamp_ns,
            sequence=sequence,
            samples=pcm,
            provenance=provenance if provenance is not None else self.provenance,
        )
        self._replay.append(chunk)
        self._inc(chunks=1, samples=len(pcm))
        return chunk

    def push_pcm(self, timestamp_ns: int, sequence: int, pcm: bytes) -> KokoroV1Chunk:
        return self.push(timestamp_ns, sequence, pcm)

    def pop(self) -> KokoroV1Chunk:
        if not self._replay:
            raise KokoroV1ReplayError("no chunks to replay")
        chunk = self._replay.pop(0)
        self._inc(replayed=1)
        return chunk

    def replay(self) -> list[KokoroV1Chunk]:
        return list(self._replay)

    def metrics_dict(self) -> dict[str, int]:
        return {
            "chunks": self._metrics.chunks,
            "samples": self._metrics.samples,
            "dropped": self._metrics.dropped,
            "cancelled": self._metrics.cancelled,
            "replayed": self._metrics.replayed,
        }


def chunk_fingerprint(chunk: KokoroV1Chunk) -> str:
    """Deterministic SHA-256 fingerprint of a chunk payload plus provenance."""
    payload = json.dumps(
        {
            "timestamp_ns": chunk.timestamp_ns,
            "sequence": chunk.sequence,
            "samples": list(chunk.samples),
            "provenance": chunk.provenance.to_dict() if chunk.provenance is not None else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
