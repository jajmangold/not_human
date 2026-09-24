"""KokoroV1 normal, bounds, missing data, and replay contract tests."""

from __future__ import annotations

import json

import pytest

from nothuman.kokoro_v1 import (
    CHANNELS,
    SAMPLE_RATE_HZ,
    SAMPLE_WIDTH,
    KokoroProvenance,
    KokoroV1Adapter,
    KokoroV1Cancellation,
    KokoroV1Chunk,
    KokoroV1ReplayError,
    KokoroV1ValidationError,
    chunk_fingerprint,
    normalize_pcm,
)


def provenance() -> KokoroProvenance:
    return KokoroProvenance(
        model="kokoro-v1.0",
        model_sha256="0" * 64,
        data="synthetic-sine",
        data_sha256="1" * 64,
        revision="a388eb7ec9befc21b13f76581435b0bfa4cc372c",
        fixture="tests/fixtures/kokoro_v1.json",
    )


def chunk() -> KokoroV1Chunk:
    return KokoroV1Chunk(
        timestamp_ns=1_000_000,
        sequence=7,
        samples=(0, 16384, -16384, 32767, -32768),
        provenance=provenance(),
    )


def test_constants_pin_24khz_mono_int16() -> None:
    assert SAMPLE_RATE_HZ == 24_000
    assert SAMPLE_WIDTH == 2
    assert CHANNELS == 1


def test_normal_case_with_all_fields() -> None:
    value = chunk()
    assert value.timestamp_ns == 1_000_000
    assert value.sequence == 7
    assert value.samples == (0, 16384, -16384, 32767, -32768)
    assert value.provenance.model == "kokoro-v1.0"
    assert value.provenance.revision.startswith("a388eb7")


def test_round_trip_and_fixture() -> None:
    fixture = json.loads(
        __import__("pathlib").Path("tests/fixtures/kokoro_v1.json").read_text()
    )
    value = KokoroV1Chunk.from_dict(fixture)
    assert value.to_json() == json.dumps(fixture, sort_keys=True, separators=(",", ":"))
    assert value == KokoroV1Chunk.from_json(value.to_json())


def test_bounds_reject_invalid_values() -> None:
    with pytest.raises(KokoroV1ValidationError):
        KokoroV1Chunk(timestamp_ns=-1, sequence=0)
    with pytest.raises(KokoroV1ValidationError):
        KokoroV1Chunk(timestamp_ns=0, sequence=-1)
    with pytest.raises(KokoroV1ValidationError):
        KokoroV1Chunk(timestamp_ns=0, sequence=0, samples=(32768,))
    with pytest.raises(KokoroV1ValidationError):
        KokoroV1Chunk(timestamp_ns=0, sequence=0, samples=(-32769,))
    with pytest.raises(KokoroV1ValidationError):
        KokoroV1Chunk(timestamp_ns=0, sequence=0, samples=(float("inf"),))
    with pytest.raises(KokoroV1ValidationError):
        KokoroV1Chunk(timestamp_ns=0, sequence=0, samples=(float("nan"),))
    with pytest.raises(KokoroV1ValidationError):
        KokoroV1Chunk(timestamp_ns=0, sequence=0, samples=(True,))


def test_missing_data_handling() -> None:
    value = KokoroV1Chunk(timestamp_ns=0, sequence=0)
    assert value.samples == ()
    assert value.provenance is None
    assert value.to_bytes() == b""
    parsed = KokoroV1Chunk.from_dict({
        "schema": "KokoroV1",
        "schema_version": 1,
        "timestamp_ns": 0,
        "sequence": 0,
        "samples": [],
        "provenance": None,
    })
    assert parsed.samples == ()
    assert parsed.provenance is None


def test_replay_is_byte_for_byte_deterministic() -> None:
    value = chunk()
    assert value.to_json() == value.to_json()
    assert value.to_bytes() == value.to_bytes()
    assert chunk_fingerprint(value) == chunk_fingerprint(chunk())


def test_normalize_pcm_is_deterministic() -> None:
    import struct

    samples = (0, 16384, -16384, 32767, -32768)
    expected = b"".join(struct.pack("<h", s) for s in samples)
    assert normalize_pcm(samples) == expected
    assert normalize_pcm(expected) == expected


def test_from_dict_rejects_invalid_schema() -> None:
    with pytest.raises(KokoroV1ValidationError):
        KokoroV1Chunk.from_dict({"schema": "WrongSchema", "schema_version": 1})
    with pytest.raises(KokoroV1ValidationError):
        KokoroV1Chunk.from_dict({"schema": "KokoroV1", "schema_version": 2})


def test_from_json_rejects_non_object() -> None:
    with pytest.raises(KokoroV1ValidationError):
        KokoroV1Chunk.from_json("[1, 2, 3]")


def test_adapter_push_and_metrics() -> None:
    adapter = KokoroV1Adapter(provenance=provenance())
    a = adapter.push(0, 0, (1, 2, 3))
    b = adapter.push(1000, 1, (4, 5))
    assert a.samples == (1, 2, 3)
    assert b.samples == (4, 5)
    assert adapter.metrics.chunks == 2
    assert adapter.metrics.samples == 5
    assert adapter.metrics_dict() == {
        "chunks": 2, "samples": 5, "dropped": 0, "cancelled": 0, "replayed": 0,
    }


def test_adapter_cancellation() -> None:
    adapter = KokoroV1Adapter(provenance=provenance())
    adapter.push(0, 0, (1,))
    adapter.cancel()
    assert adapter.is_cancelled()
    with pytest.raises(KokoroV1Cancellation):
        adapter.push(1000, 1, (2,))
    assert adapter.metrics.cancelled == 1
    assert adapter.metrics.chunks == 1


def test_adapter_replay_fifo_and_errors() -> None:
    adapter = KokoroV1Adapter(provenance=provenance())
    with pytest.raises(KokoroV1ReplayError):
        adapter.pop()
    a = adapter.push(0, 0, (1,))
    b = adapter.push(1000, 1, (2,))
    assert adapter.pop() == a
    assert adapter.pop() == b
    assert list(adapter.replay()) == []
    assert adapter.metrics.replayed == 2
