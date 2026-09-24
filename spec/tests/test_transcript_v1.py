"""TranscriptV1 normal, bounds, missing data, and replay contract tests."""

from __future__ import annotations

import json

import pytest

from nothuman.transcript_v1 import (
    SCHEMA,
    SCHEMA_VERSION,
    TranscriptProvenance,
    TranscriptV1Adapter,
    TranscriptV1Cancellation,
    TranscriptV1ReplayError,
    TranscriptV1Segment,
    TranscriptV1ValidationError,
    segment_fingerprint,
)


def provenance() -> TranscriptProvenance:
    return TranscriptProvenance(
        model="crispasr-large-v3-turbo",
        model_sha256="2" * 64,
        data="librispeech-spk6930-clip0",
        data_sha256="3" * 64,
        revision="musetalk-volta-crispasr-stt-1",
        fixture="tests/fixtures/transcript_v1.json",
    )


def segment() -> TranscriptV1Segment:
    return TranscriptV1Segment(
        timestamp_ns=2_000_000,
        sequence=3,
        text="Concord returned to its place amidst the tents.",
        start_ns=0,
        end_ns=3_505_000_000,
        is_final=True,
        no_speech_prob=0.02,
        provenance=provenance(),
    )


def test_constants_pin_schema() -> None:
    assert SCHEMA == "TranscriptV1"
    assert SCHEMA_VERSION == 1


def test_normal_case_with_all_fields() -> None:
    value = segment()
    assert value.timestamp_ns == 2_000_000
    assert value.sequence == 3
    assert value.text == "Concord returned to its place amidst the tents."
    assert value.start_ns == 0
    assert value.end_ns == 3_505_000_000
    assert value.is_final is True
    assert value.no_speech_prob == 0.02
    assert value.provenance.model == "crispasr-large-v3-turbo"
    assert value.provenance.revision == "musetalk-volta-crispasr-stt-1"


def test_round_trip_and_fixture() -> None:
    fixture = json.loads(
        __import__("pathlib").Path("tests/fixtures/transcript_v1.json").read_text()
    )
    value = TranscriptV1Segment.from_dict(fixture)
    assert value.to_json() == json.dumps(fixture, sort_keys=True, separators=(",", ":"))
    assert value == TranscriptV1Segment.from_json(value.to_json())


def test_bounds_reject_invalid_values() -> None:
    with pytest.raises(TranscriptV1ValidationError):
        TranscriptV1Segment(timestamp_ns=-1, sequence=0)
    with pytest.raises(TranscriptV1ValidationError):
        TranscriptV1Segment(timestamp_ns=0, sequence=-1)
    with pytest.raises(TranscriptV1ValidationError):
        TranscriptV1Segment(timestamp_ns=0, sequence=0, text=123)  # type: ignore[arg-type]
    with pytest.raises(TranscriptV1ValidationError):
        TranscriptV1Segment(timestamp_ns=0, sequence=0, start_ns=-1)
    with pytest.raises(TranscriptV1ValidationError):
        TranscriptV1Segment(timestamp_ns=0, sequence=0, end_ns=-1)
    with pytest.raises(TranscriptV1ValidationError):
        TranscriptV1Segment(timestamp_ns=0, sequence=0, start_ns=100, end_ns=50)
    with pytest.raises(TranscriptV1ValidationError):
        TranscriptV1Segment(timestamp_ns=0, sequence=0, is_final="true")  # type: ignore[arg-type]
    with pytest.raises(TranscriptV1ValidationError):
        TranscriptV1Segment(timestamp_ns=0, sequence=0, no_speech_prob=1.5)
    with pytest.raises(TranscriptV1ValidationError):
        TranscriptV1Segment(timestamp_ns=0, sequence=0, no_speech_prob=-0.1)
    with pytest.raises(TranscriptV1ValidationError):
        TranscriptV1Segment(timestamp_ns=0, sequence=0, no_speech_prob=float("nan"))


def test_missing_data_handling() -> None:
    value = TranscriptV1Segment(timestamp_ns=0, sequence=0)
    assert value.text == ""
    assert value.start_ns == 0
    assert value.end_ns == 0
    assert value.is_final is False
    assert value.no_speech_prob is None
    assert value.provenance is None
    parsed = TranscriptV1Segment.from_dict({
        "schema": "TranscriptV1",
        "schema_version": 1,
        "timestamp_ns": 0,
        "sequence": 0,
    })
    assert parsed.text == ""
    assert parsed.provenance is None


def test_empty_text_is_a_valid_no_speech_segment() -> None:
    # A genuine no-speech segment (real backend emits it rather than
    # nothing) must round-trip distinctly from a missing/dropped event.
    value = TranscriptV1Segment(
        timestamp_ns=0, sequence=0, text="", is_final=True, no_speech_prob=0.91
    )
    assert value.text == ""
    assert value.is_final is True
    assert value.no_speech_prob == 0.91


def test_replay_is_byte_for_byte_deterministic() -> None:
    value = segment()
    assert value.to_json() == value.to_json()
    assert segment_fingerprint(value) == segment_fingerprint(segment())


def test_from_dict_rejects_invalid_schema() -> None:
    with pytest.raises(TranscriptV1ValidationError):
        TranscriptV1Segment.from_dict({"schema": "WrongSchema", "schema_version": 1})
    with pytest.raises(TranscriptV1ValidationError):
        TranscriptV1Segment.from_dict({"schema": "TranscriptV1", "schema_version": 2})


def test_from_json_rejects_non_object() -> None:
    with pytest.raises(TranscriptV1ValidationError):
        TranscriptV1Segment.from_json("[1, 2, 3]")


def test_adapter_push_and_metrics() -> None:
    adapter = TranscriptV1Adapter(provenance=provenance())
    a = adapter.push(0, 0, text="hello", is_final=False)
    b = adapter.push(1000, 1, text="hello there", is_final=True)
    assert a.text == "hello"
    assert b.is_final is True
    assert adapter.metrics.segments == 2
    assert adapter.metrics.finals == 1
    assert adapter.metrics_dict() == {
        "segments": 2, "finals": 1, "dropped": 0, "cancelled": 0, "replayed": 0,
    }


def test_adapter_cancellation() -> None:
    adapter = TranscriptV1Adapter(provenance=provenance())
    adapter.push(0, 0, text="a")
    adapter.cancel()
    assert adapter.is_cancelled()
    with pytest.raises(TranscriptV1Cancellation):
        adapter.push(1000, 1, text="b")
    assert adapter.metrics.cancelled == 1
    assert adapter.metrics.segments == 1


def test_adapter_replay_fifo_and_errors() -> None:
    adapter = TranscriptV1Adapter(provenance=provenance())
    with pytest.raises(TranscriptV1ReplayError):
        adapter.pop()
    a = adapter.push(0, 0, text="a")
    b = adapter.push(1000, 1, text="b", is_final=True)
    assert adapter.pop() == a
    assert adapter.pop() == b
    assert list(adapter.replay()) == []
    assert adapter.metrics.replayed == 2
