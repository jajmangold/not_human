"""Tests for BodyControlV1 adapter: normal, bounds, missing data, and replay behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from nothuman.body_control_v1 import (
    CANONICAL_BODY_VERSION,
    PINNED_LANDMARKS,
    BodyControlV1,
    Landmark,
)


def test_pinned_landmarks_count():
    assert len(PINNED_LANDMARKS) == 33


def test_pinned_landmarks_unique():
    assert len(set(PINNED_LANDMARKS)) == 33


def test_landmark_valid():
    lm = Landmark(x=0.5, y=-0.3, z=0.1, visibility=0.9)
    assert lm.x == 0.5
    assert lm.y == -0.3
    assert lm.z == 0.1
    assert lm.visibility == 0.9


def test_landmark_rejects_non_finite():
    with pytest.raises(ValueError, match="must be finite"):
        Landmark(x=float("inf"), y=0.0, z=0.0, visibility=1.0)


def test_landmark_rejects_out_of_range_visibility():
    with pytest.raises(ValueError, match="between 0 and 1"):
        Landmark(x=0.0, y=0.0, z=0.0, visibility=1.5)


def test_body_control_v1_normal():
    landmarks = {name: Landmark(x=0.0, y=0.0, z=0.0, visibility=1.0) for name in PINNED_LANDMARKS}
    frame = BodyControlV1(timestamp_ns=1000, sequence=1, landmarks=landmarks)
    assert frame.timestamp_ns == 1000
    assert frame.sequence == 1
    assert len(frame.landmarks) == 33
    assert frame.valid is True
    assert frame.reason == ""


def test_body_control_v1_rejects_unknown_landmark():
    with pytest.raises(ValueError, match="not a pinned landmark"):
        BodyControlV1(
            timestamp_ns=0,
            sequence=0,
            landmarks={"unknown_joint": Landmark(x=0.0, y=0.0, z=0.0, visibility=1.0)},
        )


def test_body_control_v1_rejects_negative_timestamp():
    with pytest.raises(ValueError, match="non-negative"):
        BodyControlV1(timestamp_ns=-1, sequence=0)


def test_body_control_v1_rejects_negative_sequence():
    with pytest.raises(ValueError, match="non-negative"):
        BodyControlV1(timestamp_ns=0, sequence=-1)


def test_body_control_v1_partial_landmarks():
    landmarks = {"nose": Landmark(x=0.0, y=0.5, z=0.0, visibility=1.0)}
    frame = BodyControlV1(timestamp_ns=0, sequence=0, landmarks=landmarks)
    assert len(frame.landmarks) == 1
    assert "nose" in frame.landmarks


def test_body_control_v1_missing_landmarks_empty():
    frame = BodyControlV1(timestamp_ns=0, sequence=0)
    assert len(frame.landmarks) == 0


def test_body_control_v1_invalid_frame():
    frame = BodyControlV1(timestamp_ns=0, sequence=0, valid=False, reason="occluded")
    assert frame.valid is False
    assert frame.reason == "occluded"


def test_to_json_roundtrip():
    landmarks = {
        "nose": Landmark(x=0.1, y=0.5, z=0.0, visibility=1.0),
        "left_shoulder": Landmark(x=-0.3, y=-0.2, z=0.1, visibility=0.95),
        "right_shoulder": Landmark(x=0.3, y=-0.2, z=0.1, visibility=0.95),
    }
    frame = BodyControlV1(timestamp_ns=42, sequence=7, landmarks=landmarks)
    encoded = frame.to_json()
    decoded = BodyControlV1.from_json(encoded)
    assert decoded.timestamp_ns == 42
    assert decoded.sequence == 7
    assert decoded.to_json() == encoded


def test_to_dict_schema():
    frame = BodyControlV1(timestamp_ns=0, sequence=0)
    d = frame.to_dict()
    assert d["schema"] == "BodyControlV1"
    assert d["schema_version"] == 1
    assert d["canonical_body_version"] == CANONICAL_BODY_VERSION


def test_from_dict_rejects_wrong_schema():
    with pytest.raises(ValueError, match="unsupported BodyControlV1 schema"):
        BodyControlV1.from_dict(
            {"schema": "WrongSchema", "schema_version": 1, "timestamp_ns": 0, "sequence": 0},
        )


def test_from_json_rejects_non_object():
    with pytest.raises(ValueError, match="must be an object"):
        BodyControlV1.from_json("[1, 2, 3]")


def test_replay_determinism():
    """Replaying the same frame N times produces identical JSON."""
    landmarks = {
        name: Landmark(x=float(i) / 33.0, y=0.0, z=0.0, visibility=1.0)
        for i, name in enumerate(PINNED_LANDMARKS)
    }
    frame = BodyControlV1(timestamp_ns=100, sequence=1, landmarks=landmarks)
    results = {frame.to_json() for _ in range(10)}
    assert len(results) == 1


def test_canonical_ordering():
    """Landmarks are stored in canonical PINNED_LANDMARKS order."""
    landmarks = {
        "right_hip": Landmark(x=0.2, y=0.0, z=0.0, visibility=1.0),
        "nose": Landmark(x=0.0, y=0.5, z=0.0, visibility=1.0),
        "left_hip": Landmark(x=-0.2, y=0.0, z=0.0, visibility=1.0),
    }
    frame = BodyControlV1(timestamp_ns=0, sequence=0, landmarks=landmarks)
    keys = list(frame.landmarks.keys())
    expected = [n for n in PINNED_LANDMARKS if n in {"nose", "left_hip", "right_hip"}]
    assert keys == expected


def test_fixture_roundtrip():
    """Load the canonical fixture and verify round-trip determinism."""
    fixture_path = Path(__file__).parent / "fixtures" / "body_control_v1.json"
    raw = fixture_path.read_text()
    frame = BodyControlV1.from_json(raw)
    assert frame.timestamp_ns == 1_000_000_000
    assert frame.sequence == 1
    assert len(frame.landmarks) == 33
    assert frame.valid is True
    # Round-trip: to_json -> from_json -> to_json is stable
    reencoded = frame.to_json()
    decoded = BodyControlV1.from_json(reencoded)
    assert decoded.to_json() == reencoded
