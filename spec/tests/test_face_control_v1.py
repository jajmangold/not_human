"""FaceControlV1 normal, bounds, missing data, and replay contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nothuman.face_control_v1 import FaceControlV1, PINNED_CATEGORIES


def frame() -> FaceControlV1:
    """Create a test frame with valid data."""
    return FaceControlV1(
        timestamp_ns=1_000_000,
        sequence=7,
        blendshapes={"jawOpen": 0.25, "eyeBlinkLeft": 0.1},
        confidence={"jawOpen": 0.95, "eyeBlinkLeft": 0.85},
        transforms={"x": 0.0, "y": 0.0, "z": 0.0},
        valid=True,
        reason="",
    )


def test_pinned_categories_have_exactly_52_entries() -> None:
    """Verify the pinned category list has exactly 52 unique entries."""
    assert len(PINNED_CATEGORIES) == 52
    assert len(set(PINNED_CATEGORIES)) == 52


def test_round_trip_and_fixture() -> None:
    """Test JSON round-trip and fixture consistency."""
    value = frame()
    json_str = value.to_json()
    fixture = Path("tests/fixtures/face_control_v1.json").read_text().strip()
    assert json_str == fixture
    parsed = FaceControlV1.from_json(json_str)
    assert parsed == value
    assert value.to_json() == json_str


def test_normal_case_with_all_fields() -> None:
    """Test normal case with all fields populated."""
    value = frame()
    assert value.timestamp_ns == 1_000_000
    assert value.sequence == 7
    assert value.blendshapes["jawOpen"] == 0.25
    assert value.blendshapes["eyeBlinkLeft"] == 0.1
    assert value.confidence["jawOpen"] == 0.95
    assert value.transforms["x"] == 0.0
    assert value.valid is True
    assert value.reason == ""


def test_bounds_reject_invalid_values() -> None:
    """Test that out-of-bounds values are rejected."""
    with pytest.raises(ValueError):
        FaceControlV1(timestamp_ns=-1, sequence=0)
    with pytest.raises(ValueError):
        FaceControlV1(timestamp_ns=0, sequence=-1)
    with pytest.raises(ValueError):
        FaceControlV1(timestamp_ns=0, sequence=0, blendshapes={"jawOpen": 1.5})
    with pytest.raises(ValueError):
        FaceControlV1(timestamp_ns=0, sequence=0, blendshapes={"jawOpen": -0.1})
    with pytest.raises(ValueError):
        FaceControlV1(timestamp_ns=0, sequence=0, confidence={"jawOpen": 1.5})
    with pytest.raises(ValueError):
        FaceControlV1(timestamp_ns=0, sequence=0, confidence={"jawOpen": -0.1})
    with pytest.raises(ValueError):
        FaceControlV1(timestamp_ns=0, sequence=0, blendshapes={"jawOpen": float("inf")})
    with pytest.raises(ValueError):
        FaceControlV1(timestamp_ns=0, sequence=0, confidence={"jawOpen": float("nan")})


def test_missing_data_handling() -> None:
    """Test handling of missing or incomplete data."""
    value = FaceControlV1(timestamp_ns=0, sequence=0)
    assert value.blendshapes == {}
    assert value.confidence == {}
    assert value.transforms == {}
    value = FaceControlV1(
        timestamp_ns=0,
        sequence=0,
        blendshapes={"jawOpen": 0.5},
        confidence={},
    )
    assert value.blendshapes["jawOpen"] == 0.5
    assert value.confidence == {}


def test_replay_is_byte_for_byte_deterministic() -> None:
    """Test that replay produces identical output."""
    value = frame()
    assert value.to_json() == value.to_json()


def test_invalid_category_rejected() -> None:
    """Test that non-pinned categories are rejected."""
    with pytest.raises(ValueError):
        FaceControlV1(
            timestamp_ns=0,
            sequence=0,
            blendshapes={"invalidCategory": 0.5},
        )
    with pytest.raises(ValueError):
        FaceControlV1(
            timestamp_ns=0,
            sequence=0,
            confidence={"invalidCategory": 0.5},
        )

def test_forward_compatibility() -> None:
    """Test that unknown fields are ignored for forward compatibility."""
    value = frame()
    payload = json.loads(value.to_json())
    payload["future_field"] = {"value": 1}
    parsed = FaceControlV1.from_dict(payload)
    assert parsed == value


def test_validity_fields() -> None:
    """Test validity field handling."""
    value = FaceControlV1(timestamp_ns=0, sequence=0, valid=True, reason="")
    assert value.valid is True
    assert value.reason == ""
    value = FaceControlV1(
        timestamp_ns=0, sequence=0, valid=False, reason="face not detected"
    )
    assert value.valid is False
    assert value.reason == "face not detected"
    with pytest.raises(ValueError):
        FaceControlV1(timestamp_ns=0, sequence=0, valid="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FaceControlV1(timestamp_ns=0, sequence=0, reason=123)  # type: ignore[arg-type]


def test_transforms_validation() -> None:
    """Test transform parameter validation."""
    value = FaceControlV1(
        timestamp_ns=0,
        sequence=0,
        transforms={"x": 1.0, "y": 2.0, "z": 3.0},
    )
    assert value.transforms["x"] == 1.0
    assert value.transforms["y"] == 2.0
    assert value.transforms["z"] == 3.0
    with pytest.raises(ValueError):
        FaceControlV1(
            timestamp_ns=0,
            sequence=0,
            transforms={"x": float("inf")},
        )
    with pytest.raises(ValueError):
        FaceControlV1(
            timestamp_ns=0,
            sequence=0,
            transforms={"x": True},  # type: ignore[dict-item]
        )


def test_all_52_categories_accepted() -> None:
    """Test that all 52 pinned categories are accepted."""
    blendshapes = {name: 0.5 for name in PINNED_CATEGORIES}
    confidence = {name: 0.9 for name in PINNED_CATEGORIES}
    value = FaceControlV1(
        timestamp_ns=0, sequence=0, blendshapes=blendshapes, confidence=confidence
    )
    assert len(value.blendshapes) == 52
    assert len(value.confidence) == 52


def test_canonical_ordering() -> None:
    """Test that blendshapes and confidence are sorted in canonical order."""
    blendshapes = {"jawOpen": 0.1, "eyeBlinkLeft": 0.2, "browDownLeft": 0.3}
    value = FaceControlV1(timestamp_ns=0, sequence=0, blendshapes=blendshapes)
    expected_order = [name for name in PINNED_CATEGORIES if name in blendshapes]
    actual_order = list(value.blendshapes.keys())
    assert actual_order == expected_order


def test_json_schema_fields() -> None:
    """Test that JSON output includes correct schema fields."""
    value = frame()
    payload = json.loads(value.to_json())
    assert payload["schema"] == "FaceControlV1"
    assert payload["schema_version"] == 1
    assert "timestamp_ns" in payload
    assert "sequence" in payload
    assert "blendshapes" in payload
    assert "confidence" in payload
    assert "transforms" in payload
    assert "validity" in payload


def test_from_dict_rejects_invalid_schema() -> None:
    """Test that from_dict rejects invalid schema."""
    with pytest.raises(ValueError):
        FaceControlV1.from_dict({"schema": "WrongSchema", "schema_version": 1})
    with pytest.raises(ValueError):
        FaceControlV1.from_dict({"schema": "FaceControlV1", "schema_version": 2})


def test_from_json_rejects_non_object() -> None:
    """Test that from_json rejects non-object JSON."""
    with pytest.raises(ValueError):
        FaceControlV1.from_json("[1, 2, 3]")
    with pytest.raises(ValueError):
        FaceControlV1.from_json("42")


def test_deterministic_output_across_instances() -> None:
    """Test that identical inputs produce identical outputs across instances."""
    value1 = frame()
    value2 = frame()
    assert value1.to_json() == value2.to_json()
    assert value1 == value2
