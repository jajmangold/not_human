"""ControlFrameV1 normal, bounds, compatibility, and replay contract tests."""

import json
from pathlib import Path

import pytest

from nothuman.control_frame_v1 import ControlFrameV1


def frame() -> ControlFrameV1:
    return ControlFrameV1(
        1_000_000,
        7,
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        {"jawOpen": 0.25},
        {"face": 1.0},
    )


def test_round_trip_and_fixture() -> None:
    value = frame()
    assert value.to_json() == Path("tests/fixtures/control_frame_v1.json").read_text().strip()
    assert ControlFrameV1.from_json(value.to_json()) == value


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timestamp_ns": -1},
        {"sequence": -1},
        {"authority": {"face": 1.1}},
        {"root_position": (1.0, 2.0)},
    ],
)
def test_bounds_reject_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ControlFrameV1(**{"timestamp_ns": 0, "sequence": 0, **kwargs})


def test_unknown_fields_are_ignored_for_forward_compatibility() -> None:
    payload = json.loads(frame().to_json())
    payload["future_field"] = {"value": 1}
    assert ControlFrameV1.from_dict(payload) == frame()


def test_replay_is_byte_for_byte_deterministic() -> None:
    assert frame().to_json() == frame().to_json()


def test_generated_binding_matches_schema_contract() -> None:
    from nothuman import control_frame_v1_pb2

    assert control_frame_v1_pb2.SCHEMA == "nothuman.contract.ControlFrameV1"
    assert control_frame_v1_pb2.ControlFrameV1.field_numbers["timestamp_ns"] == 1
    assert control_frame_v1_pb2.ControlFrameV1.field_numbers["authority"] == 30

def test_versioning_and_provenance_are_recorded() -> None:
    from nothuman.control_frame_v1 import SCHEMA_REVISION
    from nothuman import control_frame_v1_pb2

    value = frame()
    payload = json.loads(value.to_json())
    assert payload["schema_revision"] == SCHEMA_REVISION
    assert payload["revision"] == SCHEMA_REVISION
    assert payload["provenance"] == {"model": "none", "data": "synthetic-fixture", "license": "none"}
    assert control_frame_v1_pb2.SCHEMA_VERSION == 1
    assert ControlFrameV1.from_json(value.to_json()) == value
