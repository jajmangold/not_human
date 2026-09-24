# Generated binding surface for proto/control_frame_v1.proto.
# This dependency-free checked-in binding is intentionally limited to the
# schema field contract; runtime serialization is owned by control_frame_v1.
"""Generated ControlFrameV1 field metadata."""

SCHEMA = "nothuman.contract.ControlFrameV1"
SCHEMA_VERSION = 1

FIELD_NUMBERS = {
    "timestamp_ns": 1,
    "sequence": 2,
    "root": 10,
    "body": 11,
    "head": 12,
    "gaze": 13,
    "face": 14,
    "speech": 15,
    "prosody": 16,
    "gesture": 17,
    "behavior": 18,
    "authority": 30,
    "validity": 31,
}


class ControlFrameV1:
    """Generated binding metadata for schema inspection and compatibility checks."""

    DESCRIPTOR = SCHEMA
    schema_version = SCHEMA_VERSION
    field_numbers = FIELD_NUMBERS.copy()
