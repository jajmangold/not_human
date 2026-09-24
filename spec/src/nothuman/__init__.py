"""Canonical package boundary for the nothuman avatar system."""

__version__ = "0.1.0"

from nothuman.rotation import (
    RotationError,
    euler_to_matrix,
    matrix_from_6d,
    matrix_to_6d,
    matrix_to_euler,
    matrix_to_quaternion,
    quaternion_to_matrix,
    slerp,
    slerp_6d,
)
from nothuman.face_control_v1 import FaceControlV1, PINNED_CATEGORIES as FACE_PINNED_CATEGORIES
from nothuman.kokoro_v1 import (
    KokoroProvenance,
    KokoroV1Adapter,
    KokoroV1Chunk,
    KokoroV1Error,
    KokoroV1ValidationError,
    KokoroV1Cancellation,
    KokoroV1ReplayError,
    SAMPLE_RATE_HZ as KOKORO_SAMPLE_RATE_HZ,
)



__all__ = [
    "__version__",
    "FaceControlV1",
    "FACE_PINNED_CATEGORIES",
    "KokoroProvenance",
    "KokoroV1Adapter",
    "KokoroV1Chunk",
    "KokoroV1Error",
    "KokoroV1ValidationError",
    "KokoroV1Cancellation",
    "KokoroV1ReplayError",
    "KOKORO_SAMPLE_RATE_HZ",
    "RotationError",
    "euler_to_matrix",
    "matrix_to_euler",
    "matrix_to_quaternion",
    "quaternion_to_matrix",
    "matrix_to_6d",
    "matrix_from_6d",
    "slerp",
    "slerp_6d",
]
