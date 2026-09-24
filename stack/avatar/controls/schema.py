"""Canonical MediaPipe/ARKit-compatible avatar control stream.

The schema deliberately contains no AdvancedLivePortrait slider names.  Those
implementation-specific values belong in a renderer adapter; the rest of the
stack exchanges this representation instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCHEMA_NAME = "musetalk-avatar-controls"
SCHEMA_VERSION = 1

# MediaPipe Face Landmarker emits these 52 ARKit-style names.  Keep this order
# stable: the column index is part of controls.npz and is checked on load.
BLENDSHAPE_NAMES = (
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
    "tongueOut",
)
BLENDSHAPE_INDEX = {name: index for index, name in enumerate(BLENDSHAPE_NAMES)}

# MediaPipe Face Landmarker's real output is 52 categories, but they are not
# BLENDSHAPE_NAMES verbatim: index 0 is a synthetic "_neutral" score (how
# neutral the face looks overall, not an ARKit shape), and the model never
# emits an actual "tongueOut" score -- there is no tongue-visibility signal in
# a monocular RGB face crop. This was verified empirically against the pinned
# `face_landmarker.task` asset (mediapipe 0.10.35): category 0 is "_neutral",
# categories 1-51 are BLENDSHAPE_NAMES[0:51] in this exact order, and
# "tongueOut" (BLENDSHAPE_NAMES[51]) is not among the returned categories.
# BLENDSHAPE_NAMES keeps the full ARKit-52 name set for renderer/interchange
# compatibility; extractors that cannot measure some of those channels must
# say so explicitly (see FaceLandmarkerExtractor.UNSUPPORTED_BLENDSHAPES)
# rather than let an always-zero value be mistaken for a measured zero.
NEUTRAL_CATEGORY_NAME = "_neutral"


def _array(value: Sequence[float], shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class AvatarFrame:
    """One timestamped frame in canonical control space.

    ``head_rotation`` is XYZ Euler rotation in radians, derived from the
    MediaPipe facial transformation matrix. ``head_translation`` is that
    matrix's translation component: a right-handed, centimeter-scale
    coordinate space with a virtual perspective camera at the origin looking
    down -Z (Google's MediaPipe Face Geometry module; see
    docs/AVATAR-CONTROLS.md for sources and caveats). Blendshape scores are
    float32 values in [0, 1].

    ``confidence`` is detection *validity*, not a graded measurement: the
    bundled MediaPipe Face Landmarker IMAGE-mode API exposes no continuous
    per-detection confidence (landmark ``presence``/``visibility`` are always
    ``None`` for this task, verified empirically), so extractors built on it
    can only report whether a face was found (1.0) or not (0.0). Treat any
    other extractor's continuous confidence as extractor-specific and
    document it there; do not assume this field is graded.

    ``neutral_score`` is MediaPipe's own "_neutral" category: a measure of how
    neutral/expressionless the face looks overall. It is not an ARKit shape
    and is deliberately kept out of ``blendshapes`` (see
    ``NEUTRAL_CATEGORY_NAME``); it is optional because non-MediaPipe
    extractors may not produce it.
    """

    timestamp: float
    blendshapes: np.ndarray
    head_rotation: np.ndarray
    head_translation: np.ndarray
    confidence: float = 1.0
    head_transform: np.ndarray | None = None
    neutral_score: float | None = None

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be a finite non-negative number")
        blendshapes = _array(self.blendshapes, (len(BLENDSHAPE_NAMES),), "blendshapes")
        if np.any(blendshapes < 0) or np.any(blendshapes > 1):
            raise ValueError("blendshapes must be in [0, 1]")
        rotation = _array(self.head_rotation, (3,), "head_rotation")
        translation = _array(self.head_translation, (3,), "head_translation")
        confidence = float(self.confidence)
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        transform = None if self.head_transform is None else _array(self.head_transform, (4, 4), "head_transform")
        neutral_score = None if self.neutral_score is None else float(self.neutral_score)
        if neutral_score is not None and (not np.isfinite(neutral_score) or not 0 <= neutral_score <= 1):
            raise ValueError("neutral_score must be in [0, 1]")
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "blendshapes", blendshapes)
        object.__setattr__(self, "head_rotation", rotation)
        object.__setattr__(self, "head_translation", translation)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "head_transform", transform)
        object.__setattr__(self, "neutral_score", neutral_score)


@dataclass(frozen=True)
class AvatarSequence:
    """A monotonic sequence of :class:`AvatarFrame` values."""

    timestamps: np.ndarray
    blendshapes: np.ndarray
    head_rotation: np.ndarray
    head_translation: np.ndarray
    confidence: np.ndarray
    head_transform: np.ndarray | None = None
    neutral_score: np.ndarray | None = None

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        if timestamps.ndim != 1 or timestamps.size == 0:
            raise ValueError("timestamps must be a non-empty one-dimensional array")
        if not np.isfinite(timestamps).all() or timestamps[0] < 0 or np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamps must be finite, non-negative, and strictly increasing")
        count = timestamps.size
        blendshapes = np.asarray(self.blendshapes, dtype=np.float32)
        rotation = np.asarray(self.head_rotation, dtype=np.float32)
        translation = np.asarray(self.head_translation, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        if blendshapes.shape != (count, len(BLENDSHAPE_NAMES)):
            raise ValueError(f"blendshapes must have shape ({count}, 52), got {blendshapes.shape}")
        if rotation.shape != (count, 3) or translation.shape != (count, 3):
            raise ValueError("head pose arrays must have shape (frame_count, 3)")
        if confidence.shape != (count,):
            raise ValueError(f"confidence must have shape ({count},), got {confidence.shape}")
        if not np.isfinite(blendshapes).all() or np.any(blendshapes < 0) or np.any(blendshapes > 1):
            raise ValueError("blendshapes must be finite and in [0, 1]")
        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("head pose arrays contain non-finite values")
        if not np.isfinite(confidence).all() or np.any(confidence < 0) or np.any(confidence > 1):
            raise ValueError("confidence must be finite and in [0, 1]")
        transform = None if self.head_transform is None else np.asarray(self.head_transform, dtype=np.float32)
        if transform is not None:
            if transform.shape != (count, 4, 4):
                raise ValueError(f"head_transform must have shape ({count}, 4, 4), got {transform.shape}")
            if not np.isfinite(transform).all():
                raise ValueError("head_transform contains non-finite values")
        neutral_score = None if self.neutral_score is None else np.asarray(self.neutral_score, dtype=np.float32)
        if neutral_score is not None:
            if neutral_score.shape != (count,):
                raise ValueError(f"neutral_score must have shape ({count},), got {neutral_score.shape}")
            if not np.isfinite(neutral_score).all() or np.any(neutral_score < 0) or np.any(neutral_score > 1):
                raise ValueError("neutral_score must be finite and in [0, 1]")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "blendshapes", blendshapes)
        object.__setattr__(self, "head_rotation", rotation)
        object.__setattr__(self, "head_translation", translation)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "head_transform", transform)
        object.__setattr__(self, "neutral_score", neutral_score)

    @classmethod
    def from_frames(cls, frames: Sequence[AvatarFrame]) -> "AvatarSequence":
        if not frames:
            raise ValueError("at least one frame is required")
        transforms = [frame.head_transform for frame in frames]
        neutral_scores = [frame.neutral_score for frame in frames]
        return cls(
            timestamps=np.asarray([frame.timestamp for frame in frames], dtype=np.float64),
            blendshapes=np.stack([frame.blendshapes for frame in frames]),
            head_rotation=np.stack([frame.head_rotation for frame in frames]),
            head_translation=np.stack([frame.head_translation for frame in frames]),
            confidence=np.asarray([frame.confidence for frame in frames], dtype=np.float32),
            head_transform=np.stack(transforms) if all(value is not None for value in transforms) else None,
            neutral_score=(
                np.asarray(neutral_scores, dtype=np.float32)
                if all(value is not None for value in neutral_scores)
                else None
            ),
        )

    @property
    def frame_count(self) -> int:
        return int(self.timestamps.size)

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])

    def frame(self, index: int) -> AvatarFrame:
        transform = None if self.head_transform is None else self.head_transform[index]
        neutral = None if self.neutral_score is None else float(self.neutral_score[index])
        return AvatarFrame(
            timestamp=float(self.timestamps[index]),
            blendshapes=self.blendshapes[index],
            head_rotation=self.head_rotation[index],
            head_translation=self.head_translation[index],
            confidence=float(self.confidence[index]),
            head_transform=transform,
            neutral_score=neutral,
        )

    def derived_motion(self) -> dict[str, np.ndarray]:
        """Return finite-difference velocity and acceleration arrays."""
        if self.frame_count < 2:
            zeros = np.zeros_like(self.blendshapes)
            return {"blendshape_velocity": zeros, "blendshape_acceleration": zeros}
        velocity = np.gradient(self.blendshapes, self.timestamps, axis=0, edge_order=1)
        acceleration = np.gradient(velocity, self.timestamps, axis=0, edge_order=1)
        return {
            "blendshape_velocity": np.asarray(velocity, dtype=np.float32),
            "blendshape_acceleration": np.asarray(acceleration, dtype=np.float32),
        }

    def save(self, directory: str | Path, metadata: Mapping[str, object] | None = None, include_derived: bool = False) -> tuple[Path, Path]:
        """Write ``controls.npz`` and its versioned ``metadata.json`` sidecar."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "timestamps": self.timestamps,
            "blendshapes": self.blendshapes,
            "head_rotation": self.head_rotation,
            "head_translation": self.head_translation,
            "confidence": self.confidence,
        }
        if self.head_transform is not None:
            arrays["head_transform"] = self.head_transform
        if self.neutral_score is not None:
            arrays["neutral_score"] = self.neutral_score
        if include_derived:
            arrays.update(self.derived_motion())
        controls_path = directory / "controls.npz"
        np.savez_compressed(controls_path, **arrays)
        invariants: dict[str, object] = {
            "schema": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
            "blendshape_names": list(BLENDSHAPE_NAMES),
            "blendshape_range": [0.0, 1.0],
            "neutral_category_name": NEUTRAL_CATEGORY_NAME,
            "head_rotation_unit": "radians",
            "head_rotation_order": "xyz_euler",
            # Right-handed, centimeter-scale space with a virtual perspective
            # camera at the origin looking down -Z (Google's MediaPipe Face
            # Geometry module). See docs/AVATAR-CONTROLS.md for sources and
            # the caveat that absolute magnitude is FOV-estimated, not a
            # calibrated physical measurement.
            "head_translation_space": "mediapipe_facial_transformation_matrix_ccs_cm",
            "frame_count": self.frame_count,
            "duration_seconds": self.duration,
        }
        # confidence_semantics is extractor-specific (see AvatarFrame's
        # docstring) so it is a soft default, not a reasserted invariant --
        # a caller-supplied value in `metadata` is meant to override it.
        document: dict[str, object] = {"confidence_semantics": "unknown", **invariants}
        if metadata:
            document.update(dict(metadata))
        # Metadata is extensible, but cannot relabel or reinterpret the packet.
        # Reassert the invariants after merging caller-provided annotations.
        document.update(invariants)
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return controls_path, metadata_path

    @classmethod
    def load(cls, directory: str | Path) -> "AvatarSequence":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema") != SCHEMA_NAME or metadata.get("version") != SCHEMA_VERSION:
            raise ValueError("unsupported avatar control schema")
        if tuple(metadata.get("blendshape_names", ())) != BLENDSHAPE_NAMES:
            raise ValueError("blendshape column order does not match the canonical schema")
        with np.load(directory / "controls.npz", allow_pickle=False) as archive:
            required = {"timestamps", "blendshapes", "head_rotation", "head_translation", "confidence"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"controls.npz missing arrays: {sorted(missing)}")
            return cls(
                timestamps=archive["timestamps"],
                blendshapes=archive["blendshapes"],
                head_rotation=archive["head_rotation"],
                head_translation=archive["head_translation"],
                confidence=archive["confidence"],
                head_transform=archive["head_transform"] if "head_transform" in archive.files else None,
                # Optional and added after the initial release: older
                # controls.npz archives simply lack this array. Its absence
                # is not itself an error -- schema/version above already
                # gate incompatible changes.
                neutral_score=archive["neutral_score"] if "neutral_score" in archive.files else None,
            )
