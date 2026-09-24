"""MediaPipe Face Landmarker -> canonical AvatarSequence extraction."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

from avatar.controls.schema import (
    BLENDSHAPE_INDEX,
    NEUTRAL_CATEGORY_NAME,
    AvatarFrame,
    AvatarSequence,
)


class FaceNotFoundError(RuntimeError):
    """Raised when an image has no detectable face."""


def _euler_xyz(matrix: np.ndarray) -> np.ndarray:
    """Decompose a MediaPipe 4x4 rotation matrix into XYZ radians."""
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    # Clamp the asin input to absorb tiny numerical drift from TFLite output.
    yaw = math.asin(float(np.clip(rotation[0, 2], -1.0, 1.0)))
    pitch = math.atan2(float(-rotation[1, 2]), float(rotation[2, 2]))
    roll = math.atan2(float(-rotation[0, 1]), float(rotation[0, 0]))
    return np.asarray((pitch, yaw, roll), dtype=np.float32)


class FaceLandmarkerExtractor:
    """Reusable IMAGE-mode extractor for stills and timestamped video frames.

    Verified empirically against the pinned ``face_landmarker.task`` asset
    (mediapipe 0.10.35): the model returns exactly 52 blendshape categories,
    but they are not schema.BLENDSHAPE_NAMES verbatim -- category 0 is
    "_neutral" (captured separately as ``AvatarFrame.neutral_score``, not an
    ARKit shape), and "tongueOut" is never emitted (there is no tongue signal
    in a monocular RGB face crop). See ``UNSUPPORTED_BLENDSHAPES``.

    ``AvatarFrame.confidence`` from this extractor is binary detection
    validity, not a graded score: ``result.face_landmarks[i].presence`` and
    ``.visibility`` are always ``None`` for the Face Landmarker task (checked
    empirically), so there is no continuous per-detection confidence to
    report -- only whether a face was found at all.
    """

    # See the class docstring: measured empirically, not assumed from the
    # ARKit-52 spec. Keep in sync with any future pinned model swap.
    UNSUPPORTED_BLENDSHAPES = frozenset({"tongueOut"})

    def __init__(self, model_path: str | Path, min_confidence: float = 0.2) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path), delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarkerExtractor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _image(frame: np.ndarray) -> mp.Image:
        if frame is None or frame.size == 0:
            raise ValueError("empty image frame")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("expected an HxWx3 BGR frame")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    @staticmethod
    def _blendshapes(result: Any) -> tuple[np.ndarray, float | None]:
        """Split MediaPipe's raw categories into (ARKit-52 array, neutral).

        Any category name that is neither a canonical ARKit shape nor
        ``NEUTRAL_CATEGORY_NAME`` is unexpected for the pinned model and is
        surfaced via an assertion rather than silently dropped, so a future
        model swap that changes the category set is caught immediately
        instead of quietly corrupting the canonical array.
        """
        values = np.zeros(len(BLENDSHAPE_INDEX), dtype=np.float32)
        neutral: float | None = None
        categories = result.face_blendshapes[0] if result.face_blendshapes else ()
        for category in categories:
            name = getattr(category, "category_name", None) or getattr(category, "display_name", None)
            score = float(np.clip(category.score, 0.0, 1.0))
            if name == NEUTRAL_CATEGORY_NAME:
                neutral = score
            elif name in BLENDSHAPE_INDEX:
                values[BLENDSHAPE_INDEX[name]] = score
            else:
                raise ValueError(
                    f"unexpected MediaPipe blendshape category {name!r}; the pinned "
                    "model's category set no longer matches BLENDSHAPE_NAMES/"
                    "NEUTRAL_CATEGORY_NAME and must be re-verified"
                )
        return values, neutral

    def detect(self, frame: np.ndarray, timestamp: float, strict: bool = True) -> AvatarFrame:
        result = self._landmarker.detect(self._image(frame))
        if not result.face_landmarks:
            if strict:
                raise FaceNotFoundError("no face detected")
            return AvatarFrame(
                timestamp=timestamp,
                blendshapes=np.zeros(len(BLENDSHAPE_INDEX), dtype=np.float32),
                head_rotation=np.zeros(3, dtype=np.float32),
                head_translation=np.zeros(3, dtype=np.float32),
                confidence=0.0,
            )
        matrices = getattr(result, "facial_transformation_matrixes", None)
        if matrices is None:
            matrices = ()
        transform = np.asarray(matrices[0], dtype=np.float32) if matrices else np.eye(4, dtype=np.float32)
        if transform.shape != (4, 4):
            transform = transform.reshape(4, 4)
        blendshapes, neutral = self._blendshapes(result)
        return AvatarFrame(
            timestamp=timestamp,
            blendshapes=blendshapes,
            head_rotation=_euler_xyz(transform),
            head_translation=transform[:3, 3],
            confidence=1.0,
            head_transform=transform,
            neutral_score=neutral,
        )

    def extract_image(self, path: str | Path, timestamp: float = 0.0) -> AvatarSequence:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(path)
        return AvatarSequence.from_frames([self.detect(frame, timestamp, strict=True)])

    def extract_video(self, path: str | Path, fps: float | None = None, strict: bool = False) -> AvatarSequence:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(path)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        rate = float(fps or source_fps or 25.0)
        if rate <= 0 or not math.isfinite(rate):
            raise ValueError("fps must be positive")
        frames: list[AvatarFrame] = []
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(self.detect(frame, index / rate, strict=strict))
                index += 1
        finally:
            capture.release()
        if not frames:
            raise ValueError("video contains no decodable frames")
        return AvatarSequence.from_frames(frames)

    def extract(self, path: str | Path, fps: float | None = None, strict: bool = False) -> AvatarSequence:
        path = Path(path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return AvatarSequence.from_frames([self.detect(image, 0.0, strict=True)])
        return self.extract_video(path, fps=fps, strict=strict)
