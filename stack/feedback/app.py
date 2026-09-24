import hashlib
import os
import threading

import cv2
import mediapipe as mp
import numpy as np
from fastapi import Body, FastAPI, HTTPException
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL", "/models/face_landmarker.task")
EXPECTED_SHA256 = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_SHA256", "")
MAX_FRAME_BYTES = int(os.environ.get("FEEDBACK_MAX_FRAME_BYTES", "2097152"))
actual_sha256 = hashlib.sha256(open(MODEL, "rb").read()).hexdigest()
if EXPECTED_SHA256 and actual_sha256 != EXPECTED_SHA256:
    raise RuntimeError("MediaPipe face landmarker checksum mismatch")

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL, delegate=python.BaseOptions.Delegate.CPU),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.20,
    min_face_presence_confidence=0.20,
)
landmarker = vision.FaceLandmarker.create_from_options(options)
landmarker_lock = threading.Lock()
app = FastAPI()


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def distance(first, second) -> float:
    """Distance between compact ``(x, y)`` landmark pairs."""
    return float(np.hypot(second[0] - first[0], second[1] - first[1]))


def mean_point(points, indices: tuple[int, ...]) -> tuple[float, float]:
    return (
        sum(points[index][0] for index in indices) / len(indices),
        sum(points[index][1] for index in indices) / len(indices),
    )


def normalized_iris(
    points,
    iris: tuple[int, ...],
    corners: tuple[int, int],
    eye_distance: float,
) -> tuple[float, float]:
    """Return iris position relative to the de-rolled eye-corner frame.

    The previous implementation normalized Y by the upper/lower lid gap. That
    makes a blink look like a vertical gaze movement and makes pupil_y appear
    coupled even when the iris is stationary. The eye-corner line is stable
    under aperture changes, so it is the appropriate reference for gazeY.
    """
    iris_x, iris_y = mean_point(points, iris)
    first, second = points[corners[0]], points[corners[1]]
    left, right = sorted((first[0], second[0]))
    span = max(0.001, right - left)
    x = (iris_x - left) / span - 0.5
    # Points are de-rolled before this helper is called, so the line between
    # the corners is horizontal. Keep the signed offset in image coordinates;
    # four eye-widths maps normal iris travel into the existing [-1, 1] range.
    corner_dx = second[0] - first[0]
    t = clamp((iris_x - first[0]) / (corner_dx if abs(corner_dx) >= 0.001 else 0.001), 0.0, 1.0)
    corner_line_y = first[1] + t * (second[1] - first[1])
    y = (iris_y - corner_line_y) / max(0.001, eye_distance)
    return clamp(x * 2.0, -1.0, 1.0), clamp(y * 4.0, -1.0, 1.0)


def derol_landmarks(points, center: tuple[float, float], roll: float) -> list[tuple[float, float]]:
    """Rotate landmarks around the eye midpoint so facial metrics ignore roll."""
    coords = np.asarray([(point.x, point.y) for point in points], dtype=np.float32)
    origin = np.asarray(center, dtype=np.float32)
    offset = coords - origin
    cosine, sine = float(np.cos(-roll)), float(np.sin(-roll))
    rotated = np.empty_like(offset)
    rotated[:, 0] = cosine * offset[:, 0] - sine * offset[:, 1]
    rotated[:, 1] = sine * offset[:, 0] + cosine * offset[:, 1]
    rotated += origin
    return [(float(point[0]), float(point[1])) for point in rotated]


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {"ok": True, "model_sha256": actual_sha256, "runtime": mp.__version__}


@app.post("/observe")
def observe(payload: bytes = Body(media_type="image/jpeg")) -> dict[str, object]:
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="invalid feedback frame size")
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="invalid feedback JPEG")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with landmarker_lock:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return {"metrics": None}
    raw_points = result.face_landmarks[0]
    left_eye, right_eye = raw_points[33], raw_points[263]
    dx, dy = right_eye.x - left_eye.x, right_eye.y - left_eye.y
    eye_distance = max(0.001, float(np.hypot(dx, dy)))
    eye_mid_x, eye_mid_y = (left_eye.x + right_eye.x) / 2, (left_eye.y + right_eye.y) / 2
    roll = float(np.arctan2(dy, dx))
    # All shape/pose metrics below use a de-rolled copy. Preserve the raw roll
    # itself as the independent roll channel and keep center coordinates in the
    # original image frame for tracking/telemetry.
    points = derol_landmarks(raw_points, (eye_mid_x, eye_mid_y), roll)
    left_eye, right_eye, nose = points[33], points[263], points[1]
    left_upper, left_lower, right_upper, right_lower = points[159], points[145], points[386], points[374]
    left_brow, right_brow = points[105], points[334]
    upper_lip, lower_lip, left_mouth, right_mouth = points[13], points[14], points[61], points[291]
    left_gaze = normalized_iris(points, (468, 469, 470, 471, 472), (33, 133), eye_distance)
    right_gaze = normalized_iris(points, (473, 474, 475, 476, 477), (362, 263), eye_distance)
    left_eye_open = clamp(distance(left_upper, left_lower) / eye_distance, 0, 0.5)
    right_eye_open = clamp(distance(right_upper, right_lower) / eye_distance, 0, 0.5)
    left_brow_value = clamp((left_upper[1] - left_brow[1]) / eye_distance, -0.2, 0.8)
    right_brow_value = clamp((right_upper[1] - right_brow[1]) / eye_distance, -0.2, 0.8)
    return {"metrics": {
        "centerX": eye_mid_x, "centerY": eye_mid_y, "eyeDistance": eye_distance,
        "roll": roll,
        "yaw": clamp((nose[0] - eye_mid_x) / eye_distance, -0.8, 0.8),
        "pitch": clamp((nose[1] - eye_mid_y) / eye_distance, 0.1, 1.8),
        "mouth": clamp(distance(upper_lip, lower_lip) / eye_distance, 0, 0.8),
        "leftEyeOpen": left_eye_open,
        "rightEyeOpen": right_eye_open,
        "eyeAsymmetry": clamp(left_eye_open - right_eye_open, -0.25, 0.25),
        "gazeX": clamp((left_gaze[0] + right_gaze[0]) / 2, -1.0, 1.0),
        "gazeY": clamp((left_gaze[1] + right_gaze[1]) / 2, -1.0, 1.0),
        "gazeVergence": clamp(left_gaze[0] - right_gaze[0], -1.0, 1.0),
        "leftBrow": left_brow_value,
        "rightBrow": right_brow_value,
        "browAsymmetry": clamp(left_brow_value - right_brow_value, -0.5, 0.5),
        "smile": clamp(((upper_lip[1] + lower_lip[1]) / 2 - (left_mouth[1] + right_mouth[1]) / 2) / eye_distance, -0.4, 0.4),
        "smirk": clamp((right_mouth[1] - left_mouth[1]) / eye_distance, -0.4, 0.4),
    }}
