"""MediaPipe-fitted real-time upper-body sidecar."""

from __future__ import annotations

import hashlib
import os
import threading

import cv2
import mediapipe as mp
import numpy as np
from fastapi import Body, FastAPI, HTTPException
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pydantic import BaseModel, Field

from motion import plan_motion


MODEL = os.environ.get("MEDIAPIPE_POSE_LANDMARKER_MODEL", "/models/pose_landmarker_lite.task")
EXPECTED_SHA256 = os.environ.get("MEDIAPIPE_POSE_LANDMARKER_SHA256", "")
MAX_FRAME_BYTES = int(os.environ.get("UPPER_BODY_MAX_FRAME_BYTES", "4194304"))
actual_sha256 = hashlib.sha256(open(MODEL, "rb").read()).hexdigest()
if EXPECTED_SHA256 and actual_sha256 != EXPECTED_SHA256:
    raise RuntimeError("MediaPipe pose landmarker checksum mismatch")

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL, delegate=python.BaseOptions.Delegate.CPU),
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.25,
    min_pose_presence_confidence=0.25,
    min_tracking_confidence=0.25,
    output_segmentation_masks=False,
)
landmarker = vision.PoseLandmarker.create_from_options(options)
landmarker_lock = threading.Lock()
app = FastAPI(title="MuseTalk upper-body controller")


class MotionRequest(BaseModel):
    seed: int
    duration: float = Field(ge=0.0, le=120.0)
    prosody: dict[str, float] = Field(default_factory=dict)
    controls: dict[str, float] = Field(default_factory=dict)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))


def _decode(payload: bytes) -> np.ndarray:
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="invalid upper-body frame size")
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="invalid upper-body image")
    return image


def _point(landmark) -> dict[str, float]:
    return {
        "x": round(_clamp(landmark.x, 0.0, 1.0), 6),
        "y": round(_clamp(landmark.y, 0.0, 1.0), 6),
        "visibility": round(_clamp(getattr(landmark, "visibility", 0.0), 0.0, 1.0), 5),
    }


def _detect(image: np.ndarray):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with landmarker_lock:
        return landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))


def _rig_from_result(result, width: int, height: int) -> dict[str, object]:
    if not result.pose_landmarks:
        # A headshot often crops the hips. The fallback is explicit and
        # conservative; it never pretends to be a detected 3D fit.
        left, right = {"x": 0.30, "y": 0.64, "visibility": 0.0}, {"x": 0.70, "y": 0.64, "visibility": 0.0}
        hips = ({"x": 0.38, "y": 0.94, "visibility": 0.0}, {"x": 0.62, "y": 0.94, "visibility": 0.0})
        mode, confidence = "inferred-headshot", 0.0
    else:
        points = result.pose_landmarks[0]
        left, right = _point(points[11]), _point(points[12])
        hips = (_point(points[23]), _point(points[24]))
        confidence = min(left["visibility"], right["visibility"])
        mode = "mediapipe-pose" if confidence >= 0.20 else "low-confidence-pose"
    center_x = (left["x"] + right["x"]) / 2
    shoulder_y = (left["y"] + right["y"]) / 2
    hip_x = (hips[0]["x"] + hips[1]["x"]) / 2
    hip_y = (hips[0]["y"] + hips[1]["y"]) / 2
    span = _clamp(abs(right["x"] - left["x"]), 0.18, 0.75)
    torso_height = _clamp(hip_y - shoulder_y, 0.18, 0.48)
    return {
        "enabled": True,
        "mode": mode,
        "confidence": round(confidence, 5),
        "image": {"width": width, "height": height},
        "leftShoulder": left,
        "rightShoulder": right,
        "neck": {"x": round(center_x, 6), "y": round(_clamp(shoulder_y - 0.15 * span, 0.28, 0.78), 6)},
        "sternum": {"x": round(center_x, 6), "y": round(_clamp(shoulder_y + 0.26 * torso_height, 0.35, 0.90), 6)},
        "hipCenter": {"x": round(hip_x, 6), "y": round(_clamp(hip_y, shoulder_y + 0.18, 0.98), 6)},
        "shoulderSpan": round(span, 6),
        "torsoHeight": round(torso_height, 6),
        "rest_profile": {
            "inhale_seconds": [1.35, 1.90],
            "exhale_seconds": [2.15, 3.05],
            "rest_seconds": [0.08, 0.32],
            "amplitude": [0.52, 0.72],
        },
        "renderer": "piecewise-affine-smplx-upper-body-v1",
    }


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {
        "ok": True,
        "runtime": mp.__version__,
        "model_sha256": actual_sha256,
        "renderer": "piecewise-affine-smplx-upper-body-v1",
        "archive_controller": "greenman/build_motion.py",
    }


@app.post("/prepare")
def prepare(payload: bytes = Body(media_type="image/png")) -> dict[str, object]:
    image = _decode(payload)
    return {"rig": _rig_from_result(_detect(image), image.shape[1], image.shape[0])}


@app.post("/observe")
def observe(payload: bytes = Body(media_type="image/jpeg")) -> dict[str, object]:
    image = _decode(payload)
    result = _detect(image)
    if not result.pose_landmarks:
        return {"metrics": None}
    points = result.pose_landmarks[0]
    left, right, left_hip, right_hip = points[11], points[12], points[23], points[24]
    # Wrists (BlazePose topology indices 15/16, same 33-point layout across
    # the lite/full/heavy model variants) -- added 2026-09-10 for the
    # LiveKit agent's wave-gesture trigger (see musetalk-volta/vision's
    # /gesture endpoint and assistant-ui-chat's WaveGestureDetector). This
    # landmarker was already computing these every call; they just weren't
    # returned, since the only consumer until now was the avatar's own
    # breathing/shoulder-sway rig, which has no use for hand position.
    left_wrist, right_wrist = points[15], points[16]
    shoulder_dx, shoulder_dy = right.x - left.x, right.y - left.y
    shoulder_span = max(0.001, float(np.hypot(shoulder_dx, shoulder_dy)))
    shoulder_y = (left.y + right.y) / 2
    hip_y = (left_hip.y + right_hip.y) / 2
    torso_height = max(0.001, hip_y - shoulder_y)
    return {"metrics": {
        "leftShoulderY": float(left.y),
        "rightShoulderY": float(right.y),
        "shoulderCenterY": float(shoulder_y),
        "shoulderSpan": shoulder_span,
        "shoulderRoll": float(np.arctan2(shoulder_dy, shoulder_dx)),
        "torsoHeight": float(torso_height),
        "torsoCenterX": float((left.x + right.x + left_hip.x + right_hip.x) / 4),
        # Shoulder expansion remains measurable in a head-and-shoulders crop
        # even when hips are absent. Keep torso confidence separate so the
        # respiratory feedback loop does not throw away its valid signal.
        "shoulderVisibility": float(min(left.visibility, right.visibility)),
        "torsoVisibility": float(min(left.visibility, right.visibility, left_hip.visibility, right_hip.visibility)),
        "visibility": float(min(left.visibility, right.visibility, left_hip.visibility, right_hip.visibility)),
        "leftWristX": float(left_wrist.x),
        "leftWristY": float(left_wrist.y),
        "leftWristVisibility": float(left_wrist.visibility),
        "rightWristX": float(right_wrist.x),
        "rightWristY": float(right_wrist.y),
        "rightWristVisibility": float(right_wrist.visibility),
    }}


@app.post("/plan")
def plan(request: MotionRequest) -> dict[str, object]:
    return plan_motion(request.seed, request.duration, request.prosody, request.controls)
