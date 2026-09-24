"""YOLO26n object detection, loaded once and kept resident.

Deliberately eager (no CUDA-graph capture), unlike the 25fps avatar-driving
render-loop budget the 2026-09-09 vision-stack spike optimized YOLO for.
This service serves a slow, ~0.5 Hz "what is the person doing / what's in
frame" situational-awareness signal for the LiveKit voice agent (see
agent/vision_client.py) -- a fundamentally different budget. The spike
measured eager YOLO26n at ~48-58ms/frame; CUDA-graph capture got that to
~5.65ms, an 8.9x win that matters at 25fps and is irrelevant at 0.5Hz. Eager
mode is the right choice here: it's correct-by-construction (uses
ultralytics' own pre/post-processing unmodified) where a hand-rolled graph
capture would need to duplicate that logic outside the graph and carries
real correctness risk for a first version. If this service's cadence is
ever pushed toward real-time, revisit yolo_cudagraph.py from the spike
(scratch/vision-stack-test-20260909/) as the starting point.
"""

from __future__ import annotations

import os

import numpy as np
import torch

# Same sm_70 cuDNN conv-engine gap the spike hit on SmolVLM2's patch-embed
# conv -- YOLO26n's own convolutions haven't been confirmed to need this,
# but it's a cheap, already-validated workaround on this fleet's hardware
# and costs only some kernel-launch overhead (im2col+GEMM instead of one
# fused conv kernel), not correctness. See the spike's gotcha #3.
torch.backends.cudnn.enabled = False

from ultralytics import YOLO  # noqa: E402

MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "/models/yolo26n.pt")
CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONFIDENCE_THRESHOLD", "0.35"))
DEVICE = os.environ.get("YOLO_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")


class YoloRuntime:
    def __init__(self, model_path: str = MODEL_PATH, device: str = DEVICE) -> None:
        self._model = YOLO(model_path)
        self._model.to(device)
        self._device = device

    def detect(self, frame_bgr: np.ndarray) -> list[dict[str, object]]:
        """Run detection on one BGR frame (as decoded by cv2.imdecode).

        Returns a list of {"label": str, "confidence": float, "box": [x1,
        y1, x2, y2]} sorted by confidence, descending. Empty list if
        nothing cleared the confidence threshold.
        """
        results = self._model.predict(
            frame_bgr, device=self._device, conf=CONFIDENCE_THRESHOLD, verbose=False
        )
        if not results:
            return []
        result = results[0]
        detections = []
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({"label": label, "confidence": confidence, "box": xyxy})
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections
