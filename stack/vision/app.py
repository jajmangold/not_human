"""Aggregated live-perception sidecar for the LiveKit voice agent.

Single integration point for agent/vision_client.py: takes one JPEG frame
from the human participant's camera and fans out to every perception layer
identified in the 2026-09-09 nothuman vision-stack spike, in parallel,
returning one combined result. Reuses the ALREADY-RUNNING `feedback` and
`upper-body` MediaPipe services for face/pose rather than loading a second
copy of those models.

Three different vision/language models split by what each actually
measured well at (2026-09-10, replacing an earlier MiniCPM-V-4.6-for-
everything design -- see docs/VISION.md for the full comparison):

- **`lfm2vl`** (LiquidAI/LFM2.5-VL-450M, a separate sidecar -- see
  lfm2vl_client.py): ambient caption (this endpoint) and on-demand
  targeted visual questions (/query). ~150-185ms/request, correct on
  every caption/count/description test tried.
- **`lfm2vl-narrate`** (LiquidAI/LFM2.5-VL-3B, same image/binary as
  `lfm2vl`, bigger weights -- see /narrate below): background
  scene-narrative consolidation on a slow (~20s) cadence, where richer
  accuracy matters more than shaving another ~100ms off a call that was
  never on the conversational hot path anyway. Measured the same latency
  class as the 450M model for short answers (~162-183ms) but meaningfully
  more accurate/detailed description.
- **MiniCPM-V-4.6** (`transformers`, in-process -- see
  minicpmv_runtime.py): classify_transcript only. LFM2.5-VL tested
  unreliable at this specific task (answered ARTIFACT for every test
  transcript, including obviously real text) -- not a like-for-like
  replacement across the board, so MiniCPM-V stays for this one call.

YOLO26n is owned in-process by this service (see yolo_runtime.py).

Important distinction from `feedback`/`upper-body`'s existing job: those
services analyze the AVATAR's own driving frames for MuseTalk rendering.
This service analyzes the HUMAN PARTICIPANT's webcam frames, a completely
separate consumer that didn't exist before -- see the spike doc and
agent/vision_client.py's docstring.

Cadence contract: the caller (agent.py) is expected to call /perceive at
~0.5Hz (one frame every ~2s), not per-frame -- see vision_client.py. This
service does not itself rate-limit /perceive; that's the caller's job,
same division of responsibility as every other sidecar in this compose
file. /classify_transcript has no cadence contract -- agent.py calls it
once per completed user turn, gated on turn boundaries, not frame time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time

import aiohttp
import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Response
from pydantic import BaseModel

import lfm2vl_client
from minicpmv_runtime import MiniCPMVRuntime
from yolo_runtime import YoloRuntime

logger = logging.getLogger("vision")

MAX_FRAME_BYTES = int(os.environ.get("VISION_MAX_FRAME_BYTES", "4194304"))
FEEDBACK_URL = os.environ.get("FEEDBACK_URL", "http://feedback:8095")
UPPER_BODY_URL = os.environ.get("UPPER_BODY_URL", "http://upper-body:8096")
# Short timeout: this endpoint is on the conversational side channel, not
# the avatar render loop, but it must still fail fast per-layer rather than
# let one slow/down sidecar block the whole aggregated response.
LANDMARK_TIMEOUT_S = float(os.environ.get("VISION_LANDMARK_TIMEOUT_S", "2.0"))

app = FastAPI(title="MuseTalk live-perception sidecar")
_yolo: YoloRuntime | None = None
_minicpmv: MiniCPMVRuntime | None = None
# Debug-only, in-memory, last-write-wins -- lets a human (or Claude) look
# at exactly what frame the model most recently received, since "is the
# captured frame actually updating" is otherwise unverifiable from the
# outside. Added 2026-09-10 debugging a live report that finger-counting
# "gets stuck" on the first answer.
_last_frame_jpeg: bytes | None = None
_last_query: dict[str, object] | None = None


@app.on_event("startup")
def _load_models() -> None:
    global _yolo, _minicpmv
    _yolo = YoloRuntime()
    _minicpmv = MiniCPMVRuntime()


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {"ok": _yolo is not None and _minicpmv is not None}


async def _observe_layer(http: aiohttp.ClientSession, url: str, jpeg: bytes) -> dict[str, object] | None:
    """POST a JPEG frame to a MediaPipe sidecar's /observe and return its
    "metrics" (or None on any failure/timeout/absence -- a missing signal
    is not an error for this aggregator, callers already treat every field
    here as optional)."""
    try:
        async with http.post(
            f"{url}/observe",
            data=jpeg,
            headers={"Content-Type": "image/jpeg"},
            timeout=aiohttp.ClientTimeout(total=LANDMARK_TIMEOUT_S),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
            return payload.get("metrics")
    except Exception:
        logger.warning("perception layer at %s failed or timed out", url, exc_info=True)
        return None


def _decode(jpeg: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="invalid JPEG frame")
    return image


@app.post("/perceive")
async def perceive(payload: bytes = Body(media_type="image/jpeg")) -> dict[str, object]:
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="invalid frame size")
    global _last_frame_jpeg
    _last_frame_jpeg = payload
    frame_bgr = _decode(payload)

    assert _yolo is not None, "startup event has not run"
    loop = asyncio.get_running_loop()
    # ultralytics' predict() is a blocking call -- run it off the event
    # loop so it doesn't stall the concurrent calls below.
    yolo_task = loop.run_in_executor(None, _yolo.detect, frame_bgr)

    async with aiohttp.ClientSession() as http:
        face_task = _observe_layer(http, FEEDBACK_URL, payload)
        pose_task = _observe_layer(http, UPPER_BODY_URL, payload)
        caption_task = lfm2vl_client.caption(http, payload)
        objects, face, pose, caption = await asyncio.gather(yolo_task, face_task, pose_task, caption_task)

    return {
        "objects": objects,
        "face": face,
        "pose": pose,
        "caption": caption,
    }


class ClassifyTranscriptRequest(BaseModel):
    text: str


@app.post("/classify_transcript")
async def classify_transcript(request: ClassifyTranscriptRequest) -> dict[str, object]:
    assert _minicpmv is not None, "startup event has not run"
    is_real_speech = await _minicpmv.classify_transcript(request.text)
    return {"is_real_speech": is_real_speech}


@app.post("/query")
async def query(question: str, payload: bytes = Body(media_type="image/jpeg")) -> dict[str, object]:
    """On-demand targeted visual question about the CURRENT frame --
    distinct from /perceive's fixed ambient caption. See
    lfm2vl_client.py's answer_visual_question() docstring: this exists
    because a generic rolling caption can't answer something specific
    ("how many fingers"), found live on 2026-09-10's first real test.
    """
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="invalid frame size")
    global _last_frame_jpeg, _last_query
    _last_frame_jpeg = payload
    _decode(payload)  # validates the JPEG is decodable before spending a round-trip on it
    async with aiohttp.ClientSession() as http:
        answer = await lfm2vl_client.answer_visual_question(http, payload, question)
    _last_query = {"question": question, "answer": answer, "at": time.time(), "frame_bytes": len(payload)}
    return {"answer": answer}


@app.post("/gesture")
async def gesture(payload: bytes = Body(media_type="image/jpeg")) -> dict[str, object]:
    """Cheap, fast pose-only probe -- proxies straight to `upper-body`
    (CPU-only MediaPipe, no GPU, no LLM) and returns nothing else, unlike
    /perceive which also runs YOLO + lfm2vl on every call. Added
    2026-09-10 for agent.py's WaveGestureDetector: real wave detection
    needs several pose samples across ~1-1.5s to see the hand actually
    swing side-to-side, not just be raised once -- calling /perceive that
    often would mean a full YOLO+caption pass every ~250ms, real GPU cost
    for zero benefit to gesture detection. This endpoint lets agent.py
    poll pose at its faster buffer-frame cadence independent of the
    slower, GPU-backed ambient-caption cadence -- see
    BUFFER_FRAME_INTERVAL_S in agent/vision_client.py.
    """
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="invalid frame size")
    _decode(payload)  # validates the JPEG is decodable before spending a round-trip on it
    async with aiohttp.ClientSession() as http:
        pose = await _observe_layer(http, UPPER_BODY_URL, payload)
    return {"pose": pose}


@app.post("/narrate")
async def narrate(
    raw_log: str = "", object_timeline: str = "", payload: bytes = Body(media_type="image/jpeg")
) -> dict[str, object]:
    """Background scene-narrative consolidation, proxied to
    `lfm2vl-narrate` (LFM2.5-VL-3B, not the 450M `lfm2vl` every other
    endpoint here uses) -- see lfm2vl_client.py's narrate() docstring.
    Not on any conversational hot path: agent.py calls this from a slow
    (~20s) background task, never gating a turn.
    """
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="invalid frame size")
    _decode(payload)  # validates the JPEG is decodable before spending a round-trip on it
    async with aiohttp.ClientSession() as http:
        narrative = await lfm2vl_client.narrate(http, payload, raw_log, object_timeline)
    return {"narrative": narrative}


@app.get("/debug/last_frame")
def debug_last_frame() -> Response:
    """Returns the most recent JPEG frame this service received (from
    either /perceive or /query), whichever was most recent -- for
    verifying by eye whether captured frames are actually updating.
    Debug-only: no auth, not part of the caller contract."""
    if _last_frame_jpeg is None:
        raise HTTPException(status_code=404, detail="no frame received yet")
    return Response(content=_last_frame_jpeg, media_type="image/jpeg")


@app.get("/debug/last_query")
def debug_last_query() -> dict[str, object]:
    """The most recent /query question+answer+timestamp, for confirming a
    targeted visual question actually re-ran against a fresh frame."""
    if _last_query is None:
        raise HTTPException(status_code=404, detail="no query received yet")
    return _last_query
