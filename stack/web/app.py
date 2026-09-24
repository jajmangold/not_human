"""Qwen -> Kokoro -> MuseTalk low-latency talking portrait service."""
from __future__ import annotations

import asyncio
import array
from collections import deque
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import struct
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from natural_motion import NaturalMotionScheduler
from optional_services import NEUTRAL_REACTION, optional_json
from speech_contract import QWEN38_SYSTEM_PROMPT, qwen38_request, release_speech_phrases
from image_normalization import PortraitDimensionsError, normalize_portrait

IO_ROOT = Path(os.environ.get("MUSE_TALK_IO_DIR", "/io"))
WEB_ROOT = IO_ROOT / "web_demo"
UPLOAD_ROOT, AUDIO_ROOT = WEB_ROOT / "uploads", WEB_ROOT / "audio"
STREAM_ROOT, SESSION_ROOT = WEB_ROOT / "streams", WEB_ROOT / "sessions"
MOTION_ROOT = WEB_ROOT / "motion"
JOBS_ROOT, MUSE_READY = IO_ROOT / "muse_jobs", IO_ROOT / "MUSE_SERVER_READY"
KOKORO_URL = os.environ.get("KOKORO_URL", "http://kokoro:8091").rstrip("/")
KOKOCLONE_URL = os.environ.get("KOKOCLONE_URL", "http://kokoclone:8098").rstrip("/")
AFFECT_URL = os.environ.get("AFFECT_URL", "http://affect:8094").rstrip("/")
FEEDBACK_URL = os.environ.get("FEEDBACK_URL", "http://feedback:8095").rstrip("/")
UPPER_BODY_URL = os.environ.get("UPPER_BODY_URL", "http://upper-body:8096").rstrip("/")
UPPER_BODY_ENABLED = os.environ.get("UPPER_BODY_ENABLED", "1") == "1"
ALP_URL = os.environ.get("ADVANCED_LIVE_PORTRAIT_URL", "http://advanced-live-portrait:8093").rstrip("/")
ALP_ENABLED = os.environ.get("ADVANCED_LIVE_PORTRAIT_ENABLED", "1") == "1"
ALP_FRAMES = int(os.environ.get("ADVANCED_LIVE_PORTRAIT_FRAMES", "28"))
ALP_READY_FRAMES = 8
ALP_REVISION = os.environ.get(
    "ADVANCED_LIVE_PORTRAIT_REVISION",
    "3bba732915e22f18af0d221b9c5c282990181f1b",
)
ALP_PROFILE = os.environ.get("ADVANCED_LIVE_PORTRAIT_PROFILE", "semantic-bank-v13-blink-brow-fix")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
LLM_MODEL, LLM_API_KEY = os.environ.get("LLM_MODEL", "qwen27b"), os.environ.get("LLM_API_KEY", "")
MAX_UPLOAD_BYTES = int(os.environ.get("MUSE_DEMO_MAX_UPLOAD_BYTES", "10485760"))
MAX_PROMPT = int(os.environ.get("MUSE_DEMO_MAX_PROMPT", "8000"))
MAX_REPLY_TOKENS = int(os.environ.get("MUSE_DEMO_MAX_REPLY_TOKENS", "512"))
JOB_TIMEOUT = float(os.environ.get("MUSE_DEMO_JOB_TIMEOUT", "300"))
SESSION_TTL = max(30.0, float(os.environ.get("MUSE_DEMO_SESSION_TTL", "600")))
START_RATE_LIMIT = max(1, int(os.environ.get("MUSE_DEMO_START_RATE_LIMIT", "6")))
FEEDBACK_RATE_LIMIT = max(1, int(os.environ.get("MUSE_DEMO_FEEDBACK_RATE_LIMIT", "900")))
MOTION_EDIT_RATE_LIMIT = max(1, int(os.environ.get("MUSE_DEMO_MOTION_EDIT_RATE_LIMIT", "240")))
PLAYGROUND_RATE_LIMIT = max(1, int(os.environ.get("MUSE_PLAYGROUND_RATE_LIMIT", "60")))
MAX_RATE_KEYS = max(128, int(os.environ.get("MUSE_DEMO_MAX_RATE_KEYS", "4096")))
AUTH_TOKEN = os.environ.get("MUSE_DEMO_AUTH_TOKEN", "").strip()
STREAM_BATCH_SIZE = int(os.environ.get("MUSE_DEMO_STREAM_BATCH_SIZE", "4"))
STREAM_FPS = int(os.environ.get("MUSE_DEMO_STREAM_FPS", "16"))
IDLE_CYCLE_FRAMES = int(os.environ.get("MUSE_DEMO_IDLE_CYCLE_FRAMES", "64"))
MEDIAPIPE_MODEL = Path(
    os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL", "/models/mediapipe/face_landmarker.task")
)
MEDIAPIPE_MODEL_SHA256 = os.environ.get(
    "MEDIAPIPE_FACE_LANDMARKER_SHA256",
    "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
)
MEDIAPIPE_MODEL_VALID = (
    MEDIAPIPE_MODEL.is_file()
    and hashlib.sha256(MEDIAPIPE_MODEL.read_bytes()).hexdigest() == MEDIAPIPE_MODEL_SHA256
)
STATIC_ROOT = Path(__file__).parent / "static"
SYSTEM_PROMPT = os.environ.get("MUSE_DEMO_SYSTEM_PROMPT", QWEN38_SYSTEM_PROMPT)
PACKET_HEADER, AUDIO_PACKET, FRAME_PACKET = struct.Struct("!BII"), 1, 2
READY_FRAME_PACKET, IDLE_FRAME_PACKET = 3, 4
READY_GESTURES = ("wink", "nod", "smile", "surprise-wink", "kiss")
READY_GESTURE_CACHE: dict[str, str] = {}
ACTIVE_SESSIONS: set[str] = set()
SESSION_EXPIRY_TASKS: dict[str, asyncio.Task[None]] = {}
RATE_EVENTS: dict[tuple[str, str], deque[float]] = {}
RATE_LOCK = asyncio.Lock()

# Keep the public calibration vocabulary close to AdvancedLivePortrait-WebUI,
# but expose it as data instead of copying its Gradio application. The live
# renderer still owns its semantic bank and uses these values as a reportable
# control vocabulary for tuning and feedback.
ALP_EXPRESSION_CONTROLS = {
    "rotate_pitch": {"label": "Rotate Pitch", "min": -20.0, "max": 20.0, "step": 0.5, "default": 0.0},
    "rotate_yaw": {"label": "Rotate Yaw", "min": -20.0, "max": 20.0, "step": 0.5, "default": 0.0},
    "rotate_roll": {"label": "Rotate Roll", "min": -20.0, "max": 20.0, "step": 0.5, "default": 0.0},
    "blink": {"label": "Blink", "min": -20.0, "max": 20.0, "step": 0.5, "default": 0.0},
    "eyebrow": {"label": "Eyebrow", "min": -40.0, "max": 20.0, "step": 0.5, "default": 0.0},
    "wink": {"label": "Wink", "min": 0.0, "max": 25.0, "step": 0.5, "default": 0.0},
    "pupil_x": {"label": "Pupil X", "min": -20.0, "max": 20.0, "step": 0.5, "default": 0.0},
    "pupil_y": {"label": "Pupil Y", "min": -20.0, "max": 20.0, "step": 0.5, "default": 0.0},
    "aaa": {"label": "AAA", "min": -30.0, "max": 120.0, "step": 1.0, "default": 0.0},
    "eee": {"label": "EEE", "min": -20.0, "max": 20.0, "step": 0.2, "default": 0.0},
    "woo": {"label": "WOO", "min": -20.0, "max": 20.0, "step": 0.2, "default": 0.0},
    "smile": {"label": "Smile", "min": -2.0, "max": 2.0, "step": 0.01, "default": 0.0},
}

for directory in (UPLOAD_ROOT, AUDIO_ROOT, STREAM_ROOT, SESSION_ROOT, MOTION_ROOT, JOBS_ROOT):
    directory.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Volta Live Portrait")
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
feedback_client = httpx.AsyncClient(timeout=3, limits=httpx.Limits(max_connections=8, max_keepalive_connections=4))


@app.on_event("shutdown")
async def close_feedback_client() -> None:
    await feedback_client.aclose()


@dataclass(frozen=True)
class PreparedPhrase:
    seq: int
    text: str
    audio_path: Path
    duration: float
    reaction: dict[str, object]
    prosody: dict[str, float]


class ClientDisconnected(Exception):
    """Internal control flow for a WebSocket that closed mid-turn."""


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)
    os.chmod(path, 0o644)


def _authorized(headers: object) -> bool:
    if not AUTH_TOKEN:
        return True
    try:
        return str(headers.get("authorization", "")) == f"Bearer {AUTH_TOKEN}"
    except AttributeError:
        return False


async def _rate_limit(scope: str, client: str, limit: int) -> None:
    now = asyncio.get_running_loop().time()
    async with RATE_LOCK:
        key = (scope, client)
        events = RATE_EVENTS.setdefault(key, deque())
        while events and events[0] <= now - 60.0:
            events.popleft()
        # Keep the in-memory limiter bounded even if many one-off addresses
        # reach the public listener. The active caller's window is retained.
        if len(RATE_EVENTS) > MAX_RATE_KEYS:
            overflow = len(RATE_EVENTS) - MAX_RATE_KEYS
            for stale_key in list(RATE_EVENTS):
                if stale_key == key:
                    continue
                RATE_EVENTS.pop(stale_key, None)
                overflow -= 1
                if overflow <= 0:
                    break
        if len(events) >= limit:
            raise HTTPException(status_code=429, detail="rate limit exceeded", headers={"Retry-After": "60"})
        events.append(now)


async def _guard_request(request: Request, scope: str, limit: int) -> None:
    if not _authorized(request.headers):
        raise HTTPException(status_code=401, detail="bearer authentication required")
    await _rate_limit(scope, request.client.host if request.client else "unknown", limit)


def _cleanup_session_artifacts(session_id: str) -> None:
    """Remove only per-session files; portrait and motion caches are shared."""
    (SESSION_ROOT / f"{session_id}.json").unlink(missing_ok=True)
    stream_session = STREAM_ROOT / session_id
    if stream_session.parent == STREAM_ROOT:
        shutil.rmtree(stream_session, ignore_errors=True)
    for artifact in (*AUDIO_ROOT.glob(f"*{session_id}*.wav"), *JOBS_ROOT.glob(f"*{session_id}*")):
        if artifact.is_file():
            artifact.unlink(missing_ok=True)


async def _expire_session(session_id: str) -> None:
    try:
        await asyncio.sleep(SESSION_TTL)
        if session_id not in ACTIVE_SESSIONS:
            await asyncio.to_thread(_cleanup_session_artifacts, session_id)
    except asyncio.CancelledError:
        return
    finally:
        task = SESSION_EXPIRY_TASKS.get(session_id)
        if task is asyncio.current_task():
            SESSION_EXPIRY_TASKS.pop(session_id, None)


async def _wait_for_cancelled_jobs(job_ids: set[str]) -> None:
    """Keep cancellation markers visible until the resident observes them."""
    if not job_ids:
        return
    deadline = asyncio.get_running_loop().time() + min(JOB_TIMEOUT, 10.0)
    while asyncio.get_running_loop().time() < deadline:
        pending = [
            job_id for job_id in job_ids
            if not (JOBS_ROOT / f"{job_id}.done").is_file()
            and not (JOBS_ROOT / f"{job_id}.err").is_file()
        ]
        if not pending:
            return
        await asyncio.sleep(0.1)


def _schedule_session_expiry(session_id: str) -> None:
    previous = SESSION_EXPIRY_TASKS.pop(session_id, None)
    if previous:
        previous.cancel()
    SESSION_EXPIRY_TASKS[session_id] = asyncio.create_task(_expire_session(session_id))


async def _read_portrait(portrait: UploadFile) -> bytes:
    if portrait.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise HTTPException(status_code=415, detail="portrait must be PNG, JPEG, or WebP")
    payload = await portrait.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="portrait is too large")
    try:
        return normalize_portrait(payload)
    except PortraitDimensionsError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="portrait is not a valid image") from exc


def _phrases(buffer: str, final: bool) -> tuple[list[str], str]:
    return release_speech_phrases(buffer, final)


def _wav_duration(payload: bytes) -> float:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        frames, rate = wav.getnframes(), wav.getframerate()
        if frames < rate * 3600:
            return frames / rate
        # Kokoro-FastAPI streams WAV headers with an unknown RIFF/data size
        # (0xffffffff). Derive duration from the actual data chunk instead.
        data_offset = payload.find(b"data")
        if data_offset >= 0:
            data_bytes = len(payload) - data_offset - 8
            return data_bytes / (wav.getnchannels() * wav.getsampwidth() * rate)
        return 0.0


def _wav_prosody(payload: bytes) -> dict[str, float]:
    with wave.open(io.BytesIO(payload), "rb") as wav:
        samples = array.array("h", wav.readframes(wav.getnframes()))
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1 or not samples:
            return {"energy": 0.0, "pause_ratio": 1.0, "onset": 0.0}
    window = 320
    rms_values = []
    for start in range(0, len(samples), window):
        block = samples[start:start + window]
        if block:
            rms_values.append((sum(value * value for value in block) / len(block)) ** 0.5 / 32768)
    peak = max(rms_values, default=0.0)
    active = [value for value in rms_values if value > max(0.006, peak * 0.12)]
    return {
        "energy": min(1.0, (sum(active) / max(1, len(active))) * 7.0),
        "pause_ratio": 1.0 - len(active) / max(1, len(rms_values)),
        "onset": min(1.0, peak * 5.0),
    }


def _reaction_plan(
    sampled: dict[str, object], prosody: dict[str, float], *, repeated: bool = False
) -> dict[str, object]:
    """Fuse phrase semantics and fast acoustic cues into one bounded trajectory."""
    plan = dict(sampled)
    reaction = str(plan.get("reaction", "neutral"))
    semantic = max(0.0, min(0.60, float(plan.get("strength", 0.0))))
    acoustic_gain = 0.68 + 0.22 * prosody["energy"] + 0.10 * prosody["onset"]
    # Semantics bias a slow conversational state; they do not authorize a
    # deliberately posed, full-strength expression anchor.
    strength = min(0.45, semantic * acoustic_gain * 0.82)
    if repeated and reaction != "neutral":
        # Repeated phrase labels represent a sustained conversational state,
        # not a new gesture and not a reason to collapse back toward neutral.
        plan["repeat_continuation"] = True
    primitives = plan.get("primitives", {})
    if isinstance(primitives, dict):
        mixtures = sorted(
            (
                {"reaction": str(name), "weight": min(0.16, max(0.0, float(value)) * 0.18)}
                for name, value in primitives.items()
                if name != reaction and float(value) >= 0.10
            ),
            key=lambda item: item["weight"],
            reverse=True,
        )
        plan["primitive_mix"] = mixtures[:2]
    plan.update(
        {
            "reaction": reaction,
            "strength": strength,
            "start_ratio": 0.18 if repeated and reaction != "neutral" else 0.0,
            "end_ratio": 0.18,
            "attack_ms": max(170, int(float(plan.get("attack_ms", 220)) * (1.0 - 0.10 * prosody["onset"]))),
            "hold_ms": max(140, int(float(plan.get("hold_ms", 320)) * (1.0 - 0.25 * prosody["pause_ratio"]))),
            "decay_ms": max(650, int(float(plan.get("decay_ms", 700)))),
            "clock": "audio",
            "semantic_sample": "phrase",
            "prosody_sample_hz": 50,
            "actuator_fps": STREAM_FPS,
        }
    )
    if reaction == "neutral" or strength < 0.08:
        plan["reaction"], plan["strength"] = "neutral", 0.0
    return plan


def _silence_wav(seconds: float = 0.5) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
        wav.writeframes(bytes(int(16000 * seconds) * 2))
    return output.getvalue()


async def _read_playground_audio(upload: UploadFile, label: str) -> tuple[bytes, str]:
    content_type = (upload.content_type or "").lower()
    if not content_type.startswith("audio/") and content_type not in {"application/octet-stream", "video/webm"}:
        raise HTTPException(status_code=415, detail=f"{label} must be an audio file")
    payload = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"{label} is too large")
    if not payload:
        raise HTTPException(status_code=400, detail=f"{label} is empty")
    suffix = Path(upload.filename or "reference.wav").suffix.lower()
    if suffix not in {".wav", ".mp3", ".ogg", ".oga", ".webm", ".flac", ".m4a"}:
        suffix = ".wav"
    return payload, suffix


async def _speech_post(client: httpx.AsyncClient, url: str, **kwargs: object) -> httpx.Response:
    """Tolerate brief Docker DNS convergence while speech services restart."""
    for attempt in range(3):
        try:
            return await client.post(url, **kwargs)
        except httpx.ConnectError:
            if attempt == 2:
                raise
            await asyncio.sleep(0.25 * (2**attempt))


async def _kokoro_speech(client: httpx.AsyncClient, text: str, voice: str, speed: float) -> httpx.Response:
    response = await _speech_post(
        client,
        f"{KOKORO_URL}/v1/audio/speech",
        json={"model": "kokoro", "input": text, "voice": voice, "speed": speed, "response_format": "wav"},
    )
    response.raise_for_status()
    return response


def _playground_text_fallback(text: str) -> str:
    """Keep a failed/disabled Gemma director from blocking direct Kokoro use."""
    text = re.sub(r"```.*?```|<[^>]+>|\[[^]]{1,80}\]|\([^)]{1,80}\)", " ", text, flags=re.S)
    text = re.sub(r"[*_`#~|]", "", text)
    return re.sub(r"\s+", " ", text).strip()[:2000]


async def _direct_speech_plan(text: str, style: str, model: str) -> dict[str, str]:
    prompt = (
        "Prepare one short passage for Kokoro TTS. Return JSON only with keys "
        "spoken_text and delivery. spoken_text must preserve the user's meaning, "
        "contain only words and ordinary punctuation, and be natural to say aloud. "
        "Do not include stage directions, markup, emotion tags, emoji, lists, or quotes. "
        "delivery is a brief human-readable direction for the operator, not text to speak.\n"
        f"Desired delivery: {style or 'natural, warm, conversational'}\n"
        f"Text: {text}"
    )
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    body = {
        "model": model,
        "messages": [{"role": "system", "content": "You are a precise speech editor."}, {"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.35,
        "top_p": 0.8,
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(25, connect=5)) as client:
        response = await client.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=body)
        response.raise_for_status()
        payload = response.json()
    content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        raise ValueError("Gemma returned no JSON speech plan")
    planned = json.loads(match.group(0))
    spoken = _playground_text_fallback(str(planned.get("spoken_text", "")))
    if not spoken:
        raise ValueError("Gemma returned empty spoken text")
    return {"spoken_text": spoken, "delivery": str(planned.get("delivery", "natural"))[:240], "model": model}


async def _wait_for_marker(job_id: str) -> None:
    done, error = JOBS_ROOT / f"{job_id}.done", JOBS_ROOT / f"{job_id}.err"
    deadline = asyncio.get_running_loop().time() + JOB_TIMEOUT
    while asyncio.get_running_loop().time() < deadline:
        if error.is_file(): raise RuntimeError(error.read_text(encoding="utf-8", errors="replace")[-2000:])
        if done.is_file(): return
        await asyncio.sleep(0.05)
    raise RuntimeError("MuseTalk job timed out")


@app.get("/")
def index() -> FileResponse: return FileResponse(STATIC_ROOT / "index.html")


@app.get("/playground")
def playground() -> FileResponse:
    return FileResponse(STATIC_ROOT / "playground.html")


@app.get("/motion-editor")
def motion_editor() -> FileResponse:
    return FileResponse(STATIC_ROOT / "motion-editor.html")


@app.get("/api/mocap/model")
def mocap_model() -> FileResponse:
    if not MEDIAPIPE_MODEL_VALID:
        raise HTTPException(status_code=503, detail="MediaPipe face-landmarker model is unavailable or invalid")
    return FileResponse(
        MEDIAPIPE_MODEL,
        media_type="application/octet-stream",
        filename="face_landmarker.task",
        headers={"X-Model-SHA256": MEDIAPIPE_MODEL_SHA256, "Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/api/motion/controls")
def motion_controls() -> dict[str, object]:
    """Return a copyable ALP-style calibration vocabulary for the live UI."""
    return {
        "profile": ALP_PROFILE,
        "renderer": "AdvancedLivePortrait semantic bank + MuseTalk mouth",
        "controls": ALP_EXPRESSION_CONTROLS,
        "ownership": {
            "rotate_pitch": "advanced-live-portrait",
            "rotate_yaw": "advanced-live-portrait",
            "rotate_roll": "upper-body-controller",
            "blink": "advanced-live-portrait",
            "eyebrow": "advanced-live-portrait",
            "wink": "advanced-live-portrait",
            "pupil_x": "advanced-live-portrait",
            "pupil_y": "calibration-only (disabled in live bank)",
            "aaa": "musetalk-mouth",
            "eee": "advanced-live-portrait",
            "woo": "advanced-live-portrait",
            "smile": "advanced-live-portrait",
        },
        "live_safe": {
            "rotate_pitch": [-3.0, 3.0],
            "rotate_yaw": [-4.0, 4.0],
            "rotate_roll": [0.0, 0.0],
            "blink": [-14.0, 6.0],
            "eyebrow": [-8.0, 5.0],
            "wink": [0.0, 0.0],
            "pupil_x": [-5.0, 5.0],
            "pupil_y": [0.0, 0.0],
            "aaa": [0.0, 0.0],
            "eee": [0.0, 0.0],
            "woo": [0.0, 0.0],
            "smile": [-0.35, 0.7],
        },
        "note": "Runtime values appear in the browser Motion values panel and are safe to paste into a tuning report. pupil_y and mouth-shape sliders remain available for calibration only; the live semantic bank leaves them neutral.",
    }


@app.post("/api/motion/edit")
async def motion_edit(
    request: Request,
    portrait: UploadFile = File(...),
    rotate_pitch: float = Form(default=0.0, ge=-20.0, le=20.0),
    rotate_yaw: float = Form(default=0.0, ge=-20.0, le=20.0),
    rotate_roll: float = Form(default=0.0, ge=-20.0, le=20.0),
    blink: float = Form(default=0.0, ge=-20.0, le=20.0),
    eyebrow: float = Form(default=0.0, ge=-40.0, le=20.0),
    wink: float = Form(default=0.0, ge=0.0, le=25.0),
    pupil_x: float = Form(default=0.0, ge=-20.0, le=20.0),
    pupil_y: float = Form(default=0.0, ge=-20.0, le=20.0),
    aaa: float = Form(default=0.0, ge=-30.0, le=120.0),
    eee: float = Form(default=0.0, ge=-20.0, le=20.0),
    woo: float = Form(default=0.0, ge=-20.0, le=20.0),
    smile: float = Form(default=0.0, ge=-2.0, le=2.0),
) -> Response:
    """Proxy one live slider edit to the resident ALP GPU sidecar."""
    await _guard_request(request, "motion-edit", MOTION_EDIT_RATE_LIMIT)
    normalized = await _read_portrait(portrait)
    fields = {
        "rotate_pitch": rotate_pitch,
        "rotate_yaw": rotate_yaw,
        "rotate_roll": rotate_roll,
        "blink": blink,
        "eyebrow": eyebrow,
        "wink": wink,
        "pupil_x": pupil_x,
        "pupil_y": pupil_y,
        "aaa": aaa,
        "eee": eee,
        "woo": woo,
        "smile": smile,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10)) as client:
            response = await client.post(
                f"{ALP_URL}/edit",
                files={"portrait": ("portrait.png", normalized, "image/png")},
                data={key: str(value) for key, value in fields.items()},
            )
        if not response.is_success:
            raise HTTPException(status_code=response.status_code, detail=response.text[:800])
        return Response(
            content=response.content,
            media_type="image/png",
            headers={
                "X-ALP-Edit": response.headers.get("X-ALP-Edit", "single-expression-v1"),
                "X-ALP-Encoder": response.headers.get("X-ALP-Encoder", "lossless-rgb-v1"),
                "Cache-Control": "no-store",
            },
        )
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"AdvancedLivePortrait unavailable: {exc}") from exc


@app.get("/api/healthz")
async def healthz() -> JSONResponse:
    async with httpx.AsyncClient() as client:
        try:
            kokoro_response = await client.get(f"{KOKORO_URL}/health", timeout=3)
            kokoro_ok = kokoro_response.is_success
            kokoro_health = kokoro_response.json() if kokoro_ok else {}
        except (httpx.HTTPError, ValueError):
            kokoro_ok, kokoro_health = False, {}
        try:
            llm_ok = (await client.get(f"{LLM_BASE_URL}/models", timeout=3)).is_success
        except httpx.HTTPError: llm_ok = False
        try:
            affect_response = await client.get(f"{AFFECT_URL}/healthz", timeout=3)
            affect_ok = affect_response.is_success
            affect_health = affect_response.json() if affect_ok else {}
        except (httpx.HTTPError, ValueError):
            affect_ok, affect_health = False, {}
        try:
            feedback_response = await client.get(f"{FEEDBACK_URL}/healthz", timeout=3)
            feedback_ok = feedback_response.is_success
            feedback_health = feedback_response.json() if feedback_ok else {}
        except (httpx.HTTPError, ValueError):
            feedback_ok, feedback_health = False, {}
        if UPPER_BODY_ENABLED:
            try:
                upper_body_response = await client.get(f"{UPPER_BODY_URL}/healthz", timeout=3)
                upper_body_ok = upper_body_response.is_success
                upper_body_health = upper_body_response.json() if upper_body_ok else {}
            except (httpx.HTTPError, ValueError):
                upper_body_ok, upper_body_health = False, {}
        else:
            upper_body_ok, upper_body_health = True, {"renderer": "disabled"}
        if ALP_ENABLED:
            try:
                alp_ok = (await client.get(f"{ALP_URL}/healthz", timeout=3)).is_success
            except httpx.HTTPError:
                alp_ok = False
        else:
            alp_ok = True
    muse_ok = MUSE_READY.is_file() and JOBS_ROOT.is_dir()
    payload = {
        "ok": muse_ok and kokoro_ok and llm_ok and alp_ok and affect_ok and feedback_ok and upper_body_ok,
        "musetalk": muse_ok,
        "kokoro": kokoro_ok,
        "kokoro_g2p": kokoro_health.get("g2p"),
        "advanced_live_portrait": alp_ok,
        "advanced_live_portrait_enabled": ALP_ENABLED,
        "mediapipe_face_landmarker": MEDIAPIPE_MODEL_VALID,
        "mediapipe_face_landmarker_sha256": MEDIAPIPE_MODEL_SHA256,
        "qwen": llm_ok,
        "affect": affect_ok,
        "affect_model_revision": affect_health.get("model_revision"),
        "feedback": feedback_ok,
        "feedback_runtime": feedback_health.get("runtime"),
        "upper_body": upper_body_ok,
        "upper_body_enabled": UPPER_BODY_ENABLED,
        "upper_body_renderer": upper_body_health.get("renderer"),
        "model": LLM_MODEL,
        "model_family": "Qwen3.8-27B",
        "generation_mode": "non-thinking speech",
    }
    return JSONResponse(status_code=200 if payload["ok"] else 503, content=payload)


@app.get("/api/voices")
async def voices() -> dict[str, list[str]]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{KOKORO_URL}/v1/audio/voices", timeout=10)
            response.raise_for_status()
            payload = response.json()
            return {"voices": [item.get("id", item.get("name")) if isinstance(item, dict) else item for item in payload.get("voices", [])]}
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Kokoro unavailable: {exc}") from exc


@app.get("/api/playground/capabilities")
async def playground_capabilities() -> dict[str, object]:
    result: dict[str, object] = {"kokoro": False, "voice_lab": False, "kokoclone": False, "gemma_models": [], "modes": ["preset", "voice-lab", "clone", "convert"]}
    async with httpx.AsyncClient(timeout=3) as client:
        try:
            kokoro = await client.get(f"{KOKORO_URL}/health")
            result["kokoro"] = kokoro.is_success
            result["voice_lab"] = False
            if kokoro.is_success:
                result["kokoro_health"] = kokoro.json()
        except (httpx.HTTPError, ValueError):
            pass
        try:
            clone = await client.get(f"{KOKOCLONE_URL}/healthz")
            result["kokoclone"] = clone.is_success and bool(clone.json().get("enabled"))
            result["kokoclone_health"] = clone.json() if clone.is_success else {"error": clone.text[:240]}
        except (httpx.HTTPError, ValueError):
            result["kokoclone_health"] = {"error": "optional service unavailable"}
        try:
            models = await client.get(f"{LLM_BASE_URL}/models")
            if models.is_success:
                result["gemma_models"] = [item.get("id") for item in models.json().get("data", []) if "gemma" in str(item.get("id", "")).lower()]
        except (httpx.HTTPError, ValueError):
            pass
    return result


@app.post("/api/playground/voice-lab")
async def playground_voice_lab(
    request: Request,
    text: str = Form(...),
    source: str = Form(...),
    target: str = Form(...),
    factor: float = Form(default=0.0),
    name: str = Form(default=""),
    speed: float = Form(default=1.0),
    lang: str = Form(default="en-us"),
) -> dict[str, object]:
    await _guard_request(request, "playground-voice-lab", PLAYGROUND_RATE_LIMIT)
    if not text.strip() or len(text) > 2000:
        raise HTTPException(status_code=400, detail="text must contain 1–2000 characters")
    if not -2.0 <= factor <= 2.0:
        raise HTTPException(status_code=422, detail="factor must be between -2 and 2")
    if not 0.5 <= speed <= 2.0:
        raise HTTPException(status_code=422, detail="speed must be between 0.5 and 2.0")
    if lang not in {"en-us", "en-gb"}:
        raise HTTPException(status_code=422, detail="language must be en-us or en-gb")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=5)) as client:
            normalized_response = await client.post(f"{KOKORO_URL}/normalize", json={"text": text.strip()})
            normalized_response.raise_for_status()
            normalized = str(normalized_response.json()["text"])
            response = await client.post(
                f"{KOKORO_URL}/voice-lab",
                json={"text": normalized, "source": source, "target": target,
                      "factor": factor, "name": name.strip() or None, "speed": speed, "lang": lang},
            )
            if not response.is_success:
                raise HTTPException(status_code=502, detail=response.text[:600])
            audio = response.content
    except HTTPException:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"Kokoro voice lab unavailable: {exc}") from exc
    digest = hashlib.sha256(audio).hexdigest()[:24]
    output = AUDIO_ROOT / f"playground-{digest}.wav"
    if not output.exists():
        _write_atomic(output, audio)
    return {"audio_url": f"/api/playground/audio/{output.name}", "duration": _wav_duration(audio),
            "mode": "voice-lab", "voice": name.strip() or "ephemeral", "source": source,
            "target": target, "factor": factor, "speed": speed, "normalized_text": normalized,
            "saved": bool(name.strip()), "plan": {"delivery": "Kokoro style-vector interpolation", "model": "Kokoro-82M"}}


@app.post("/api/playground/generate")
async def playground_generate(
    request: Request,
    text: str = Form(...),
    mode: str = Form(default="preset"),
    voice: str = Form(default="af_bella"),
    speed: float = Form(default=1.0),
    lang: str = Form(default="en-us"),
    style: str = Form(default=""),
    director: bool = Form(default=False),
    director_model: str = Form(default="gemma4-26b"),
    reference: UploadFile | None = File(default=None),
    source: UploadFile | None = File(default=None),
) -> dict[str, object]:
    await _guard_request(request, "playground", PLAYGROUND_RATE_LIMIT)
    if not text.strip() or len(text) > 2000:
        raise HTTPException(status_code=400, detail="text must contain 1–2000 characters")
    if mode not in {"preset", "clone", "convert"}:
        raise HTTPException(status_code=400, detail="mode must be preset, clone, or convert")
    if not 0.5 <= speed <= 2.0:
        raise HTTPException(status_code=422, detail="speed must be between 0.5 and 2.0")
    if lang not in {"en-us", "en-gb"}:
        raise HTTPException(status_code=422, detail="language must be en-us or en-gb")
    plan: dict[str, str] = {"spoken_text": text.strip(), "delivery": "direct Kokoro text", "model": "none"}
    if director:
        try:
            plan = await _direct_speech_plan(text.strip(), style, director_model)
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            plan = {"spoken_text": _playground_text_fallback(text), "delivery": f"Gemma unavailable; fallback used ({type(exc).__name__})", "model": "fallback"}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=5)) as client:
            normalized = plan["spoken_text"]
            if mode == "preset":
                audio_response = await _kokoro_speech(client, normalized, voice, speed)
            else:
                if reference is None:
                    raise HTTPException(status_code=400, detail="reference audio is required for cloning and conversion")
                reference_bytes, reference_suffix = await _read_playground_audio(reference, "reference audio")
                if mode == "clone":
                    audio_response = await _speech_post(
                        client, f"{KOKOCLONE_URL}/clone", params={"text": normalized, "lang": lang[:2]},
                        files={"reference": (f"reference{reference_suffix}", reference_bytes, reference.content_type or "audio/wav")},
                    )
                else:
                    if source is None:
                        raise HTTPException(status_code=400, detail="source audio is required for conversion")
                    source_bytes, source_suffix = await _read_playground_audio(source, "source audio")
                    audio_response = await _speech_post(
                        client, f"{KOKOCLONE_URL}/convert",
                        files={
                            "source": (f"source{source_suffix}", source_bytes, source.content_type or "audio/wav"),
                            "reference": (f"reference{reference_suffix}", reference_bytes, reference.content_type or "audio/wav"),
                        },
                    )
            if not audio_response.is_success:
                detail = audio_response.text[:600]
                raise HTTPException(status_code=503 if mode != "preset" else 502, detail=detail)
            audio = audio_response.content
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"speech service unavailable: {exc}") from exc
    digest = hashlib.sha256(audio).hexdigest()[:24]
    output = AUDIO_ROOT / f"playground-{digest}.wav"
    if not output.exists():
        _write_atomic(output, audio)
    return {
        "audio_url": f"/api/playground/audio/{output.name}",
        "duration": _wav_duration(audio),
        "mode": mode,
        "voice": voice,
        "speed": speed,
        "normalized_text": normalized,
        "plan": plan,
        "model": "Kokoro-82M" if mode == "preset" else "KokoClone + Kanade",
    }


@app.get("/api/playground/audio/{name}")
async def playground_audio(request: Request, name: str) -> FileResponse:
    await _guard_request(request, "playground-audio", PLAYGROUND_RATE_LIMIT * 4)
    if Path(name).name != name or not name.startswith("playground-") or Path(name).suffix != ".wav":
        raise HTTPException(status_code=404, detail="audio not found")
    path = AUDIO_ROOT / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(path, media_type="audio/wav", filename=name)


@app.post("/api/feedback")
async def feedback(request: Request, body: bool = True) -> dict[str, object]:
    await _guard_request(request, "feedback", FEEDBACK_RATE_LIMIT)
    payload = await request.body()
    if not payload or len(payload) > 2 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="invalid feedback frame size")
    try:
        face_request = feedback_client.post(
            f"{FEEDBACK_URL}/observe", content=payload, headers={"Content-Type": "image/jpeg"}
        )
        if UPPER_BODY_ENABLED and body:
            face_response, body_result = await asyncio.gather(
                face_request,
                feedback_client.post(
                    f"{UPPER_BODY_URL}/observe", content=payload,
                    headers={"Content-Type": "image/jpeg"},
                ),
                return_exceptions=True,
            )
            if isinstance(body_result, httpx.Response) and body_result.is_success:
                try:
                    body_payload = body_result.json()
                    body_metrics = body_payload.get("metrics") if isinstance(body_payload, dict) else None
                except ValueError:
                    body_metrics = None
            else:
                body_metrics = None
        else:
            face_response, body_metrics = await face_request, None
        if not isinstance(face_response, httpx.Response):
            raise HTTPException(status_code=502, detail="feedback observer unavailable")
        face_response.raise_for_status()
        result = face_response.json()
        result["body_metrics"] = body_metrics
        result["body_skipped"] = not body
        return result
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=f"feedback observer unavailable: {exc}") from exc


@app.post("/api/live/start")
async def live_start(request: Request, portrait: UploadFile = File(...)) -> dict[str, str]:
    await _guard_request(request, "start", START_RATE_LIMIT)
    normalized = await _read_portrait(portrait)
    session_id, digest = secrets.token_urlsafe(10), hashlib.sha256(normalized).hexdigest()[:20]
    portrait_path, avatar_id = UPLOAD_ROOT / f"portrait-{digest}.png", f"web-{digest}"
    warm_job, prime_job = f"prepare-{session_id}", f"prime-{session_id}"
    prime_audio = AUDIO_ROOT / f"prime-{session_id}.wav"
    prime_stream = STREAM_ROOT / session_id / "prime"
    _write_atomic(portrait_path, normalized); (STREAM_ROOT / session_id).mkdir(parents=True, exist_ok=True)
    upper_body_rig: dict[str, object] | None = None
    if UPPER_BODY_ENABLED:
        try:
            response = await feedback_client.post(
                f"{UPPER_BODY_URL}/prepare", content=normalized,
                headers={"Content-Type": "image/png"},
            )
            response.raise_for_status()
            upper_body_rig = response.json().get("rig")
        except (httpx.HTTPError, ValueError):
            # This stage is optional. A transient pose-sidecar failure must not
            # prevent MuseTalk from preparing and serving the portrait.
            upper_body_rig = None
    # Keep one settled ALP/MuseTalk avatar per portrait in this resident. The
    # first session still gets a random readiness gesture, while later sessions
    # reuse the same motion-bank identity instead of multiplying GPU avatars.
    gesture = READY_GESTURE_CACHE.setdefault(digest, secrets.choice(READY_GESTURES))
    if len(READY_GESTURE_CACHE) > 256:
        READY_GESTURE_CACHE.pop(next(iter(READY_GESTURE_CACHE)))
    prime_stream.mkdir(parents=True, exist_ok=True)
    startup_frames = IDLE_CYCLE_FRAMES + (ALP_READY_FRAMES if ALP_ENABLED else 0)
    _write_atomic(prime_audio, _silence_wav(startup_frames / STREAM_FPS))
    _write_atomic(
        SESSION_ROOT / f"{session_id}.json",
        json.dumps(
            {
                "portrait": str(portrait_path),
                "motion": str(
                    MOTION_ROOT
                    / f"motion-{digest}-{ALP_PROFILE}-{gesture}-f{ALP_FRAMES}-{ALP_REVISION[:8]}.mp4"
                ),
                "avatar_id": (
                    f"{avatar_id}-alp-{ALP_PROFILE}-{gesture}-settled-f{ALP_FRAMES}-{ALP_REVISION[:8]}"
                    if ALP_ENABLED
                    else avatar_id
                ),
                "warm_job": warm_job,
                "prime_job": prime_job,
                "prime_audio": str(prime_audio),
                "prime_stream": str(prime_stream),
                "gesture": gesture,
                "upper_body_rig": upper_body_rig,
            }
        ).encode(),
    )
    _schedule_session_expiry(session_id)
    return {"session_id": session_id, "websocket": f"/ws/live/{session_id}"}


@app.websocket("/ws/live/{session_id}")
async def live(websocket: WebSocket, session_id: str) -> None:
    session_file = SESSION_ROOT / f"{session_id}.json"
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,32}", session_id) or not session_file.is_file():
        await websocket.close(code=1008, reason="unknown live session"); return
    if not _authorized(websocket.headers):
        await websocket.close(code=1008, reason="bearer authentication required"); return
    try:
        await _rate_limit(
            "websocket",
            websocket.client.host if websocket.client else "unknown",
            START_RATE_LIMIT,
        )
    except HTTPException:
        await websocket.close(code=1013, reason="rate limit exceeded"); return
    try:
        session = json.loads(session_file.read_text()); portrait_path = Path(session["portrait"])
        motion_path = Path(session["motion"])
        avatar_id, warm_job, prime_job = (
            str(session["avatar_id"]),
            str(session["warm_job"]),
            str(session["prime_job"]),
        )
        prime_audio, prime_stream = Path(session["prime_audio"]), Path(session["prime_stream"])
        gesture = str(session["gesture"])
        upper_body_rig = session.get("upper_body_rig")
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        await websocket.close(code=1008, reason="invalid live session"); return
    if portrait_path.parent != UPLOAD_ROOT or not portrait_path.is_file():
        await websocket.close(code=1008, reason="invalid live portrait"); return
    await websocket.accept(); send_lock = asyncio.Lock()
    ACTIVE_SESSIONS.add(session_id)
    expiry_task = SESSION_EXPIRY_TASKS.pop(session_id, None)
    if expiry_task:
        expiry_task.cancel()
    session_job_ids: set[str] = set()

    async def send_json(message: dict[str, object]) -> None:
        async with send_lock: await websocket.send_json(message)

    async def send_packet(kind: int, seq: int, index: int, payload: bytes) -> None:
        async with send_lock: await websocket.send_bytes(PACKET_HEADER.pack(kind, seq, index) + payload)

    async def watch_disconnect() -> None:
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    return
        except asyncio.CancelledError:
            return
        except Exception:
            return

    async def run_with_disconnect(coroutine):
        """Race a send/render operation against closure; watcher is the sole receiver."""
        work = asyncio.create_task(coroutine)
        watcher = asyncio.create_task(watch_disconnect())
        done, _ = await asyncio.wait({work, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            raise ClientDisconnected()
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        return await work

    try:
        async def prepare_avatar() -> None:
            source_path = portrait_path
            settle_from = 0
            if ALP_ENABLED:
                await send_json({"type": "motion_preparing"})
                cache = "web"
                generation_seconds = 0.0
                if not motion_path.is_file() or motion_path.stat().st_size < 1024:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(JOB_TIMEOUT, connect=10)) as client:
                        with portrait_path.open("rb") as handle:
                            response = await client.post(
                                f"{ALP_URL}/animate",
                                files={"portrait": (portrait_path.name, handle, "image/png")},
                                data={"fps": str(STREAM_FPS), "frames": str(ALP_FRAMES), "gesture": gesture},
                            )
                    if not response.is_success:
                        raise RuntimeError(f"AdvancedLivePortrait failed: {response.text[:500]}")
                    if len(response.content) < 1024:
                        raise RuntimeError("AdvancedLivePortrait returned an invalid motion clip")
                    _write_atomic(motion_path, response.content)
                    cache = response.headers.get("X-ALP-Cache", "miss")
                    generation_seconds = float(
                        response.headers.get("X-ALP-Generation-Seconds", "0")
                    )
                source_path = motion_path
                settle_from = ALP_READY_FRAMES
                await send_json(
                    {
                        "type": "motion_ready",
                        "cache": cache,
                        "generation_seconds": generation_seconds,
                        "frames": ALP_FRAMES,
                        "gesture": gesture,
                    }
                )
            _write_atomic(
                JOBS_ROOT / f"{warm_job}.json",
                json.dumps(
                    {
                        "video": str(source_path),
                        "avatar_id": avatar_id,
                        "bbox_shift": 0,
                        "source_start_frame": 0,
                        "bank_start_frame": settle_from,
                        "cancel_path": str(JOBS_ROOT / f"{warm_job}.cancel"),
                        "prepare_only": True,
                    }
                ).encode(),
            )
            session_job_ids.add(warm_job)
            _write_atomic(
                JOBS_ROOT / f"{prime_job}.json",
                json.dumps(
                    {
                        "video": str(source_path),
                        "audio": str(prime_audio),
                        "stream_dir": str(prime_stream),
                        "avatar_id": avatar_id,
                        "bbox_shift": 0,
                        "source_start_frame": 0,
                        "bank_start_frame": settle_from,
                        "fps": STREAM_FPS,
                        "batch_size": STREAM_BATCH_SIZE,
                        "stream_format": "jpg",
                        "cancel_path": str(JOBS_ROOT / f"{prime_job}.cancel"),
                        "reaction": {
                            "reaction": "startup" if ALP_ENABLED else "idle",
                            "ready_frames": ALP_READY_FRAMES if ALP_ENABLED else 0,
                            "strength": 0.0,
                        },
                    }
                ).encode(),
            )
            session_job_ids.add(prime_job)

        await run_with_disconnect(prepare_avatar())
        await send_json({"type": "avatar_warming"})
        await run_with_disconnect(_wait_for_marker(prime_job))
        startup_frames = sorted(prime_stream.glob("*.jpg"))
        ready_frames = startup_frames[:ALP_READY_FRAMES] if ALP_ENABLED else []
        idle_frames = startup_frames[ALP_READY_FRAMES:] if ALP_ENABLED else startup_frames
        if not idle_frames or (ALP_ENABLED and len(ready_frames) != ALP_READY_FRAMES):
            raise RuntimeError("MuseTalk prime produced no idle frames")
        face_bbox = None
        try:
            candidate = json.loads((prime_stream / "face_bbox.json").read_text())
            if (isinstance(candidate, list) and len(candidate) == 4
                    and all(isinstance(value, (int, float)) for value in candidate)
                    and candidate[2] > candidate[0] and candidate[3] > candidate[1]):
                face_bbox = candidate
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        if ready_frames:
            await send_json({"type": "ready_gesture", "gesture": gesture, "fps": STREAM_FPS, "frames": len(ready_frames)})
            for index, frame in enumerate(ready_frames):
                await send_packet(READY_FRAME_PACKET, 0, index, await asyncio.to_thread(frame.read_bytes))
            await send_json({"type": "ready_gesture_loaded", "gesture": gesture})
        await send_json({"type": "idle_start", "fps": STREAM_FPS, "frames": len(idle_frames)})
        for index, frame in enumerate(idle_frames):
            await send_packet(IDLE_FRAME_PACKET, 0, index, await asyncio.to_thread(frame.read_bytes))
        await send_json({
            "type": "portrait_ready",
            "gesture": gesture,
            "idle_frames": len(idle_frames),
            "idle_gesture_start": max(0, len(idle_frames) - 4),
            "face_bbox": face_bbox,
            "upper_body_rig": upper_body_rig,
        })

        turn = 0
        next_seq = 0
        while True:
            request = await websocket.receive_json(); prompt = str(request.get("prompt", "")).strip()
            voice, speed = str(request.get("voice", "af_bella")), float(request.get("speed", 1.0))
            if not prompt or len(prompt) > MAX_PROMPT: raise ValueError(f"prompt must contain 1-{MAX_PROMPT} characters")
            if not 0.5 <= speed <= 2.0: raise ValueError("speed must be between 0.5 and 2.0")
            phrase_queue: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue(maxsize=4)
            media_queue: asyncio.Queue[PreparedPhrase | None] = asyncio.Queue(maxsize=2)

            async def produce_qwen() -> None:
                nonlocal next_seq
                headers = {"Content-Type": "application/json"}
                if LLM_API_KEY: headers["Authorization"] = f"Bearer {LLM_API_KEY}"
                body = qwen38_request(LLM_MODEL, prompt, MAX_REPLY_TOKENS)
                body["messages"][0]["content"] = SYSTEM_PROMPT
                buffer = full_text = ""; phrase_count = 0
                await send_json({
                    "type": "assistant_start", "model": LLM_MODEL,
                    "model_family": "Qwen3.8-27B", "mode": "non-thinking speech", "turn": turn,
                    "sampling": {key: body[key] for key in (
                        "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty"
                    )},
                })
                async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10)) as client:
                    async with client.stream("POST", f"{LLM_BASE_URL}/chat/completions", headers=headers, json=body) as response:
                        if not response.is_success:
                            detail = (await response.aread()).decode(errors="replace")[:800]
                            raise RuntimeError(f"Qwen failed ({response.status_code}): {detail}")
                        async for line in response.aiter_lines():
                            if not line.startswith("data: ") or line == "data: [DONE]": continue
                            choices = json.loads(line[6:]).get("choices") or []
                            delta = choices[0].get("delta", {}).get("content", "") if choices else ""
                            if not delta: continue
                            full_text += delta; buffer += delta
                            await send_json({"type": "assistant_delta", "text": delta})
                            phrases, buffer = _phrases(buffer, False)
                            for phrase in phrases:
                                await phrase_queue.put((next_seq, phrase)); await send_json({"type": "phrase_queued", "seq": next_seq, "text": phrase})
                                next_seq += 1; phrase_count += 1
                phrases, _ = _phrases(buffer, True)
                for phrase in phrases:
                    await phrase_queue.put((next_seq, phrase)); await send_json({"type": "phrase_queued", "seq": next_seq, "text": phrase})
                    next_seq += 1; phrase_count += 1
                await phrase_queue.put(None); await send_json({"type": "assistant_done", "text": full_text, "phrases": phrase_count})

            async def synthesize_phrases() -> None:
                previous_reaction = "neutral"
                natural_motion = NaturalMotionScheduler(secrets.randbits(64), STREAM_FPS)
                async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10)) as client:
                    while True:
                        item = await phrase_queue.get()
                        if item is None: await media_queue.put(None); return
                        seq, phrase = item
                        audio_request = client.post(f"{KOKORO_URL}/synthesize", json={"text": phrase, "voice": voice, "speed": speed, "lang": "en-us"})
                        affect_request = client.post(
                            f"{AFFECT_URL}/classify", json={"text": phrase}, timeout=1.5
                        )
                        response, affect_response = await asyncio.gather(
                            audio_request, affect_request, return_exceptions=True
                        )
                        if isinstance(response, BaseException):
                            raise RuntimeError(f"Kokoro failed: {response}") from response
                        if not response.is_success: raise RuntimeError(f"Kokoro failed: {response.text[:500]}")
                        sampled_reaction = optional_json(affect_response, NEUTRAL_REACTION)
                        audio_path = AUDIO_ROOT / f"{session_id}-{turn:03d}-{seq:04d}.wav"; _write_atomic(audio_path, response.content)
                        prosody = _wav_prosody(response.content)
                        reaction = _reaction_plan(
                            sampled_reaction,
                            prosody,
                            repeated=str(sampled_reaction.get("reaction", "neutral")) == previous_reaction,
                        )
                        previous_reaction = str(reaction["reaction"])
                        duration = _wav_duration(response.content)
                        if UPPER_BODY_ENABLED:
                            stable_seed = int.from_bytes(
                                hashlib.sha256(f"{session_id}:{turn}:{seq}".encode()).digest()[:8], "big"
                            )
                            try:
                                upper_response = await client.post(
                                    f"{UPPER_BODY_URL}/plan",
                                    json={
                                        "seed": stable_seed,
                                        "duration": duration,
                                        "prosody": prosody,
                                        "controls": reaction.get("controls", {}),
                                    }, timeout=1.5,
                                )
                                upper_plan = optional_json(upper_response)
                                if upper_plan:
                                    reaction["upper_body"] = upper_plan
                            except (httpx.HTTPError, TypeError, ValueError):
                                # Upper-body motion is enhancement-only. Speech and
                                # lip sync remain live when the sidecar is restarting.
                                pass
                        motion_plan = natural_motion.schedule(
                            duration, prosody, reaction.get("controls", {})
                        )
                        reaction["blinks"] = motion_plan.pop("blinks")
                        reaction["gaze_events"] = motion_plan.pop("gaze_events")
                        reaction["head_events"] = motion_plan.pop("head_events")
                        if not ALP_ENABLED:
                            # The browser's whole-canvas head fallback is disabled;
                            # never advertise events that would silently be ignored.
                            reaction["head_events"] = []
                        if ALP_ENABLED and reaction["head_events"] and isinstance(reaction.get("upper_body"), dict):
                            # A scheduled ALP Euler event owns pitch/yaw for
                            # this phrase. Do not also approximate the same nod
                            # with a whole-canvas shear in the browser.
                            upper_pose = reaction["upper_body"].get("pose")
                            if isinstance(upper_pose, dict) and any(
                                event.get("axis") == "pitch" for event in reaction["head_events"]
                            ):
                                upper_pose["head_pitch"] = 0.0
                                upper_pose["nod_impulse"] = 0.0
                                reaction["upper_body"]["head_pose_route"] = "advanced-live-portrait"
                        reaction["motion_prior"] = motion_plan
                        prepared = PreparedPhrase(seq, phrase, audio_path, duration, reaction, prosody)
                        await send_json({"type": "media_start", "seq": seq, "text": phrase, "duration": prepared.duration, "fps": STREAM_FPS, "reaction": reaction, "prosody": prepared.prosody})
                        await send_packet(AUDIO_PACKET, seq, 0, response.content); await media_queue.put(prepared)

            async def render_phrases() -> None:
                while True:
                    prepared = await media_queue.get()
                    if prepared is None: return
                    seq = prepared.seq; stream_dir = STREAM_ROOT / session_id / f"{turn:03d}-{seq:04d}"; stream_dir.mkdir(parents=True, exist_ok=True)
                    job_id = f"live-{session_id}-{turn:03d}-{seq:04d}"
                    source_path = motion_path if ALP_ENABLED else portrait_path
                    session_job_ids.add(job_id)
                    _write_atomic(JOBS_ROOT / f"{job_id}.json", json.dumps({"video": str(source_path), "audio": str(prepared.audio_path), "stream_dir": str(stream_dir), "avatar_id": avatar_id, "bbox_shift": 0, "source_start_frame": 0, "bank_start_frame": ALP_READY_FRAMES if ALP_ENABLED else 0, "fps": STREAM_FPS, "batch_size": STREAM_BATCH_SIZE, "stream_format": "jpg", "cancel_path": str(JOBS_ROOT / f"{job_id}.cancel"), "reaction": prepared.reaction}).encode())
                    await send_json({"type": "render_start", "seq": seq})
                    done, error = JOBS_ROOT / f"{job_id}.done", JOBS_ROOT / f"{job_id}.err"; emitted: set[str] = set()
                    deadline = asyncio.get_running_loop().time() + JOB_TIMEOUT
                    async def emit_pending_frames() -> None:
                        for frame in sorted(stream_dir.glob("*.jpg")):
                            if frame.name in emitted:
                                continue
                            payload = await asyncio.to_thread(frame.read_bytes)
                            emitted.add(frame.name)
                            await send_packet(FRAME_PACKET, seq, int(frame.stem), payload)
                            frame.unlink(missing_ok=True)

                    while asyncio.get_running_loop().time() < deadline:
                        if error.is_file(): raise RuntimeError(error.read_text(encoding="utf-8", errors="replace")[-2000:])
                        await emit_pending_frames()
                        if done.is_file():
                            # MuseTalk joins its blend workers before writing .done,
                            # but drain twice to cover filesystem visibility races.
                            for _ in range(2):
                                await asyncio.sleep(0.02)
                                await emit_pending_frames()
                            prepared.audio_path.unlink(missing_ok=True)
                            await send_json({"type": "media_done", "seq": seq, "frames": len(emitted)}); break
                        await asyncio.sleep(0.025)
                    else: raise RuntimeError("MuseTalk live render timed out")

            async def run_turn_tasks() -> None:
                async with asyncio.TaskGroup() as tasks:
                    tasks.create_task(produce_qwen())
                    tasks.create_task(synthesize_phrases())
                    tasks.create_task(render_phrases())

            await run_with_disconnect(run_turn_tasks())
            await send_json({"type": "complete", "turn": turn})
            turn += 1
    except (WebSocketDisconnect, ClientDisconnected): return
    except Exception as exc:
        detail = str(exc.exceptions[0]) if isinstance(exc, ExceptionGroup) and exc.exceptions else str(exc)
        try: await send_json({"type": "error", "detail": detail[:1000]})
        except Exception: pass
    finally:
        ACTIVE_SESSIONS.discard(session_id)
        for job_id in session_job_ids:
            _write_atomic(JOBS_ROOT / f"{job_id}.cancel", b"cancel\n")
        if session_job_ids:
            await _wait_for_cancelled_jobs(session_job_ids)
        expiry_task = SESSION_EXPIRY_TASKS.pop(session_id, None)
        if expiry_task:
            expiry_task.cancel()
        try: await websocket.close()
        except Exception: pass
        await asyncio.to_thread(_cleanup_session_artifacts, session_id)
