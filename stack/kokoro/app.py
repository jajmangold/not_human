"""Small HTTP wrapper around the Kokoro ONNX text-to-speech model."""

from __future__ import annotations

import io
import os
import re
import threading
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from kokoro_onnx import Kokoro
from kokoro_onnx import MAX_PHONEME_LENGTH, SAMPLE_RATE
from misaki import en, espeak
from pydantic import BaseModel, Field
from text_normalization import normalize_for_speech


MODEL_DIR = os.environ.get("KOKORO_MODEL_DIR", "/models/kokoro")
MODEL_FILE = os.environ.get("KOKORO_MODEL_FILE", "model.onnx")
MODEL_PATH = os.path.join(MODEL_DIR, "onnx", MODEL_FILE)
VOICES_PATH = os.path.join(MODEL_DIR, "voices.npz")
CUSTOM_VOICE_DIR = os.environ.get("KOKORO_CUSTOM_VOICE_DIR", "/models/custom-voices")
INTRA_OP_THREADS = int(os.environ.get("KOKORO_INTRA_OP_THREADS", "12"))
INTER_OP_THREADS = int(os.environ.get("KOKORO_INTER_OP_THREADS", "1"))

_tts: Kokoro | None = None
_g2p_us: en.G2P | None = None
_g2p_gb: en.G2P | None = None
_synthesis_lock = threading.Lock()
_custom_voices: dict[str, np.ndarray] = {}


def _create_audio_float_speed(self: Kokoro, phonemes: str, voice: np.ndarray, speed: float):
    """Run newer exports whose speed input is float (0.4.7 sends int32)."""
    phonemes = phonemes[:MAX_PHONEME_LENGTH]
    tokens = np.array(self.tokenizer.tokenize(phonemes), dtype=np.int64)
    voice = voice[len(tokens)]
    inputs = {
        "input_ids": [[0, *tokens, 0]],
        "style": np.array(voice, dtype=np.float32),
        "speed": np.array([speed], dtype=np.float32),
    }
    # The q8f16 HF export returns [1, samples], while kokoro-onnx concatenates
    # phrase batches along axis 0. Flatten each batch so long text joins safely.
    audio = np.asarray(self.sess.run(None, inputs)[0]).squeeze()
    return audio, SAMPLE_RATE


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    voice: str = Field(default="af_bella", min_length=2, max_length=96)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    lang: Literal["en-us", "en-gb"] = "en-us"


class VoiceLabRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    source: str = Field(min_length=2, max_length=32)
    target: str = Field(min_length=2, max_length=32)
    factor: float = Field(default=0.0, ge=-2.0, le=2.0)
    name: str | None = Field(default=None, max_length=48)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    lang: Literal["en-us", "en-gb"] = "en-us"


def _voice_style(name: str) -> np.ndarray:
    if name in _custom_voices:
        return _custom_voices[name]
    if _tts is None or name not in _tts.get_voices() or not name.startswith(("af_", "am_", "bf_", "bm_")):
        raise HTTPException(status_code=400, detail=f"unknown English voice: {name}")
    return np.asarray(_tts.get_voice_style(name), dtype=np.float32)


def _interpolate_styles(source: np.ndarray, target: np.ndarray, factor: float) -> np.ndarray:
    # factor 0 is the midpoint, -1/+1 reproduce source/target, and +/-2
    # provide restrained extrapolation beyond the preset pair.
    return ((source + target) / 2.0 + (target - source) * factor / 2.0).astype(np.float32)


def _load_custom_voices() -> None:
    _custom_voices.clear()
    os.makedirs(CUSTOM_VOICE_DIR, exist_ok=True)
    for path in sorted(os.scandir(CUSTOM_VOICE_DIR), key=lambda item: item.name):
        if not path.is_file() or not path.name.startswith("custom_") or not path.name.endswith(".npy"):
            continue
        try:
            value = np.asarray(np.load(path.path, allow_pickle=False), dtype=np.float32)
            if value.ndim == 1 and value.size:
                _custom_voices[path.name[:-4]] = value
        except (OSError, ValueError):
            continue


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _tts, _g2p_us, _g2p_gb
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(f"Kokoro model missing: {MODEL_PATH}")
    if not os.path.isfile(VOICES_PATH):
        raise RuntimeError(f"Kokoro voices archive missing: {VOICES_PATH}")
    _tts = Kokoro(MODEL_PATH, VOICES_PATH)
    _load_custom_voices()
    _g2p_us = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False))
    _g2p_gb = en.G2P(trf=False, british=True, fallback=espeak.EspeakFallback(british=True))
    # kokoro-onnx does not expose ONNX Runtime SessionOptions. Replace its
    # default all-core session with a measured, bounded pool. On the dual
    # E5-2650L v3 host, 12 intra-op threads avoids cross-socket/SMT overhead.
    default_session = _tts.sess
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = INTRA_OP_THREADS
    session_options.inter_op_num_threads = INTER_OP_THREADS
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _tts.sess = ort.InferenceSession(
        MODEL_PATH,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    del default_session
    input_types = {item.name: item.type for item in _tts.sess.get_inputs()}
    if input_types.get("speed") == "tensor(float)":
        import types

        _tts._create_audio = types.MethodType(_create_audio_float_speed, _tts)
    print(
        f"KOKORO_READY model={MODEL_PATH} voices={len(_tts.get_voices())} "
        f"custom_voices={len(_custom_voices)} intra_op_threads={INTRA_OP_THREADS} inter_op_threads={INTER_OP_THREADS}",
        flush=True,
    )
    yield
    _tts = None
    _g2p_us = _g2p_gb = None


app = FastAPI(title="Volta Kokoro TTS", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {
        "ok": _tts is not None,
        "model": MODEL_FILE,
        "g2p": "misaki-0.7.4",
        "intra_op_threads": INTRA_OP_THREADS,
        "inter_op_threads": INTER_OP_THREADS,
    }


@app.get("/voices")
def voices() -> dict[str, list[str]]:
    if _tts is None:
        raise HTTPException(status_code=503, detail="Kokoro is still loading")
    presets = sorted(voice for voice in _tts.get_voices() if voice.startswith(("af_", "am_", "bf_", "bm_")))
    return {"voices": presets + sorted(_custom_voices)}


@app.post("/normalize")
def normalize(request: dict[str, str]) -> dict[str, str]:
    text = normalize_for_speech(str(request.get("text", "")))
    if not text:
        raise HTTPException(status_code=400, detail="text contains no speakable content")
    return {"text": text}


@app.post("/synthesize")
def synthesize(request: SynthesisRequest) -> Response:
    if _tts is None or _g2p_us is None or _g2p_gb is None:
        raise HTTPException(status_code=503, detail="Kokoro is still loading")
    voices = [item.strip() for item in request.voice.split(",") if item.strip()]
    if not voices or len(voices) > 3 or any(
        item not in _tts.get_voices() and item not in _custom_voices for item in voices
    ):
        raise HTTPException(status_code=400, detail=f"unknown English voice mix: {request.voice}")
    try:
        spoken_text = normalize_for_speech(request.text)
        if not spoken_text:
            raise HTTPException(status_code=400, detail="text contains no speakable content")
        g2p = _g2p_gb if request.lang == "en-gb" else _g2p_us
        phonemes, _ = g2p(spoken_text)
        with _synthesis_lock:
            # kokoro-onnx accepts one voice name or one style vector; blend
            # multiple selected presets into a single style vector here.
            # This keeps the public playground API useful across package
            # versions that do not implement comma-separated voice names.
            style = np.mean([_voice_style(voice) for voice in voices], axis=0).astype(np.float32)
            samples, sample_rate = _tts.create(
                phonemes,
                voice=style,
                speed=request.speed,
                is_phonemes=True,
            )
        output = io.BytesIO()
        sf.write(output, np.asarray(samples).squeeze(), sample_rate, format="WAV", subtype="PCM_16")
        return Response(content=output.getvalue(), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime/model-specific errors
        raise HTTPException(status_code=500, detail=f"Kokoro synthesis failed: {exc}") from exc


@app.post("/voice-lab")
def voice_lab(request: VoiceLabRequest) -> Response:
    if _tts is None or _g2p_us is None or _g2p_gb is None:
        raise HTTPException(status_code=503, detail="Kokoro is still loading")
    if request.source == request.target:
        raise HTTPException(status_code=400, detail="source and target voices must differ")
    if request.name is not None and not re.fullmatch(r"custom_[a-z0-9][a-z0-9_-]{1,46}", request.name):
        raise HTTPException(status_code=422, detail="name must start with custom_ and contain lowercase letters, numbers, _ or -")
    try:
        spoken_text = normalize_for_speech(request.text)
        if not spoken_text:
            raise HTTPException(status_code=400, detail="text contains no speakable content")
        with _synthesis_lock:
            style = _interpolate_styles(_voice_style(request.source), _voice_style(request.target), request.factor)
            if request.name:
                os.makedirs(CUSTOM_VOICE_DIR, exist_ok=True)
                destination = os.path.join(CUSTOM_VOICE_DIR, f"{request.name}.npy")
                temporary = f"{destination}.tmp.npy"
                np.save(temporary, style, allow_pickle=False)
                os.replace(temporary, destination)
                _custom_voices[request.name] = style
            g2p = _g2p_gb if request.lang == "en-gb" else _g2p_us
            phonemes, _ = g2p(spoken_text)
            samples, sample_rate = _tts.create(phonemes, voice=style, speed=request.speed, is_phonemes=True)
        output = io.BytesIO()
        sf.write(output, np.asarray(samples).squeeze(), sample_rate, format="WAV", subtype="PCM_16")
        response = Response(content=output.getvalue(), media_type="audio/wav")
        response.headers["X-Kokoro-Voice"] = request.name or "ephemeral"
        response.headers["X-Kokoro-Factor"] = str(request.factor)
        return response
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - runtime/model-specific errors
        raise HTTPException(status_code=500, detail=f"Kokoro voice lab failed: {exc}") from exc
