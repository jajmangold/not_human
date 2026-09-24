"""Optional KokoClone adapter; baseline Kokoro never depends on this service."""
from __future__ import annotations

import io
import os
import tempfile
import threading
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

ENABLED = os.environ.get("KOKOCLONE_ENABLED", "1") == "1"
MAX_BYTES = int(os.environ.get("KOKOCLONE_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
_cloner = None
_load_error: str | None = None
_lock = threading.Lock()
_inference_lock = threading.Lock()
app = FastAPI(title="Volta KokoClone adapter")


def _load():
    global _cloner, _load_error
    if not ENABLED:
        raise RuntimeError("KokoClone is disabled")
    if _cloner is None:
        with _lock:
            if _cloner is None:
                try:
                    from core.cloner import KokoClone
                    _cloner = KokoClone()
                except Exception as exc:  # model download/driver failures are optional
                    _load_error = f"{type(exc).__name__}: {exc}"
                    raise RuntimeError(_load_error) from exc
    return _cloner


async def _read_audio(upload: UploadFile, label: str) -> bytes:
    if upload.content_type not in {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/ogg", "audio/webm"}:
        raise HTTPException(status_code=415, detail=f"{label} must be WAV, MP3, OGG, or WebM audio")
    payload = await upload.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"{label} is too large")
    if not payload:
        raise HTTPException(status_code=400, detail=f"{label} is empty")
    return payload


def _run(mode: str, text: str | None, reference: bytes, source: bytes | None, lang: str) -> bytes:
    cloner = _load()
    with tempfile.TemporaryDirectory(prefix="kokoclone-") as directory:
        root = Path(directory)
        ref_path = root / "reference.audio"
        ref_path.write_bytes(reference)
        output = root / "output.wav"
        if mode == "clone":
            if not text or not text.strip():
                raise HTTPException(status_code=400, detail="text is required for text-to-clone")
            cloner.generate(text.strip(), lang, str(ref_path), str(output))
        elif mode == "convert":
            if source is None:
                raise HTTPException(status_code=400, detail="source audio is required for re-voicing")
            source_path = root / "source.audio"
            source_path.write_bytes(source)
            cloner.convert(str(source_path), str(ref_path), str(output))
        else:
            raise HTTPException(status_code=400, detail="mode must be clone or convert")
        return output.read_bytes()


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {"ok": _cloner is not None, "enabled": ENABLED, "loaded": _cloner is not None, "error": _load_error}


@app.post("/clone")
async def clone(text: str, lang: str = "en", reference: UploadFile = File(...)) -> Response:
    reference_bytes = await _read_audio(reference, "reference audio")
    try:
        with _inference_lock:
            payload = _run("clone", text, reference_bytes, None, lang)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"KokoClone unavailable or failed to load: {exc}") from exc
    return Response(payload, media_type="audio/wav", headers={"X-Voice-Mode": "kokoclone-kanade"})


@app.post("/convert")
async def convert(source: UploadFile = File(...), reference: UploadFile = File(...)) -> Response:
    source_bytes, reference_bytes = await _read_audio(source, "source audio"), await _read_audio(reference, "reference audio")
    try:
        with _inference_lock:
            payload = _run("convert", None, reference_bytes, source_bytes, "en")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"KokoClone unavailable or failed to load: {exc}") from exc
    return Response(payload, media_type="audio/wav", headers={"X-Voice-Mode": "kokoclone-kanade-convert"})
