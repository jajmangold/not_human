"""Resident adapter for the pinned AdvancedLivePortrait ComfyUI node."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import os
import subprocess
import sys
import threading
import time
import types
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from safetensors.torch import load_file as load_safetensors

from face_validation import NoUsableFaceError, require_usable_face


UPSTREAM_ROOT = Path(os.environ.get("ALP_SOURCE_DIR", "/opt/advanced-live-portrait"))
MODEL_ROOT = Path(os.environ.get("ALP_MODEL_DIR", "/models"))
CACHE_ROOT = Path(os.environ.get("ALP_CACHE_DIR", "/cache"))
MAX_UPLOAD_BYTES = int(os.environ.get("ALP_MAX_UPLOAD_BYTES", "10485760"))
MAX_SOURCE_EDGE = int(os.environ.get("ALP_MAX_SOURCE_EDGE", "512"))
UPSTREAM_REVISION = os.environ.get(
    "ALP_REVISION", "3bba732915e22f18af0d221b9c5c282990181f1b"
)
MOTION_PROFILE_REVISION = os.environ.get("ALP_MOTION_PROFILE", "semantic-bank-v13-blink-brow-fix")
READY_FRAME_COUNT = 8
REACTION_BANK_SIZE = 20
MOTION_ENCODER_REVISION = "lossless-rgb-v1"

_nodes = None
_ready = False
_generation_lock = threading.Lock()


def _install_comfy_shims() -> None:
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(MODEL_ROOT)
    folder_paths.output_directory = str(CACHE_ROOT / "output")
    folder_paths.temp_directory = str(CACHE_ROOT / "temp")

    def get_folder_paths(name: str) -> list[str]:
        return [str(MODEL_ROOT / name)]

    def get_save_image_path(filename: str, directory: str):
        return directory, filename, 0, "", filename

    folder_paths.get_folder_paths = get_folder_paths
    folder_paths.get_save_image_path = get_save_image_path
    sys.modules["folder_paths"] = folder_paths

    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")

    def load_torch_file(path: str):
        if path.endswith(".safetensors"):
            return load_safetensors(path, device="cpu")
        return torch.load(path, map_location="cpu")

    class ProgressBar:
        def __init__(self, _: int):
            pass

        def update_absolute(self, *_args, **_kwargs) -> None:
            pass

    comfy_utils.load_torch_file = load_torch_file
    comfy_utils.ProgressBar = ProgressBar
    comfy.utils = comfy_utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = comfy_utils


def _load_upstream():
    _install_comfy_shims()
    package = types.ModuleType("advanced_live_portrait_upstream")
    package.__path__ = [str(UPSTREAM_ROOT)]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.nodes",
        UPSTREAM_ROOT / "nodes.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned AdvancedLivePortrait source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _expression(
    nodes,
    *,
    pitch=0.0,
    yaw=0.0,
    roll=0.0,
    blink=0.0,
    eyebrow=0.0,
    wink=0.0,
    pupil_x=0.0,
    pupil_y=0.0,
    mouth=0.0,
    eee=0.0,
    woo=0.0,
    smile=0.0,
):
    expression = nodes.ExpressionSet()
    expression.r = nodes.g_engine.calc_fe(
        expression.e,
        blink,
        eyebrow,
        wink,
        pupil_x,
        pupil_y,
        mouth,
        eee,
        woo,
        smile,
        pitch,
        yaw,
        roll,
    )
    return expression


def _normalized_source(payload: bytes) -> tuple[torch.Tensor, bytes]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image = image.convert("RGB")
            image.thumbnail((MAX_SOURCE_EDGE, MAX_SOURCE_EDGE), Image.Resampling.LANCZOS)
            width = image.width - image.width % 2
            height = image.height - image.height % 2
            if width < 64 or height < 64:
                raise ValueError("portrait is too small")
            if (width, height) != image.size:
                image = image.crop((0, 0, width, height))
            array = np.asarray(image, dtype=np.float32) / 255.0
            normalized = io.BytesIO()
            image.save(normalized, format="PNG")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid portrait: {exc}") from exc
    return torch.from_numpy(array).unsqueeze(0), normalized.getvalue()


def _encode_mp4(frames: np.ndarray, output_path: Path, fps: int) -> None:
    height, width = frames.shape[1:3]
    writer = imageio_ffmpeg.write_frames(
        str(output_path),
        (width, height),
        fps=fps,
        # MuseTalk consumes this source bank for every later phrase. Keep the
        # RGB pixels lossless so H.264 chroma subsampling cannot seed mouth or
        # eye ringing before the final compositor runs.
        codec="libx264rgb",
        pix_fmt_in="rgb24",
        pix_fmt_out="rgb24",
        output_params=["-preset", "veryfast", "-crf", "0"],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()


def _encode_frame_archive(frames: np.ndarray, output_path: Path) -> None:
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for index, frame in enumerate(frames):
            encoded = io.BytesIO()
            Image.fromarray(frame).save(encoded, format="PNG", compress_level=3)
            archive.writestr(f"{index:08d}.png", encoded.getvalue())


def _gesture_motion(nodes, gesture: str):
    transition = _expression(
        nodes, pitch=0.10, yaw=0.15, roll=0.0, eyebrow=0.15,
        pupil_x=0.10, smile=0.09,
    )
    gestures = {
        "wink": _expression(nodes, yaw=4.0, roll=0.0, wink=18.0, smile=0.45),
        "nod": _expression(nodes, pitch=8.0, eyebrow=1.5, smile=0.2),
        "smile": _expression(nodes, pitch=1.0, eyebrow=2.0, smile=1.0),
        "surprise-wink": _expression(nodes, pitch=-2.0, roll=0.0, blink=3.0, eyebrow=10.0, mouth=24.0),
        "kiss": _expression(nodes, pitch=2.0, blink=-3.0, woo=12.0, mouth=5.0, smile=0.12),
    }
    if gesture not in gestures:
        raise ValueError(f"unsupported ready gesture: {gesture}")
    if gesture == "surprise-wink":
        transition = _expression(nodes, yaw=0.20, roll=0.0, wink=6.0, smile=0.18)
    # The retained frames are a random-access expression bank, not a motion
    # sequence. MuseTalk interpolates these subtle upper-face sources on its
    # audio timeline and repaints the mouth last.
    bank = [
        # Three mouth-stable fixation anchors: camera, left/up, right/down.
        _expression(nodes, eyebrow=0.08, smile=0.03),
        # The calibrated eye anchors deliberately leave pupil_y at neutral: on
        # this identity that actuator changes lid aperture and brows instead of
        # producing an isolated vertical gaze.
        _expression(nodes, eyebrow=0.10, pupil_x=-4.5, smile=0.03),
        # Full closure (-20, not a partial -14) measures lowest brow leak of
        # any depth on the 3 identities in evidence/issue-30/blink-brow-fix/:
        # worst-case brow max_abs_delta 0.117, vs 0.250 at -14/eyebrow=0 and
        # 0.858 at the previous -14/eyebrow=-32 "compensation." That term is
        # removed: across every depth and identity tested, any negative
        # eyebrow counter-value made the brow-down leak *larger*, monotonically
        # -- it was never actually canceling the lift, it was compounding it.
        _expression(nodes, blink=-20.0, smile=0.03),
        _expression(nodes, eyebrow=0.07, pupil_x=3.8, smile=0.035),
        _expression(nodes, pitch=0.10, roll=0.0, blink=-0.25, eyebrow=0.45, smile=0.28),
        _expression(nodes, pitch=0.26, roll=0.0, blink=-0.7, eyebrow=1.2, smile=0.62),
        _expression(nodes, pitch=-0.18, blink=1.0, eyebrow=2.2),
        _expression(nodes, pitch=-0.65, blink=2.2, eyebrow=4.5),
        _expression(nodes, yaw=0.20, roll=0.0, blink=-0.25, eyebrow=-1.0, pupil_x=0.35),
        _expression(nodes, yaw=0.65, roll=0.0, blink=-0.9, eyebrow=-2.4, pupil_x=0.75),
        _expression(nodes, pitch=0.12, blink=0.25, eyebrow=-1.1, smile=-0.06),
        _expression(nodes, pitch=0.38, blink=0.7, eyebrow=-2.6, smile=-0.16),
        _expression(nodes, pitch=0.55, blink=-0.15, eyebrow=0.8, smile=0.30),
        _expression(nodes, yaw=-0.55, roll=0.0, blink=-0.35, eyebrow=-1.7, pupil_x=-0.40, smile=-0.10),
        _expression(nodes, pitch=0.25, eyebrow=1.8, blink=0.65, smile=0.16),
        _expression(nodes, yaw=0.62, roll=0.0, blink=-0.25, eyebrow=-1.3, pupil_x=0.60),
        # Real Euler head-pose endpoints. Runtime scheduling uses only a
        # fraction of these anchors and never loops them as a sway oscillator.
        _expression(nodes, yaw=-4.0, eyebrow=0.08, smile=0.03),
        _expression(nodes, yaw=4.0, eyebrow=0.08, smile=0.03),
        _expression(nodes, pitch=-3.0, eyebrow=0.08, smile=0.03),
        _expression(nodes, pitch=3.0, eyebrow=0.08, smile=0.03),
    ]
    return gestures[gesture], transition, bank


def _generate(
    payload: bytes, fps: int, frames: int, gesture: str, *, use_cache: bool = True
) -> tuple[bytes, bytes, float, bool]:
    source, normalized = _normalized_source(payload)
    digest = hashlib.sha256(
        normalized
        + f"|{fps}|{frames}|{gesture}|{UPSTREAM_REVISION}|{MOTION_PROFILE_REVISION}|{MOTION_ENCODER_REVISION}".encode()
    ).hexdigest()
    output_path = (
        CACHE_ROOT / f"{digest}.mp4"
        if use_cache
        else CACHE_ROOT / f"warmup-{os.getpid()}.mp4"
    )
    archive_path = output_path.with_suffix(".frames.zip")
    started = time.perf_counter()
    with _generation_lock:
        assert _nodes is not None
        source_rgb = (source[0] * 255.0).byte().numpy()
        require_usable_face(_nodes.g_engine.get_face_bboxes(source_rgb))
        # Validate before trusting a cache entry. Older adapter revisions could
        # cache a whole-image fallback even when upstream detected no face.
        if use_cache and output_path.is_file() and archive_path.is_file():
            return output_path.read_bytes(), archive_path.read_bytes(), 0.0, True
        psi = _nodes.g_engine.prepare_source(source, 1.7)
        ready, transition, bank = _gesture_motion(_nodes, gesture)
        if frames != READY_FRAME_COUNT + REACTION_BANK_SIZE:
            raise ValueError(
                f"semantic expression profile requires {READY_FRAME_COUNT + REACTION_BANK_SIZE} frames"
            )
        motion_link = [psi, ready, transition, *bank]
        changes = [READY_FRAME_COUNT // 2, READY_FRAME_COUNT // 2, *([1] * len(bank))]
        command = "\n".join(
            f"{motion_index}={length}:0"
            for motion_index, length in zip(range(1, len(motion_link)), changes)
        )
        animator = _nodes.AdvancedLivePortrait()
        result = animator.run(
            0.0,
            0.0,
            True,
            False,
            True,
            command,
            1.7,
            src_images=None,
            driving_images=None,
            motion_link=motion_link,
        )[0]
        if result is None or len(result) != frames:
            raise RuntimeError(f"AdvancedLivePortrait returned {0 if result is None else len(result)} frames")
        rgb = np.clip(result.numpy() * 255.0, 0, 255).astype(np.uint8)
        temporary = output_path.with_suffix(".tmp.mp4")
        temporary_archive = output_path.with_suffix(".tmp.frames.zip")
        _encode_mp4(rgb, temporary, fps)
        _encode_frame_archive(rgb, temporary_archive)
        temporary.replace(output_path)
        temporary_archive.replace(archive_path)
    video = output_path.read_bytes()
    frame_archive = archive_path.read_bytes()
    if not use_cache:
        output_path.unlink(missing_ok=True)
        archive_path.unlink(missing_ok=True)
    return video, frame_archive, time.perf_counter() - started, False


def _edit_expression(payload: bytes, controls: dict[str, float]) -> bytes:
    """Render one ALP expression frame for the interactive calibration editor."""
    source, _ = _normalized_source(payload)
    with _generation_lock:
        assert _nodes is not None
        source_rgb = (source[0] * 255.0).byte().numpy()
        require_usable_face(_nodes.g_engine.get_face_bboxes(source_rgb))
        psi = _nodes.g_engine.prepare_source(source, 1.7)
        expression = _expression(
            _nodes,
            pitch=controls["rotate_pitch"],
            yaw=controls["rotate_yaw"],
            roll=controls["rotate_roll"],
            blink=controls["blink"],
            eyebrow=controls["eyebrow"],
            wink=controls["wink"],
            pupil_x=controls["pupil_x"],
            pupil_y=controls["pupil_y"],
            mouth=controls["aaa"],
            eee=controls["eee"],
            woo=controls["woo"],
            smile=controls["smile"],
        )
        animator = _nodes.AdvancedLivePortrait()
        result = animator.run(
            0.0,
            0.0,
            True,
            False,
            True,
            "1=1:0",
            1.7,
            src_images=None,
            driving_images=None,
            motion_link=[psi, expression],
        )[0]
        if result is None or len(result) != 1:
            raise RuntimeError(f"AdvancedLivePortrait returned {0 if result is None else len(result)} frames")
        frame = np.clip(result[0].numpy() * 255.0, 0, 255).astype(np.uint8)
    output = io.BytesIO()
    Image.fromarray(frame).save(output, format="PNG", compress_level=3)
    return output.getvalue()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _nodes, _ready
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / "output" / "exp_data").mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / "temp").mkdir(parents=True, exist_ok=True)
    required = [
        MODEL_ROOT / "liveportrait" / "appearance_feature_extractor.safetensors",
        MODEL_ROOT / "liveportrait" / "motion_extractor.safetensors",
        MODEL_ROOT / "liveportrait" / "warping_module.safetensors",
        MODEL_ROOT / "liveportrait" / "spade_generator.safetensors",
        MODEL_ROOT / "liveportrait" / "stitching_retargeting_module.safetensors",
        MODEL_ROOT / "ultralytics" / "face_yolov8n.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"AdvancedLivePortrait assets missing: {missing}")
    actual_revision = subprocess.run(
        ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_revision != UPSTREAM_REVISION:
        raise RuntimeError(
            f"AdvancedLivePortrait revision mismatch: {actual_revision} != {UPSTREAM_REVISION}"
        )
    _nodes = _load_upstream()
    torch.backends.cudnn.benchmark = True
    _nodes.g_engine.get_pipeline()
    _nodes.g_engine.get_detect_model()
    _, _, warmup_seconds, _ = _generate(
        (UPSTREAM_ROOT / "sample" / "source_image.png").read_bytes(),
        8,
        READY_FRAME_COUNT + REACTION_BANK_SIZE,
        "smile",
        use_cache=False,
    )
    _ready = True
    print(
        f"ALP_READY upstream={UPSTREAM_REVISION} device={_nodes.get_device()} "
        f"max_source_edge={MAX_SOURCE_EDGE} warmup_seconds={warmup_seconds:.3f}",
        flush=True,
    )
    yield
    _ready = False
    _nodes = None


app = FastAPI(title="AdvancedLivePortrait resident adapter", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {
        "ok": _ready,
        "upstream_revision": UPSTREAM_REVISION,
        "motion_profile": MOTION_PROFILE_REVISION,
        "device": str(_nodes.get_device()) if _nodes is not None else None,
    }


@app.post("/animate")
def animate(
    portrait: UploadFile = File(...),
    fps: int = Form(default=8, ge=4, le=30),
    frames: int = Form(default=28, ge=28, le=28),
    gesture: str = Form(default="smile"),
    output: str = Form(default="mp4"),
) -> Response:
    if not _ready:
        raise HTTPException(status_code=503, detail="AdvancedLivePortrait is still loading")
    payload = portrait.file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="portrait is too large")
    try:
        if output not in {"mp4", "frames"}:
            raise HTTPException(status_code=400, detail="output must be mp4 or frames")
        video, frame_archive, generation_seconds, cache_hit = _generate(
            payload, fps, frames, gesture
        )
    except HTTPException:
        raise
    except NoUsableFaceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"motion generation failed: {exc}") from exc
    return Response(
        content=video if output == "mp4" else frame_archive,
        media_type="video/mp4" if output == "mp4" else "application/zip",
        headers={
            "X-ALP-Cache": "hit" if cache_hit else "miss",
            "X-ALP-Generation-Seconds": f"{generation_seconds:.3f}",
            "X-ALP-Frames": str(frames),
            "X-ALP-Encoder": MOTION_ENCODER_REVISION,
        },
    )


@app.post("/edit")
def edit(
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
    """Interactive single-image ALP editor used by the web calibration page."""
    if not _ready:
        raise HTTPException(status_code=503, detail="AdvancedLivePortrait is still loading")
    payload = portrait.file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="portrait is too large")
    controls = {
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
        rendered = _edit_expression(payload, controls)
    except HTTPException:
        raise
    except NoUsableFaceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"expression edit failed: {exc}") from exc
    return Response(
        content=rendered,
        media_type="image/png",
        headers={
            "X-ALP-Encoder": MOTION_ENCODER_REVISION,
            "X-ALP-Edit": "single-expression-v1",
        },
    )
