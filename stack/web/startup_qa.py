"""Non-blocking startup calibration for the resident portrait stack.

The live web service never depends on this one-shot process.  It renders the same
portrait through controlled ALP/MuseTalk conditions, measures the final frames,
and optionally asks the configured vision-capable Qwen endpoint for a verdict.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
import uuid
import wave
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat


IO_ROOT = Path(os.environ.get("MUSE_TALK_IO_DIR", "/io"))
PORTRAIT = Path(
    os.environ.get(
        "STARTUP_QA_PORTRAIT",
        "/io/web_demo/uploads/portrait-d08aa48606ebf5e43a2a.png",
    )
)
ALP_URL = os.environ.get("ADVANCED_LIVE_PORTRAIT_URL", "http://advanced-live-portrait:8093").rstrip("/")
FEEDBACK_URL = os.environ.get("FEEDBACK_URL", "http://feedback:8095").rstrip("/")
VISION_BASE_URL = os.environ.get("STARTUP_QA_VISION_BASE_URL", "http://localhost:8000/v1").rstrip("/")
VISION_MODEL = os.environ.get("STARTUP_QA_VISION_MODEL", "qwen27b")
VISION_API_KEY = os.environ.get("STARTUP_QA_VISION_API_KEY", os.environ.get("LLM_API_KEY", ""))
TIMEOUT = float(os.environ.get("STARTUP_QA_TIMEOUT", "300"))
FPS = int(os.environ.get("STARTUP_QA_FPS", "16"))
JOBS_ROOT = IO_ROOT / "muse_jobs"
EVIDENCE_ROOT = IO_ROOT / "web_demo" / "startup_qa"

ALP_BANK = (
    "neutral", "gaze-left", "blink", "gaze-right", "amusement-low",
    "amusement", "surprise-low", "surprise", "skepticism-low",
    "skepticism", "concern-low", "concern", "agreement",
    "disagreement", "interest", "thinking", "head-yaw-left",
    "head-yaw-right", "head-pitch-up", "head-pitch-down",
)
FINAL_CONTROLS = {
    "neutral": {"reaction": "neutral", "strength": 0.0},
    "gaze-left": {
        "reaction": "neutral", "strength": 0.0,
        "gaze_events": [{"at_ms": 0, "direction": -1, "move_ms": 60, "hold_ms": 900, "return_ms": 180}],
    },
    "gaze-right": {
        "reaction": "neutral", "strength": 0.0,
        "gaze_events": [{"at_ms": 0, "direction": 1, "move_ms": 60, "hold_ms": 900, "return_ms": 180}],
    },
    "blink": {
        "reaction": "neutral", "strength": 0.0,
        "blinks": [{"at_ms": 380, "close_ms": 55, "hold_ms": 20, "open_ms": 95, "peak": 0.84}],
    },
    "amusement": {"reaction": "amusement", "strength": 0.68, "attack_ms": 80, "hold_ms": 900, "decay_ms": 300},
    "surprise": {"reaction": "surprise", "strength": 0.68, "attack_ms": 80, "hold_ms": 900, "decay_ms": 300},
    "skepticism": {"reaction": "skepticism", "strength": 0.68, "attack_ms": 100, "hold_ms": 900, "decay_ms": 300},
    "concern": {"reaction": "concern", "strength": 0.68, "attack_ms": 100, "hold_ms": 900, "decay_ms": 300},
    "agreement": {"reaction": "agreement", "strength": 0.68, "attack_ms": 100, "hold_ms": 900, "decay_ms": 300},
    "disagreement": {"reaction": "disagreement", "strength": 0.68, "attack_ms": 100, "hold_ms": 900, "decay_ms": 300},
    "interest": {"reaction": "interest", "strength": 0.68, "attack_ms": 100, "hold_ms": 900, "decay_ms": 300},
    "thinking": {"reaction": "thinking", "strength": 0.68, "attack_ms": 100, "hold_ms": 900, "decay_ms": 300},
    "head-yaw-left": {
        "reaction": "neutral", "strength": 0.0,
        "head_events": [{"axis": "yaw", "direction": -1, "at_ms": 0,
                         "move_ms": 160, "hold_ms": 900, "return_ms": 240, "amplitude": 0.72}],
    },
    "head-yaw-right": {
        "reaction": "neutral", "strength": 0.0,
        "head_events": [{"axis": "yaw", "direction": 1, "at_ms": 0,
                         "move_ms": 160, "hold_ms": 900, "return_ms": 240, "amplitude": 0.72}],
    },
    "head-pitch-up": {
        "reaction": "neutral", "strength": 0.0,
        "head_events": [{"axis": "pitch", "direction": -1, "at_ms": 0,
                         "move_ms": 160, "hold_ms": 900, "return_ms": 240, "amplitude": 0.72}],
    },
    "head-pitch-down": {
        "reaction": "neutral", "strength": 0.0,
        "head_events": [{"axis": "pitch", "direction": 1, "at_ms": 0,
                         "move_ms": 160, "hold_ms": 900, "return_ms": 240, "amplitude": 0.72}],
    },
}


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def silence_wav(seconds: float) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * int(16000 * seconds))
    return output.getvalue()


def wait_for_job(name: str) -> None:
    deadline = time.monotonic() + TIMEOUT
    done, error = JOBS_ROOT / f"{name}.done", JOBS_ROOT / f"{name}.err"
    while time.monotonic() < deadline:
        if done.is_file():
            return
        if error.is_file():
            raise RuntimeError(error.read_text(errors="replace")[:1000])
        time.sleep(0.05)
    raise TimeoutError(f"MuseTalk job {name} did not finish in {TIMEOUT:g}s")


def submit_job(name: str, spec: dict[str, object]) -> None:
    for suffix in (".done", ".err"):
        (JOBS_ROOT / f"{name}{suffix}").unlink(missing_ok=True)
    atomic_write(JOBS_ROOT / f"{name}.json", json.dumps(spec, sort_keys=True).encode())
    wait_for_job(name)


def observe(client: httpx.Client, image_path: Path) -> dict[str, float] | None:
    response = client.post(
        f"{FEEDBACK_URL}/observe",
        content=image_path.read_bytes(),
        headers={"Content-Type": "image/jpeg"},
    )
    response.raise_for_status()
    metrics = response.json().get("metrics")
    return metrics if isinstance(metrics, dict) else None


def label_image(image: Image.Image, label: str, width: int = 320) -> Image.Image:
    image = image.convert("RGB")
    height = max(1, round(image.height * width / image.width))
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (width, height + 34), "#151922")
    tile.paste(image, (0, 34))
    ImageDraw.Draw(tile).text((10, 9), label, fill="white", font=ImageFont.load_default())
    return tile


def contact_sheet(samples: list[tuple[str, Path]], columns: int = 3, width: int = 320) -> bytes:
    tiles = [label_image(Image.open(path), label, width=width) for label, path in samples]
    rows = (len(tiles) + columns - 1) // columns
    cell_height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (columns * width, rows * cell_height), "#0b0d12")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * width, (index // columns) * cell_height))
    output = io.BytesIO()
    sheet.save(output, format="JPEG", quality=91, optimize=True)
    return output.getvalue()


def lower_mouth_delta(first: Path, second: Path, bbox: list[float]) -> float:
    a = Image.open(first).convert("RGB")
    b = Image.open(second).convert("RGB")
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    left, right = int(x1 + 0.18 * width), int(x1 + 0.82 * width)
    top, bottom = int(y1 + 0.54 * height), int(y1 + 0.84 * height)
    box = (left, top, right, bottom)
    channel_means = ImageStat.Stat(ImageChops.difference(a.crop(box), b.crop(box))).mean
    return float(sum(channel_means) / len(channel_means))


def feature_crop(source: Path, bbox: list[float], feature: str) -> Image.Image:
    image = Image.open(source).convert("RGB")
    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1
    if feature == "eyes":
        box = (x1 + 0.02 * width, y1 + 0.08 * height, x2 - 0.02 * width, y1 + 0.48 * height)
    elif feature == "face":
        box = (x1 - 0.08 * width, y1 - 0.05 * height, x2 + 0.08 * width, y2 + 0.05 * height)
    else:
        raise ValueError(f"unsupported feature crop: {feature}")
    left = max(0, int(box[0])); top = max(0, int(box[1]))
    right = min(image.width, int(box[2])); bottom = min(image.height, int(box[3]))
    if right <= left or bottom <= top:
        raise ValueError(f"empty {feature} crop for bbox {bbox}")
    return image.crop((left, top, right, bottom))


def qwen_verdict(client: httpx.Client, sheet: bytes) -> dict[str, object]:
    prompt = """You are QA for one fixed talking-portrait identity. This labeled contact sheet uses the same source and silent audio in every cell. Return JSON only with keys: summary, checks, blocking_artifacts. For each non-neutral label, checks must contain {label, intended_change_visible, natural_and_subtle, identity_stable, mouth_stable_when_expected, confidence, note}. Check whether gaze-left and gaze-right visibly differ without changing the lips; blink is quick-looking and not crushed; amusement reads as a subtle smile; surprise, skepticism, concern, agreement, disagreement, interest, and thinking are distinguishable but not theatrical. The four head-yaw/head-pitch labels must show a genuine small 3D orientation change rather than flat translation or shear, while retaining a neutral mouth. Also say whether disagreement and thinking show opposite small head directions. Do not infer motion timing from this static sheet."""
    encoded = base64.b64encode(sheet).decode()
    headers = {"Content-Type": "application/json"}
    if VISION_API_KEY:
        headers["Authorization"] = f"Bearer {VISION_API_KEY}"
    response = client.post(
        f"{VISION_BASE_URL}/chat/completions",
        headers=headers,
        json={
            "model": VISION_MODEL,
            "temperature": 0.1,
            "max_tokens": 1800,
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ]}],
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    cleaned = str(content).strip().removeprefix("```json").removesuffix("```").strip()
    try:
        verdict = json.loads(cleaned)
    except json.JSONDecodeError:
        verdict = {"raw": content}
    return {
        "model": VISION_MODEL,
        "prompt": prompt,
        "contact_sheet_sha256": hashlib.sha256(sheet).hexdigest(),
        "response": verdict,
        "usage": payload.get("usage"),
    }


def qwen_focused_verdict(
    client: httpx.Client,
    gaze_sheet: bytes,
    head_sheet: bytes,
    yaw_sheet: bytes,
    pitch_sheet: bytes,
) -> dict[str, object]:
    prompt = """Perform four exact ordered portrait checks. Picture 1 is gaze-left, neutral, gaze-right: judge iris/pupil motion and lip stability, not head direction. Picture 2 is disagreement, neutral, thinking: judge their small opposed semantic face directions. Picture 3 is head-yaw-left, neutral, head-yaw-right. Picture 4 is head-pitch-up, neutral, head-pitch-down. For pictures 3 and 4 require genuine changes in 3D head orientation (perspective, feature geometry, and face plane), not flat translation, shear, or moving the whole crop. The mouth should retain the same closed neutral shape even though its image position can move with the head. Return JSON only with boolean keys gaze_axis_visible, gaze_mouth_stable, head_opposition_visible, yaw_orientation_visible, yaw_direction_opposed, pitch_orientation_visible, pitch_direction_opposed, identity_stable, mouth_stable, natural_and_subtle; numeric confidence; and a short evidence string. Judge only visible pixels."""
    headers = {"Content-Type": "application/json"}
    if VISION_API_KEY:
        headers["Authorization"] = f"Bearer {VISION_API_KEY}"
    images = [gaze_sheet, head_sheet, yaw_sheet, pitch_sheet]
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(image).decode()}"}}
        for image in images
    )
    response = client.post(
        f"{VISION_BASE_URL}/chat/completions",
        headers=headers,
        json={
            "model": VISION_MODEL,
            "temperature": 0.1,
            "max_tokens": 500,
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
            "messages": [{"role": "user", "content": content}],
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    answer = payload["choices"][0]["message"]["content"]
    cleaned = str(answer).strip().removeprefix("```json").removesuffix("```").strip()
    try:
        verdict = json.loads(cleaned)
    except json.JSONDecodeError:
        verdict = {"raw": answer}
    return {
        "model": VISION_MODEL,
        "prompt": prompt,
        "gaze_sheet_sha256": hashlib.sha256(gaze_sheet).hexdigest(),
        "head_sheet_sha256": hashlib.sha256(head_sheet).hexdigest(),
        "yaw_sheet_sha256": hashlib.sha256(yaw_sheet).hexdigest(),
        "pitch_sheet_sha256": hashlib.sha256(pitch_sheet).hexdigest(),
        "response": verdict,
        "usage": payload.get("usage"),
    }


def run() -> dict[str, object]:
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    output_root = EVIDENCE_ROOT / run_id
    output_root.mkdir(parents=True, exist_ok=False)
    report: dict[str, object] = {
        "schema": "musetalk-startup-expression-qa/v1",
        "started_at": started.isoformat(),
        "portrait": str(PORTRAIT),
        "portrait_sha256": None,
        "status": "running",
        "warnings": [],
    }
    try:
        if not PORTRAIT.is_file():
            raise FileNotFoundError(f"default QA portrait is missing: {PORTRAIT}")
        portrait_payload = PORTRAIT.read_bytes()
        digest = hashlib.sha256(portrait_payload).hexdigest()
        report["portrait_sha256"] = digest
        client = httpx.Client(timeout=httpx.Timeout(TIMEOUT, connect=10))
        health_response = client.get(f"{ALP_URL}/healthz")
        health_response.raise_for_status()
        alp_identity = health_response.json()
        if not alp_identity.get("ok") or not alp_identity.get("motion_profile"):
            raise RuntimeError(f"ALP identity is incomplete: {alp_identity}")
        report["alp_identity"] = alp_identity
        identity_payload = json.dumps(
            {
                "motion_profile": alp_identity.get("motion_profile"),
                "upstream_revision": alp_identity.get("upstream_revision"),
            },
            sort_keys=True,
        ).encode()
        identity_tag = hashlib.sha256(identity_payload).hexdigest()[:10]

        request_files = {"portrait": (PORTRAIT.name, portrait_payload, "image/png")}
        data = {"fps": str(FPS), "frames": "28", "gesture": "smile"}
        archive_response = client.post(f"{ALP_URL}/animate", files=request_files, data={**data, "output": "frames"})
        archive_response.raise_for_status()
        alp_root = output_root / "alp-bank"
        alp_root.mkdir()
        with zipfile.ZipFile(io.BytesIO(archive_response.content)) as archive:
            members = sorted(archive.namelist())
            if len(members) != 28:
                raise RuntimeError(f"ALP returned {len(members)} frames instead of 28")
            for index, label in enumerate(ALP_BANK, start=8):
                atomic_write(alp_root / f"{label}.jpg", archive.read(members[index]))

        video_response = client.post(f"{ALP_URL}/animate", files=request_files, data={**data, "output": "mp4"})
        video_response.raise_for_status()
        motion_path = IO_ROOT / "web_demo" / "motion" / f"startup-qa-{digest[:20]}-{identity_tag}.mp4"
        atomic_write(motion_path, video_response.content)
        audio_path = output_root / "silence.wav"
        atomic_write(audio_path, silence_wav(1.25))
        avatar_id = f"startup-qa-{digest[:20]}-{identity_tag}"
        token = uuid.uuid4().hex[:10]
        submit_job(
            f"startup-qa-prepare-{token}",
            {"video": str(motion_path), "avatar_id": avatar_id, "bbox_shift": 0,
             "source_start_frame": 0, "bank_start_frame": 8, "prepare_only": True},
        )

        final_root = output_root / "final"
        final_root.mkdir()
        samples: dict[str, Path] = {}
        metrics: dict[str, dict[str, float] | None] = {}
        face_bbox: list[float] | None = None
        for label, reaction in FINAL_CONTROLS.items():
            stream_dir = final_root / label
            stream_dir.mkdir()
            submit_job(
                f"startup-qa-{label}-{token}",
                {"video": str(motion_path), "audio": str(audio_path), "stream_dir": str(stream_dir),
                 "avatar_id": avatar_id, "bbox_shift": 0, "source_start_frame": 0,
                 "bank_start_frame": 8, "fps": FPS, "batch_size": 4,
                 "stream_format": "jpg", "reaction": reaction},
            )
            frames = sorted(stream_dir.glob("*.jpg"))
            if not frames:
                raise RuntimeError(f"{label} produced no frames")
            sample_index = 7 if label == "blink" else 8
            sample = frames[min(sample_index, len(frames) - 1)]
            samples[label] = sample
            metrics[label] = observe(client, sample)
            if face_bbox is None:
                candidate = json.loads((stream_dir / "face_bbox.json").read_text())
                if isinstance(candidate, list) and len(candidate) == 4:
                    face_bbox = [float(value) for value in candidate]

        ordered_samples = [(label, samples[label]) for label in FINAL_CONTROLS]
        sheet = contact_sheet(ordered_samples)
        atomic_write(output_root / "contact-sheet.jpg", sheet)
        pair_root = output_root / "pairs"
        pair_root.mkdir()
        for label, sample in ordered_samples[1:]:
            atomic_write(pair_root / f"neutral-vs-{label}.jpg", contact_sheet([("neutral", samples["neutral"]), (label, sample)], columns=2))
        if face_bbox is None:
            raise RuntimeError("MuseTalk did not return a face bbox for focused QA")
        focused_paths: dict[str, Path] = {}
        for label in ("gaze-left", "neutral", "gaze-right"):
            focused_paths[f"eyes-{label}"] = pair_root / f"eyes-{label}.jpg"
            feature_crop(samples[label], face_bbox, "eyes").save(focused_paths[f"eyes-{label}"], quality=95)
        for label in ("disagreement", "neutral", "thinking"):
            focused_paths[f"face-{label}"] = pair_root / f"face-{label}.jpg"
            feature_crop(samples[label], face_bbox, "face").save(focused_paths[f"face-{label}"], quality=95)
        for label in ("head-yaw-left", "head-yaw-right", "head-pitch-up", "head-pitch-down"):
            focused_paths[f"face-{label}"] = pair_root / f"face-{label}.jpg"
            feature_crop(samples[label], face_bbox, "face").save(focused_paths[f"face-{label}"], quality=95)
        gaze_sheet = contact_sheet(
            [("gaze-left eyes", focused_paths["eyes-gaze-left"]), ("neutral eyes", focused_paths["eyes-neutral"]), ("gaze-right eyes", focused_paths["eyes-gaze-right"])],
            columns=3,
            width=480,
        )
        head_sheet = contact_sheet(
            [("disagreement face", focused_paths["face-disagreement"]), ("neutral face", focused_paths["face-neutral"]), ("thinking face", focused_paths["face-thinking"])],
            columns=3,
            width=480,
        )
        yaw_sheet = contact_sheet(
            [("head-yaw-left", focused_paths["face-head-yaw-left"]),
             ("neutral", focused_paths["face-neutral"]),
             ("head-yaw-right", focused_paths["face-head-yaw-right"])],
            columns=3,
            width=480,
        )
        pitch_sheet = contact_sheet(
            [("head-pitch-up", focused_paths["face-head-pitch-up"]),
             ("neutral", focused_paths["face-neutral"]),
             ("head-pitch-down", focused_paths["face-head-pitch-down"])],
            columns=3,
            width=480,
        )
        atomic_write(pair_root / "gaze-axis.jpg", gaze_sheet)
        atomic_write(pair_root / "head-axis.jpg", head_sheet)
        atomic_write(pair_root / "head-yaw-axis.jpg", yaw_sheet)
        atomic_write(pair_root / "head-pitch-axis.jpg", pitch_sheet)

        checks: dict[str, object] = {"face_detected": all(value is not None for value in metrics.values())}
        if face_bbox is not None:
            checks["gaze_left_mouth_pixel_delta"] = lower_mouth_delta(samples["neutral"], samples["gaze-left"], face_bbox)
            checks["gaze_right_mouth_pixel_delta"] = lower_mouth_delta(samples["neutral"], samples["gaze-right"], face_bbox)
            checks["gaze_mouth_stable"] = max(
                float(checks["gaze_left_mouth_pixel_delta"]),
                float(checks["gaze_right_mouth_pixel_delta"]),
            ) <= 3.0
        neutral = metrics.get("neutral") or {}
        left = metrics.get("gaze-left") or {}
        right = metrics.get("gaze-right") or {}
        blink = metrics.get("blink") or {}
        checks["gaze_separation"] = abs(float(left.get("gazeX", 0)) - float(right.get("gazeX", 0)))
        checks["gaze_distinct"] = float(checks["gaze_separation"]) >= 0.08
        neutral_eye = (float(neutral.get("leftEyeOpen", 0)) + float(neutral.get("rightEyeOpen", 0))) / 2
        blink_eye = (float(blink.get("leftEyeOpen", 0)) + float(blink.get("rightEyeOpen", 0))) / 2
        checks["blink_eye_open_delta"] = blink_eye - neutral_eye
        checks["blink_closes_both_eyes"] = blink_eye < neutral_eye - 0.015
        checks["head_direction_separation"] = float((metrics.get("thinking") or {}).get("yaw", 0)) - float((metrics.get("disagreement") or {}).get("yaw", 0))
        checks["opposite_head_directions_visible"] = abs(float(checks["head_direction_separation"])) >= 0.01
        yaw_left = metrics.get("head-yaw-left") or {}
        yaw_right = metrics.get("head-yaw-right") or {}
        pitch_up = metrics.get("head-pitch-up") or {}
        pitch_down = metrics.get("head-pitch-down") or {}
        checks["head_yaw_separation"] = float(yaw_right.get("yaw", 0)) - float(yaw_left.get("yaw", 0))
        checks["head_yaw_distinct"] = abs(float(checks["head_yaw_separation"])) >= 0.012
        checks["head_pitch_separation"] = float(pitch_down.get("pitch", 0)) - float(pitch_up.get("pitch", 0))
        checks["head_pitch_distinct"] = abs(float(checks["head_pitch_separation"])) >= 0.01
        neutral_mouth = float(neutral.get("mouth", 0))
        pose_mouth_deltas = {
            label: abs(float((metrics.get(label) or {}).get("mouth", 0)) - neutral_mouth)
            for label in ("head-yaw-left", "head-yaw-right", "head-pitch-up", "head-pitch-down")
        }
        checks["head_pose_mouth_deltas"] = pose_mouth_deltas
        checks["head_pose_mouth_stable"] = max(pose_mouth_deltas.values()) <= 0.025
        report["metrics"] = metrics
        report["checks"] = checks
        visual_blockers: list[object] = []
        visual_misses: list[str] = []
        try:
            report["qwen"] = qwen_verdict(client, sheet)
            visual_response = report["qwen"].get("response", {})
            if isinstance(visual_response, dict):
                candidate_blockers = visual_response.get("blocking_artifacts", [])
                if isinstance(candidate_blockers, list):
                    visual_blockers = candidate_blockers
                candidate_checks = visual_response.get("checks", [])
                if isinstance(candidate_checks, list):
                    visual_misses = [
                        str(check.get("label", "unknown"))
                        for check in candidate_checks
                        if isinstance(check, dict) and check.get("intended_change_visible") is False
                    ]
            report["qwen_focused"] = qwen_focused_verdict(
                client, gaze_sheet, head_sheet, yaw_sheet, pitch_sheet
            )
            focused = report["qwen_focused"].get("response", {})
            if isinstance(focused, dict):
                if focused.get("gaze_axis_visible") is True:
                    visual_misses = [label for label in visual_misses if label not in {"gaze-left", "gaze-right"}]
                    visual_blockers = [item for item in visual_blockers if "gaze" not in str(item).lower()]
                elif focused.get("gaze_axis_visible") is False:
                    visual_misses.extend(label for label in ("gaze-left", "gaze-right") if label not in visual_misses)
                if focused.get("head_opposition_visible") is False:
                    visual_misses.append("head-direction")
                elif focused.get("head_opposition_visible") is True:
                    visual_misses = [
                        label for label in visual_misses
                        if label not in {"disagreement", "thinking", "head-direction"}
                    ]
                    visual_blockers = [
                        item for item in visual_blockers
                        if not any(word in str(item).lower() for word in ("disagreement", "thinking"))
                    ]
                if focused.get("yaw_orientation_visible") is False or focused.get("yaw_direction_opposed") is False:
                    visual_misses.append("head-yaw")
                elif focused.get("yaw_orientation_visible") is True and focused.get("yaw_direction_opposed") is True:
                    visual_misses = [
                        label for label in visual_misses
                        if label not in {"head-yaw-left", "head-yaw-right", "head-yaw"}
                    ]
                    visual_blockers = [item for item in visual_blockers if "head-yaw" not in str(item).lower()]
                if focused.get("pitch_orientation_visible") is False or focused.get("pitch_direction_opposed") is False:
                    visual_misses.append("head-pitch")
                elif focused.get("pitch_orientation_visible") is True and focused.get("pitch_direction_opposed") is True:
                    visual_misses = [
                        label for label in visual_misses
                        if label not in {"head-pitch-up", "head-pitch-down", "head-pitch"}
                    ]
                    visual_blockers = [item for item in visual_blockers if "head-pitch" not in str(item).lower()]
                if focused.get("mouth_stable") is False:
                    visual_misses.append("head-pose-mouth")
        except Exception as exc:  # The live system is deliberately independent of its judge.
            report["warnings"].append(f"Qwen visual QA unavailable: {type(exc).__name__}: {exc}")
        structural = (
            checks["face_detected"]
            and checks.get("gaze_mouth_stable", False)
            and checks.get("head_pose_mouth_stable", False)
            and checks.get("head_yaw_distinct", False)
            and checks.get("head_pitch_distinct", False)
        )
        report["structural_status"] = "pass" if structural else "fail"
        report["visual_status"] = "warn" if visual_blockers or visual_misses else "pass"
        report["visual_misses"] = visual_misses
        report["status"] = "fail" if not structural else report["visual_status"]
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write(output_root / "report.json", json.dumps(report, indent=2, sort_keys=True).encode())
    atomic_write(EVIDENCE_ROOT / "latest.json", json.dumps({"run": run_id, "report": str(output_root / "report.json")}, indent=2).encode())
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    run()
