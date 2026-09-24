#!/usr/bin/env python3
"""Measure ALP controllability and face-region cross-coupling (#30).

For each identity portrait: render repeated neutral baselines (noise floor),
each of the 12 controls at its six named points (avatar.controls.calibration.
named_grid), and bounded multi-control combinations held out for later
adapter evaluation (#31). Every render is extracted through the canonical
MediaPipe FaceLandmarkerExtractor (#28) -- raw transforms, neutral score, and
confidence included, never just a derived summary metric. Rendered or
detection failures are recorded, not silently dropped.

Writes, per identity, under <output>/<identity_hash>/:
  controls.npz, metadata.json  -- AvatarSequence.save(), temporal_sequence=false
  samples.jsonl                -- one record per attempted sample, incl. failures
  contact_sheets/<control>.jpg -- six-panel named-point strip (if PIL available)

And one consolidated <output>/report.json across all identities: per-identity
and per-control noise floor, eye-to-mouth / blink-to-brow leakage, and a
per-identity + cross-identity-conservative safe envelope. Optionally, one
Qwen visual review per control per identity if LLM_BASE_URL/LLM_MODEL are set
(best-effort: numeric analysis is the required deliverable, this supplements
it -- a Qwen failure is recorded and does not abort the run).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from avatar.controls.calibration import (
    CONTROL_RANGES,
    bounded_combined_controls,
    named_grid,
    neutral_controls,
)
from avatar.controls.controllability import (
    blink_to_brow_leakage,
    channel_deltas,
    eye_to_mouth_leakage,
    noise_floor,
    region_leakage,
    safe_envelope,
    top_response_channels,
    BROW_INDICES,
    CROSS_COUPLING_REGIONS,
    MOUTH_INDICES,
)
from avatar.controls.schema import BLENDSHAPE_NAMES, AvatarFrame, AvatarSequence
from avatar.extractors.mediapipe_face import FaceLandmarkerExtractor
from avatar.render.alp_client import render_alp

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - PIL is present in musetalk-feedback
    Image = None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _label_tile(png: bytes, label: str, width: int = 220) -> "Image.Image":
    image = Image.open(io.BytesIO(png)).convert("RGB")
    height = int(image.height * width / image.width)
    image = image.resize((width, height))
    canvas = Image.new("RGB", (width, height + 24), (21, 25, 34))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, height + 4), label, fill=(230, 230, 230), font=ImageFont.load_default())
    return canvas


def _contact_sheet(tiles: list["Image.Image"]) -> bytes:
    width = sum(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
    sheet = Image.new("RGB", (width, height), (11, 13, 18))
    x = 0
    for tile in tiles:
        sheet.paste(tile, (x, 0))
        x += tile.width
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=88)
    return buffer.getvalue()


def _qwen_review(base_url: str, model: str, api_key: str, prompt: str, jpeg: bytes) -> dict[str, object]:
    body = json.dumps({
        "model": model,
        "temperature": 0.1,
        "max_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}"
                }},
            ],
        }],
    }).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=60.0) as response:
        payload = json.loads(response.read())
    content = payload["choices"][0]["message"]["content"]
    match = re.search(r"```json\s*(.*?)```", content, re.DOTALL)
    text = match.group(1) if match else content
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": content}


def _collect_identity(
    portrait_path: Path,
    alp_url: str,
    extractor: FaceLandmarkerExtractor,
    neutral_repeats: int,
    combined_samples: int,
    seed: int,
    output_dir: Path,
    qwen: tuple[str, str, str] | None,
    only: list[str] | None = None,
) -> dict[str, object]:
    portrait = portrait_path.read_bytes()
    identity_hash = _sha256(portrait)[:12]
    identity_dir = output_dir / identity_hash
    contact_dir = identity_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    plan: list[dict[str, object]] = []
    for repeat in range(neutral_repeats):
        plan.append({"control": None, "point": f"repeat_{repeat}", "requested": neutral_controls()})
    plan.extend(named_grid(only))
    for index, requested in enumerate(bounded_combined_controls(combined_samples, seed=seed)):
        plan.append({"control": "combined", "point": f"combined_{index}", "requested": requested})

    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    frames: list[AvatarFrame] = []
    tiles_by_control: dict[str, list[tuple[str, bytes]]] = {}

    for order, entry in enumerate(plan):
        requested = entry["requested"]
        try:
            png = render_alp(alp_url, portrait, requested)
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            failures.append({"control": entry["control"], "point": entry["point"], "reason": "render_error", "detail": str(exc)})
            continue
        image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            failures.append({"control": entry["control"], "point": entry["point"], "reason": "undecodable_frame"})
            continue
        frame = extractor.detect(image, order / 25.0, strict=False)
        if frame.confidence == 0.0:
            failures.append({"control": entry["control"], "point": entry["point"], "reason": "no_face_detected"})
        frames.append(frame)
        records.append({
            "control": entry["control"], "point": entry["point"], "requested": requested,
            "confidence": frame.confidence, "neutral_score": frame.neutral_score,
            "blendshapes": frame.blendshapes.tolist(), "head_rotation": frame.head_rotation.tolist(),
            "head_translation": frame.head_translation.tolist(),
        })
        if entry["control"] not in (None, "combined") and Image is not None:
            tiles_by_control.setdefault(str(entry["control"]), []).append((str(entry["point"]), png))

    if not frames:
        raise RuntimeError(f"no successful renders for {portrait_path}")
    sequence = AvatarSequence.from_frames(frames)
    sequence.save(
        identity_dir,
        metadata={
            "source": str(portrait_path), "portrait_sha256": _sha256(portrait),
            "renderer": "advanced-live-portrait", "renderer_url": alp_url,
            "temporal_sequence": False, "confidence_semantics": "binary_detection_validity",
            "unmeasured_blendshapes": sorted(FaceLandmarkerExtractor.UNSUPPORTED_BLENDSHAPES),
            "neutral_repeats": neutral_repeats, "combined_samples": combined_samples, "seed": seed,
        },
        include_derived=False,
    )
    (identity_dir / "samples.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8",
    )
    (identity_dir / "failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    qwen_reviews: dict[str, object] = {}
    if Image is not None:
        for control, tiles in tiles_by_control.items():
            point_order = {"min": 0, "q1": 1, "default": 2, "mid": 3, "q3": 4, "max": 5}
            ordered = sorted(tiles, key=lambda item: point_order.get(item[0], 99))
            sheet = _contact_sheet([_label_tile(png, point) for point, png in ordered])
            (contact_dir / f"{control}.jpg").write_bytes(sheet)
            if qwen is not None:
                base_url, model, api_key = qwen
                try:
                    qwen_reviews[control] = _qwen_review(
                        base_url, model, api_key,
                        f"This strip shows the '{control}' AdvancedLivePortrait control swept across "
                        "min, q1, default, mid, q3, max (left to right) on the same identity. Reply as "
                        'JSON: {"progression_smooth": bool, "identity_stable": bool, '
                        '"mouth_leak_visible": bool, "notes": str}.',
                        sheet,
                    )
                except Exception as exc:  # noqa: BLE001 - best-effort visual QA must never abort the run
                    qwen_reviews[control] = {"error": str(exc)}

    return {
        "identity_hash": identity_hash, "portrait": str(portrait_path),
        "frames": frames, "records": records, "failures": failures, "qwen_reviews": qwen_reviews,
    }


def _identity_report(identity: dict[str, object], threshold_multiplier: float, only: list[str] | None = None) -> dict[str, object]:
    frames: list[AvatarFrame] = identity["frames"]
    records: list[dict[str, object]] = identity["records"]
    repeat_indices = [i for i, r in enumerate(records) if r["point"].startswith("repeat_") and r["confidence"] == 1.0]
    if len(repeat_indices) < 2:
        return {"identity_hash": identity["identity_hash"], "error": "insufficient neutral repeats to estimate noise floor"}
    repeat_blendshapes = np.stack([frames[i].blendshapes for i in repeat_indices])
    repeat_rotation = np.stack([frames[i].head_rotation for i in repeat_indices])
    noise = noise_floor(repeat_blendshapes, repeat_rotation)
    baseline = np.mean(repeat_blendshapes, axis=0)

    per_control: dict[str, object] = {}
    for control in (only if only is not None else list(CONTROL_RANGES)):
        # (point label, requested slider value, measured blendshapes) for
        # every successfully detected render of this control.
        control_records = [
            (r["point"], r["requested"][control], np.asarray(r["blendshapes"], dtype=np.float32))
            for r in records if r["control"] == control and r["confidence"] == 1.0
        ]
        if not control_records:
            per_control[control] = {"error": "no successful renders for this control"}
            continue
        deltas_by_point = {point: channel_deltas(baseline, blendshapes) for point, _, blendshapes in control_records}
        values_by_point = {point: value for point, value, _ in control_records}
        low, high = CONTROL_RANGES[control]
        mouth = {point: region_leakage(deltas, noise, MOUTH_INDICES, threshold_multiplier)
                 for point, deltas in deltas_by_point.items()}
        brow = {point: region_leakage(deltas, noise, BROW_INDICES, threshold_multiplier)
                for point, deltas in deltas_by_point.items()}
        # Only the regions NOT this control's own intended target count
        # toward its own safe envelope -- e.g. eyebrow moving brow channels
        # is its job, not leakage; see CROSS_COUPLING_REGIONS's docstring.
        cross_coupling_regions = CROSS_COUPLING_REGIONS.get(control, ("mouth", "brow"))
        region_reports = {"mouth": mouth, "brow": brow}
        ordered_points = [p for p in ("min", "q1", "default", "mid", "q3", "max") if p in deltas_by_point]
        # Magnitude-based severity per point: the largest single-channel
        # cross-coupling delta among this control's non-home regions. A
        # channel-count criterion turned out uninformative here -- almost
        # every control shows *some* nonzero cross-region delta by q1/q3, so
        # a zero-tolerance count collapses every control to the same [0, 0]
        # envelope. Reporting three magnitude tolerance tiers differentiates
        # "barely measurable" from "a clearly visible cross-region change."
        severity = {
            p: max((region_reports[region][p]["max_abs_delta"] for region in cross_coupling_regions), default=0.0)
            for p in ordered_points
        }
        envelopes = {}
        for tier, tolerance in (("negligible", 0.05), ("mild", 0.15), ("severe", 0.30)):
            envelopes[tier] = safe_envelope(
                [values_by_point[p] for p in ordered_points],
                [1 if severity[p] > tolerance else 0 for p in ordered_points],
                default_value=0.0, max_leaking_channels=0,
            ) if ordered_points else (0.0, 0.0)
        max_point = max(deltas_by_point, key=lambda p: float(np.max(np.abs(deltas_by_point[p]))))
        per_control[control] = {
            "range": [low, high], "mouth_leakage_by_point": mouth, "brow_leakage_by_point": brow,
            "cross_coupling_regions": list(cross_coupling_regions),
            "eye_to_mouth_leakage": eye_to_mouth_leakage(control, deltas_by_point, noise, threshold_multiplier),
            "safe_envelope_by_tolerance": {tier: list(bounds) for tier, bounds in envelopes.items()},
            "top_response_channels_at_strongest_point": {
                "point": max_point, "channels": top_response_channels(deltas_by_point[max_point]),
            },
        }

    blink_records = [(r["point"], np.asarray(r["blendshapes"], dtype=np.float32))
                     for r in records if r["control"] == "blink" and r["confidence"] == 1.0]
    blink_deltas = {point: channel_deltas(baseline, blendshapes) for point, blendshapes in blink_records}

    return {
        "identity_hash": identity["identity_hash"],
        "noise_floor": {"blendshape_std_max": float(np.max(noise.blendshape_std)), "sample_count": noise.sample_count},
        "per_control": per_control,
        "blink_to_brow_leakage": blink_to_brow_leakage(blink_deltas, noise, threshold_multiplier) if blink_deltas else None,
        "failure_count": len(identity["failures"]),
        "qwen_reviews": identity["qwen_reviews"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("portraits", type=Path, nargs="+", help="at least 3 portrait images (distinct identities)")
    model_path = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL")
    parser.add_argument("--model", type=Path, default=Path(model_path) if model_path else None)
    parser.add_argument("--alp-url", default=os.environ.get("ADVANCED_LIVE_PORTRAIT_URL", "http://advanced-live-portrait:8093"))
    parser.add_argument("--output", type=Path, default=Path("controllability"))
    parser.add_argument("--neutral-repeats", type=int, default=6)
    parser.add_argument("--combined-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--threshold-multiplier", type=float, default=3.0)
    parser.add_argument("--qwen", action="store_true", help="also run one Qwen visual review per control per identity")
    parser.add_argument("--only", action="append", choices=tuple(CONTROL_RANGES),
                        help="limit the named-point sweep; repeat for multiple controls (smoke tests)")
    args = parser.parse_args()
    if args.model is None:
        parser.error("--model or MEDIAPIPE_FACE_LANDMARKER_MODEL is required")
    if len(args.portraits) < 3:
        parser.error("at least 3 distinct-identity portraits are required (#30's own acceptance criteria)")

    qwen = None
    if args.qwen:
        base_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
        model = os.environ.get("LLM_MODEL", "qwen27b")
        api_key = os.environ.get("LLM_API_KEY", "")
        qwen = (base_url, model, api_key)

    args.output.mkdir(parents=True, exist_ok=True)
    identity_reports = []
    with urllib.request.urlopen(args.alp_url.rstrip("/") + "/healthz", timeout=10.0) as health:
        health.read()
    with FaceLandmarkerExtractor(args.model) as extractor:
        for portrait_path in args.portraits:
            started = time.monotonic()
            identity = _collect_identity(
                portrait_path, args.alp_url, extractor, args.neutral_repeats,
                args.combined_samples, args.seed, args.output, qwen, args.only,
            )
            identity_reports.append(_identity_report(identity, args.threshold_multiplier, args.only))
            print(f"identity={identity['identity_hash']} samples={len(identity['records'])} "
                  f"failures={len(identity['failures'])} elapsed={time.monotonic() - started:.1f}s")

    # Cross-identity conservative envelope: intersection of every identity's
    # own per-control safe range, i.e. the widest span every measured
    # identity agrees is leak-free.
    cross_identity: dict[str, object] = {}
    for control in (args.only if args.only is not None else list(CONTROL_RANGES)):
        per_tier: dict[str, list[float]] = {}
        for tier in ("negligible", "mild", "severe"):
            bounds = [
                report["per_control"][control]["safe_envelope_by_tolerance"][tier]
                for report in identity_reports
                if "per_control" in report and "safe_envelope_by_tolerance" in report["per_control"].get(control, {})
            ]
            if bounds:
                per_tier[tier] = [max(b[0] for b in bounds), min(b[1] for b in bounds)]
        if per_tier:
            cross_identity[control] = per_tier

    report = {
        "schema": "musetalk-controllability-report/v1",
        "identities": identity_reports,
        "cross_identity_safe_envelope": cross_identity,
        "threshold_multiplier": args.threshold_multiplier,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    print(f"wrote {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
