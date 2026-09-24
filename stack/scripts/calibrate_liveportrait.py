#!/usr/bin/env python3
"""Build a measured ALP-slider -> canonical-face calibration set.

Rendered frames are deliberately written to the caller-selected output
directory. Keep that directory in runtime storage; it is evidence/training data,
not source.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from avatar.controls.schema import BLENDSHAPE_NAMES, AvatarSequence
from avatar.controls.calibration import CONTROL_RANGES, control_grid, random_controls
from avatar.extractors.mediapipe_face import FaceLandmarkerExtractor
from avatar.render.alp_client import render_alp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("portrait", type=Path)
    model_path = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL")
    parser.add_argument("--model", type=Path, default=Path(model_path) if model_path else None)
    parser.add_argument("--alp-url", default=os.environ.get("ADVANCED_LIVE_PORTRAIT_URL", "http://127.0.0.1:8093"))
    parser.add_argument("--output", type=Path, default=Path("alp-calibration"))
    parser.add_argument("--points", type=int, default=5, help="values per one-dimensional sweep")
    parser.add_argument("--random-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--only", action="append", choices=tuple(CONTROL_RANGES),
                        help="limit the sweep; repeat for multiple controls (useful for smoke tests)")
    parser.add_argument("--keep-frames", action="store_true", help="retain rendered PNGs alongside records")
    args = parser.parse_args()
    if args.model is None:
        parser.error("--model or MEDIAPIPE_FACE_LANDMARKER_MODEL is required")
    if not args.portrait.is_file():
        parser.error(f"portrait does not exist: {args.portrait}")

    controls = control_grid(args.points, args.only) + random_controls(args.random_samples, args.seed)
    portrait = args.portrait.read_bytes()
    args.output.mkdir(parents=True, exist_ok=True)
    frame_root = args.output / "frames"
    if args.keep_frames:
        frame_root.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []
    timestamps: list[float] = []
    blendshapes: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    confidences: list[float] = []
    transforms: list[np.ndarray] = []
    neutral_scores: list[float] = []
    with urllib.request.urlopen(args.alp_url.rstrip("/") + "/healthz", timeout=10.0) as health:
        health.read()
    with FaceLandmarkerExtractor(args.model) as extractor:
        for index, requested in enumerate(controls):
            payload = render_alp(args.alp_url, portrait, requested)
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"ALP returned an undecodable image at sample {index}")
            # `index` is only an arbitrary strictly-increasing sort key so
            # AvatarSequence's monotonic-timestamp invariant holds -- these
            # samples are independently requested slider settings, not
            # consecutive frames of one clip, so they carry no real temporal
            # spacing. See the `temporal_sequence: false` metadata flag below
            # and do not pass `include_derived=True` here: velocity/
            # acceleration computed across unrelated samples (e.g. across a
            # sweep boundary) would be fictitious motion, not measured motion.
            frame = extractor.detect(image, index / 25.0, strict=True)
            timestamps.append(frame.timestamp)
            blendshapes.append(frame.blendshapes)
            rotations.append(frame.head_rotation)
            translations.append(frame.head_translation)
            confidences.append(frame.confidence)
            transforms.append(frame.head_transform)
            neutral_scores.append(frame.neutral_score)
            records.append({
                "index": index, "controls": requested, "timestamp": frame.timestamp,
                "blendshapes": frame.blendshapes.tolist(),
                "head_rotation": frame.head_rotation.tolist(),
                "head_translation": frame.head_translation.tolist(),
                "head_transform": frame.head_transform.tolist() if frame.head_transform is not None else None,
                "confidence": frame.confidence,
                "neutral_score": frame.neutral_score,
            })
            if args.keep_frames:
                (frame_root / f"{index:06d}.png").write_bytes(payload)

    sequence = AvatarSequence(
        timestamps=np.asarray(timestamps), blendshapes=np.stack(blendshapes),
        head_rotation=np.stack(rotations), head_translation=np.stack(translations),
        confidence=np.asarray(confidences),
        head_transform=np.stack(transforms) if all(t is not None for t in transforms) else None,
        neutral_score=np.asarray(neutral_scores, dtype=np.float32) if all(s is not None for s in neutral_scores) else None,
    )
    sequence.save(
        args.output,
        metadata={
            "source": str(args.portrait), "renderer": "advanced-live-portrait",
            "renderer_url": args.alp_url, "sample_rate_hz": 25.0,
            "sample_count": len(records), "seed": args.seed,
            "sweep_points": args.points, "random_samples": args.random_samples,
            "control_names": list(CONTROL_RANGES), "blendshape_names": list(BLENDSHAPE_NAMES),
            # These are independent slider samples, not a temporal sequence:
            # consecutive rows can be unrelated control settings. Consumers
            # must not compute or trust motion derivatives across them.
            "temporal_sequence": False,
            "confidence_semantics": "binary_detection_validity",
            "unmeasured_blendshapes": sorted(FaceLandmarkerExtractor.UNSUPPORTED_BLENDSHAPES),
        },
        include_derived=False,
    )
    (args.output / "samples.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8",
    )
    print(f"samples={len(records)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
