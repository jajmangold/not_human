#!/usr/bin/env python3
"""Extract canonical MediaPipe face controls from an image or video."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from avatar.extractors.mediapipe_face import FaceLandmarkerExtractor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="portrait image or video")
    parser.add_argument("--model", type=Path, default=Path(os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL", "")), help="face_landmarker.task path")
    parser.add_argument("--output", type=Path, default=Path("controls"), help="directory for controls.npz and metadata.json")
    parser.add_argument("--fps", type=float, default=None, help="video sampling rate; defaults to the source rate")
    parser.add_argument("--strict", action="store_true", help="fail on a video frame without a face")
    parser.add_argument("--derived", action="store_true", help="also store blendshape velocity and acceleration")
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or MEDIAPIPE_FACE_LANDMARKER_MODEL is required")
    with FaceLandmarkerExtractor(args.model) as extractor:
        sequence = extractor.extract(args.input, fps=args.fps, strict=args.strict)
    controls_path, metadata_path = sequence.save(
        args.output,
        metadata={
            "source": str(args.input), "source_fps": args.fps,
            # A still image is a single frame with no motion to speak of; a
            # video's consecutive frames are genuinely temporally spaced, so
            # derived velocity/acceleration (when requested) are real.
            "temporal_sequence": sequence.frame_count > 1,
            "confidence_semantics": "binary_detection_validity",
            "unmeasured_blendshapes": sorted(FaceLandmarkerExtractor.UNSUPPORTED_BLENDSHAPES),
        },
        include_derived=args.derived,
    )
    print(f"frames={sequence.frame_count} duration={sequence.duration:.3f}s")
    print(controls_path)
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
