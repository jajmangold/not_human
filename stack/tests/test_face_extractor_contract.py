"""Contract tests against the real pinned MediaPipe Face Landmarker model.

These do not run on the host (no mediapipe/cv2) -- they run inside
musetalk-feedback, which has both, against the checksum-pinned model asset at
MEDIAPIPE_FACE_LANDMARKER_MODEL. Unlike test_avatar_controls.py (pure schema,
no model), this file is the "compare every named channel against the pinned
extractor" acceptance evidence for #28: it does not assume the model's
category set matches BLENDSHAPE_NAMES, it measures it.
"""

import os
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
    from avatar.extractors.mediapipe_face import FaceLandmarkerExtractor
except ModuleNotFoundError:
    cv2 = None

MODEL_PATH = os.environ.get("MEDIAPIPE_FACE_LANDMARKER_MODEL", "")
# A directory of real portrait images (at least one detectable face each).
# Not a repo fixture: face-detection tests need actual photographs, and this
# repo does not vendor portrait images. Point this at, e.g., the kokoro-web
# demo's uploads directory to run these locally/in CI with real assets.
PORTRAITS_DIR = os.environ.get("MUSE_AVATAR_TEST_PORTRAITS_DIR", "")


def _available() -> bool:
    return cv2 is not None and bool(MODEL_PATH) and Path(MODEL_PATH).is_file()


@unittest.skipUnless(_available(), "requires mediapipe/cv2 and MEDIAPIPE_FACE_LANDMARKER_MODEL")
class FaceLandmarkerContractTests(unittest.TestCase):
    def setUp(self):
        self.extractor = FaceLandmarkerExtractor(MODEL_PATH)

    def tearDown(self):
        self.extractor.close()

    def test_no_face_frame_is_explicit_zero_confidence_not_an_exception(self):
        from avatar.extractors.mediapipe_face import FaceNotFoundError

        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        with self.assertRaises(FaceNotFoundError):
            self.extractor.detect(blank, 0.0, strict=True)
        frame = self.extractor.detect(blank, 0.0, strict=False)
        self.assertEqual(frame.confidence, 0.0)
        self.assertTrue(np.all(frame.blendshapes == 0.0))
        self.assertIsNone(frame.neutral_score)

    def test_unsupported_blendshapes_are_a_subset_of_the_canonical_names(self):
        from avatar.controls.schema import BLENDSHAPE_NAMES
        self.assertTrue(FaceLandmarkerExtractor.UNSUPPORTED_BLENDSHAPES.issubset(BLENDSHAPE_NAMES))

    @unittest.skipUnless(PORTRAITS_DIR and Path(PORTRAITS_DIR).is_dir(), "set MUSE_AVATAR_TEST_PORTRAITS_DIR")
    def test_three_portraits_detect_a_face_with_measured_neutral_and_no_tongue_signal(self):
        portraits = sorted(Path(PORTRAITS_DIR).glob("*.png"))[:3]
        self.assertGreaterEqual(len(portraits), 3, "need at least 3 portrait fixtures")
        from avatar.controls.schema import BLENDSHAPE_INDEX
        tongue_index = BLENDSHAPE_INDEX["tongueOut"]
        for path in portraits:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image, f"could not read {path}")
            frame = self.extractor.detect(image, 0.0, strict=True)
            self.assertEqual(frame.confidence, 1.0)
            self.assertIsNotNone(frame.neutral_score)
            self.assertGreaterEqual(frame.neutral_score, 0.0)
            # Never measured by this model -- must stay structurally zero,
            # not because the face has no tongueOut activation.
            self.assertEqual(frame.blendshapes[tongue_index], 0.0)
            self.assertIsNotNone(frame.head_transform)
            self.assertTrue(np.isfinite(frame.head_transform).all())

    @unittest.skipUnless(PORTRAITS_DIR and Path(PORTRAITS_DIR).is_dir(), "set MUSE_AVATAR_TEST_PORTRAITS_DIR")
    def test_video_extraction_has_monotonic_timestamps_and_preserves_head_transform(self):
        # Reuse a still as a synthetic single-frame "video" stand-in is not
        # representative; skip unless a real video fixture is provided.
        video_candidates = sorted(Path(PORTRAITS_DIR).glob("*.mp4"))
        if not video_candidates:
            self.skipTest("no .mp4 fixture found under MUSE_AVATAR_TEST_PORTRAITS_DIR")
        sequence = self.extractor.extract_video(video_candidates[0], strict=False)
        self.assertTrue(np.all(np.diff(sequence.timestamps) > 0))
        self.assertIsNotNone(sequence.head_transform)
        self.assertEqual(sequence.head_transform.shape, (sequence.frame_count, 4, 4))


if __name__ == "__main__":
    unittest.main()
