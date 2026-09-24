import unittest
from pathlib import Path

from advanced_live_portrait.face_validation import NoUsableFaceError, require_usable_face


ROOT = Path(__file__).resolve().parents[1]


class FaceValidationTests(unittest.TestCase):
    def test_accepts_one_finite_box_at_upstream_size_gate(self):
        require_usable_face([[2, 3, 32, 40]])

    def test_rejects_empty_small_malformed_and_nonfinite_boxes(self):
        invalid = ([], [[0, 0, 29, 100]], [[1, 2, 3]], [[0, 0, float("nan"), 50]])
        for boxes in invalid:
            with self.subTest(boxes=boxes), self.assertRaises(NoUsableFaceError):
                require_usable_face(boxes)

    def test_musetalk_has_an_independent_empty_detection_guard(self):
        source = (ROOT / "muse_patch/preprocessing.py").read_text()
        self.assertEqual(source.count('raise ValueError("no usable face detected in source frames")'), 2)
        self.assertLess(
            source.index('raise ValueError("no usable face detected in source frames")'),
            source.index('text_range=f"Total frame:'),
        )


if __name__ == "__main__":
    unittest.main()
