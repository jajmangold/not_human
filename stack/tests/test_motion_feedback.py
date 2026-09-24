import unittest

import numpy as np

from scripts.analyze_motion_feedback import analyze_frames


class MotionFeedbackTests(unittest.TestCase):
    def test_upper_face_blink_passes_without_mouth_motion(self):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(8)]
        for index, value in enumerate((0, 65, 150, 120, 80, 45, 20, 0)):
            frames[index][20:45, 25:75] = value
        report = analyze_frames(frames, [20, 10, 80, 90], 60)
        self.assertTrue(report["passes_mouth_isolation"])
        self.assertTrue(report["reopening_slower_than_closure"])
        self.assertLess(report["lower_face_leak_ratio"], 0.01)

    def test_mouth_motion_during_blink_fails(self):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
        for index, value in enumerate((0, 60, 140, 50, 0)):
            frames[index][20:80, 25:75] = value
        report = analyze_frames(frames, [20, 10, 80, 90], 60)
        self.assertFalse(report["passes_mouth_isolation"])
        self.assertGreater(report["lower_face_leak_ratio"], 0.18)


if __name__ == "__main__":
    unittest.main()
