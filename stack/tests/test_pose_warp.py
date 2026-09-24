import unittest

import numpy as np

try:
    import cv2
    from muse_patch.pose_warp import backward_pose_flow, warp_with_pose_flow
except ModuleNotFoundError:
    cv2 = None


@unittest.skipIf(cv2 is None, "OpenCV is installed in the MuseTalk runtime")
class PoseWarpTests(unittest.TestCase):
    def test_motion_compensation_avoids_two_exposed_geometries(self):
        base = np.zeros((128, 128, 3), dtype=np.uint8)
        for y in range(16, 112, 16):
            for x in range(16, 112, 16):
                color = 255 if (x // 16 + y // 16) % 2 else 100
                cv2.circle(base, (x, y), 5, (color, 255 - color, 180), -1)
        target = cv2.warpAffine(
            base, np.float32([[1, 0, 8], [0, 1, 0]]), (128, 128),
            borderMode=cv2.BORDER_REFLECT_101,
        )
        expected = cv2.warpAffine(
            base, np.float32([[1, 0, 4], [0, 1, 0]]), (128, 128),
            borderMode=cv2.BORDER_REFLECT_101,
        )
        flow = backward_pose_flow(base, target)
        warped = warp_with_pose_flow(base, flow, 0.5)
        dissolved = cv2.addWeighted(base, 0.5, target, 0.5, 0)
        warp_error = np.abs(warped.astype(np.float32) - expected).mean()
        dissolve_error = np.abs(dissolved.astype(np.float32) - expected).mean()
        self.assertLess(warp_error, dissolve_error * 0.20)

    def test_zero_amount_preserves_the_exact_input_object(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        flow = np.zeros((8, 8, 2), dtype=np.float32)
        self.assertIs(warp_with_pose_flow(frame, flow, 0), frame)


if __name__ == "__main__":
    unittest.main()
