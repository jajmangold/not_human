import unittest
from pathlib import Path

import numpy as np

from muse_patch.motion_compositor import gaze_region_only, upper_face_only


class MotionCompositorTests(unittest.TestCase):
    def test_blink_is_removed_from_lower_face(self):
        base = np.zeros((100, 100, 3), dtype=np.uint8)
        animated = np.full_like(base, 200)
        result = upper_face_only(base, animated, [20, 10, 80, 90])
        self.assertGreater(result[30, 50].mean(), 150)
        self.assertEqual(result[80, 50].mean(), 0)
        self.assertEqual(result[30, 5].mean(), 0)

    def test_zero_strength_is_exact_base(self):
        base = np.full((20, 20, 3), 17, dtype=np.uint8)
        animated = np.full_like(base, 240)
        self.assertIs(upper_face_only(base, animated, [2, 2, 18, 18], 0), base)

    def test_blink_layers_over_gaze_without_reaching_the_mouth(self):
        base = np.zeros((100, 100, 3), dtype=np.uint8)
        gaze = np.full_like(base, 100)
        blink = np.full_like(base, 200)
        gaze_result = gaze_region_only(base, gaze, [20, 10, 80, 90])
        result = upper_face_only(gaze_result, blink, [20, 10, 80, 90], 0.5)
        self.assertEqual(result[30, 50].mean(), 150)
        self.assertEqual(result[80, 50].mean(), 0)

    def test_blink_fade_ends_before_the_mouth_boundary(self):
        base = np.full((100, 100, 3), 17, dtype=np.uint8)
        blink = np.full_like(base, 240)
        result = upper_face_only(base, blink, [20, 10, 80, 90])
        # The fade ends at y=10 + .58*80 = 56.4. Everything below the
        # rounded boundary is exactly MuseTalk's mouth-owned base frame.
        np.testing.assert_array_equal(result[58:, :, :], base[58:, :, :])

    def test_head_gaze_and_blink_overlap_preserves_musetalk_lower_face(self):
        head = np.full((100, 100, 3), 33, dtype=np.uint8)
        gaze = np.full_like(head, 100)
        blink = np.full_like(head, 220)
        with_gaze = gaze_region_only(head, gaze, [20, 10, 80, 90])
        combined = upper_face_only(with_gaze, blink, [20, 10, 80, 90], 0.6)
        np.testing.assert_array_equal(combined[58:, :, :], head[58:, :, :])
        self.assertGreater(combined[30, 50].mean(), head[30, 50].mean())

    def test_gaze_mask_has_an_exact_mouth_exclusion_zone(self):
        base = np.full((100, 100, 3), 17, dtype=np.uint8)
        gaze = np.full_like(base, 240)
        result = gaze_region_only(base, gaze, [20, 10, 80, 90])
        self.assertGreater(result[30, 50].mean(), 200)
        # 52 is 52.5% down this face bbox: below the gaze mask's hard edge.
        np.testing.assert_array_equal(result[52:, :, :], base[52:, :, :])

    def test_gaze_does_not_condition_musetalk_mouth_inputs(self):
        server = (Path(__file__).resolve().parents[1] / "muse_patch/muse_server.py").read_text()
        self.assertIn('sources["coords"].append(coord_cycle[QUIET_SLOTS[0]])', server)
        self.assertIn('sources["latents"].append(latent_cycle[QUIET_SLOTS[0]])', server)
        self.assertIn('sources["coords"].append(base_coord)', server)
        self.assertIn('sources["latents"].append(base_latent)', server)
        self.assertIn('gaze_region_only(base_combine, gaze_combine, bbox)', server)
        gaze_block = server.split(
            "gaze_direction, gaze_amount = gaze_envelope(index, fps, reaction)", 1
        )[1].split('sources["base_frames"].append(base_frame)', 1)[0]
        self.assertNotIn("semantic_coord =", gaze_block)
        self.assertNotIn("semantic_latent =", gaze_block)

    def test_real_head_pose_warps_rgb_and_conditions_musetalk_inputs(self):
        server = (Path(__file__).resolve().parents[1] / "muse_patch/muse_server.py").read_text()
        pose_block = server.split(
            "head_axis, head_direction, head_amount = head_pose_envelope(index, fps, reaction)", 1
        )[1].split("base_frame, base_coord, base_latent =", 1)[0]
        self.assertIn("self._warp_pose_frame(", pose_block)
        self.assertIn("coord_cycle[head_index]", pose_block)
        self.assertIn("latent_cycle[head_index]", pose_block)
        self.assertNotIn("semantic_frame = self._add_source_delta(", pose_block)

    def test_blink_is_additive_so_it_cannot_erase_head_pose(self):
        server = (Path(__file__).resolve().parents[1] / "muse_patch/muse_server.py").read_text()
        render_block = server.split(
            "head_axis, head_direction, head_amount = head_pose_envelope(index, fps, reaction)", 1
        )[1].split('sources["coords"].append(base_coord)', 1)[0]
        self.assertIn("visible_frame = self._add_source_delta(", render_block)
        self.assertIn("frame_cycle[BLINK_SLOT]", render_block)
        self.assertIn('sources["frames"].append(visible_frame)', render_block)
        self.assertNotIn("self._mix_source_value(", render_block)


if __name__ == "__main__":
    unittest.main()
