import unittest

from upper_body.motion import plan_motion


class UpperBodyMotionTests(unittest.TestCase):
    def test_plan_is_deterministic_bounded_and_audio_clocked(self):
        first = plan_motion(17, 2.4, {"energy": 0.7, "pause_ratio": 0.1}, {"nod": 0.4, "tilt": -0.3})
        second = plan_motion(17, 2.4, {"energy": 0.7, "pause_ratio": 0.1}, {"nod": 0.4, "tilt": -0.3})
        self.assertEqual(first, second)
        self.assertEqual(first["clock"], "audio")
        self.assertEqual(first["respiration"]["speech_phase"], "exhale")
        self.assertTrue(11.0 <= first["respiration"]["rate_bpm"] <= 17.5)
        self.assertTrue(0.33 <= first["respiration"]["inhale_ratio"] <= 0.42)
        self.assertLessEqual(abs(first["pose"]["head_roll"]), 0.007)
        self.assertEqual(first["pose"]["head_pitch"], 0.0)
        self.assertLessEqual(first["pose"]["nod_impulse"], 0.008)

    def test_plan_has_no_open_loop_sway_channel(self):
        plan = plan_motion(2, 3.0, {}, {})
        self.assertNotIn("sway", plan)
        self.assertNotIn("bob", plan)
        self.assertEqual(plan["ownership"]["mouth_jaw_articulation"], "musetalk")

    def test_ordinary_phrase_controls_clear_the_subpixel_floor_without_exceeding_caps(self):
        plan = plan_motion(
            11,
            2.0,
            {"energy": 0.55, "pause_ratio": 0.2},
            {"lean": 0.16, "nod": 0.16, "tilt": 0.16, "gaze_x": 0.16},
        )
        pose = plan["pose"]
        self.assertGreaterEqual(pose["sternum_y"], 0.002)
        self.assertGreaterEqual(pose["shoulder_lift"], 0.003)
        self.assertGreaterEqual(abs(pose["torso_yaw"]), 0.0035)
        self.assertGreaterEqual(abs(pose["head_roll"]), 0.0035)
        self.assertGreaterEqual(pose["nod_impulse"], 0.004)
        self.assertLessEqual(abs(pose["sternum_y"]), 0.004)
        self.assertLessEqual(abs(pose["torso_yaw"]), 0.006)
        self.assertLessEqual(abs(pose["head_roll"]), 0.007)
        self.assertLessEqual(pose["nod_impulse"], 0.008)


if __name__ == "__main__":
    unittest.main()
