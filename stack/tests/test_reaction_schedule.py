import unittest

from muse_patch.reaction_schedule import HEAD_POSE_SLOTS, blink_envelope, head_pose_envelope, reaction_envelope


class ReactionEnvelopeTests(unittest.TestCase):
    def test_neutral_and_unknown_reactions_stay_zero(self):
        self.assertEqual(reaction_envelope(8, 32, 16, {"reaction": "neutral", "strength": 1}), 0)
        self.assertEqual(reaction_envelope(8, 32, 16, {"reaction": "anger", "strength": 1}), 0)

    def test_envelope_attacks_holds_decays_and_settles(self):
        plan = {
            "reaction": "surprise", "strength": 0.7,
            "attack_ms": 200, "hold_ms": 300, "decay_ms": 500,
        }
        values = [reaction_envelope(index, 32, 16, plan) for index in range(32)]
        self.assertEqual(values[0], 0)
        self.assertGreater(max(values), 0.67)
        self.assertGreater(values[5], values[1])
        self.assertLess(values[14], values[8])
        self.assertGreater(values[-1], 0)
        self.assertLess(values[-1], values[8])

    def test_short_phrases_scale_the_curve_and_keep_the_bound(self):
        plan = {
            "reaction": "amusement", "strength": 2.0,
            "attack_ms": 260, "hold_ms": 500, "decay_ms": 700,
        }
        values = [reaction_envelope(index, 12, 16, plan) for index in range(12)]
        self.assertLessEqual(max(values), 0.68)
        self.assertEqual(values[0], 0)
        self.assertLess(values[-1], values[len(values) // 2])

    def test_expression_floor_is_subtle_and_bounded_for_phrase_continuity(self):
        first = {"reaction": "interest", "strength": 0.4, "start_ratio": 0.0, "end_ratio": 0.18}
        repeated = {"reaction": "interest", "strength": 0.4, "start_ratio": 0.18, "end_ratio": 0.18}
        first_values = [reaction_envelope(index, 32, 16, first) for index in range(32)]
        repeated_values = [reaction_envelope(index, 32, 16, repeated) for index in range(32)]
        self.assertEqual(first_values[0], 0.0)
        self.assertAlmostEqual(repeated_values[0], 0.072, places=4)
        self.assertAlmostEqual(first_values[-1], repeated_values[-1], places=4)
        self.assertGreaterEqual(min(repeated_values), 0.072)
        self.assertLessEqual(max(repeated_values), 0.4)

    def test_blink_is_independent_and_opens_more_slowly_than_it_closes(self):
        plan = {"reaction": "neutral", "strength": 0, "blinks": [
            {"at_ms": 100, "close_ms": 55, "hold_ms": 24, "open_ms": 90},
        ]}
        values = [blink_envelope(index, 80, plan) for index in range(28)]
        self.assertEqual(values[0], 0)
        self.assertGreater(max(values), 0.9)
        self.assertGreater(values[17], 0)
        self.assertEqual(values[-1], 0)

    def test_real_head_pose_event_moves_holds_and_returns_with_a_hard_amplitude_cap(self):
        plan = {"head_events": [{
            "axis": "yaw", "direction": -1, "at_ms": 100,
            "move_ms": 200, "hold_ms": 200, "return_ms": 300,
            "amplitude": 4.0,
        }]}
        values = [head_pose_envelope(index, 20, plan) for index in range(20)]
        self.assertEqual(values[0], (None, 0, 0.0))
        self.assertEqual(values[6][:2], ("yaw", -1))
        self.assertLessEqual(max(value[2] for value in values), 0.72)
        self.assertGreater(values[8][2], values[3][2])
        self.assertEqual(values[16], (None, 0, 0.0))

    def test_head_follow_slots_match_visual_gaze_direction(self):
        # ALP's Euler yaw sign is opposite its pupil_x image-space sign on the
        # calibrated portrait; the mapping must pair visual directions.
        self.assertEqual(HEAD_POSE_SLOTS[("yaw", -1)], 17)
        self.assertEqual(HEAD_POSE_SLOTS[("yaw", 1)], 16)


if __name__ == "__main__":
    unittest.main()
