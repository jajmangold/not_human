import unittest

from web.natural_motion import NaturalMotionScheduler


class NaturalMotionSchedulerTests(unittest.TestCase):
    def test_events_share_one_clock_and_avoid_pre_gaze_blinks(self):
        saw_gaze = saw_blink = False
        for seed in range(80):
            scheduler = NaturalMotionScheduler(seed, fps=16)
            plans = [scheduler.schedule(3.0, {"pause_ratio": 0.2}, {}) for _ in range(6)]
            saw_gaze = saw_gaze or any(plan["gaze_events"] for plan in plans)
            saw_blink = saw_blink or any(plan["blinks"] for plan in plans)
            for plan in plans:
                for blink in plan["blinks"]:
                    for gaze in plan["gaze_events"]:
                        delta = blink["at_ms"] - gaze["at_ms"]
                        self.assertFalse(-480 <= delta < 0)
        self.assertTrue(saw_gaze)
        self.assertTrue(saw_blink)

    def test_motion_prior_is_bounded_correlated_and_stateful(self):
        scheduler = NaturalMotionScheduler(7)
        states = []
        for _ in range(12):
            plan = scheduler.schedule(
                1.1, {"energy": 0.4, "pause_ratio": 0.15}, {"gaze_focus": 0.3}
            )
            states.append(plan["state"])
            self.assertLessEqual(plan["gain"], 0.75)
            self.assertGreaterEqual(plan["gain"], 0.0)
            self.assertTrue(all(-1 <= value <= 1 for value in plan["latent"].values()))
        self.assertLess(len(set(states)), len(states))

    def test_blink_shapes_are_subtle_and_frame_aligned(self):
        events = NaturalMotionScheduler(13, fps=16).schedule(30.0)["blinks"]
        self.assertGreaterEqual(len(events), 3)
        for event in events:
            self.assertTrue(52 <= event["close_ms"] <= 72)
            self.assertTrue(6 <= event["hold_ms"] <= 18)
            self.assertTrue(105 <= event["open_ms"] <= 145)
            self.assertTrue(0.86 <= event["peak"] <= 0.97)
            peak = event["at_ms"] + event["close_ms"]
            self.assertLessEqual(abs(peak / 62.5 - round(peak / 62.5)), 0.01)

    def test_gaze_amplitudes_do_not_use_the_full_bank_extreme(self):
        scheduler = NaturalMotionScheduler(19)
        events = []
        for _ in range(10):
            events.extend(scheduler.schedule(3.0)["gaze_events"])
        self.assertTrue(events)
        self.assertTrue(all(0.46 <= event["amplitude"] <= 0.74 for event in events))

    def test_real_head_events_are_sparse_bounded_and_finish_inside_the_phrase(self):
        events = []
        for seed in range(50):
            scheduler = NaturalMotionScheduler(seed)
            for _ in range(8):
                plan = scheduler.schedule(2.2, {"energy": 0.4}, {"nod": 0.0})
                events.extend(plan["head_events"])
                for event in plan["head_events"]:
                    self.assertLessEqual(
                        event["at_ms"] + event["move_ms"] + event["hold_ms"] + event["return_ms"],
                        2150,
                    )
        self.assertTrue(events)
        self.assertTrue(all(event["axis"] == "yaw" for event in events))
        self.assertTrue(all(0.38 <= event["amplitude"] <= 0.58 for event in events))

    def test_semantic_nod_gets_one_real_pitch_event_without_a_phrase_boundary_snap(self):
        plan = NaturalMotionScheduler(3).schedule(1.2, {}, {"nod": 0.2})
        self.assertEqual(len(plan["head_events"]), 1)
        event = plan["head_events"][0]
        self.assertEqual((event["axis"], event["direction"], event["source"]), ("pitch", 1, "semantic-nod"))
        self.assertLessEqual(event["at_ms"] + event["move_ms"] + event["hold_ms"] + event["return_ms"], 1150)


if __name__ == "__main__":
    unittest.main()
