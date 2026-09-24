import unittest

from muse_patch.reaction_schedule import gaze_envelope
from web.gaze_schedule import GazeScheduler


class GazeScheduleTests(unittest.TestCase):
    def test_events_span_phrases_and_have_human_scale_durations(self):
        scheduler = GazeScheduler(7)
        events = []
        for duration in (0.7, 1.1, 2.4, 3.2):
            phrase = scheduler.schedule(duration)
            events.extend(phrase)
            for event in phrase:
                self.assertGreaterEqual(event["at_ms"], 180)
                self.assertIn(event["direction"], (-1, 1))
                self.assertGreaterEqual(event["move_ms"], 75)
                self.assertLessEqual(event["move_ms"], 125)
                self.assertGreaterEqual(event["hold_ms"], 380)
        self.assertTrue(events)

    def test_envelope_moves_holds_and_returns_to_camera(self):
        plan = {"gaze_events": [{
            "at_ms": 200, "move_ms": 100, "hold_ms": 400,
            "return_ms": 200, "direction": -1,
        }]}
        self.assertEqual(gaze_envelope(0, 20, plan), (0, 0.0))
        self.assertEqual(gaze_envelope(8, 20, plan), (-1, 1.0))
        direction, amount = gaze_envelope(15, 20, plan)
        self.assertEqual(direction, -1)
        self.assertGreater(amount, 0)
        self.assertLess(amount, 1)
        self.assertEqual(gaze_envelope(20, 20, plan), (0, 0.0))


if __name__ == "__main__":
    unittest.main()
