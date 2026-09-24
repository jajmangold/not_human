import unittest

from muse_patch.reaction_schedule import blink_envelope
from web.blink_schedule import BlinkScheduler


class BlinkSchedulerTests(unittest.TestCase):
    def test_blinks_span_short_phrases_and_stay_away_from_cuts(self):
        scheduler = BlinkScheduler(42)
        events = []
        for _ in range(8):
            phrase_events = scheduler.schedule(1.25)
            for event in phrase_events:
                total = event["close_ms"] + event["hold_ms"] + event["open_ms"]
                self.assertGreaterEqual(event["at_ms"], 160)
                self.assertLessEqual(event["at_ms"] + total, 1190)
                values = [blink_envelope(index, 16, {"blinks": [event]}) for index in range(20)]
                self.assertGreater(max(values), 0.95)
            events.extend(phrase_events)
        self.assertGreaterEqual(len(events), 1)
        self.assertLessEqual(len(events), 3)

    def test_shapes_are_bounded_and_varied(self):
        events = BlinkScheduler(7).schedule(20)
        self.assertGreaterEqual(len(events), 2)
        self.assertTrue(all(55 <= event["close_ms"] <= 78 for event in events))
        self.assertTrue(all(10 <= event["hold_ms"] <= 24 for event in events))
        self.assertTrue(all(115 <= event["open_ms"] <= 155 for event in events))
        self.assertGreater(len({tuple(event.values()) for event in events}), 1)
