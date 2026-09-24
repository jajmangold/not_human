"""Unit tests for NaturalMotionScheduler -- ported verbatim from
musetalk-volta's `/ws/live/` demo (internal issue #45's "why is it
still crappy" follow-up: this is what drives the avatar's blinking and
gaze/head micro-motion, previously never wired into this agent at all).
"""

from __future__ import annotations

from natural_motion import NaturalMotionScheduler


def test_schedule_returns_the_expected_shape():
    scheduler = NaturalMotionScheduler(seed=1, fps=16)
    plan = scheduler.schedule(2.0, {"energy": 0.5, "pause_ratio": 0.2}, {})
    assert set(plan) >= {"blinks", "gaze_events", "head_events", "state", "gain", "latent"}
    assert isinstance(plan["blinks"], list)
    assert isinstance(plan["gaze_events"], list)
    assert isinstance(plan["head_events"], list)


def test_a_long_enough_phrase_eventually_produces_a_blink():
    # Blink timing is randomized (2.8-5.6s initial interval) -- a single
    # short phrase can legitimately produce zero blinks. Across enough
    # cumulative duration on one continuous clock, at least one must land.
    scheduler = NaturalMotionScheduler(seed=7, fps=16)
    all_blinks = []
    for _ in range(10):
        plan = scheduler.schedule(3.0, {"energy": 0.3, "pause_ratio": 0.3}, {})
        all_blinks.extend(plan["blinks"])
    assert all_blinks, "expected at least one blink across 30s of continuous conversation"


def test_clock_is_continuous_across_calls_not_reset_per_phrase():
    # Two schedulers seeded identically: one gets one big call, the other
    # gets the same total duration split into several smaller calls. If the
    # clock is genuinely continuous (not reset per call), the split-call
    # scheduler should still eventually produce blinks/gaze events -- it
    # must not restart its random draw window on every call.
    seed = 42
    single = NaturalMotionScheduler(seed=seed, fps=16)
    single_plan = single.schedule(12.0, {"energy": 0.4, "pause_ratio": 0.2}, {})

    split = NaturalMotionScheduler(seed=seed, fps=16)
    split_blinks: list = []
    split_gaze: list = []
    for _ in range(6):
        plan = split.schedule(2.0, {"energy": 0.4, "pause_ratio": 0.2}, {})
        split_blinks.extend(plan["blinks"])
        split_gaze.extend(plan["gaze_events"])

    # Both must actually produce events over 12s of continuous conversation
    # -- if the clock reset every call, most 2s calls would produce nothing
    # (blink intervals start at 2.8-5.6s) and this would flake toward zero.
    assert len(single_plan["blinks"]) + len(single_plan["gaze_events"]) > 0
    assert split_blinks or split_gaze


def test_events_never_extend_past_their_own_call_duration():
    scheduler = NaturalMotionScheduler(seed=3, fps=16)
    duration = 4.0
    plan = scheduler.schedule(duration, {"energy": 0.6, "pause_ratio": 0.1}, {"nod": 0.3})
    for blink in plan["blinks"]:
        end_ms = blink["at_ms"] + blink["close_ms"] + blink["hold_ms"] + blink["open_ms"]
        assert end_ms <= duration * 1000
    for gaze in plan["gaze_events"]:
        end_ms = gaze["at_ms"] + gaze["move_ms"] + gaze["hold_ms"] + gaze["return_ms"]
        assert end_ms <= duration * 1000
    for head in plan["head_events"]:
        end_ms = head["at_ms"] + head["move_ms"] + head["hold_ms"] + head["return_ms"]
        assert end_ms <= duration * 1000
