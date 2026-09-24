"""Contract tests for the 50 Hz timeline ring buffer (P0-004).

Covers normal operation, bounds, missing data, replay determinism,
late-packet policy, jitter, drops, and reordering.
"""

import pytest

from nothuman.timeline import TimelineBuffer, TimelineConfig, TimelineMetrics


def _cfg(capacity=8, max_late=3, max_reorder=2, interpolate=True) -> TimelineConfig:
    return TimelineConfig(
        capacity=capacity, max_late=max_late, max_reorder=max_reorder, interpolate=interpolate
    )


def _fill(buf: TimelineBuffer, n: int) -> None:
    for i in range(n):
        assert buf.push(i, i * 10)


# ---------------------------------------------------------------- normal


def test_normal_push_and_sample_roundtrip() -> None:
    buf = TimelineBuffer(config=_cfg())
    for i in range(5):
        assert buf.push(i, i * 10)
    for i in range(5):
        assert buf.sample(i) == i * 10
    assert len(buf) == 5
    m = buf.metrics
    assert m.accepted == 5 and m.late_drops == 0 and m.reordered_drops == 0


def test_sample_holds_last_frame_beyond_last_accepted() -> None:
    buf = TimelineBuffer(config=_cfg())
    _fill(buf, 5)
    assert buf.sample(99) == 40


def test_sample_before_first_frame_holds_first_value() -> None:
    buf = TimelineBuffer(config=_cfg())
    _fill(buf, 3)
    assert buf.sample(0) == 0


def test_sample_empty_timeline_raises() -> None:
    buf = TimelineBuffer(config=_cfg())
    with pytest.raises(LookupError):
        buf.sample(0)


def test_negative_timestamp_rejected() -> None:
    buf = TimelineBuffer(config=_cfg())
    with pytest.raises(ValueError):
        buf.push(-1, 0)
    with pytest.raises(ValueError):
        buf.sample(-1)


# ---------------------------------------------------------------- bounds


def test_config_bounds_rejected() -> None:
    with pytest.raises(ValueError):
        TimelineConfig(capacity=0, max_late=1, max_reorder=0)
    with pytest.raises(ValueError):
        TimelineConfig(capacity=4, max_late=-1, max_reorder=0)
    with pytest.raises(ValueError):
        TimelineConfig(capacity=4, max_late=1, max_reorder=-1)
    with pytest.raises(ValueError):
        TimelineConfig(capacity=4, max_late=1, max_reorder=2)


def test_ring_overwrite_counts_overwritten_frames() -> None:
    buf = TimelineBuffer(config=_cfg(capacity=4))
    for i in range(8):
        assert buf.push(i, i)
    assert len(buf) == 4
    assert buf.metrics.overwritten == 4
    assert buf.sample(7) == 7


def test_capacity_one_buffer() -> None:
    buf = TimelineBuffer(config=_cfg(capacity=1, max_late=0, max_reorder=0))
    assert buf.push(0, "a")
    assert buf.sample(0) == "a"
    assert buf.push(1, "b")
    assert buf.sample(1) == "b"
    assert len(buf) == 1


# ---------------------------------------------------------------- missing data


def test_missing_intermediate_frame_interpolated() -> None:
    buf = TimelineBuffer(config=_cfg(capacity=16, max_late=8, max_reorder=8))
    _fill(buf, 10)
    # Frame 5 was never pushed: linear interpolation between 4 and 6.
    assert buf.sample(5) == 50.0
    # Non-numeric payloads fall back to hold-last.
    buf2 = TimelineBuffer(config=_cfg(capacity=16, max_late=8, max_reorder=8))
    for i in (0, 2, 4):
        assert buf2.push(i, f"v{i}")
    assert buf2.sample(1) == "v0"
    assert buf2.sample(3) == "v2"


def test_non_interpolating_mode_holds_last() -> None:
    buf = TimelineBuffer(config=_cfg(interpolate=False))
    _fill(buf, 10)
    assert buf.sample(5) == 50
    assert buf.sample(5) == 50


# ---------------------------------------------------------------- late policy


def test_late_packet_within_window_accepted() -> None:
    buf = TimelineBuffer(config=_cfg(max_late=3, max_reorder=2))
    _fill(buf, 10)
    # ts 7 is 3 behind last accepted (9): within max_late, accepted.
    assert buf.push(7, 77)
    assert buf.metrics.accepted == 11
    assert buf.sample(7) == 77


def test_late_packet_beyond_window_dropped() -> None:
    buf = TimelineBuffer(config=_cfg(max_late=3, max_reorder=2))
    _fill(buf, 10)
    assert not buf.push(5, 55)  # gap 4 > max_late
    assert buf.metrics.late_drops == 1
    assert buf.sample(5) == 50  # original retained value untouched


def test_duplicate_last_frame_dropped_as_late() -> None:
    buf = TimelineBuffer(config=_cfg())
    _fill(buf, 5)
    assert not buf.push(4, 999)
    assert buf.metrics.late_drops == 1
    assert buf.sample(4) == 40


def test_far_future_packet_dropped_as_late() -> None:
    buf = TimelineBuffer(config=_cfg(max_late=3))
    _fill(buf, 5)
    assert not buf.push(9, 99)  # 4 ahead of last accepted
    assert buf.metrics.future_drops == 1
    assert buf.sample(5) == 40


# ---------------------------------------------------------------- reordering


def test_reordered_packet_dropped_when_beyond_reorder_window() -> None:
    buf = TimelineBuffer(config=_cfg(max_late=3, max_reorder=2))
    _fill(buf, 10)
    # ts 6: gap 3, within max_late but beyond max_reorder -> reordered drop.
    assert not buf.push(6, 66)
    assert buf.metrics.reordered_drops == 1
    assert buf.metrics.late_drops == 0


def test_reordered_packet_within_reorder_window_accepted() -> None:
    buf = TimelineBuffer(config=_cfg(max_late=3, max_reorder=2))
    _fill(buf, 10)
    assert buf.push(8, 88)  # gap 2 <= max_reorder
    assert buf.metrics.accepted == 11
    assert buf.metrics.reordered_drops == 0
    assert buf.sample(8) == 88
    assert buf.sample(8) == 88


# ---------------------------------------------------------------- replay


def test_replay_is_deterministic() -> None:
    def run() -> tuple[int, int, int, int]:
        buf = TimelineBuffer(config=_cfg(capacity=16, max_late=3, max_reorder=2))
        for ts in [0, 1, 2, 4, 5, 3, 6, 7, 8, 2, 9, 10, 5, 11]:
            buf.push(ts, ts * 10)
        m = buf.metrics
        return (m.accepted, m.late_drops, m.reordered_drops, m.overwritten)

    first = run()
    second = run()
    assert first == second
    # Exact expected counters for the fixed packet sequence.
    assert first == (12, 2, 0, 0)


def test_reset_clears_state_for_replay() -> None:
    buf = TimelineBuffer(config=_cfg())
    _fill(buf, 5)
    buf.reset()
    assert len(buf) == 0
    assert buf.metrics == TimelineMetrics()
    _fill(buf, 5)
    assert buf.sample(4) == 40


def test_jitter_accounted_by_drop_counters() -> None:
    # Deterministic jitter: a fixed out-of-order, gappy stream.
    buf = TimelineBuffer(config=_cfg(capacity=32, max_late=4, max_reorder=2))
    stream = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
    for ts in stream:
        buf.push(ts, ts)
    m = buf.metrics
    assert m.accepted + m.late_drops + m.reordered_drops == len(stream)
    assert m.accepted == 13  # every frame eventually lands in-window
    assert m.late_drops == 0
    assert m.reordered_drops == 0
