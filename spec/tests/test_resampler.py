"""Contract tests for the 24 kHz -> 16 kHz timestamp-preserving resampler (P1-004).

Covers normal operation, bounds, missing data, and replay determinism.
"""

from __future__ import annotations

import math

import pytest

from nothuman.resampler import (
    INPUT_RATE_HZ,
    OUTPUT_RATE_HZ,
    ResamplerMetrics,
    input_timestamp,
    output_length,
    resample,
    resample_stream,
)


def _sine(n: int, freq: float = 440.0) -> list[float]:
    return [math.sin(2 * math.pi * freq * i / INPUT_RATE_HZ) for i in range(n)]


def test_normal_output_length_is_exact_ratio() -> None:
    for n in (0, 1, 2, 3, 4, 7, 24, 240, 24000):
        assert output_length(n) == (n * OUTPUT_RATE_HZ + INPUT_RATE_HZ - 1) // INPUT_RATE_HZ


def test_normal_resample_matches_anchored_interpolation() -> None:
    samples = _sine(240)
    out = resample(samples)
    assert len(out) == output_length(240)
    for i, value in enumerate(out):
        pos = i * (INPUT_RATE_HZ / OUTPUT_RATE_HZ)
        lo = int(pos)
        frac = pos - lo
        a = samples[lo]
        b = samples[lo + 1] if lo + 1 < len(samples) else a
        assert value == pytest.approx(a + (b - a) * frac)


def test_normal_zero_input_is_zero_output() -> None:
    assert resample([0.0] * 10) == [0.0] * output_length(10)


def test_normal_dc_is_preserved() -> None:
    out = resample([0.5] * 100)
    assert out == [0.5] * output_length(100)


def test_bounds_negative_counts_rejected() -> None:
    with pytest.raises(ValueError):
        output_length(-1)
    with pytest.raises(ValueError):
        input_timestamp(-1)


def test_bounds_unsupported_rates_rejected() -> None:
    with pytest.raises(ValueError):
        resample([0.0], input_rate=16000, output_rate=16000)
    with pytest.raises(ValueError):
        resample([0.0], input_rate=24000, output_rate=8000)


def test_bounds_empty_input_is_empty_output() -> None:
    assert resample([]) == []
    assert output_length(0) == 0


def test_bounds_single_sample_held() -> None:
    out = resample([0.75])
    assert len(out) == 1
    assert out[0] == pytest.approx(0.75)


def test_missing_tail_input_holds_last_value() -> None:
    out = resample([1.0, 2.0, 3.0])
    assert len(out) == 2
    # pos 0.0 -> 1.0 ; pos 1.5 -> 2.0 + (3.0 - 2.0) * 0.5 = 2.5
    assert all(v == pytest.approx(x) for v, x in zip(out, [1.0, 2.5]))


def test_missing_intermediate_input_linearly_interpolated() -> None:
    samples = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    out = resample(samples)
    for i, value in enumerate(out):
        pos = i * 1.5
        lo = int(pos)
        frac = pos - lo
        expected = samples[lo] + (samples[lo + 1] - samples[lo]) * frac
        assert value == pytest.approx(expected)


def test_replay_is_deterministic() -> None:
    samples = _sine(480)
    first = resample(samples)
    second = resample(samples)
    assert first == second
    assert resample_stream(samples)[1] == resample_stream(samples)[1]


def test_replay_metrics_are_deterministic() -> None:
    out, metrics = resample_stream(_sine(2400))
    assert isinstance(metrics, ResamplerMetrics)
    assert metrics.input_samples == 2400
    assert metrics.output_samples == len(out) == 1600
    assert metrics.duration_in_s == pytest.approx(2400 / INPUT_RATE_HZ)
    assert metrics.duration_out_s == pytest.approx(1600 / OUTPUT_RATE_HZ)


def test_replay_no_duration_drift() -> None:
    # 1 s of 24 kHz input must map to exactly 16000 output samples (1 s).
    out, metrics = resample_stream(_sine(INPUT_RATE_HZ))
    assert metrics.output_samples == OUTPUT_RATE_HZ
    assert metrics.duration_out_s == pytest.approx(1.0)
    assert input_timestamp(OUTPUT_RATE_HZ - 1) == pytest.approx(INPUT_RATE_HZ - 1.5)
