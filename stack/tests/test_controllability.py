import unittest

import numpy as np

from avatar.controls.schema import BLENDSHAPE_NAMES
from avatar.controls.calibration import CONTROL_RANGES
from avatar.controls.controllability import (
    BROW_INDICES,
    CROSS_COUPLING_REGIONS,
    MOUTH_INDICES,
    blink_to_brow_leakage,
    channel_deltas,
    eye_to_mouth_leakage,
    noise_floor,
    region_leakage,
    safe_envelope,
    top_response_channels,
)


def _zeros() -> np.ndarray:
    return np.zeros(len(BLENDSHAPE_NAMES), dtype=np.float32)


class NoiseFloorTests(unittest.TestCase):
    def test_noise_floor_reflects_repeat_variance(self):
        rng = np.random.default_rng(30)
        blendshapes = rng.normal(0.0, 0.01, size=(8, len(BLENDSHAPE_NAMES))).astype(np.float32)
        rotation = rng.normal(0.0, 0.001, size=(8, 3)).astype(np.float32)
        noise = noise_floor(blendshapes, rotation)
        self.assertEqual(noise.sample_count, 8)
        self.assertTrue(np.all(noise.blendshape_std > 0))
        self.assertTrue(np.all(noise.blendshape_std < 0.05))

    def test_noise_floor_rejects_too_few_samples(self):
        with self.assertRaises(ValueError):
            noise_floor(_zeros()[None, :], np.zeros((1, 3)))

    def test_noise_floor_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            noise_floor(np.zeros((4, len(BLENDSHAPE_NAMES))), np.zeros((3, 3)))


class LeakageTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(30)
        repeats = rng.normal(0.0, 0.01, size=(6, len(BLENDSHAPE_NAMES))).astype(np.float32)
        self.noise = noise_floor(repeats, np.zeros((6, 3), dtype=np.float32))

    def test_region_leakage_flags_only_channels_above_threshold(self):
        deltas = _zeros()
        deltas[MOUTH_INDICES[0]] = 0.5  # a large, real mouth response
        report = region_leakage(deltas, self.noise, MOUTH_INDICES, threshold_multiplier=3.0)
        self.assertEqual(report["leaking_channel_count"], 1)
        self.assertIn(BLENDSHAPE_NAMES[int(MOUTH_INDICES[0])], report["leaking_channels"])

    def test_region_leakage_uses_absolute_floor_when_noise_is_exactly_zero(self):
        # Empirically, the resident ALP + MediaPipe pipeline is exactly
        # deterministic for repeated identical renders: real repeat std is
        # 0.0, not just small. A purely multiplicative threshold would then
        # call any nonzero delta "leakage."
        zero_noise = noise_floor(np.zeros((3, len(BLENDSHAPE_NAMES)), dtype=np.float32), np.zeros((3, 3)))
        deltas = _zeros()
        deltas[MOUTH_INDICES[0]] = 0.01  # below the default 0.02 absolute floor
        report = region_leakage(deltas, zero_noise, MOUTH_INDICES, threshold_multiplier=3.0, absolute_floor=0.02)
        self.assertEqual(report["leaking_channel_count"], 0)
        deltas[MOUTH_INDICES[0]] = 0.05  # above it
        report = region_leakage(deltas, zero_noise, MOUTH_INDICES, threshold_multiplier=3.0, absolute_floor=0.02)
        self.assertEqual(report["leaking_channel_count"], 1)

    def test_region_leakage_ignores_noise_level_deltas(self):
        deltas = _zeros()
        # Perturb every mouth channel by roughly one noise-floor unit -- below
        # the 3x threshold, so nothing should be flagged as leaking.
        deltas[MOUTH_INDICES] = self.noise.blendshape_std[MOUTH_INDICES]
        report = region_leakage(deltas, self.noise, MOUTH_INDICES, threshold_multiplier=3.0)
        self.assertEqual(report["leaking_channel_count"], 0)

    def test_eye_to_mouth_leakage_only_applies_to_eye_region_controls(self):
        self.assertIsNone(eye_to_mouth_leakage("smile", {}, self.noise))
        deltas = _zeros()
        deltas[MOUTH_INDICES[0]] = 0.4
        result = eye_to_mouth_leakage("blink", {"min": deltas, "max": _zeros()}, self.noise)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["max_abs_delta"], 0.4 - 1e-6)

    def test_blink_to_brow_leakage_picks_the_worst_sweep_point(self):
        mild = _zeros()
        mild[BROW_INDICES[0]] = 0.05
        severe = _zeros()
        severe[BROW_INDICES[0]] = 0.6
        result = blink_to_brow_leakage({"min": mild, "max": severe}, self.noise)
        self.assertAlmostEqual(result["max_abs_delta"], 0.6, places=5)

    def test_channel_deltas_shape_validation(self):
        with self.assertRaises(ValueError):
            channel_deltas(np.zeros(10), _zeros())


class SafeEnvelopeTests(unittest.TestCase):
    def test_envelope_stops_at_the_first_breaching_point_each_direction(self):
        values = [-20.0, -10.0, 0.0, 10.0, 20.0]
        leaking = [2, 0, 0, 0, 1]  # breaches at the far negative and far positive ends
        low, high = safe_envelope(values, leaking, default_value=0.0, max_leaking_channels=0)
        self.assertEqual((low, high), (-10.0, 10.0))

    def test_envelope_handles_asymmetric_default_position(self):
        # eyebrow-style range: breaches only at the far negative and far
        # positive ends, default (0.0) sits off-center among the six points.
        values = [-40.0, -25.0, -10.0, 0.0, 5.0, 20.0]
        leaking = [3, 0, 0, 0, 0, 1]
        low, high = safe_envelope(values, leaking, default_value=0.0, max_leaking_channels=0)
        self.assertEqual((low, high), (-25.0, 5.0))

    def test_envelope_stops_immediately_when_the_point_adjacent_to_default_breaches(self):
        values = [-40.0, -25.0, -10.0, 0.0, 5.0, 20.0]
        leaking = [3, 0, 1, 0, 0, 0]  # breach right next to default on the low side
        low, high = safe_envelope(values, leaking, default_value=0.0, max_leaking_channels=0)
        self.assertEqual((low, high), (0.0, 20.0))

    def test_envelope_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            safe_envelope([0.0, 1.0], [0])


class TopResponseChannelsTests(unittest.TestCase):
    def test_returns_the_largest_signed_deltas_first(self):
        deltas = _zeros()
        deltas[0] = 0.1
        deltas[5] = -0.5
        deltas[10] = 0.3
        top = top_response_channels(deltas, count=3)
        self.assertEqual([name for name, _ in top], [BLENDSHAPE_NAMES[5], BLENDSHAPE_NAMES[10], BLENDSHAPE_NAMES[0]])
        self.assertAlmostEqual(top[0][1], -0.5)


class CrossCouplingRegionsTests(unittest.TestCase):
    def test_every_key_is_a_real_control_and_every_region_is_known(self):
        self.assertTrue(set(CROSS_COUPLING_REGIONS).issubset(CONTROL_RANGES))
        for regions in CROSS_COUPLING_REGIONS.values():
            self.assertTrue(set(regions).issubset({"mouth", "brow"}))

    def test_a_controls_own_target_region_is_excluded_from_its_own_cross_coupling_set(self):
        # eyebrow's job is brow motion -- brow must not count as leakage for it.
        self.assertNotIn("brow", CROSS_COUPLING_REGIONS["eyebrow"])
        # smile/aaa/eee/woo's job is mouth shape -- mouth must not count as
        # leakage for them.
        for control in ("smile", "aaa", "eee", "woo"):
            self.assertNotIn("mouth", CROSS_COUPLING_REGIONS[control])


if __name__ == "__main__":
    unittest.main()
