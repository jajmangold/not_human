import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from avatar.controls.schema import (
    BLENDSHAPE_NAMES,
    NEUTRAL_CATEGORY_NAME,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    AvatarFrame,
    AvatarSequence,
)
from avatar.controls.calibration import (
    CONTROL_RANGES,
    bounded_combined_controls,
    control_grid,
    named_control_points,
    named_grid,
    neutral_controls,
    random_controls,
)


class AvatarControlSchemaTests(unittest.TestCase):
    def test_canonical_names_are_stable_and_unique(self):
        self.assertEqual(len(BLENDSHAPE_NAMES), 52)
        self.assertEqual(len(set(BLENDSHAPE_NAMES)), 52)
        self.assertEqual(BLENDSHAPE_NAMES[0], "browDownLeft")
        self.assertEqual(BLENDSHAPE_NAMES[-1], "tongueOut")

    def test_round_trip_preserves_arrays_and_metadata(self):
        count = 3
        frames = [
            AvatarFrame(
                timestamp=index / 25,
                blendshapes=np.full(52, index / 4, dtype=np.float32),
                head_rotation=np.asarray([index, 0, -index], dtype=np.float32),
                head_translation=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
                confidence=0.9,
                head_transform=np.eye(4, dtype=np.float32),
            )
            for index in range(count)
        ]
        sequence = AvatarSequence.from_frames(frames)
        with tempfile.TemporaryDirectory() as temporary:
            controls, metadata = sequence.save(temporary, {"test": "round-trip"}, include_derived=True)
            loaded = AvatarSequence.load(temporary)
            self.assertEqual(controls.name, "controls.npz")
            self.assertEqual(metadata.name, "metadata.json")
            self.assertTrue(np.array_equal(sequence.timestamps, loaded.timestamps))
            self.assertTrue(np.array_equal(sequence.blendshapes, loaded.blendshapes))
            self.assertTrue(np.array_equal(sequence.head_transform, loaded.head_transform))
            document = json.loads(Path(metadata).read_text())
            self.assertEqual(document["schema"], SCHEMA_NAME)
            self.assertEqual(document["version"], 1)
            with np.load(controls, allow_pickle=False) as archive:
                self.assertIn("blendshape_velocity", archive.files)
                self.assertIn("blendshape_acceleration", archive.files)

    def test_rejects_non_monotonic_timestamps_and_bad_blendshapes(self):
        kwargs = dict(
            timestamps=np.asarray([0.0, 0.0]),
            blendshapes=np.zeros((2, 52), dtype=np.float32),
            head_rotation=np.zeros((2, 3), dtype=np.float32),
            head_translation=np.zeros((2, 3), dtype=np.float32),
            confidence=np.ones(2, dtype=np.float32),
        )
        with self.assertRaises(ValueError):
            AvatarSequence(**kwargs)
        with self.assertRaises(ValueError):
            AvatarFrame(0, np.full(52, 1.1), np.zeros(3), np.zeros(3))

    def test_derived_motion_has_expected_linear_velocity(self):
        timestamps = np.arange(4, dtype=np.float64) / 10
        values = np.stack([np.full(52, value, dtype=np.float32) for value in timestamps])
        sequence = AvatarSequence(timestamps, values, np.zeros((4, 3)), np.zeros((4, 3)), np.ones(4))
        derived = sequence.derived_motion()
        self.assertTrue(np.allclose(derived["blendshape_velocity"], 1.0, atol=1e-5))
        self.assertTrue(np.allclose(derived["blendshape_acceleration"], 0.0, atol=1e-5))

    def test_neutral_category_is_excluded_from_the_measured_channel_names(self):
        # "_neutral" is MediaPipe's own synthetic category, not an ARKit
        # shape; it must never collide with a real canonical channel name.
        self.assertNotIn(NEUTRAL_CATEGORY_NAME, BLENDSHAPE_NAMES)

    def test_neutral_score_round_trips_and_validates_range(self):
        frames = [
            AvatarFrame(
                timestamp=index / 25,
                blendshapes=np.zeros(52, dtype=np.float32),
                head_rotation=np.zeros(3, dtype=np.float32),
                head_translation=np.zeros(3, dtype=np.float32),
                neutral_score=0.25 * index,
            )
            for index in range(3)
        ]
        sequence = AvatarSequence.from_frames(frames)
        self.assertTrue(np.allclose(sequence.neutral_score, [0.0, 0.25, 0.5]))
        with tempfile.TemporaryDirectory() as temporary:
            sequence.save(temporary)
            loaded = AvatarSequence.load(temporary)
            self.assertTrue(np.allclose(loaded.neutral_score, sequence.neutral_score))
        with self.assertRaises(ValueError):
            AvatarFrame(0, np.zeros(52), np.zeros(3), np.zeros(3), neutral_score=1.5)

    def test_loading_an_archive_without_neutral_score_is_backward_compatible(self):
        # Older controls.npz files predate the neutral_score array. Loading
        # them must not fail just because the optional array is missing.
        frames = [
            AvatarFrame(0, np.zeros(52, dtype=np.float32), np.zeros(3), np.zeros(3)),
            AvatarFrame(1, np.zeros(52, dtype=np.float32), np.zeros(3), np.zeros(3)),
        ]
        sequence = AvatarSequence.from_frames(frames)
        self.assertIsNone(sequence.neutral_score)
        with tempfile.TemporaryDirectory() as temporary:
            sequence.save(temporary)
            with np.load(Path(temporary) / "controls.npz") as archive:
                self.assertNotIn("neutral_score", archive.files)
            loaded = AvatarSequence.load(temporary)
            self.assertIsNone(loaded.neutral_score)

    def test_load_rejects_schema_version_and_column_order_mismatch(self):
        frame = AvatarFrame(0, np.zeros(52, dtype=np.float32), np.zeros(3), np.zeros(3))
        sequence = AvatarSequence.from_frames([frame])
        with tempfile.TemporaryDirectory() as temporary:
            _, metadata_path = sequence.save(temporary)
            document = json.loads(metadata_path.read_text())

            stale = dict(document, version=SCHEMA_VERSION + 1)
            metadata_path.write_text(json.dumps(stale))
            with self.assertRaises(ValueError):
                AvatarSequence.load(temporary)

            reordered = dict(document, version=SCHEMA_VERSION,
                              blendshape_names=list(reversed(BLENDSHAPE_NAMES)))
            metadata_path.write_text(json.dumps(reordered))
            with self.assertRaises(ValueError):
                AvatarSequence.load(temporary)

    def test_nonfinite_head_transform_is_rejected(self):
        bad_transform = np.eye(4, dtype=np.float32)
        bad_transform[0, 3] = np.nan
        with self.assertRaises(ValueError):
            AvatarFrame(0, np.zeros(52, dtype=np.float32), np.zeros(3), np.zeros(3),
                        head_transform=bad_transform)
        with self.assertRaises(ValueError):
            AvatarSequence(
                timestamps=np.asarray([0.0, 1.0]),
                blendshapes=np.zeros((2, 52), dtype=np.float32),
                head_rotation=np.zeros((2, 3), dtype=np.float32),
                head_translation=np.zeros((2, 3), dtype=np.float32),
                confidence=np.ones(2, dtype=np.float32),
                head_transform=np.stack([np.eye(4, dtype=np.float32), bad_transform]),
            )

    def test_derived_motion_uses_actual_irregular_spacing(self):
        # Timestamps here are not evenly spaced (0, 0.1, 0.4); a correct
        # finite-difference implementation must use the real gaps rather than
        # assume a uniform sample rate, or the derivative is fictitious.
        timestamps = np.asarray([0.0, 0.1, 0.4])
        values = np.stack([np.full(52, t * 2.0, dtype=np.float32) for t in timestamps])
        sequence = AvatarSequence(timestamps, values, np.zeros((3, 3)), np.zeros((3, 3)), np.ones(3))
        velocity = sequence.derived_motion()["blendshape_velocity"]
        # True slope is a constant 2.0 everywhere; np.gradient's one-sided
        # edge estimate and central estimate should both recover it exactly
        # for this perfectly linear signal, regardless of the uneven spacing.
        self.assertTrue(np.allclose(velocity, 2.0, atol=1e-5))

    def test_neutral_controls_are_all_zero(self):
        neutral = neutral_controls()
        self.assertEqual(set(neutral), set(CONTROL_RANGES))
        self.assertTrue(all(value == 0.0 for value in neutral.values()))

    def test_named_control_points_are_labeled_and_bounded(self):
        for name, (low, high) in CONTROL_RANGES.items():
            points = named_control_points(name)
            self.assertEqual(list(points), ["min", "q1", "default", "mid", "q3", "max"])
            self.assertEqual(points["min"], low)
            self.assertEqual(points["max"], high)
            # Every control's Form default is 0.0 and every range spans zero.
            self.assertEqual(points["default"], 0.0)
            # All six points fall within the control's own range...
            self.assertTrue(all(low <= value <= high for value in points.values()))
            # ...and the three range-derived points (independent of where
            # zero falls) are themselves monotonic, since q1/q3 are quartiles
            # around mid. "default" is not required to fall in this sequence:
            # it's anchored at zero, which sits off-center for asymmetric
            # ranges (e.g. eyebrow, wink, aaa).
            self.assertTrue(points["min"] <= points["q1"] <= points["mid"] <= points["q3"] <= points["max"])

    def test_named_grid_covers_every_control_at_six_points_with_others_zero(self):
        grid = named_grid()
        self.assertEqual(len(grid), 6 * len(CONTROL_RANGES))
        for entry in grid:
            requested = entry["requested"]
            self.assertEqual(requested[entry["control"]], entry["value"])
            others = {k: v for k, v in requested.items() if k != entry["control"]}
            self.assertTrue(all(value == 0.0 for value in others.values()))

    def test_bounded_combined_controls_stay_within_the_bounded_fraction(self):
        samples = bounded_combined_controls(20, seed=30, max_active=3, fraction=0.5)
        self.assertEqual(len(samples), 20)
        for sample in samples:
            active = [name for name, value in sample.items() if value != 0.0]
            self.assertTrue(2 <= len(active) <= 3)
            for name in active:
                low, high = CONTROL_RANGES[name]
                mid = (low + high) / 2.0
                span = (high - low) * 0.5 / 2.0
                self.assertTrue(mid - span - 1e-6 <= sample[name] <= mid + span + 1e-6)
        self.assertEqual(bounded_combined_controls(5, seed=30), bounded_combined_controls(5, seed=30))

    def test_calibration_grid_is_deterministic_and_covers_each_control(self):
        grid = control_grid(3)
        self.assertEqual(len(grid), 3 * len(CONTROL_RANGES))
        for index, name in enumerate(CONTROL_RANGES):
            rows = grid[index * 3:(index + 1) * 3]
            self.assertEqual([row[name] for row in rows], list(np.linspace(*CONTROL_RANGES[name], 3)))
            self.assertTrue(all(all(value == 0.0 for key, value in row.items() if key != name) for row in rows))
        self.assertEqual(random_controls(2, 28), random_controls(2, 28))


if __name__ == "__main__":
    unittest.main()
