import unittest

from scripts.analyze_landmark_series import CHANNELS, analyze_rows


class LandmarkSeriesTests(unittest.TestCase):
    def test_reports_derivatives_neutral_occupancy_and_covariance(self):
        rows = []
        for index in range(40):
            signal = (index % 10 - 5) * 0.001
            metrics = {name: 0.2 + signal for name in CHANNELS}
            metrics["gazeX"] = signal * 6
            metrics["yaw"] = signal * 3
            rows.append({"time": index / 20, "state": "attend", "metrics": metrics})
        report = analyze_rows(rows)
        self.assertEqual(report["observations"], 40)
        self.assertAlmostEqual(report["sample_rate_hz"], 20.0)
        self.assertIn("jerk_rms", report["channels"]["gazeX"])
        self.assertGreaterEqual(report["near_neutral_fraction"], 0)
        self.assertTrue(report["strongest_correlations"])
        self.assertEqual(report["states"], {"attend": 40})

    def test_rejects_sparse_input(self):
        with self.assertRaises(ValueError):
            analyze_rows([{"time": 0, "metrics": {}}])


if __name__ == "__main__":
    unittest.main()
