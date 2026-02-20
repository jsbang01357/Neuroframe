import unittest
import datetime as dt
from neuroframe.engine import UserBaseline, tune_user_weights

class TestWeightTuning(unittest.TestCase):
    def setUp(self):
        self.baseline = UserBaseline(
            baseline_sleep_start=dt.time(23, 30),
            baseline_wake=dt.time(7, 30),
            caffeine_sensitivity=1.0,
            circadian_weight=1.0
        )
        self.lr = 0.05

    def test_empty_logs(self):
        w = tune_user_weights(self.baseline, [])
        self.assertEqual(w, {})

    def test_high_caffeine_low_clarity_decreases_sensitivity(self):
        # 300mg caffeine, clarity 3.0 -> should decrease sensitivity
        logs = [
            {
                "subjective_clarity": "3.0",
                "doses_json": '[{"hour": 9, "minute": 0, "amount_mg": 300.0}]'
            }
        ]
        w = tune_user_weights(self.baseline, logs, learning_rate=self.lr)
        self.assertLess(w["caffeine_sensitivity"], 1.0)
        self.assertEqual(w["caffeine_sensitivity"], 0.95)
        # Low clarity also increases circadian weight according to heuristics
        self.assertGreater(w["circadian_weight"], 1.0)

    def test_high_caffeine_high_clarity_increases_sensitivity(self):
        # 300mg caffeine, clarity 8.5 -> should increase sensitivity
        logs = [
            {
                "subjective_clarity": "8.5",
                "doses_json": '[{"hour": 9, "minute": 0, "amount_mg": 300.0}]'
            }
        ]
        w = tune_user_weights(self.baseline, logs, learning_rate=self.lr)
        self.assertGreater(w["caffeine_sensitivity"], 1.0)
        self.assertEqual(w["caffeine_sensitivity"], 1.05)
        # High clarity decreases circadian weight constraint
        self.assertLess(w["circadian_weight"], 1.0)
        
    def test_no_change_for_average_day(self):
        # 150mg caffeine, clarity 5.5 -> no conditions met, weights stay same
        logs = [
            {
                "subjective_clarity": "5.5",
                "doses_json": '[{"hour": 9, "minute": 0, "amount_mg": 150.0}]'
            }
        ]
        w = tune_user_weights(self.baseline, logs, learning_rate=self.lr)
        self.assertEqual(w["caffeine_sensitivity"], 1.0)
        self.assertEqual(w["circadian_weight"], 1.0)

if __name__ == '__main__':
    unittest.main()
