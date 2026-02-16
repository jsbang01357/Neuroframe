from __future__ import annotations

import datetime as dt
import unittest
from zoneinfo import ZoneInfo

from neuroframe.engine import DayInputs, Dose, UserBaseline, calibrate_baseline_offset, predict_day


class EngineTests(unittest.TestCase):
    def test_predict_day_returns_consistent_shapes_and_bounds(self):
        tz = ZoneInfo("Asia/Seoul")
        baseline = UserBaseline(
            baseline_sleep_start=dt.time(23, 30),
            baseline_wake=dt.time(7, 30),
            chronotype_shift_hours=0.0,
            caffeine_half_life_hours=5.0,
            caffeine_sensitivity=1.0,
            baseline_offset=0.0,
            circadian_weight=1.0,
            sleep_pressure_weight=1.2,
            drug_weight=0.004,
            load_weight=0.2,
        )
        day = DayInputs(
            date=dt.date(2026, 2, 15),
            timezone=tz,
            sleep_override=None,
            doses=[],
            workload_level=1.0,
        )

        out = predict_day(baseline, day, step_minutes=10)

        self.assertEqual(len(out.t), 144)
        self.assertEqual(len(out.net), len(out.t))
        self.assertEqual(len(out.raw), len(out.t))
        self.assertEqual(len(out.circadian), len(out.t))
        self.assertEqual(len(out.sleep_pressure), len(out.t))
        self.assertEqual(len(out.drug), len(out.t))
        self.assertEqual(len(out.load), len(out.t))

        for v in out.net:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

        for key in ("prime", "crash", "sleep_gate"):
            self.assertIn(key, out.zones)
            self.assertEqual(len(out.zones[key]), len(out.t))

    def test_calibrate_baseline_offset_moves_toward_feedback(self):
        raised = calibrate_baseline_offset(
            current_offset=0.0,
            subjective_clarity_0_10=9.0,
            predicted_net_mean_0_1=0.4,
            eta=0.1,
            k=0.5,
        )
        lowered = calibrate_baseline_offset(
            current_offset=0.0,
            subjective_clarity_0_10=2.0,
            predicted_net_mean_0_1=0.7,
            eta=0.1,
            k=0.5,
        )
        self.assertGreater(raised, 0.0)
        self.assertLess(lowered, 0.0)

    def test_predict_day_marks_rebound_candidate_for_mph(self):
        tz = ZoneInfo("Asia/Seoul")
        baseline = UserBaseline(
            baseline_sleep_start=dt.time(23, 30),
            baseline_wake=dt.time(7, 30),
            drug_weight=0.004,
            load_weight=0.2,
            sleep_pressure_weight=1.2,
        )
        day = DayInputs(
            date=dt.date(2026, 2, 15),
            timezone=tz,
            doses=[
                Dose(time=dt.datetime(2026, 2, 15, 8, 0, tzinfo=tz), amount_mg=20.0, dose_type="mph_ir"),
                Dose(time=dt.datetime(2026, 2, 15, 9, 0, tzinfo=tz), amount_mg=100.0, dose_type="caffeine"),
            ],
            workload_level=1.0,
        )
        out = predict_day(baseline, day, step_minutes=10)
        self.assertIn("rebound_candidate", out.zones)
        self.assertTrue(any(out.zones["rebound_candidate"]))
        self.assertTrue(any(out.zones["crash"]))


if __name__ == "__main__":
    unittest.main()
