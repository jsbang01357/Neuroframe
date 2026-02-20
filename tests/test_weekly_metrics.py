from __future__ import annotations

import unittest

from shared.weekly_metrics import compute_caffeine_total_mg_from_doses_json, compute_sleep_debt_hours


class WeeklyMetricsTests(unittest.TestCase):
    def test_compute_sleep_debt_hours(self):
        # target 8h/day for 7 days = 56h, actual 50h => 6h debt
        debt = compute_sleep_debt_hours(8.0, [7.0, 8.0, 6.0, 7.0, 8.0, 7.0, 7.0])
        self.assertAlmostEqual(debt, 6.0, places=6)

    def test_compute_caffeine_total_mg_from_doses_json(self):
        rows = [
            '[{"hh":9,"mm":0,"mg":150.0,"type":"caffeine"},{"hh":14,"mm":0,"mg":70.0,"type":"caffeine"}]',
            '[{"hh":9,"mm":0,"mg":20.0,"type":"mph_ir"},{"hh":13,"mm":30,"mg":80.0,"type":"caffeine"}]',
            "[]",
        ]
        total = compute_caffeine_total_mg_from_doses_json(rows)
        self.assertAlmostEqual(total, 300.0, places=6)


if __name__ == "__main__":
    unittest.main()
