from __future__ import annotations

from typing import List

from .today_input import doses_from_json


def compute_sleep_debt_hours(target_sleep_h: float, sleep_hours_by_day: List[float]) -> float:
    total_target = max(0.0, float(target_sleep_h)) * max(0, len(sleep_hours_by_day))
    actual = sum(max(0.0, float(h)) for h in sleep_hours_by_day)
    return max(0.0, total_target - actual)


def compute_caffeine_total_mg_from_doses_json(doses_json_by_day: List[str]) -> float:
    total = 0.0
    for raw in doses_json_by_day:
        for d in doses_from_json(raw or "[]"):
            if str(getattr(d, "dose_type", "caffeine") or "caffeine").strip().lower() == "caffeine":
                total += max(0.0, float(d.amount_mg))
    return total
