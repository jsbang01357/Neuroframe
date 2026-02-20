from __future__ import annotations

import datetime as dt
import json
import unittest
from zoneinfo import ZoneInfo

from shared.today_input import DoseDraft, doses_from_json, doses_to_json, drafts_to_engine_doses


class TodayInputTests(unittest.TestCase):
    def test_doses_to_json_includes_type_schema(self):
        tz = ZoneInfo("Asia/Seoul")
        payload = doses_to_json(
            dt.date(2026, 2, 16),
            tz,
            [
                DoseDraft(9, 0, 150.0, "caffeine"),
                DoseDraft(13, 0, 10.0, "mph_ir"),
            ],
        )
        arr = json.loads(payload)
        self.assertEqual(arr[0], {"hh": 9, "mm": 0, "mg": 150.0, "type": "caffeine"})
        self.assertEqual(arr[1], {"hh": 13, "mm": 0, "mg": 10.0, "type": "mph_ir"})

    def test_doses_from_json_legacy_without_type_falls_back_to_caffeine(self):
        raw = '[{"hh":9,"mm":0,"mg":150.0}]'
        out = doses_from_json(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].dose_type, "caffeine")

    def test_doses_from_json_legacy_field_names_supported(self):
        raw = '[{"hour":9,"minute":30,"amount_mg":120.0,"dose_type":"stimulant_xr"}]'
        out = doses_from_json(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].hour, 9)
        self.assertEqual(out[0].minute, 30)
        self.assertEqual(out[0].amount_mg, 120.0)
        self.assertEqual(out[0].dose_type, "mph_xr")

    def test_drafts_to_engine_doses_sets_dose_type(self):
        tz = ZoneInfo("Asia/Seoul")
        doses = drafts_to_engine_doses(
            dt.date(2026, 2, 16),
            tz,
            [DoseDraft(8, 30, 18.0, "mph_ir"), DoseDraft(9, 0, 150.0, "caffeine")],
        )
        self.assertEqual(len(doses), 2)
        self.assertEqual(doses[0].dose_type, "mph_ir")
        self.assertEqual(doses[1].dose_type, "caffeine")


if __name__ == "__main__":
    unittest.main()
