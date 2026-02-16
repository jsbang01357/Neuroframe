from __future__ import annotations

import unittest

from neuroframe_app import _parse_shift_blocks_json


class AppParserTests(unittest.TestCase):
    def test_parse_shift_blocks_json_filters_invalid_entries(self):
        raw = '[{"start":"08:00","end":"18:00","type":"day"},{"start":"bad","end":"18:00"},{"x":1}]'
        out = _parse_shift_blocks_json(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["start"], "08:00")
        self.assertEqual(out[0]["end"], "18:00")
        self.assertEqual(out[0]["type"], "day")

    def test_parse_shift_blocks_json_returns_empty_on_invalid_json(self):
        out = _parse_shift_blocks_json("{not-json")
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
