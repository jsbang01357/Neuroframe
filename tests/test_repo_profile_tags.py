from __future__ import annotations

import unittest

from storage.repo import NeuroRepo


class RepoProfileTagsTests(unittest.TestCase):
    def test_profile_json_tags_are_parsed(self):
        repo = NeuroRepo(None)
        row = {
            "profile_json": '{"atomoxetine": true, "ssri": false, "aripiprazole": true, "beta_blocker": false}'
        }
        tags = repo._row_to_medication_tags(row)
        self.assertTrue(tags["atomoxetine"])
        self.assertFalse(tags["ssri"])
        self.assertTrue(tags["aripiprazole"])
        self.assertFalse(tags["beta_blocker"])

    def test_legacy_tag_columns_fallback_when_profile_json_missing(self):
        repo = NeuroRepo(None)
        row = {
            "atomoxetine": "true",
            "ssri": "false",
            "aripiprazole": "yes",
            "beta_blocker": "0",
        }
        tags = repo._row_to_medication_tags(row)
        self.assertTrue(tags["atomoxetine"])
        self.assertFalse(tags["ssri"])
        self.assertTrue(tags["aripiprazole"])
        self.assertFalse(tags["beta_blocker"])


if __name__ == "__main__":
    unittest.main()
