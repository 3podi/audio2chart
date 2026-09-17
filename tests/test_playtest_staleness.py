"""A built playtest folder has to notice when its inputs move.

Robert played the generated charts and reported that Diary of Jane had the wrong audio
-- it sounded like the guitar stem alone, with drums only in the intro. The song ships
as seven stems, and the manifest was repointed at their mix six hours after that folder
was built. The data was fixed and the chart could never see it, because build_playtest
skipped any folder that already had a notes.chart.
"""

import os
import tempfile
import unittest
from pathlib import Path

from build_playtest import stale_inputs


class StalenessTests(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.generated = Path(self.directory.name) / "notes.chart"
        self.generated.write_text("[Song]\n", encoding="utf-8")
        os.utime(self.generated, (1_000_000, 1_000_000))
        self.addCleanup(self.directory.cleanup)

    def test_nothing_changed_means_nothing_to_rebuild(self):
        self.assertEqual(stale_inputs(self.generated, 999_000, 999_000), [])

    def test_newer_audio_is_the_diary_of_jane_case(self):
        """The exact failure: the mix was cached after the chart was written."""
        self.assertEqual(stale_inputs(self.generated, 1_000_001, 999_000), ["audio"])

    def test_newer_model_rebuilds_too(self):
        """Otherwise a retrained model quietly leaves the old charts in the library."""
        self.assertEqual(stale_inputs(self.generated, 999_000, 1_000_001), ["model"])

    def test_both_are_reported_so_the_message_says_why(self):
        self.assertEqual(stale_inputs(self.generated, 1_000_001, 1_000_002),
                         ["audio", "model"])

    def test_a_missing_stamp_is_not_treated_as_stale(self):
        """An empty model directory gives None, which must not force endless rebuilds."""
        self.assertEqual(stale_inputs(self.generated, None, None), [])

    def test_equal_timestamps_are_not_stale(self):
        """Copying preserves mtime, so an input exactly as old as the chart is fine."""
        self.assertEqual(stale_inputs(self.generated, 1_000_000, 1_000_000), [])


if __name__ == "__main__":
    unittest.main()
