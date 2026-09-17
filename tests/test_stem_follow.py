"""Detecting which line a chart follows has to refuse to guess.

The point is to turn Robert's overchart rule -- vocals for crucial parts, instrumental
where there are none, never both -- into labels. A detector that always names a winner
would manufacture that pattern whether or not it is there.
"""

import unittest

import numpy as np

from chart.stem_follow import label_windows, summarise, switches, windows


class LabellingTests(unittest.TestCase):

    def test_a_clear_winner_is_named(self):
        self.assertEqual(["vocals"], label_windows([{"vocals": 0.6, "other": 0.1}]))

    def test_a_tie_is_not_a_winner(self):
        """Correlated stems score alike; that is ignorance, not following both."""
        self.assertEqual(["none"], label_windows([{"vocals": 0.42, "other": 0.40}]))

    def test_negative_scores_never_win(self):
        self.assertEqual(["none"], label_windows([{"vocals": -0.1, "other": -0.5}]))

    def test_empty_window_is_none(self):
        self.assertEqual(["none"], label_windows([{}]))


class SummaryTests(unittest.TestCase):

    def test_shares_sum_to_one(self):
        shares = summarise(["vocals", "vocals", "other", "none"])
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertAlmostEqual(shares["vocals"], 0.5)

    def test_switches_ignore_unclear_windows(self):
        """A gap in the measurement is not a change of line."""
        self.assertEqual(1, switches(["vocals", "none", "vocals", "other"]))

    def test_a_chart_following_one_line_never_switches(self):
        self.assertEqual(0, switches(["other"] * 8))


class WindowTests(unittest.TestCase):

    def test_windows_cover_the_song(self):
        spans = windows(95.0, window=10.0)
        self.assertEqual(9, len(spans))
        self.assertEqual(0.0, spans[0][0])

    def test_a_short_song_still_gets_one_window(self):
        self.assertEqual(1, len(windows(4.0, window=10.0)))


if __name__ == "__main__":
    unittest.main()
