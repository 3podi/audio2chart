"""The scorecard has to flag the defects that actually occur.

The first version reported chord_1, the single-note rate. A chart with no chords at
all reads 1.000 against a human 0.872, and a relative tolerance on a number that close
to 1 passed it silently -- hiding the most conspicuous thing wrong with the output.
"""

import unittest

import evaluate_chart


class ScorecardTests(unittest.TestCase):

    def test_a_missing_feature_is_flagged(self):
        """Zero taps against a human 0.194 must not pass quietly."""
        self.assertTrue(evaluate_chart.line("pct_tap", 0.0, 0.194).endswith("!"))

    def test_no_chords_at_all_is_flagged(self):
        """The case the first version missed: chord rate 0 against 0.128."""
        self.assertTrue(evaluate_chart.line("pct_chord", 0.0, 0.128).endswith("!"))

    def test_a_close_value_is_not_flagged(self):
        self.assertFalse(evaluate_chart.line("pct_tap", 0.190, 0.194).endswith("!"))

    def test_tiny_targets_do_not_flag_on_noise(self):
        """A 25% band on a near-zero target would flag every rounding difference."""
        self.assertFalse(evaluate_chart.line("lane_open", 0.010, 0.002).endswith("!"))


class OverlapTests(unittest.TestCase):

    def test_detects_a_sustain_running_into_the_next_note(self):
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp()) / "notes.chart"
        path.write_text("[ExpertSingle]\n{\n"
                        "  0 = N 0 200\n  100 = N 0 0\n  400 = N 1 0\n}\n",
                        encoding="utf-8")
        self.assertEqual((1, 1), evaluate_chart.overlapping_sustains(path))

    def test_a_sustain_ending_before_the_next_note_is_clean(self):
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp()) / "notes.chart"
        path.write_text("[ExpertSingle]\n{\n"
                        "  0 = N 0 50\n  100 = N 0 0\n}\n", encoding="utf-8")
        self.assertEqual((0, 1), evaluate_chart.overlapping_sustains(path))


if __name__ == "__main__":
    unittest.main()


class PhrasingTests(unittest.TestCase):
    """A human Expert chart breathes constantly; the current model does not."""

    def test_counts_rests_per_minute(self):
        times = [0.0, 0.1, 0.2, 5.0, 5.1, 10.0]      # two gaps over 0.25s in 10s
        per_minute, typical = evaluate_chart.phrasing(times)
        self.assertAlmostEqual(per_minute, 12.0, places=5)
        self.assertAlmostEqual(typical, 4.85, places=2)

    def test_continuous_play_has_no_rests(self):
        times = [i * 0.1 for i in range(200)]
        self.assertEqual((0.0, 0.0), evaluate_chart.phrasing(times))

    def test_duplicate_timestamps_are_one_position(self):
        """Chords share a timestamp and must not read as a zero-length gap."""
        times = [0.0, 0.0, 0.0, 1.0, 1.0]
        per_minute, typical = evaluate_chart.phrasing(times)
        self.assertAlmostEqual(typical, 1.0, places=5)

    def test_too_few_notes_is_zero_not_an_error(self):
        self.assertEqual((0.0, 0.0), evaluate_chart.phrasing([]))
        self.assertEqual((0.0, 0.0), evaluate_chart.phrasing([1.0]))
