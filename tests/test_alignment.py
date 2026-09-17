import unittest

import numpy as np

from chart.alignment import DEFAULT_TOLERANCE, agreement_with_reference, alignment


class AlignmentTest(unittest.TestCase):
    def test_perfect_match(self):
        times = [0.0, 0.5, 1.0, 1.5]
        result = alignment(times, times)
        self.assertAlmostEqual(1.0, result.precision)
        self.assertAlmostEqual(1.0, result.recall)
        self.assertAlmostEqual(1.0, result.f1)

    def test_a_shifted_chart_loses_alignment(self):
        onsets = [0.0, 0.5, 1.0, 1.5]
        shifted = [t + 0.3 for t in onsets]          # well beyond tolerance
        result = alignment(shifted, onsets)
        self.assertEqual(0.0, result.precision)
        self.assertEqual(0.0, result.recall)

    def test_a_shift_inside_tolerance_still_matches(self):
        onsets = [0.0, 0.5, 1.0]
        nudged = [t + DEFAULT_TOLERANCE * 0.8 for t in onsets]
        self.assertAlmostEqual(1.0, alignment(nudged, onsets).precision)

    def test_median_offset_reveals_a_systematic_lag(self):
        onsets = [0.0, 0.5, 1.0, 1.5]
        late = [t + 0.02 for t in onsets]
        result = alignment(late, onsets)
        # Notes are late, so the nearest onset lies behind them: negative offset.
        self.assertAlmostEqual(-0.02, result.median_offset, places=6)

    def test_precision_and_recall_come_apart(self):
        onsets = [i * 0.5 for i in range(20)]
        # One note, perfectly placed: precision 1, recall almost 0.
        sparse = alignment([0.0], onsets)
        self.assertAlmostEqual(1.0, sparse.precision)
        self.assertLess(sparse.recall, 0.1)

        # A note every 10 ms: recall 1, precision poor.
        spam = alignment([i * 0.01 for i in range(1000)], onsets)
        self.assertAlmostEqual(1.0, spam.recall)
        self.assertLess(spam.precision, 0.5)

    def test_empty_inputs_are_not_fatal(self):
        self.assertEqual(0.0, alignment([], [1.0]).f1)
        self.assertEqual(0.0, alignment([1.0], []).f1)
        self.assertEqual(0.0, alignment([], []).f1)

    def test_overcharting_shows_as_high_recall_low_precision(self):
        reference = [i * 0.5 for i in range(20)]
        overcharted = sorted(reference + [i * 0.5 + 0.25 for i in range(20)])
        result = agreement_with_reference(overcharted, reference)
        self.assertAlmostEqual(1.0, result.recall)
        self.assertAlmostEqual(0.5, result.precision)

    def test_undercharting_shows_as_high_precision_low_recall(self):
        reference = [i * 0.5 for i in range(20)]
        result = agreement_with_reference(reference[:5], reference)
        self.assertAlmostEqual(1.0, result.precision)
        self.assertAlmostEqual(0.25, result.recall)

    def test_counts_are_reported(self):
        result = alignment([0.0, 1.0], [0.0, 1.0, 2.0])
        self.assertEqual(2, result.notes)
        self.assertEqual(3, result.onsets)
        self.assertEqual(DEFAULT_TOLERANCE, result.tolerance)


if __name__ == "__main__":
    unittest.main()
