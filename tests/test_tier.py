"""Tier features have to separate charts that average density cannot.

The difficulty ladder showed notes per second rising only 1.8x from Easy to Expert
while mean run length quadrupled, so a grader built on average rate would put a
sustained shred chart and a sparse one in the same place.
"""

import unittest

from chart.tier import hand_movement, longest_fast_stretch, peak_density


class PeakDensityTests(unittest.TestCase):

    def test_finds_the_busiest_window_not_the_average(self):
        """A quiet chart with one hard passage must score on the passage."""
        burst = [100.0 + i * 0.05 for i in range(200)]      # 20 nps for 10s
        sparse = [i * 2.0 for i in range(50)]               # 0.5 nps for 100s
        self.assertGreater(peak_density(sparse + burst), 15.0)

    def test_empty_input_is_zero(self):
        self.assertEqual(0.0, peak_density([]))
        self.assertEqual(0.0, peak_density([1.0]))


class EnduranceTests(unittest.TestCase):

    def test_measures_the_longest_unbroken_fast_run(self):
        fast = [i * 0.05 for i in range(100)]               # 5s at 20 nps
        later = [50.0 + i * 0.05 for i in range(20)]        # 1s at 20 nps
        self.assertAlmostEqual(longest_fast_stretch(fast + later), 4.95, places=2)

    def test_a_slow_chart_has_no_fast_stretch(self):
        self.assertEqual(0.0, longest_fast_stretch([i * 0.5 for i in range(20)]))

    def test_a_run_reaching_the_end_still_counts(self):
        """The stretch is closed at the final note, not dropped."""
        self.assertGreater(longest_fast_stretch([i * 0.05 for i in range(40)]), 1.8)


class HandMovementTests(unittest.TestCase):

    def test_wide_jumps_need_a_reposition(self):
        _, wide = hand_movement([[0, 4, 0, 4]])
        self.assertEqual(1.0, wide)

    def test_adjacent_frets_are_finger_rolls(self):
        mean, wide = hand_movement([[0, 1, 2, 3, 4]])
        self.assertEqual(0.0, wide)
        self.assertEqual(1.0, mean)

    def test_no_runs_is_zero_not_an_error(self):
        self.assertEqual((0.0, 0.0), hand_movement([]))


if __name__ == "__main__":
    unittest.main()


class AnchorTests(unittest.TestCase):
    """Anchoring is a technique across a passage, not a shape.

    Robert: hold the lowest note down and do not let it up until it is safe. Unlike
    slide-versus-anchor, which is invisible because both produce identical notes, this
    is measurable from the chart.
    """

    def test_a_quad_zig_anchors_on_its_lowest_fret(self):
        """Robert's example: R Y B Y R G repeated, index parked on green."""
        from chart.tier import anchorable
        self.assertTrue(anchorable([1, 2, 3, 2, 1, 0] * 4))

    def test_a_passage_wider_than_the_hand_cannot_anchor(self):
        """G R Y B is one position; adding O forces a shift."""
        from chart.tier import anchorable
        self.assertTrue(anchorable([0, 1, 2, 3, 0, 1, 2, 3]))
        self.assertFalse(anchorable([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]))

    def test_the_lowest_fret_must_actually_recur(self):
        from chart.tier import anchorable
        self.assertFalse(anchorable([0, 1, 2, 3, 2, 1, 2, 3, 2, 1]))

    def test_a_long_gap_means_letting_go(self):
        from chart.tier import anchorable
        self.assertFalse(anchorable([0] + [1, 2, 3] * 9 + [0], window=4))

    def test_share_counts_notes_not_runs(self):
        from chart.tier import anchor_share
        held = [1, 2, 3, 2, 1, 0] * 2          # 12 notes, anchorable
        loose = [0, 4, 0, 4]                    # 4 notes, too wide
        self.assertAlmostEqual(anchor_share([held, loose]), 12 / 16)
