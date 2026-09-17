"""Catalogue coverage needs its own chance floor to mean anything.

The catalogue holds two-delta shapes that random fret motion produces constantly, so
coverage alone rewards a chart for wandering the neck. These check that the floor
actually moves with the chart, and that authored structure scores above it.
"""

import unittest

from chart.discover import catalogue_lift


class CatalogueLiftTests(unittest.TestCase):

    def test_a_real_pattern_scores_above_its_floor(self):
        """A quint zig repeated is pure catalogue; shuffling it should score lower."""
        run = [0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0] * 3
        coverage, floor = catalogue_lift([run])
        self.assertGreater(coverage, floor)
        self.assertGreater(coverage - floor, 0.05)

    def test_the_floor_is_not_a_constant(self):
        """Two charts using the neck differently must get different floors, which is
        why raw coverage cannot be compared between charts."""
        wide = [[0, 4, 1, 3, 2, 4, 0, 3, 1, 2] * 4]
        narrow = [[0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 4]
        _, wide_floor = catalogue_lift(wide)
        _, narrow_floor = catalogue_lift(narrow)
        self.assertNotAlmostEqual(wide_floor, narrow_floor, places=2)

    def test_empty_input_is_zero_not_an_error(self):
        self.assertEqual((0.0, 0.0), catalogue_lift([]))
        self.assertEqual((0.0, 0.0), catalogue_lift([[3]]))

    def test_shuffling_is_deterministic(self):
        run = [0, 1, 2, 1, 0, 2, 3, 2, 1, 0] * 3
        self.assertEqual(catalogue_lift([run]), catalogue_lift([run]))


if __name__ == "__main__":
    unittest.main()
