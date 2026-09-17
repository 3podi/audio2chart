import unittest

from chart.discover import FRAGMENT_LIFT, PATTERN_LIFT, bar_lift, classify


class BarLiftTest(unittest.TestCase):
    def test_lift_is_observed_over_expected(self):
        # 20 of 100 instances on a bar line, where the charts predict 5%.
        self.assertAlmostEqual(4.0, bar_lift(20, 100, 0.05))

    def test_chance_is_one(self):
        self.assertAlmostEqual(1.0, bar_lift(5, 100, 0.05))

    def test_degenerate_inputs_are_zero_not_an_error(self):
        self.assertEqual(0.0, bar_lift(0, 0, 0.05))
        self.assertEqual(0.0, bar_lift(5, 100, 0.0))

    def test_the_measured_cases_classify_correctly(self):
        # Tremolo and the split ladder are real; the G B R G family is residue.
        self.assertEqual("pattern-like", classify(2.97))   # G G G G G G G G
        self.assertEqual("pattern-like", classify(2.63))   # O Y B R Y G
        self.assertEqual("fragment", classify(0.09))       # G B R G O
        self.assertEqual("fragment", classify(0.08))       # G B R G Y

    def test_the_middle_is_reported_as_unclear_not_forced(self):
        self.assertEqual("unclear", classify(1.0))
        self.assertEqual("unclear", classify(FRAGMENT_LIFT))
        self.assertEqual("pattern-like", classify(PATTERN_LIFT))


if __name__ == "__main__":
    unittest.main()
