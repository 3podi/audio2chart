"""The two corpus references answer different questions and must not be mixed.

CORPUS_REFERENCE is pooled over note positions; PER_CHART_REFERENCE is the unweighted
mean over charts. A ChartProfile describes one chart, so it belongs against the
per-chart numbers -- scoring it against the pooled ones overstates how far off density
is, which is a mistake the module's own docstring warns about and I made anyway.
"""

import unittest

from chart.metrics import CORPUS_REFERENCE, PER_CHART_MEDIAN, PER_CHART_REFERENCE


class ReferenceTests(unittest.TestCase):

    def test_the_two_references_disagree_where_it_matters(self):
        """If these ever coincided, the distinction would not need documenting."""
        self.assertGreater(PER_CHART_REFERENCE["nps"], CORPUS_REFERENCE["nps"] * 1.3)
        self.assertGreater(PER_CHART_REFERENCE["pct_sustain"],
                           CORPUS_REFERENCE["pct_sustain"] * 2)

    def test_tap_and_sustain_are_skewed(self):
        """Mean well above median is why a single chart cannot be judged by the mean."""
        self.assertGreater(PER_CHART_REFERENCE["pct_tap"], PER_CHART_MEDIAN["pct_tap"] * 1.5)
        self.assertGreater(PER_CHART_REFERENCE["pct_sustain"],
                           PER_CHART_MEDIAN["pct_sustain"] * 2.5)

    def test_both_carry_the_same_keys(self):
        for key in PER_CHART_MEDIAN:
            self.assertIn(key, PER_CHART_REFERENCE)

    def test_proportions_are_proportions(self):
        for name, table in (("mean", PER_CHART_REFERENCE), ("median", PER_CHART_MEDIAN)):
            for key, value in table.items():
                if key == "nps":
                    continue
                self.assertTrue(0.0 <= value <= 1.0, f"{name}[{key}] = {value}")


if __name__ == "__main__":
    unittest.main()


class GeneratedBaselineTests(unittest.TestCase):
    """Where this model stood, so a future one can be compared against it."""

    def test_taps_are_not_a_defect(self):
        """Recorded because "zero taps" was claimed from a single chart and is wrong."""
        from chart.metrics import GENERATED_BASELINE, PER_CHART_MEDIAN
        self.assertGreater(GENERATED_BASELINE["pct_tap"], PER_CHART_MEDIAN["pct_tap"])

    def test_the_real_defects_are_recorded(self):
        from chart.metrics import GENERATED_BASELINE, PER_CHART_MEDIAN
        self.assertLess(GENERATED_BASELINE["pct_sustain"], PER_CHART_MEDIAN["pct_sustain"])
        self.assertGreater(GENERATED_BASELINE["nps"], 3 * PER_CHART_MEDIAN["nps"])
        self.assertLess(GENERATED_BASELINE["rests_per_minute"], 5.0)
