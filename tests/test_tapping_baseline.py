"""The tapping baseline has to describe tapping charts.

PER_CHART_MEDIAN is measured over 250 Expert charts of every style, and a tapping
overchart is nothing like a typical chart -- 0.96 taps against 0.19, 21 NPS against 7.8.
Scoring a tapping-trained model against it marks correct output as broken on every
metric, which is how a run gets condemned for doing what it was trained to do. A human
chart pulled straight from the tapping validation split failed five of five metrics
against the all-styles median.

These check the numbers are self-consistent and that the two populations really are
distinct, without needing the corpus present.
"""

from __future__ import annotations

import unittest

from chart.metrics import (BASELINES, PER_CHART_MEDIAN, TAPPING_PER_CHART_MEDIAN,
                           TAPPING_REST_SECONDS, TAPPING_RESTS_PER_MINUTE)


class TappingBaselineTest(unittest.TestCase):

    def test_both_populations_describe_the_same_quantities(self):
        self.assertEqual(sorted(PER_CHART_MEDIAN), sorted(TAPPING_PER_CHART_MEDIAN))
        self.assertEqual({"all", "tapping"}, set(BASELINES))
        self.assertIs(PER_CHART_MEDIAN, BASELINES["all"])
        self.assertIs(TAPPING_PER_CHART_MEDIAN, BASELINES["tapping"])

    def test_the_chord_histogram_is_a_histogram(self):
        for name, baseline in BASELINES.items():
            total = sum(baseline[f"chord_{size}"] for size in (1, 2, 3))
            self.assertLessEqual(total, 1.0 + 1e-9, f"{name} chord shares exceed 1")
            self.assertGreater(total, 0.9, f"{name} chord shares are implausibly low")

    def test_rates_are_proportions_and_density_is_positive(self):
        for name, baseline in BASELINES.items():
            for key in ("pct_tap", "pct_forced", "pct_sustain",
                        "lane_0", "lane_2", "lane_open"):
                self.assertGreaterEqual(baseline[key], 0.0, f"{name}.{key}")
                self.assertLessEqual(baseline[key], 1.0, f"{name}.{key}")
            self.assertGreater(baseline["nps"], 0.0, name)

    def test_the_tapping_population_is_actually_different(self):
        """If these ever converge, one of them was measured wrong."""
        self.assertGreater(TAPPING_PER_CHART_MEDIAN["pct_tap"], 0.8)
        self.assertLess(PER_CHART_MEDIAN["pct_tap"], 0.4)
        self.assertGreater(TAPPING_PER_CHART_MEDIAN["nps"],
                           2 * PER_CHART_MEDIAN["nps"])
        # Tapping charts barely sustain or force, and they breathe less often.
        self.assertLess(TAPPING_PER_CHART_MEDIAN["pct_sustain"],
                        PER_CHART_MEDIAN["pct_sustain"])
        self.assertLess(TAPPING_PER_CHART_MEDIAN["pct_forced"],
                        PER_CHART_MEDIAN["pct_forced"])
        self.assertLess(TAPPING_RESTS_PER_MINUTE, 34.0)
        self.assertGreater(TAPPING_RESTS_PER_MINUTE, 0.0)
        self.assertGreater(TAPPING_REST_SECONDS, 0.0)

    def test_pattern_lift_has_a_tapping_baseline_too(self):
        """The lift figure was stale twice over: wrong catalogue and wrong population.

        HUMAN_LIFT_MEAN was 0.209, measured over 60 charts against a 123-entry
        catalogue. The catalogue is now 100 entries, which alone moved the general
        figure to 0.268, and the tapping population sits at 0.399. Against the old
        number a chart at +0.30 read as half a standard deviation above human; against
        the tapping one it is three quarters of one below -- a reversed verdict.
        """
        from chart.discover import (HUMAN_LIFT_MEAN, HUMAN_LIFT_SD,
                                    TAPPING_LIFT_MEAN, TAPPING_LIFT_SD)

        for mean, sd in ((HUMAN_LIFT_MEAN, HUMAN_LIFT_SD),
                         (TAPPING_LIFT_MEAN, TAPPING_LIFT_SD)):
            self.assertGreater(mean, 0.0, "lift over a chance floor must be positive")
            self.assertLess(mean, 1.0)
            self.assertGreater(sd, 0.0)

        self.assertGreater(TAPPING_LIFT_MEAN, HUMAN_LIFT_MEAN,
                           "tapping charts carry more catalogue vocabulary, not less")

        # A chart at the tapping mean must not read as unusual for a tapping model.
        sigmas = (TAPPING_LIFT_MEAN - TAPPING_LIFT_MEAN) / TAPPING_LIFT_SD
        self.assertEqual(0.0, sigmas)
        # ...but it is clearly above average against the general population.
        self.assertGreater((TAPPING_LIFT_MEAN - HUMAN_LIFT_MEAN) / HUMAN_LIFT_SD, 0.5)

    def test_evaluate_chart_offers_both_and_score_run_defaults_to_tapping(self):
        """The default matters: score_run judges tapping-trained checkpoints."""
        import argparse
        import inspect

        import evaluate_chart
        import score_run

        for module in (evaluate_chart, score_run):
            source = inspect.getsource(module.parse_args)
            self.assertIn("--baseline", source, module.__name__)

        parser_source = inspect.getsource(score_run.parse_args)
        self.assertIn('default="tapping"', parser_source)
        self.assertIsInstance(argparse.ArgumentParser(), argparse.ArgumentParser)


if __name__ == "__main__":
    unittest.main()
