"""Pattern grades mean nothing without their null.

Expert sections hold far more notes than Easy ones, so a pattern with no difficulty
preference at all still scores 2.76 on a 1-4 scale. Reading a grade as "above 2.5 means
hard" would call almost the whole catalogue a difficulty marker.
"""

import unittest

from chart.catalogue import CATALOGUE
from chart.grades import (MIN_OCCURRENCES, NULL_GRADE, PATTERN_GRADES, markers,
                          relative_grade)


class GradeTests(unittest.TestCase):

    def test_the_null_is_well_above_the_midpoint(self):
        """If it were 2.5, no correction would be needed and none would be applied."""
        self.assertGreater(NULL_GRADE, 2.6)

    def test_most_of_the_catalogue_marks_nothing(self):
        """Anchors and sweeps appear at every level; only a minority indicate Expert."""
        self.assertLess(len(markers()), len(PATTERN_GRADES) / 3)

    def test_split_zigs_lead_the_markers(self):
        self.assertIn("split", markers()[0])

    def test_universal_vocabulary_sits_below_the_null(self):
        self.assertLess(relative_grade("shape B-G-B"), 0.0)

    def test_grades_are_on_the_level_scale(self):
        for name, (grade, count) in PATTERN_GRADES.items():
            self.assertTrue(1.0 <= grade <= 4.0, name)
            self.assertGreaterEqual(count, MIN_OCCURRENCES, name)

    def test_every_graded_pattern_is_known(self):
        """Grades track the catalogue and the shapes demoted out of it."""
        from chart.catalogue import UNCONFIRMED_SHAPES

        known = set(CATALOGUE) | set(UNCONFIRMED_SHAPES)
        for name in PATTERN_GRADES:
            self.assertIn(name, known)

    def test_an_unknown_pattern_has_no_grade(self):
        self.assertIsNone(relative_grade("not a pattern"))


if __name__ == "__main__":
    unittest.main()


class MarkerShareTests(unittest.TestCase):
    """Expert charts are arrangements of ordinary vocabulary, not exotic patterns."""

    def test_human_expert_charts_barely_use_marker_patterns(self):
        from chart.grades import HUMAN_MARKER_SHARE
        self.assertLess(HUMAN_MARKER_SHARE, 0.05)

    def test_the_generated_overuse_is_several_sd_out(self):
        """14.9% against a human 2.7% -- the number this baseline exists to catch."""
        from chart.grades import HUMAN_MARKER_SHARE, HUMAN_MARKER_SHARE_SD
        sigmas = (0.149 - HUMAN_MARKER_SHARE) / HUMAN_MARKER_SHARE_SD
        self.assertGreater(sigmas, 3.0)


class RetractedClaimsTests(unittest.TestCase):
    """Two supports for the 'no dynamics' reading did not survive their baselines.

    Pinned because both looked obviously damning in isolation, and the second was
    contradicted by a measurement already in this repo.
    """

    def test_expert_runs_really_are_that_long(self):
        """235 notes was cited as abnormal; Expert's own mean is 232."""
        from chart.metrics import DIFFICULTY_LADDER
        self.assertGreater(DIFFICULTY_LADDER["Expert"]["mean_run"], 200)
