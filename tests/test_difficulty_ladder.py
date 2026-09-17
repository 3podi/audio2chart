"""Difficulty is qualitative, and the ladder has to keep saying so.

Measured over 120 songs charted at all four levels by the same charter. The tempting
model -- harder means more notes -- is wrong, and these pin the specific ways it is
wrong so a future difficulty control is not built on it.
"""

import unittest

from chart.metrics import DIFFICULTY_LADDER

ORDER = ["Easy", "Medium", "Hard", "Expert"]


class DifficultyLadderTests(unittest.TestCase):

    def test_density_alone_cannot_separate_the_levels(self):
        """nps rises less than 2x across the whole ladder."""
        ratio = DIFFICULTY_LADDER["Expert"]["nps"] / DIFFICULTY_LADDER["Easy"]["nps"]
        self.assertLess(ratio, 2.0)

    def test_expert_is_defined_by_run_length(self):
        """Flat through Hard, then quadruples -- the real Expert signature."""
        hard = DIFFICULTY_LADDER["Hard"]["mean_run"]
        easy = DIFFICULTY_LADDER["Easy"]["mean_run"]
        expert = DIFFICULTY_LADDER["Expert"]["mean_run"]
        self.assertLess(abs(hard - easy) / easy, 0.15)
        self.assertGreater(expert / hard, 3.0)

    def test_chords_peak_below_expert(self):
        """Chord rate is not a difficulty axis: Hard uses more of them than Expert."""
        self.assertGreater(DIFFICULTY_LADDER["Hard"]["pct_chord"],
                           DIFFICULTY_LADDER["Expert"]["pct_chord"])

    def test_sustains_run_the_other_way(self):
        """Easier charts hold notes longer, so sustain is inversely related."""
        values = [DIFFICULTY_LADDER[level]["pct_sustain"] for level in ORDER]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_taps_and_patterns_rise_with_difficulty(self):
        for key in ("pct_tap", "pattern_lift"):
            values = [DIFFICULTY_LADDER[level][key] for level in ORDER]
            self.assertEqual(values, sorted(values), key)


if __name__ == "__main__":
    unittest.main()
