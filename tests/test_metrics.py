import unittest

from chart.metrics import (
    CORPUS_REFERENCE,
    ChartProfile,
    compare,
    paired_deltas,
    profile_from_timed,
    scorecard,
    total_variation,
)
from chart.tokenizer import SimpleTokenizerGuitar


class ProfileTest(unittest.TestCase):
    def setUp(self):
        self.tok = SimpleTokenizerGuitar()

    def _timed(self, specs):
        """specs: list of (seconds, lanes, flag, sustain_bucket)."""
        out = []
        for seconds, lanes, flag, sustain in specs:
            chord = self.tok.chord_map[tuple(sorted(lanes))]
            out.append((seconds, self.tok.compose(chord, flag, sustain), 0, {}))
        return out

    def test_empty_chart_profiles_to_zero(self):
        profile = profile_from_timed([], self.tok)
        self.assertEqual(0, profile.positions)
        self.assertEqual(0.0, profile.nps)

    def test_counts_positions_and_density(self):
        specs = [(i * 0.5, (0,), 0, 0) for i in range(21)]      # 21 notes over 10s
        profile = profile_from_timed(self._timed(specs), self.tok)
        self.assertEqual(21, profile.positions)
        self.assertAlmostEqual(10.0, profile.duration_seconds, places=6)
        self.assertAlmostEqual(2.1, profile.nps, places=6)

    def test_chord_histogram(self):
        specs = [(0.0, (0,), 0, 0), (1.0, (0, 1), 0, 0), (2.0, (0, 1, 2), 0, 0), (3.0, (1,), 0, 0)]
        profile = profile_from_timed(self._timed(specs), self.tok)
        self.assertAlmostEqual(0.5, profile.chord_hist[1])
        self.assertAlmostEqual(0.25, profile.chord_hist[2])
        self.assertAlmostEqual(0.25, profile.chord_hist[3])

    def test_flags_and_sustains(self):
        specs = [(0.0, (0,), 2, 0), (1.0, (0,), 1, 0), (2.0, (0,), 0, 3), (3.0, (0,), 0, 0)]
        profile = profile_from_timed(self._timed(specs), self.tok)
        self.assertAlmostEqual(0.25, profile.pct_tap)
        self.assertAlmostEqual(0.25, profile.pct_forced)
        self.assertAlmostEqual(0.25, profile.pct_sustain)

    def test_lane_histogram_counts_every_lane_of_a_chord(self):
        specs = [(0.0, (0, 1), 0, 0)]
        profile = profile_from_timed(self._timed(specs), self.tok)
        self.assertAlmostEqual(0.5, profile.lane_hist[0])
        self.assertAlmostEqual(0.5, profile.lane_hist[1])

    def test_runs_break_on_a_gap(self):
        specs = [(0.0, (0,), 0, 0), (0.1, (0,), 0, 0), (0.2, (0,), 0, 0),
                 (5.0, (0,), 0, 0), (5.1, (0,), 0, 0)]
        profile = profile_from_timed(self._timed(specs), self.tok)
        self.assertEqual(3, profile.longest_run)
        self.assertAlmostEqual(2.5, profile.mean_run_length)

    def test_inter_onset_buckets(self):
        specs = [(0.0, (0,), 0, 0), (0.05, (0,), 0, 0), (2.05, (0,), 0, 0)]
        profile = profile_from_timed(self._timed(specs), self.tok)
        self.assertAlmostEqual(0.5, profile.ioi_hist["<=60ms"])
        self.assertAlmostEqual(0.5, profile.ioi_hist[">1s"])

    def test_nps_variation_is_zero_for_a_flat_chart(self):
        specs = [(i * 0.1, (0,), 0, 0) for i in range(300)]     # uniform over 30s
        profile = profile_from_timed(self._timed(specs), self.tok)
        self.assertLess(profile.nps_variation, 0.05)

    def test_nps_variation_detects_dynamics(self):
        dense = [(i * 0.05, (0,), 0, 0) for i in range(200)]    # busy first 10s
        sparse = [(20.0 + i * 2.0, (0,), 0, 0) for i in range(5)]
        profile = profile_from_timed(self._timed(dense + sparse), self.tok)
        self.assertGreater(profile.nps_variation, 0.5)

    def test_total_variation_orders_degenerate_above_typical(self):
        # Measured over 40 real charts this metric is mean 0.120 / max 0.596, so a single
        # chart's score is not a verdict -- only the ordering is meaningful.
        degenerate = ChartProfile(chord_hist={1: 0.992, 2: 0.008})
        healthy = ChartProfile(chord_hist={1: 0.873, 2: 0.089, 3: 0.027, 4: 0.006, 5: 0.005})
        self.assertGreater(total_variation(degenerate), total_variation(healthy))
        self.assertLess(total_variation(healthy), 0.01)

    def test_paired_deltas_cancel_for_identical_charts(self):
        specs = [(0.0, (0,), 0, 0), (0.5, (1, 2), 2, 0), (3.0, (3,), 0, 4)]
        profile = profile_from_timed(self._timed(specs), self.tok)
        for name, delta in paired_deltas(profile, profile).items():
            self.assertAlmostEqual(0.0, delta, places=9, msg=name)

    def test_paired_deltas_isolate_the_model_not_the_song(self):
        chords = profile_from_timed(self._timed(
            [(i * 0.5, (0, 1), 0, 0) for i in range(10)]), self.tok)
        singles = profile_from_timed(self._timed(
            [(i * 0.5, (0,), 0, 0) for i in range(10)]), self.tok)
        deltas = paired_deltas(singles, chords)
        self.assertAlmostEqual(1.0, deltas["chord_1"])
        self.assertAlmostEqual(-1.0, deltas["chord_2"])

    def test_compare_reports_every_reference_metric(self):
        profile = profile_from_timed(self._timed([(0.0, (0,), 0, 0), (1.0, (1,), 0, 0)]), self.tok)
        report = compare(profile)
        self.assertEqual(set(CORPUS_REFERENCE), set(report))
        for values in report.values():
            self.assertIn("delta", values)

    def test_scorecard_renders(self):
        profile = profile_from_timed(self._timed([(0.0, (0,), 0, 0), (1.0, (1,), 2, 0)]), self.tok)
        text = scorecard(profile)
        self.assertIn("chord-size total variation", text)
        self.assertIn("pct_tap", text)


if __name__ == "__main__":
    unittest.main()
