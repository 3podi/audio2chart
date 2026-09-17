import unittest

from chart.discover import (
    Discovery, claim_run, known_signatures, render, report, unexplained,
)


class DiscoverTest(unittest.TestCase):
    def test_render_places_a_shape_on_the_neck(self):
        self.assertEqual("G R Y", render((1, 1)))
        self.assertEqual("G B R G Y", render((3, -2, -1, 2)))

    def test_render_reports_shapes_that_do_not_fit(self):
        self.assertIn("does not fit", render((4, 4)))

    def test_longest_first_prevents_a_short_shape_eating_a_long_one(self):
        deltas = [1, 1, 1, 1]
        # (1,1,1,1) present as well as (1,1); the long one must win.
        claimed = claim_run(deltas, [(1, 1, 1, 1), (1, 1)])
        self.assertEqual([True] * 4, claimed)

    def test_unclaimed_regions_are_returned(self):
        deltas = [1, 1, 9, 9, 9, 1, 1]
        claimed = claim_run(deltas, [(1, 1)])
        self.assertEqual([(9, 9, 9)], unexplained(claimed, deltas, min_length=2))

    def test_short_residue_is_filtered_out(self):
        deltas = [1, 1, 9, 1, 1]
        claimed = claim_run(deltas, [(1, 1)])
        self.assertEqual([], unexplained(claimed, deltas, min_length=2))
        self.assertEqual([(9,)], unexplained(claimed, deltas, min_length=1))

    def test_nothing_known_means_everything_is_a_candidate(self):
        deltas = [1, 2, 3]
        claimed = claim_run(deltas, [])
        self.assertEqual([(1, 2, 3)], unexplained(claimed, deltas, min_length=1))

    def test_known_signatures_uses_bodies_not_whole_sequences(self):
        known = known_signatures({"zig": "B Y | R Y B Y R"})
        self.assertIn((1, 1, -1, -1), known)          # body only
        self.assertNotIn((-1, -1, 1, 1, -1, -1), known)

    def test_coverage_is_zero_for_an_empty_scan(self):
        self.assertEqual(0.0, Discovery().coverage)

    def test_report_renders(self):
        d = Discovery(deltas_seen=10, deltas_claimed=7)
        d.candidates[(3, -2, -1, 2)] = 5
        d.charts_per_candidate[(3, -2, -1, 2)] = 2
        text = report(d)
        self.assertIn("70.0%", text)
        self.assertIn("G B R G Y", text)


if __name__ == "__main__":
    unittest.main()
