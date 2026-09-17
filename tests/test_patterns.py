import unittest

from chart.patterns import (
    describe_signature,
    find_in_run,
    find_patterns,
    signature_histogram,
    single_note_runs,
)
from chart.tokenizer import SimpleTokenizerGuitar

G, R, Y, B, O = 0, 1, 2, 3, 4


class PatternTest(unittest.TestCase):
    def setUp(self):
        self.tok = SimpleTokenizerGuitar()

    def _timed(self, lanes_seq, spacing=0.1, flag=0):
        out = []
        for i, lanes in enumerate(lanes_seq):
            lanes = (lanes,) if isinstance(lanes, int) else tuple(sorted(lanes))
            out.append((i * spacing, self.tok.compose(self.tok.chord_map[lanes], flag, 0), 0, {}))
        return out

    def test_descending_triplet_signature(self):
        # Robert's example: Y R G is a descending triplet.
        run = [(0.0, Y), (0.1, R), (0.2, G)]
        found = find_in_run(run, min_repeats=2)
        self.assertEqual(1, len(found))
        self.assertEqual((-1,), found[0].signature)
        self.assertEqual(2, found[0].repeats)

    def test_transposition_gives_the_same_signature(self):
        # Y R G and B Y R are the same shape on different frets.
        low = find_in_run([(0.0, Y), (0.1, R), (0.2, G)])
        high = find_in_run([(0.0, B), (0.1, Y), (0.2, R)])
        self.assertEqual(low[0].signature, high[0].signature)
        self.assertNotEqual(low[0].start_fret, high[0].start_fret)

    def test_trip_zig_is_found_as_a_repeating_cycle(self):
        # Robert's example: O B R B O B R B O B R.
        # Frets 4,3,1,3 -> deltas -1,-2,+2 and then +1 back up to O. The repeating unit
        # must include that return step, and it sums to zero -- which is what makes this a
        # zigzag rather than a run.
        lanes = [O, B, R, B, O, B, R, B, O, B, R]
        found = find_in_run([(i * 0.1, l) for i, l in enumerate(lanes)])
        self.assertTrue(found)
        self.assertEqual((-1, -2, 2, 1), found[0].signature)
        self.assertEqual(2, found[0].repeats)
        self.assertEqual(0, sum(found[0].signature), "a zigzag returns to where it started")

    def test_a_run_and_a_zigzag_are_distinguishable_by_net_motion(self):
        run = find_in_run([(i * 0.1, f) for i, f in enumerate([G, R, Y, B, O])])
        zig = find_in_run([(i * 0.1, f) for i, f in enumerate([O, B, R, B, O, B, R, B, O])])
        self.assertNotEqual(0, sum(run[0].signature))
        self.assertEqual(0, sum(zig[0].signature))

    def test_alternation_reports_the_short_unit(self):
        lanes = [G, Y, G, Y, G, Y, G, Y]
        found = find_in_run([(i * 0.1, l) for i, l in enumerate(lanes)])
        self.assertEqual((2, -2), found[0].signature)

    def test_chords_break_a_run(self):
        timed = self._timed([G, R, Y, (G, R), B, O])
        runs = single_note_runs(timed, self.tok)
        self.assertEqual(2, len(runs))
        self.assertEqual([G, R, Y], [f for _, f in runs[0]])
        self.assertEqual([B, O], [f for _, f in runs[1]])

    def test_a_long_gap_breaks_a_run(self):
        timed = self._timed([G, R, Y], spacing=0.1) + [
            (5.0, self.tok.compose(self.tok.chord_map[(B,)], 0, 0), 0, {}),
            (5.1, self.tok.compose(self.tok.chord_map[(O,)], 0, 0), 0, {}),
        ]
        self.assertEqual(2, len(single_note_runs(timed, self.tok)))

    def test_taps_do_not_break_a_run(self):
        timed = self._timed([G, R, Y, B], flag=2)
        self.assertEqual(1, len(single_note_runs(timed, self.tok)))

    def test_no_repetition_yields_nothing(self):
        found = find_in_run([(0.0, G), (0.1, O), (0.2, R)])
        self.assertEqual([], found)

    def test_histogram_counts_shapes(self):
        timed = self._timed([G, R, Y, B, O])
        hist = signature_histogram(find_patterns(timed, self.tok))
        self.assertEqual(1, hist[(1,)])

    def test_instance_reports_timing(self):
        found = find_in_run([(0.0, G), (0.25, R), (0.5, Y)])
        self.assertAlmostEqual(0.5, found[0].duration)
        self.assertGreater(found[0].notes_per_second, 0)

    def test_describe_signature(self):
        self.assertIn("ascending", describe_signature((1, 1)))
        self.assertIn("descending", describe_signature((-1,)))
        self.assertIn("alternation", describe_signature((2, -2)))
        self.assertIn("tremolo", describe_signature((0,)))
        self.assertIn("tremolo", describe_signature((0, 0)))
        self.assertIn("closed cycle", describe_signature((-1, -2, 2, 1)))


if __name__ == "__main__":
    unittest.main()
