import tempfile
import unittest
from pathlib import Path

from chart.bars import (
    bar_boundaries, bar_length_ticks, bars_for_chart, beat_boundaries,
    beat_length_ticks, beats_for_chart, group_by_bar, parse_time_signatures,
    soft_bar_cuts,
)

CHART = """[Song]
{
  Resolution = 192
}
[SyncTrack]
{
  0 = TS 4
  0 = B 120000
  1536 = TS 3
}
[ExpertSingle]
{
  0 = N 0 0
}
"""


class BarTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "notes.chart"
        self.path.write_text(CHART, encoding="utf-8")

    def tearDown(self):
        self.dir.cleanup()

    def test_parses_time_signature_changes(self):
        self.assertEqual([(0, 4, 4), (1536, 3, 4)], parse_time_signatures(self.path))

    def test_missing_denominator_defaults_to_four(self):
        # ".. = TS 4" with no second value means 4/4, not 4/1.
        self.assertEqual(4, parse_time_signatures(self.path)[0][2])

    def test_explicit_denominator_exponent_is_a_power_of_two(self):
        p = Path(self.dir.name) / "eighths.chart"
        p.write_text("[SyncTrack]\n{\n  0 = TS 6 3\n}\n", encoding="utf-8")
        self.assertEqual((0, 6, 8), parse_time_signatures(p)[0])

    def test_defaults_to_four_four_when_absent(self):
        p = Path(self.dir.name) / "none.chart"
        p.write_text("[SyncTrack]\n{\n  0 = B 120000\n}\n", encoding="utf-8")
        self.assertEqual([(0, 4, 4)], parse_time_signatures(p))

    def test_bar_length(self):
        self.assertEqual(768, bar_length_ticks(4, 4, 192))     # 4/4 at 192 ppq
        self.assertEqual(576, bar_length_ticks(3, 4, 192))     # 3/4
        self.assertEqual(576, bar_length_ticks(6, 8, 192))     # 6/8

    def test_boundaries_follow_a_time_signature_change(self):
        b = bar_boundaries(self.path, resolution=192, last_tick=3000)
        self.assertEqual([0, 768, 1536, 2112, 2688], b[:5])
        # 4/4 bars of 768 until 1536, then 3/4 bars of 576.

    def test_grouping_puts_notes_in_the_right_bar(self):
        encoded = [(0, 1, 0, {}), (700, 2, 0, {}), (800, 3, 0, {}), (1600, 4, 0, {})]
        bars = group_by_bar(encoded, [0, 768, 1536, 2304])
        self.assertEqual([0, 700], [e[0] for e in bars[0]])
        self.assertEqual([800], [e[0] for e in bars[1]])
        self.assertEqual([1600], [e[0] for e in bars[2]])   # 1536 <= 1600 < 2304

    def test_empty_bars_are_preserved_so_index_is_bar_number(self):
        bars = group_by_bar([(0, 1, 0, {}), (2400, 2, 0, {})], [0, 768, 1536, 2304])
        self.assertEqual(4, len(bars))
        self.assertEqual([], bars[1])
        self.assertEqual([], bars[2])

    def test_bars_for_chart_end_to_end(self):
        encoded = [(0, 1, 0, {}), (900, 2, 0, {})]
        bars = bars_for_chart(self.path, encoded, resolution=192)
        self.assertEqual([0], [e[0] for e in bars[0]])
        self.assertEqual([900], [e[0] for e in bars[1]])

    def test_no_notes_gives_no_bars(self):
        self.assertEqual([], bars_for_chart(self.path, [], resolution=192))


class SoftCutTest(unittest.TestCase):
    """A bar line is a candidate boundary, confirmed only if the motion changes there."""

    def test_tremolo_is_not_cut(self):
        ticks = list(range(0, 1600, 100))              # crosses a 768-tick bar line
        deltas = [0] * (len(ticks) - 1)                # same fret throughout
        self.assertEqual([], soft_bar_cuts(ticks, deltas, [0, 768, 1536]))

    def test_a_steady_run_is_not_cut(self):
        ticks = list(range(0, 1600, 100))
        deltas = [1] * (len(ticks) - 1)
        self.assertEqual([], soft_bar_cuts(ticks, deltas, [0, 768, 1536]))

    def test_an_oscillation_carrying_through_is_not_cut(self):
        ticks = list(range(0, 1600, 100))
        deltas = [1, -1] * ((len(ticks) - 1) // 2)
        self.assertEqual([], soft_bar_cuts(ticks, deltas[:len(ticks) - 1], [0, 768, 1536]))

    def test_a_change_of_shape_at_the_line_is_cut(self):
        ticks = list(range(0, 1600, 100))
        n = len(ticks) - 1
        crossing = next(i for i in range(n) if ticks[i] < 768 <= ticks[i + 1])
        deltas = [1] * (crossing + 1) + [-3] * (n - crossing - 1)
        self.assertIn(crossing + 1, soft_bar_cuts(ticks, deltas, [0, 768, 1536]))

    def test_bar_lines_outside_the_run_are_ignored(self):
        ticks = [0, 100, 200]
        self.assertEqual([], soft_bar_cuts(ticks, [1, 1], [0, 768, 1536]))

    def test_a_single_note_has_no_cuts(self):
        self.assertEqual([], soft_bar_cuts([0], [], [0, 768]))


if __name__ == "__main__":
    unittest.main()


class BeatSegmentationTest(unittest.TestCase):
    """Patterns run one to two beat lines, not one to two measures.

    Robert, on the white lines: the thick bright one is the measure line, the thinner
    evenly spaced ones between are beat lines, and "most patterns are USUALLY 1-2 beat
    lines in length". Segmenting those by measure groups four times too much at 4/4 and
    merges separate figures into one unit.
    """

    def test_a_beat_is_a_quarter_of_a_four_four_measure(self):
        self.assertEqual(192, beat_length_ticks(4, 192))
        self.assertEqual(768, bar_length_ticks(4, 4, 192))
        self.assertEqual(4 * beat_length_ticks(4, 192), bar_length_ticks(4, 4, 192))

    def test_the_denominator_sets_the_beat_not_the_numerator(self):
        """6/8 counts eighths, so its beat is half a quarter note."""
        self.assertEqual(96, beat_length_ticks(8, 192))
        self.assertEqual(192, beat_length_ticks(4, 192))

    def test_beat_lines_include_every_measure_line(self):
        import tempfile
        from pathlib import Path as _Path

        text = (
            '[Song]\n{\n  Resolution = 192\n  Offset = "0"\n}\n'
            '[SyncTrack]\n{\n  0 = TS 4\n  0 = B 120000\n}\n'
            '[ExpertSingle]\n{\n  0 = N 0 0\n}\n')
        folder = _Path(tempfile.mkdtemp())
        path = folder / 'notes.chart'
        path.write_text(text, encoding='utf-8')

        beats = set(beat_boundaries(path, resolution=192, last_tick=3000))
        measures = set(bar_boundaries(path, resolution=192, last_tick=3000))
        self.assertTrue(measures <= beats, 'every measure line is also a beat line')
        self.assertGreater(len(beats), len(measures))

    def test_grouping_by_beat_splits_what_a_measure_would_merge(self):
        # Two separate figures, one per beat, inside a single 4/4 measure.
        encoded = [(0, 1, 0, {}), (48, 2, 0, {}),          # beat 0
                   (192, 3, 0, {}), (240, 4, 0, {})]       # beat 1
        by_measure = group_by_bar(encoded, [0, 768])
        by_beat = group_by_bar(encoded, [0, 192, 384, 576])
        self.assertEqual(1, sum(1 for group in by_measure if group),
                         "a measure swallows both figures")
        self.assertEqual(2, sum(1 for group in by_beat if group),
                         "beats keep them apart")

