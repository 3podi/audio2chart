"""One dense run must not cost the rest of the chart its grid.

choose_snap asks a single question of a whole chart -- "is there one subdivision that
keeps every note distinct?" -- and answers "no" as soon as any passage is denser than
the finest choice. "No" means snap nothing. On all three generated playtest charts it
returned 0, so none of them were snapped at all: Diary of Jane had 11 of 4,495 notes on
a beat line and 41 on a 1/16.

Robert, reading it in Moonscraper: "at tick 65286 the yellow note is supposed to be at
65280 ... a lot of notes are placed in a completely weird step."
"""

import unittest

from chart.time_conversion import SNAP_CHOICES, choose_snap, snap_passages

RESOLUTION = 480
SIXTEENTH = RESOLUTION // 4


class SnapPassageTests(unittest.TestCase):

    def test_a_dense_run_no_longer_disables_the_whole_chart(self):
        """The bug, reduced: clean 1/16s, a rest, then a passage too dense to snap.

        Globally there is no answer -- 13-tick spacing fits no subdivision a charter
        writes -- so the old rule snapped nothing and the clean half lost its grid too.
        """
        clean = [i * SIXTEENTH + 3 for i in range(16)]
        dense = [20000 + i * 13 for i in range(16)]
        ticks = clean + dense

        self.assertEqual(choose_snap(ticks, RESOLUTION), 0,
                         "precondition: one subdivision cannot serve the whole chart")

        mapping = snap_passages(ticks, RESOLUTION)
        for tick in clean:
            self.assertEqual(mapping[tick] % SIXTEENTH, 0,
                             "the clean passage should have landed on 1/16")
        for tick in dense:
            self.assertEqual(mapping[tick], tick,
                             "the dense passage fits nothing and must be left alone")

    def test_snapping_never_merges_two_notes(self):
        """Merging is silent note loss, which is worse than leaving a note off-grid."""
        ticks = [0, 13, 26, 40, 53, 1000, 1120, 1240, 5000, 5007]
        mapping = snap_passages(ticks, RESOLUTION)
        self.assertEqual(len(set(mapping.values())), len(set(ticks)))

    def test_a_passage_that_cannot_be_snapped_is_left_alone(self):
        """Not snapped is a fair outcome; moved onto a wrong grid is not."""
        ticks = [0, 2, 4, 6, 8]
        mapping = snap_passages(ticks, RESOLUTION)
        self.assertEqual(mapping, {t: t for t in ticks})

    def test_each_passage_gets_its_own_subdivision(self):
        """A 1/16 verse and a 1/48 run should not be forced onto one grid."""
        sixteenths = [i * SIXTEENTH + 2 for i in range(8)]
        fortyeighths = [10000 + i * 40 + 1 for i in range(8)]
        mapping = snap_passages(sixteenths + fortyeighths, RESOLUTION)
        for tick in sixteenths:
            self.assertEqual(mapping[tick] % SIXTEENTH, 0)
        for tick in fortyeighths:
            self.assertEqual(mapping[tick] % 40, 0)

    def test_passages_are_split_at_rests_not_at_fixed_offsets(self):
        """Two runs a beat apart are separate; notes within a beat are one passage."""
        together = [0, 40, 80, 120]
        apart = [0, 40, 80, 80 + 2 * RESOLUTION]
        self.assertEqual(len(set(snap_passages(together, RESOLUTION).values())), 4)
        self.assertEqual(len(set(snap_passages(apart, RESOLUTION).values())), 4)

    def test_the_choices_stay_musical(self):
        """A finer ruler is not a fix, and was tried.

        Adding 1/128 and 1/192 stops choose_snap returning 0, which looks like progress
        until you read the output: a run spaced 10 ticks apart snaps onto a 4-tick grid
        as 0, 12, 20, 28, 40, inventing unevenness the model never emitted.
        test_tick_exactness already pinned that case. Leaving an unfittable passage
        alone is the honest outcome.
        """
        self.assertEqual(SNAP_CHOICES, (4, 8, 12, 16, 24, 32, 48, 64, 96))

    def test_a_single_note_is_returned_unchanged(self):
        self.assertEqual(snap_passages([7], RESOLUTION), {7: 7})
        self.assertEqual(snap_passages([], RESOLUTION), {})


if __name__ == "__main__":
    unittest.main()
