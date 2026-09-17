"""Chord-aware pattern identity.

Single-note signatures made every chord figure invisible, which is how the H pattern
was reported as never occurring when it appears in 8.9% of charts.
"""

import unittest

from chart.catalogue import chord_signature, is_chorded


class ChordSignatureTests(unittest.TestCase):

    def test_the_same_shape_at_a_different_place_matches(self):
        """RB -> RYB -> RB and GO -> GYO -> GO are the same H."""
        rb = chord_signature([(1, 3), (1, 2, 3), (1, 3)])
        gy = chord_signature([(0, 2), (0, 1, 2), (0, 2)])
        self.assertEqual(rb, gy)

    def test_different_rail_widths_are_different_patterns(self):
        """GO rails span four frets, RB spans two -- a different hand position."""
        wide = chord_signature([(0, 4), (0, 2, 4), (0, 4)])
        narrow = chord_signature([(1, 3), (1, 2, 3), (1, 3)])
        self.assertNotEqual(wide, narrow)

    def test_single_notes_reduce_to_root_movement(self):
        signature = chord_signature([(0,), (1,), (2,)])
        self.assertEqual([shape for shape, _ in signature], [(0,), (0,), (0,)])
        self.assertEqual([move for _, move in signature], [0, 1, 1])

    def test_a_moving_chord_records_where_the_hand_went(self):
        signature = chord_signature([(0, 1), (2, 3)])
        self.assertEqual(signature[1], ((0, 1), 2))

    def test_chorded_detects_a_chord(self):
        self.assertTrue(is_chorded(chord_signature([(0, 2), (0, 1, 2)])))
        self.assertFalse(is_chorded(chord_signature([(0,), (1,), (2,)])))

    def test_every_chorded_entry_survives_a_signature_based_filter(self):
        """A chorded entry must sign to something a pattern search can use.

        `parse` resolves one letter at a time, so a chord token like 'RB' matches no
        fret and 'RB RYB RB' signs as the empty tuple. Anything keyed on the single-note
        signature therefore collapsed all four chorded entries together and then dropped
        them below its `len(sig) >= 2` filter -- which is how audit_catalogue reported
        them as one duplicate body and simultaneously never searched for any of them,
        while printing "ABSENT: none".
        """
        from chart.catalogue import CATALOGUE, entry_signature, is_chorded_entry

        chorded = {name: text for name, text in CATALOGUE.items()
                   if is_chorded_entry(text)}
        self.assertTrue(chorded, "the catalogue is supposed to hold tap-chord patterns")

        seen = {}
        for name, text in chorded.items():
            sig = entry_signature(text)
            self.assertGreaterEqual(
                len(sig), 2, f"{name} signs too short to be searchable: {sig!r}")
            self.assertNotIn(sig, seen,
                             f"{name} collides with {seen.get(sig)}")
            seen[sig] = name

    def test_empty_input_is_empty(self):
        self.assertEqual((), chord_signature([]))


if __name__ == "__main__":
    unittest.main()
