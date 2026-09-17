import unittest

from chart.style import (
    PRESETS,
    StyleConstraint,
    allowed_note_tokens,
    allowed_tokens,
    describe,
    resolve,
)
from chart.tokenizer import SimpleTokenizerGuitar


class StyleConstraintTest(unittest.TestCase):
    def setUp(self):
        self.tok = SimpleTokenizerGuitar()

    def test_unconstrained_permits_the_whole_note_vocabulary(self):
        self.assertEqual(self.tok.n_notes, len(allowed_note_tokens(self.tok, StyleConstraint())))

    def test_wii_permits_exactly_the_tap_tokens(self):
        tokens = allowed_note_tokens(self.tok, PRESETS["wii"])
        self.assertEqual(self.tok.n_notes // 2, len(tokens))
        for token in tokens:
            _, flag, _ = self.tok.split(token)
            self.assertTrue(flag & 2, f"token {token} is not a tap")

    def test_wii_excludes_every_strum(self):
        allowed = set(allowed_note_tokens(self.tok, PRESETS["wii"]))
        plain_strum = self.tok.compose(0, 0, 0)          # green, no flags, no sustain
        self.assertNotIn(plain_strum, allowed)

    def test_single_notes_excludes_chords(self):
        for token in allowed_note_tokens(self.tok, PRESETS["single_notes"]):
            lanes = [l for l in self.tok.lanes_of(token) if l != 7]
            self.assertLessEqual(len(lanes), 1)

    def test_chording_excludes_single_notes(self):
        for token in allowed_note_tokens(self.tok, PRESETS["chording"]):
            lanes = [l for l in self.tok.lanes_of(token) if l != 7]
            self.assertGreaterEqual(len(lanes), 2)

    def test_one_hand_keeps_frets_within_reach(self):
        for token in allowed_note_tokens(self.tok, PRESETS["one_hand"]):
            frets = [l for l in self.tok.lanes_of(token) if l != 7]
            if frets:
                self.assertLessEqual(max(frets) - min(frets), 2)

    def test_no_sustains_permits_only_bucket_zero(self):
        for token in allowed_note_tokens(self.tok, PRESETS["no_sustains"]):
            self.assertEqual(0, self.tok.split(token)[2])

    def test_no_opens_excludes_the_open_lane(self):
        for token in allowed_note_tokens(self.tok, PRESETS["no_opens"]):
            self.assertNotIn(7, self.tok.lanes_of(token))

    def test_pad_is_always_permitted(self):
        # The grid is ~90% padding; forbidding it would force a note every 20 ms.
        for name in PRESETS:
            self.assertIn(self.tok.pad_id, allowed_tokens(self.tok, PRESETS[name]), name)

    def test_eos_is_permitted(self):
        self.assertIn(self.tok.eos_id, allowed_tokens(self.tok, PRESETS["wii"]))

    def test_bos_is_not_offered_to_the_sampler(self):
        self.assertNotIn(self.tok.bos_id, allowed_tokens(self.tok, PRESETS["wii"]))

    def test_constraints_compose(self):
        both = StyleConstraint(taps_only=True, forbid_sustains=True, max_lanes=1)
        for token in allowed_note_tokens(self.tok, both):
            chord, flag, sustain = self.tok.split(token)
            self.assertTrue(flag & 2)
            self.assertEqual(0, sustain)
            self.assertLessEqual(len([l for l in self.tok.reverse_chord[chord] if l != 7]), 1)

    def test_an_impossible_style_is_rejected_not_silently_empty(self):
        # A mask with nothing in it would produce NaN in softmax.
        impossible = StyleConstraint(min_lanes=4, max_fret_span=1)
        with self.assertRaises(ValueError):
            allowed_tokens(self.tok, impossible)

    def test_contradictory_flags_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            StyleConstraint(taps_only=True, forbid_taps=True)
        with self.assertRaises(ValueError):
            StyleConstraint(min_lanes=3, max_lanes=2)

    def test_legacy_vocabulary_is_refused(self):
        with self.assertRaises(ValueError):
            allowed_note_tokens(SimpleTokenizerGuitar(expressive=False), PRESETS["wii"])

    def test_resolve_by_name_and_passthrough(self):
        self.assertIs(PRESETS["wii"], resolve("wii"))
        custom = StyleConstraint(max_lanes=2)
        self.assertIs(custom, resolve(custom))
        with self.assertRaises(ValueError):
            resolve("nonexistent-style")

    def test_describe_reports_the_narrowing(self):
        self.assertIn("50.0%", describe(self.tok, PRESETS["wii"]))


if __name__ == "__main__":
    unittest.main()
