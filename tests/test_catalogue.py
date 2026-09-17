import unittest

from chart.catalogue import (
    CATALOGUE, body_signature, collisions, parse, positions, signature, signatures,
    split_transition,
)


class CatalogueTest(unittest.TestCase):
    def test_parse_handles_both_separators_and_bar_markers(self):
        self.assertEqual((0, 1, 2), parse("G R Y"))
        self.assertEqual((0, 1, 2), parse("G-R-Y"))
        self.assertEqual((2, 1, 0, 1), parse("Y R | G R"))

    def test_signature_is_transposition_invariant(self):
        self.assertEqual(signature(parse("G R Y")), signature(parse("R Y B")))
        self.assertEqual(signature(parse("Y R G")), signature(parse("B Y R")))

    def test_signature_distinguishes_interval_structure(self):
        # Contiguous and gapped trips are different shapes, not transpositions.
        self.assertNotEqual(signature(parse("G R Y")), signature(parse("G Y B")))

    def test_every_entry_parses_to_at_least_two_positions(self):
        """`parse` only sees single notes, so chorded entries need `positions`."""
        from chart.catalogue import positions

        for name, text in CATALOGUE.items():
            self.assertGreaterEqual(len(positions(text)), 2, name)

    def test_every_entry_has_a_signature(self):
        """Single-note entries keep delta signatures; chorded ones cannot have them."""
        from chart.catalogue import entry_signature, is_chorded_entry

        for name, text in CATALOGUE.items():
            self.assertTrue(entry_signature(text), name)

    def test_a_chorded_entry_is_invisible_to_the_single_note_parser(self):
        """Why the H pattern was reported absent from 1,200 charts: written as single
        notes it matched nothing, and parsed as single notes it is empty."""
        from chart.catalogue import is_chorded_entry, positions

        text = CATALOGUE["H pattern standard"]
        self.assertTrue(is_chorded_entry(text))
        self.assertEqual((), parse(text))
        self.assertEqual(3, len(positions(text)))

    def test_no_two_entries_hold_the_same_sequence(self):
        """This used to assert the opposite, pinning a known duplicate in place.

        'triangle slide 8-note' and 'sweep 8-note' both held 'G R Y B O B Y R', which
        double-counted the shape in catalogue coverage. The triangle-slide one was
        misfiled rather than an alias: the families differ at six notes, where a triangle
        turns around on an interior peak (G R Y B Y R) and a sweep runs to the end of the
        neck and back (G R Y B O B). The eight-note sequence reaches O, so it is a sweep.
        """
        import collections

        bodies = collections.defaultdict(list)
        for name, text in CATALOGUE.items():
            bodies[tuple(positions(text))].append(name)
        duplicates = {body: names for body, names in bodies.items() if len(names) > 1}
        self.assertEqual({}, duplicates,
                         f"entries sharing a sequence: {list(duplicates.values())}")

    def test_transposed_families_collide_as_expected(self):
        found = collisions()
        trip_zigs = signature(parse(CATALOGUE["trip zig G-R-Y"]))
        self.assertIn(trip_zigs, found)
        self.assertIn("trip zig R-Y-B", found[trip_zigs])

    def test_roberts_original_trip_zig_example_is_the_split_variant(self):
        # He first described O B R B O B R B O B R as a "trip zig"; in this catalogue that
        # is the split R-B-O form, not the plain G-R-Y one.
        original = signature(parse("O B R B O B R B O B R"))
        self.assertEqual(original, signature(parse(CATALOGUE["trip zig split R-B-O"])))
        self.assertNotEqual(original, signature(parse(CATALOGUE["trip zig G-R-Y"])))


    def test_split_transition_separates_at_the_bar(self):
        transition, body = split_transition("B Y | R Y B Y R")
        self.assertEqual((3, 2), transition)
        self.assertEqual((1, 2, 3, 2, 1), body)

    def test_no_bar_means_everything_is_body(self):
        transition, body = split_transition("G R Y")
        self.assertEqual((), transition)
        self.assertEqual((0, 1, 2), body)

    def test_body_signature_excludes_the_transition(self):
        # A transition note belongs to the join between patterns, not to either pattern's
        # identity, so signing it makes the body findable only after that exact lead-in.
        whole = signature(parse(CATALOGUE["trip zig R-Y-B"]))
        body = body_signature(CATALOGUE["trip zig R-Y-B"])
        self.assertNotEqual(whole, body)
        self.assertLess(len(body), len(whole))

    def test_body_signature_is_still_transposition_invariant(self):
        self.assertEqual(body_signature(CATALOGUE["trip zig G-R-Y"]),
                         body_signature(CATALOGUE["trip zig R-Y-B"]))

    def test_body_is_a_suffix_of_the_whole_sequence(self):
        for name, text in CATALOGUE.items():
            transition, body = split_transition(text)
            self.assertEqual(parse(text), transition + body, name)


if __name__ == "__main__":
    unittest.main()
