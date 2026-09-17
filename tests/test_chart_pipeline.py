import math
import unittest
from pathlib import Path

from chart.chart_processor import ChartProcessor
from chart.time_conversion import convert_notes_to_seconds, preprocess_bpm_segments, tick_to_seconds
from chart.tokenizer import SimpleTokenizerGuitar

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic.chart"


class ChartProcessorTests(unittest.TestCase):
    def test_scalar_selections_and_target_section(self):
        processor = ChartProcessor("Expert", "Single")
        processor.read_chart(FIXTURE, target_sections="ExpertSingle")
        self.assertEqual(set(processor.notes), {"ExpertSingle"})
        self.assertEqual(processor.song_metadata["Resolution"], "192")
        self.assertEqual(processor.synctrack, [(0, 120000), (384, 180000)])

    def test_standard_instrument_sections(self):
        processor = ChartProcessor(["Medium", "Easy"], ["DoubleBass", "Drums"])
        processor.read_chart(FIXTURE)
        self.assertEqual(processor.notes["MediumDoubleBass"], [(96, "N", 2, 0)])
        self.assertEqual(processor.notes["EasyDrums"], [(48, "N", 3, 0)])

    def test_empty_chart_text(self):
        processor = ChartProcessor("Expert", "Single")
        processor.read_chart(None, chart_text="")
        self.assertEqual(processor.notes, {})


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        processor = ChartProcessor("Expert", "Single")
        processor.read_chart(FIXTURE)
        self.notes = processor.notes["ExpertSingle"]

    def test_default_ids_chords_modifiers_and_star_power(self):
        tokenizer = SimpleTokenizerGuitar(expressive=False)
        encoded = tokenizer.encode(self.notes)
        self.assertEqual(tokenizer.mapping_noteseqs2int[(0,)], 0)
        self.assertEqual(tokenizer.mapping_noteseqs2int[(7,)], 31)
        self.assertEqual(encoded[0][0:3], (0, tokenizer.mapping_noteseqs2int[(0, 1)], 96))
        self.assertTrue(encoded[0][3]["is5"])
        self.assertEqual(encoded[1][1], tokenizer.mapping_noteseqs2int[(7,)])
        self.assertTrue(encoded[1][3]["isS"])
        self.assertTrue(encoded[2][3]["is6"])

    def test_open_chords_can_be_enabled(self):
        tokenizer = SimpleTokenizerGuitar(exclude_open_chords=False, expressive=False)
        self.assertIn((0, 7), tokenizer.mapping_noteseqs2int)
        self.assertEqual(len(tokenizer.mapping_noteseqs2int), 63)

    def test_expressive_vocabulary_layout(self):
        tokenizer = SimpleTokenizerGuitar()
        self.assertEqual(tokenizer.n_chords, 32)
        self.assertEqual(tokenizer.vocab_size, 32 * 4 * 10 + 3)
        # legacy token c must land on c*40 so pretrained embedding rows can be broadcast
        for chord in range(tokenizer.n_chords):
            self.assertEqual(tokenizer.legacy_to_expressive(chord), chord * 40)
            self.assertEqual(tokenizer.split(chord * 40), (chord, 0, 0))
        # the pretrained checkpoint depends on this id staying put
        self.assertEqual(SimpleTokenizerGuitar(expressive=False).pad_id, 34)

    def test_expressive_tokens_carry_flags_and_sustain(self):
        tokenizer = SimpleTokenizerGuitar()
        notes = [(0, "N", 0, 192), (0, "N", 6, 0), (192, "N", 1, 0)]
        encoded = tokenizer.encode(notes, resolution=192)
        chord, flag, sustain = tokenizer.split(encoded[0][1])
        self.assertEqual(tokenizer.reverse_chord[chord], (0,))
        self.assertEqual(flag, 2)                      # tap
        self.assertEqual(tokenizer.sustain_beats(encoded[0][1]), 1.0)
        self.assertEqual(tokenizer.sustain_ticks(encoded[0][1], 192), 192)
        self.assertEqual(tokenizer.split(encoded[1][1])[2], 0)   # no sustain

    def test_decode_roundtrip_preserves_encoded_notes(self):
        tokenizer = SimpleTokenizerGuitar()
        encoded = tokenizer.encode(self.notes)
        self.assertEqual(tokenizer.encode(tokenizer.decode(encoded)), encoded)

    def test_discretization_validates_grid_arguments(self):
        tokenizer = SimpleTokenizerGuitar()
        with self.assertRaises(TypeError):
            tokenizer.discretize_time([], [], "pad", 20, 1)
        with self.assertRaises(ValueError):
            tokenizer.discretize_time([], [], 0, 30, 1)


class TimingTests(unittest.TestCase):
    def test_bpm_change_and_sustain(self):
        bpm = [(0, 120000), (384, 180000)]
        segments = preprocess_bpm_segments(bpm, 192)
        self.assertTrue(math.isclose(tick_to_seconds(384, segments, 192), 1.0))
        converted = convert_notes_to_seconds([(192, 0, 384, {})], bpm, 192, offset=0.05)
        self.assertTrue(math.isclose(converted[0][0], 0.55))
        self.assertTrue(math.isclose(converted[0][2], 5 / 6))


if __name__ == "__main__":
    unittest.main()
