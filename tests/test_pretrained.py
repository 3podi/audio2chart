import unittest

import torch

from chart.tokenizer import SimpleTokenizerGuitar
from modules.pretrained import expand_vocab_rows, read_transformer_state


class ExpandVocabRowsTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = SimpleTokenizerGuitar()
        self.legacy = SimpleTokenizerGuitar(expressive=False)
        self.rows = torch.arange(self.legacy.vocab_size * 4, dtype=torch.float32).reshape(
            self.legacy.vocab_size, 4
        )

    def test_shape_matches_expressive_vocabulary(self):
        expanded = expand_vocab_rows(self.rows, self.tokenizer, jitter=0.0)
        self.assertEqual(expanded.shape, (self.tokenizer.vocab_size, 4))
        self.assertEqual(self.tokenizer.vocab_size, 1283)

    def test_every_expressive_token_inherits_its_legacy_row(self):
        expanded = expand_vocab_rows(self.rows, self.tokenizer, jitter=0.0)
        for chord in range(self.legacy.n_chords):
            for flag in range(self.tokenizer.N_FLAG):
                for sustain in range(self.tokenizer.N_SUSTAIN):
                    token = self.tokenizer.compose(chord, flag, sustain)
                    self.assertTrue(torch.equal(expanded[token], self.rows[chord]),
                                    f"token {token} (chord {chord}) did not inherit row {chord}")

    def test_special_tokens_map_one_to_one(self):
        expanded = expand_vocab_rows(self.rows, self.tokenizer, jitter=0.0)
        for legacy_id, new_id in (
            (self.legacy.bos_id, self.tokenizer.bos_id),
            (self.legacy.eos_id, self.tokenizer.eos_id),
            (self.legacy.pad_id, self.tokenizer.pad_id),
        ):
            self.assertTrue(torch.equal(expanded[new_id], self.rows[legacy_id]))

    def test_jitter_perturbs_copies_but_not_the_base_rows(self):
        generator = torch.Generator().manual_seed(0)
        expanded = expand_vocab_rows(self.rows, self.tokenizer, jitter=0.01, generator=generator)
        for chord in range(self.legacy.n_chords):
            base = self.tokenizer.compose(chord, 0, 0)
            self.assertTrue(torch.equal(expanded[base], self.rows[chord]))
            self.assertFalse(torch.equal(expanded[base + 1], self.rows[chord]))
        self.assertTrue(torch.equal(expanded[self.tokenizer.pad_id], self.rows[self.legacy.pad_id]))

    def test_rejects_a_parameter_that_is_not_legacy_sized(self):
        with self.assertRaises(ValueError):
            expand_vocab_rows(torch.zeros(64, 4), self.tokenizer, jitter=0.0)


class ReadTransformerStateTest(unittest.TestCase):
    def test_exported_checkpoint_gains_the_lightning_prefix(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "pytorch_model.bin"
            torch.save({"token_embedding.weight": torch.zeros(35, 4)}, path)
            state = read_transformer_state(path)
            self.assertEqual(list(state), ["transformer.token_embedding.weight"])

    def test_lightning_checkpoint_keeps_only_transformer_parameters(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "last.ckpt"
            torch.save({"state_dict": {
                "transformer.token_embedding.weight": torch.zeros(35, 4),
                "audio_encoder.model.weight": torch.zeros(2, 2),
            }}, path)
            state = read_transformer_state(path)
            self.assertEqual(list(state), ["transformer.token_embedding.weight"])


if __name__ == "__main__":
    unittest.main()
