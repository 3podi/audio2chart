"""pad_bias shifts only the pad token, and is a no-op when zero.

It does not fix note density -- see the docstring for the measured 22x collapse over
one unit -- but it is kept as a diagnostic, so it has to be inert by default and
touch nothing but pad when set.
"""

import unittest

import torch

from inference.engine import Charter


class _Config:
    vocab_size = 20
    pad_token_id = 3


class PadBiasTests(unittest.TestCase):

    def setUp(self):
        self.charter = Charter.__new__(Charter)
        self.charter.config = _Config()

    def _greedy(self, logits, pad_bias):
        sample = Charter._make_sampler(self.charter, -1.0, 0, torch.device("cpu"),
                                       None, pad_bias)
        return int(sample(logits)[0, 0])

    def test_zero_bias_changes_nothing(self):
        logits = torch.zeros(1, 20)
        logits[0, 7] = 5.0
        self.assertEqual(self._greedy(logits, 0.0), 7)

    def test_positive_bias_can_select_pad(self):
        logits = torch.zeros(1, 20)
        logits[0, 7] = 5.0
        self.assertEqual(self._greedy(logits, 10.0), _Config.pad_token_id)

    def test_negative_bias_suppresses_pad(self):
        logits = torch.zeros(1, 20)
        logits[0, _Config.pad_token_id] = 5.0
        logits[0, 7] = 1.0
        self.assertEqual(self._greedy(logits, -10.0), 7)

    def test_only_pad_is_shifted(self):
        """A bias that moved other logits would change what a note looks like too."""
        logits = torch.zeros(1, 20)
        logits[0, 7] = 5.0
        logits[0, 9] = 4.0
        self.assertEqual(self._greedy(logits, 2.0), 7)


if __name__ == "__main__":
    unittest.main()
