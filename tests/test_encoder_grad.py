"""The discrete path must not retain an Encodec graph it cannot use.

Measured before this was fixed: 4.173 GiB peak with enable_grad against 0.807 GiB
under no_grad, producing byte-identical codes. That 3.4 GiB was the VRAM pressure
that forced freeze_layers: 8, which cost decoder capacity for nothing.

The mistake is easy to reintroduce, because freeze_encoder: false looks like it
enables encoder training and in this path it cannot -- gradient does not flow through
an integer index.
"""

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class EncoderGradientTests(unittest.TestCase):

    def test_discrete_config_freezes_the_encoder(self):
        config = yaml.safe_load((ROOT / "configs/model/audio_discrete.yaml").read_text())
        self.assertTrue(
            config["freeze_encoder"],
            "The discrete path returns integer codes, so an unfrozen encoder cannot "
            "learn -- it only retains ~3.4 GiB of activations per forward pass.",
        )

    def test_encodec_codes_carry_no_gradient(self):
        """The reason the setting cannot matter: there is nothing to differentiate."""
        torch = __import__("torch")
        try:
            from modules.models import Encodec
        except ImportError as error:
            self.skipTest(f"torch/transformers unavailable: {error}")

        encoder = Encodec(bandwidth=3.0).eval()
        audio = torch.randn(1, 1, 24000 * 2)
        mask = torch.ones(1, 24000 * 2, dtype=torch.long)
        with torch.enable_grad():
            codes = encoder(audio, mask, return_embeddings=False)[0]

        self.assertFalse(codes.is_floating_point())
        self.assertIsNone(codes.grad_fn)
        self.assertFalse(codes.requires_grad)


if __name__ == "__main__":
    unittest.main()
