"""Freezing the encoder must not freeze parameters the encoder itself added.

MERT's blend over its 13 hidden states is 13 scalars modules.models.MERT adds -- not
something the pretrained checkpoint shipped. freeze_encoder set requires_grad False on
every encoder parameter including those, and configure_optimizers only takes
requires_grad parameters, so the blend was never in the optimizer at all.

A full 1000-step run finished with every weight still at 0.0769, four decimal places
unchanged from initialisation.

Two earlier attempts to verify this feature missed it, both by testing the encoder in
isolation rather than through the trainer: the first checked the forward shapes, the
second called encoder(...) inside no_grad and read .grad -- which passes, because a
standalone MERT has requires_grad True. Only the trainer freezes it. So these tests go
through WaveformTransformerDiscrete.
"""

import unittest

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _available():
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download("m-a-p/MERT-v1-95M", "config.json")
        return True
    except Exception:
        return False


AVAILABLE = _available()


@unittest.skipUnless(AVAILABLE, "MERT checkpoint not reachable")
class FrozenEncoderBlendTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from hydra import compose, initialize_config_dir

        from chart.tokenizer import SimpleTokenizerGuitar
        from modules.trainer import WaveformTransformerDiscrete

        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="audio", overrides=["model=audio_mert"])
        cls.cfg = cfg
        tokenizer = SimpleTokenizerGuitar(expressive=True)
        cls.model = WaveformTransformerDiscrete(
            vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_id,
            eos_token_id=tokenizer.eos_id, cfg_model=cfg.model,
            cfg_optimizer=cfg.optimizer)

    def test_the_encoder_is_actually_frozen_in_this_config(self):
        self.assertTrue(self.cfg.model.freeze_encoder, "precondition")

    def test_the_blend_still_wants_gradients(self):
        weights = self.model.audio_encoder.layer_weights
        self.assertTrue(weights.requires_grad,
                        "the layer blend was frozen and cannot learn")

    def test_the_pretrained_backbone_does_not(self):
        trainable = [name for name, param
                     in self.model.audio_encoder.model.named_parameters()
                     if param.requires_grad]
        self.assertEqual(trainable, [], "the 95M backbone should stay frozen")

    def test_the_blend_reaches_the_optimizer(self):
        """requires_grad is necessary and not sufficient -- it has to be in a group."""
        weights = self.model.audio_encoder.layer_weights
        configured = self.model.configure_optimizers()
        optimizer = (configured["optimizer"] if isinstance(configured, dict)
                     else configured[0][0] if isinstance(configured, tuple)
                     else configured)
        in_optimizer = any(
            any(param is weights for param in group["params"])
            for group in optimizer.param_groups)
        self.assertTrue(in_optimizer, "the layer blend never reaches the optimizer")


if __name__ == "__main__":
    unittest.main()
