"""Nothing in a MERT run may need trust_remote_code.

modules.models.MERT loads the weights as plain HuBERT precisely so no downloaded Python
is executed. That covers the model and missed the dataloader, which separately calls
AutoProcessor.from_pretrained(model.encoder.checkpoint) -- and for MERT that resolves
through AutoConfig, hits the repo's auto_map, and raises:

    ValueError: The repository m-a-p/MERT-v1-95M contains custom code which must be
    executed to correctly load the model ... Please pass trust_remote_code=True.

A 1000-step run died on it at startup. The fix is that MERT needs no processor at all:
its preprocessor only normalises the waveform, which MERT.forward does itself.
"""

import unittest
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class MertNoRemoteCodeTests(unittest.TestCase):

    def _model(self, name):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            return compose(config_name="audio", overrides=[f"model={name}"]).model

    def test_mert_asks_for_no_processor(self):
        """use_processor drives AutoProcessor on the encoder checkpoint."""
        self.assertFalse(self._model("audio_mert").use_processor)

    def test_encodec_still_asks_for_one(self):
        """The fix must not quietly change the Encodec runs it is compared against."""
        self.assertTrue(self._model("audio_codec").use_processor)

    def test_building_the_encoder_touches_no_auto_class(self):
        """AutoModel/AutoConfig on this checkpoint is what demands remote code.

        Asserted by making those calls fail rather than by grepping the source. A first
        version searched inspect.getsource(MERT) for the string "AutoModel" and failed on
        the docstring that explains why MERT does not use it -- the same mistake the
        autocast test made by searching for "enabled=False".
        """
        import transformers

        try:
            from modules.models import MERT
        except ImportError as error:
            self.skipTest(f"torch unavailable: {error}")

        def refuse(*args, **kwargs):
            raise AssertionError(
                "MERT routed through an Auto class, which resolves the repo's auto_map "
                "and demands trust_remote_code")

        patched = []
        for name in ("AutoModel", "AutoConfig", "AutoProcessor",
                     "AutoFeatureExtractor"):
            holder = getattr(transformers, name, None)
            if holder is not None:
                patched.append((holder, holder.from_pretrained))
                holder.from_pretrained = refuse
        try:
            encoder = MERT()
        finally:
            for holder, original in patched:
                holder.from_pretrained = original

        self.assertEqual(encoder.output_dim, 768)


if __name__ == "__main__":
    unittest.main()
