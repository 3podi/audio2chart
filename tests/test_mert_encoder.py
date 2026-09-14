"""MERT has to load as plain HuBERT, and has to line up with what the decoder expects.

The published recipe for MERT is AutoModel(..., trust_remote_code=True), which executes
arbitrary Python from a downloaded repository. Robert's standing rule for the same class
of risk, on weights_only=False: "only for a checkpoint you produced yourself, never for
one that was downloaded."

Reading the repo's code rather than running it shows MERTModel subclasses HubertModel,
and its only addition -- MERTFeatureProjection -- is identical to HubertFeatureProjection
whenever feature_extractor_cqt is false. So the weights can go into the transformers
library's own audited HuBERT. These check that the substitution is exact rather than
approximately right, because a silently half-loaded encoder would look like a training
result.

Networked, and skipped when the model is not cached.
"""

import unittest

import torch


def _cached():
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download("m-a-p/MERT-v1-95M", "config.json")
        return True
    except Exception:
        return False


AVAILABLE = _cached()


@unittest.skipUnless(AVAILABLE, "MERT checkpoint not reachable")
class MertEncoderTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from modules.models import MERT
        cls.encoder = MERT().eval()

    def test_every_published_weight_maps_onto_hubert(self):
        """Nothing missing and nothing left over, or it is a different model."""
        import json

        from huggingface_hub import hf_hub_download
        from transformers import HubertConfig, HubertModel

        raw = json.load(open(hf_hub_download("m-a-p/MERT-v1-95M", "config.json")))
        raw.pop("auto_map", None)
        raw.pop("architectures", None)
        config = HubertConfig(**{k: v for k, v in raw.items() if k != "model_type"})
        fresh = HubertModel(config)

        try:
            from safetensors.torch import load_file
            state = load_file(hf_hub_download("m-a-p/MERT-v1-95M", "model.safetensors"))
        except Exception:
            state = torch.load(
                hf_hub_download("m-a-p/MERT-v1-95M", "pytorch_model.bin"),
                map_location="cpu", weights_only=True)

        missing, unexpected = fresh.load_state_dict(state, strict=False)
        # masked_spec_embed is a training-time buffer the released weights omit.
        self.assertEqual([k for k in missing if "masked_spec_embed" not in k], [])
        self.assertEqual(list(unexpected), [])

    def test_the_frame_rate_matches_encodec(self):
        """75 fps at 24 kHz, so the cross-attention length is unchanged.

        This is what makes MERT a substitution rather than a rebuild: conv_stride
        products to 320, and 24000/320 is 75 -- the same rate Encodec 24khz emits.
        """
        with torch.no_grad():
            features = self.encoder(torch.randn(1, 1, 24000 * 4))
        self.assertEqual(features.shape[0], 1)
        self.assertAlmostEqual(features.shape[1] / 4.0, 75.0, delta=1.0)

    def test_it_declares_the_features_contract_the_trainer_reads(self):
        """The trainer branches on output_kind and sizes cross-attention by output_dim."""
        self.assertEqual(self.encoder.output_kind, "features")
        self.assertEqual(self.encoder.output_dim, 768)

    def test_a_pinned_layer_is_honoured(self):
        from modules.models import MERT

        pinned = MERT(layer=6).eval()
        self.assertIsNone(pinned.layer_weights)
        with torch.no_grad():
            direct = pinned.model(
                _normalised(torch.randn(1, 24000 * 2)), output_hidden_states=True)
            self.assertEqual(
                tuple(pinned(torch.randn(1, 1, 24000 * 2)).shape),
                tuple(direct.hidden_states[6].shape))

    def test_layer_mixing_starts_uniform(self):
        """Zero logits mean an equal blend, so nothing is favoured before training."""
        weights = torch.softmax(self.encoder.layer_weights, dim=0)
        self.assertEqual(weights.numel(), 13)
        self.assertTrue(torch.allclose(weights, torch.full_like(weights, 1 / 13)))

    def test_a_cqt_checkpoint_is_refused_rather_than_silently_reshaped(self):
        """CQT is the one part of MERT that is not plain HuBERT."""
        from unittest.mock import patch

        from modules.models import MERT

        with patch("pathlib.Path.read_text",
                   return_value='{"feature_extractor_cqt": true, "sample_rate": 24000}'):
            with self.assertRaises(ValueError):
                MERT()


def _normalised(values):
    mean = values.mean(dim=-1, keepdim=True)
    variance = values.var(dim=-1, keepdim=True, unbiased=False)
    return (values - mean) / torch.sqrt(variance + 1e-7)


if __name__ == "__main__":
    unittest.main()
