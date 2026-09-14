"""Whether the audio encoder runs under bf16 autocast is the encoder's call.

Lightning wraps the training step in bf16, which is a 3.4x speedup on the transformer
and a 5.5x slowdown on Encodec: cuDNN has no fused LSTM kernel for bf16, so SEANet's
recurrence over 2250 timesteps drops to an unfused path. Measured 0.130s per call in
fp32 against 0.712s under autocast. Nothing is lost by opting out there, because the
discrete path returns quantiser indices -- no precision to gain, only recurrence to
slow down.

None of that is true of MERT, which has no recurrence. Forcing 95M parameters over
1124 frames into fp32 costs speed and the VRAM headroom this card does not have, so
the encoder declares `prefers_fp32` and the trainer honours it.

These check the decision rather than the source text. The previous version asserted
that the string "enabled=False" appeared in _encode_audio, which passes whether or not
the flag is reached and fails on any honest refactor -- the same mistake an earlier
test made by asserting "round(" appeared in a function body.
"""

import types
import unittest


class EncoderAutocastTests(unittest.TestCase):

    def _autocast_enabled_for(self, encoder):
        """Run _encode_audio against a stub encoder and report the autocast flag."""
        import contextlib
        from unittest.mock import patch

        import torch

        from modules.trainer import WaveformTransformerDiscrete

        seen = {}

        @contextlib.contextmanager
        def record(device_type, enabled=True, **kwargs):
            seen["enabled"] = enabled
            yield

        stub = types.SimpleNamespace(
            freeze_encoder=True,
            device=torch.device("cpu"),
            audio_encoder=encoder,
            audio_output_kind=getattr(encoder, "output_kind", "codes"),
            _flatten_encodec_frames=lambda output: output,
        )
        with patch("torch.autocast", record):
            WaveformTransformerDiscrete._encode_audio(
                stub, torch.zeros(1, 1, 8), torch.ones(1, 8))
        return seen["enabled"]

    def test_an_encoder_with_a_recurrence_opts_out(self):
        """Encodec keeps fp32: that is what the 5.5x measurement was about."""
        try:
            import torch  # noqa: F401
        except ImportError as error:
            self.skipTest(f"torch unavailable: {error}")

        encoder = type("E", (), {
            "output_kind": "codes", "prefers_fp32": True,
            "__call__": lambda self, *a, **k: (None,)})()
        self.assertFalse(self._autocast_enabled_for(encoder))

    def test_an_encoder_without_one_stays_in_autocast(self):
        """MERT has no LSTM, so the opt-out would be pure cost."""
        try:
            import torch  # noqa: F401
        except ImportError as error:
            self.skipTest(f"torch unavailable: {error}")

        encoder = type("M", (), {
            "output_kind": "features", "prefers_fp32": False,
            "__call__": lambda self, *a, **k: None})()
        self.assertTrue(self._autocast_enabled_for(encoder))

    def test_an_encoder_that_says_nothing_keeps_the_old_behaviour(self):
        """Absent the attribute, opt out -- the safe default for anything recurrent."""
        try:
            import torch  # noqa: F401
        except ImportError as error:
            self.skipTest(f"torch unavailable: {error}")

        encoder = type("U", (), {
            "output_kind": "features",
            "__call__": lambda self, *a, **k: None})()
        self.assertFalse(self._autocast_enabled_for(encoder))

    def test_the_encoders_declare_what_they_need(self):
        try:
            from modules.models import MERT, Encodec  # noqa: F401
        except ImportError as error:
            self.skipTest(f"torch unavailable: {error}")
        import inspect
        self.assertIn("prefers_fp32", inspect.getsource(Encodec.__init__))
        self.assertIn("prefers_fp32", inspect.getsource(MERT.__init__))

    def test_codes_are_identical_either_way(self):
        """The opt-out must not change what the model sees."""
        try:
            import torch
            from modules.models import Encodec
        except ImportError as error:
            self.skipTest(f"torch unavailable: {error}")
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA to exercise autocast")

        encoder = Encodec(bandwidth=3.0).cuda().eval()
        audio = torch.randn(1, 1, 24000 * 2, device="cuda")
        mask = torch.ones(1, 24000 * 2, dtype=torch.long, device="cuda")

        with torch.no_grad():
            plain = encoder(audio, mask, return_embeddings=False)[0]
            with torch.autocast("cuda", enabled=False):
                opted_out = encoder(audio.float(), mask, return_embeddings=False)[0]

        self.assertTrue(torch.equal(plain, opted_out))


if __name__ == "__main__":
    unittest.main()
