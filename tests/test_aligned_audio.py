"""Give each token step the audio at its own timestamp, instead of searching for it.

Cross-attention has to find the right audio frame. Measured, it does not: sliding a song
two seconds against its own chart cost 23% of what a different song cost, so three
quarters of what the model took from the audio was time-invariant. Putting both sequences
on one clock for RoPE doubled short-shift sensitivity -- 10% to 21% at a quarter second --
and left it far from solved.

But the two sequences are aligned by construction. Token step i and audio time
i * grid_ms are the same instant, exactly known. Resampling the audio onto the token grid
and adding it in gives every step its own audio directly.

The gate starts at zero, so at initialisation the model is bit-for-bit the one without
it. That property is the point: the RoPE change altered what positions meant under a
checkpoint that had adapted to the old meaning, the loss rose, and the run could not be
read cleanly. This cannot do that.
"""

import unittest

import torch

from modules.training_transformer import TransformerDecoderAudioConditioned

ARGS = dict(vocab_size=35, pad_token_id=-1, eos_token_id=33, d_model=32, n_heads=4,
            num_kv_heads=2, n_layers=2, dropout=0.0, audio_drop=0.0, use_flash=False,
            codebook_size=64)


def _pair():
    torch.manual_seed(0)
    plain = TransformerDecoderAudioConditioned(**ARGS).eval()
    torch.manual_seed(0)
    gated = TransformerDecoderAudioConditioned(**ARGS, aligned_audio=True).eval()
    gated.load_state_dict(plain.state_dict(), strict=False)
    return plain, gated


class AlignedAudioTests(unittest.TestCase):

    def test_off_by_default(self):
        """It adds parameters, so it must not appear in an existing run unasked."""
        model = TransformerDecoderAudioConditioned(**ARGS)
        self.assertFalse(model.aligned_audio)
        self.assertFalse(any("align" in name for name, _ in model.named_parameters()))

    def test_a_closed_gate_reproduces_the_model_without_it(self):
        plain, gated = _pair()
        tokens = torch.randint(0, 35, (2, 12))
        codes = torch.randint(0, 64, (2, 4, 9))
        with torch.no_grad():
            torch.testing.assert_close(plain(tokens, codes), gated(tokens, codes))

    def test_opening_the_gate_changes_the_output(self):
        plain, gated = _pair()
        tokens = torch.randint(0, 35, (2, 12))
        codes = torch.randint(0, 64, (2, 4, 9))
        with torch.no_grad():
            gated.align_gate.fill_(0.5)
            self.assertFalse(torch.allclose(plain(tokens, codes), gated(tokens, codes)))

    def test_the_gate_learns(self):
        """A gate with no gradient is a fixed zero, which is the whole path disabled."""
        _, gated = _pair()
        gated.train()
        tokens = torch.randint(0, 35, (2, 12))
        codes = torch.randint(0, 64, (2, 4, 9))
        gated(tokens, codes).sum().backward()
        self.assertIsNotNone(gated.align_gate.grad)
        self.assertGreater(float(gated.align_gate.grad.abs().sum()), 0.0)

    def test_the_audio_lands_at_the_right_token(self):
        """The claim is alignment, so check where a feature actually arrives.

        An impulse at audio frame j must show up at token round(j * T_tok / T_aud), not
        at j, and not smeared across the sequence.
        """
        _, gated = _pair()
        tokens_len, frames = 120, 45
        with torch.no_grad():
            gated.align_gate.fill_(1.0)
            torch.nn.init.eye_(gated.align_projection.weight)

            audio = torch.zeros(1, frames, ARGS["d_model"])
            spike = 20
            audio[0, spike, :] = 10.0
            aligned = torch.nn.functional.interpolate(
                audio.transpose(1, 2), size=tokens_len,
                mode="linear", align_corners=True).transpose(1, 2)

        landed = int(aligned[0].abs().sum(dim=-1).argmax())
        expected = round(spike * (tokens_len - 1) / (frames - 1))
        self.assertLessEqual(abs(landed - expected), 1,
                             f"impulse at frame {spike} landed at token {landed}, "
                             f"expected about {expected}")
        self.assertNotEqual(landed, spike,
                            "landing at the frame index would mean no rescaling at all")

    def test_the_ends_line_up(self):
        """First and last audio frames must map to first and last token steps."""
        frames, tokens_len = 45, 120
        audio = torch.arange(frames, dtype=torch.float32).view(1, frames, 1)
        aligned = torch.nn.functional.interpolate(
            audio.transpose(1, 2), size=tokens_len,
            mode="linear", align_corners=True).transpose(1, 2)
        self.assertAlmostEqual(float(aligned[0, 0, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(aligned[0, -1, 0]), frames - 1.0, places=4)


if __name__ == "__main__":
    unittest.main()


class TrainingAndInferenceAgreeTests(unittest.TestCase):
    """Training, full inference and cached decoding must all mean the same thing.

    Today's repeated failure was divergence between the two transformer implementations:
    the RoPE rescale went into training alone and a checkpoint quietly meant two
    different things. This path has a sharper version of the same trap -- training's
    input is 1499 steps because of the input/target shift, and a cached generation step
    is 1, so resampling onto "whatever length this forward received" would place the
    audio differently in each. Both resample onto the full window and slice.
    """

    STEPS, FRAMES = 12, 9

    def _models(self):
        from inference.model_inference import (
            TransformerDecoderAudioConditioned as Inference)

        arguments = dict(ARGS, aligned_audio=True, grid_ms=10,
                         window_seconds=self.STEPS / 100.0)
        torch.manual_seed(1)
        training = TransformerDecoderAudioConditioned(**arguments).eval()
        torch.manual_seed(1)
        inference = Inference(**arguments).eval()
        inference.load_state_dict(training.state_dict(), strict=True)
        with torch.no_grad():
            training.align_gate.fill_(0.7)
            inference.align_gate.fill_(0.7)
        return training, inference

    def test_both_derive_the_same_window_length(self):
        training, inference = self._models()
        self.assertEqual(training.token_length, self.STEPS)
        self.assertEqual(inference.token_length, self.STEPS)

    def test_inference_matches_training_with_the_gate_open(self):
        training, inference = self._models()
        tokens = torch.randint(0, 35, (1, self.STEPS))
        codes = torch.randint(0, 64, (1, 4, self.FRAMES))
        with torch.no_grad():
            expected = training(tokens, codes)
            audio = sum(inference.codes_embedding[i](codes[:, i]) for i in range(4))
            actual = inference(tokens, inference.norm_audio(audio))
        torch.testing.assert_close(expected, actual)

    def test_cached_decoding_matches_a_full_forward_with_the_gate_open(self):
        """The half that would break if the aligned slice used the wrong offset."""
        _, inference = self._models()
        tokens = torch.randint(0, 35, (1, self.STEPS))
        codes = torch.randint(0, 64, (1, 4, self.FRAMES))
        with torch.no_grad():
            audio = sum(inference.codes_embedding[i](codes[:, i]) for i in range(4))
            audio = inference.norm_audio(audio)
            whole = inference(tokens, audio)

            self_cache = [None] * inference.n_layers
            cross_cache = [None] * inference.n_layers
            stepwise = []
            for step in range(self.STEPS):
                logits, self_cache, cross_cache = inference(
                    tokens[:, step:step + 1], audio, step=step, use_cache=True,
                    self_cache=self_cache, cross_cache=cross_cache)
                stepwise.append(logits[:, -1])
            stepwise = torch.stack(stepwise, dim=1)
        torch.testing.assert_close(whole, stepwise, atol=2e-4, rtol=2e-3)
