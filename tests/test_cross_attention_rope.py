"""Cross-attention has to put both sequences on one clock before applying RoPE.

RoPE encodes relative position, which assumes a fixed relationship shows up at a fixed
offset. The decoder's two sequences are not on one clock: a token step is 10 ms and an
audio frame after compression 2 is 26.7 ms. So a time-aligned pair is (token i, audio
frame 0.375 i), and the raw index offset for it grows across the window -- 0, 235, 469,
937 -- meaning "attend to the audio at my own timestamp" lands somewhere different at
every position and is not a rule the model can express.

Measured on a model trained without the rescale: sliding a song two whole seconds against
its own chart cost 23% of what handing it a different song cost, and a quarter second
cost 10%. Three quarters of what it took from the audio was time-invariant -- it knew
what kind of song it was and not what was happening.
"""

import unittest

import torch

from modules.training_transformer import apply_rotary_emb

TOKENS = 1500      # 15 s at a 10 ms grid
FRAMES = 562       # 15 s at 75 fps, halved by compression 2
SCALE = TOKENS / FRAMES


def aligned_offsets(scale):
    """|token position - audio position| for pairs at the same instant."""
    out = []
    for token in range(0, TOKENS, 17):
        frame = round(token * FRAMES / TOKENS)
        out.append(abs(token - frame * scale))
    return out


class CrossAttentionRopeTests(unittest.TestCase):

    def test_rescaling_bounds_the_offset_for_an_aligned_pair(self):
        """Bounded by half a frame, which is the floor the rate difference imposes."""
        worst = max(aligned_offsets(SCALE))
        self.assertLessEqual(worst, SCALE / 2 + 1e-6,
                             "an aligned pair should sit within half an audio frame")

    def test_without_rescaling_the_offset_grows_across_the_window(self):
        """The bug, kept as a regression guard rather than described in prose."""
        offsets = aligned_offsets(1.0)
        self.assertGreater(max(offsets), 900)
        self.assertGreater(offsets[-1], offsets[len(offsets) // 2],
                           "the offset should have grown with position")

    def test_the_scale_reaches_apply_rotary_emb(self):
        """Position n at scale s must rotate like position n*s at scale 1."""
        dim = 64
        plain = apply_rotary_emb(torch.ones(1, 1, 64, dim), dim=dim)
        scaled = apply_rotary_emb(torch.ones(1, 1, 32, dim), dim=dim, position_scale=2.0)
        for index in range(32):
            self.assertTrue(
                torch.allclose(scaled[0, 0, index], plain[0, 0, index * 2], atol=1e-5),
                f"frame {index} at scale 2 should match token {index * 2}")

    def test_self_attention_is_untouched(self):
        """Default scale 1.0, or every existing checkpoint's positions move."""
        dim = 32
        x = torch.randn(1, 1, 40, dim)
        self.assertTrue(torch.equal(apply_rotary_emb(x, dim=dim),
                                    apply_rotary_emb(x, dim=dim, position_scale=1.0)))

    def test_cross_attention_still_runs_and_keeps_its_shape(self):
        from modules.training_transformer import CrossAttention

        layer = CrossAttention(d_model=64, n_heads=4, num_kv_heads=2).eval()
        query = torch.randn(2, 150, 64)
        key_value = torch.randn(2, 56, 64)
        with torch.no_grad():
            out = layer(query, key_value)
        out = out[0] if isinstance(out, tuple) else out
        self.assertEqual(tuple(out.shape), (2, 150, 64))

    def test_an_empty_key_sequence_does_not_divide_by_zero(self):
        """T_kv is used as a divisor for the scale."""
        from modules.training_transformer import CrossAttention

        layer = CrossAttention(d_model=32, n_heads=2, num_kv_heads=1).eval()
        with torch.no_grad():
            out = layer(torch.randn(1, 8, 32), torch.randn(1, 0, 32))
        out = out[0] if isinstance(out, tuple) else out
        self.assertEqual(out.shape[1], 8)


if __name__ == "__main__":
    unittest.main()


class InferenceMatchesTrainingTests(unittest.TestCase):
    """The rescale has to reach inference, including the cached decoding path.

    Training and inference keep separate implementations, and the training change alone
    made them disagree -- test_training_pipeline caught it. Cached decoding is the
    harder half: the query is one step long there, so T_q/T_kv cannot supply the ratio
    and the window length has to come from the config.
    """

    def _models(self, **extra):
        from inference.model_inference import (
            TransformerDecoderAudioConditioned as Inference)
        from modules.training_transformer import (
            TransformerDecoderAudioConditioned as Training)

        arguments = dict(vocab_size=35, pad_token_id=-1, eos_token_id=33,
                         d_model=32, n_heads=4, num_kv_heads=2, n_layers=2,
                         dropout=0.0, audio_drop=0.0, conditional=False,
                         use_flash=False, codebook_size=64)
        torch.manual_seed(1)
        training = Training(**arguments).eval()
        inference = Inference(**arguments, **extra).eval()
        inference.load_state_dict(training.state_dict(), strict=True)
        return training, inference

    def test_cached_decoding_matches_a_full_forward(self):
        """Step-by-step with a cache must equal one pass over the whole sequence."""
        steps, frames = 12, 9
        # grid_ms/window_seconds chosen so token_length is `steps`.
        training, inference = self._models(grid_ms=10, window_seconds=steps / 100.0)
        self.assertEqual(inference.token_length, steps)

        tokens = torch.randint(0, 35, (1, steps))
        codes = torch.randint(0, 64, (1, 4, frames))
        with torch.no_grad():
            audio = sum(inference.codes_embedding[i](codes[:, i]) for i in range(4))
            audio = inference.norm_audio(audio)
            whole = inference(tokens, audio)

            self_cache = [None] * inference.n_layers
            cross_cache = [None] * inference.n_layers
            stepwise = []
            for step in range(steps):
                logits, self_cache, cross_cache = inference(
                    tokens[:, step:step + 1], audio, step=step, use_cache=True,
                    self_cache=self_cache, cross_cache=cross_cache)
                stepwise.append(logits[:, -1])
            stepwise = torch.stack(stepwise, dim=1)

        torch.testing.assert_close(whole, stepwise, atol=2e-4, rtol=2e-3)

    def test_a_full_inference_forward_matches_training(self):
        training, inference = self._models()
        tokens = torch.randint(0, 35, (2, 5))
        codes = torch.randint(0, 64, (2, 4, 7))
        with torch.no_grad():
            expected = training(tokens, codes)
            audio = sum(inference.codes_embedding[i](codes[:, i]) for i in range(4))
            actual = inference(tokens, inference.norm_audio(audio))
        torch.testing.assert_close(expected, actual)
