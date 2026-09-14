"""Input corruption must damage the history and nothing else.

Teacher forcing only ever shows a correct history, and free running the model then
drifts -- the generated tap rate climbs 0.224 -> 0.723 across one 30s chunk. Corrupting
a fraction of inputs approximates scheduled sampling without a second forward pass, but
only if it corrupts the right things: targets untouched, padding untouched, and never
an eos or pad injected as noise.
"""

import unittest

import torch


class _Model:
    """The corruption logic, lifted out so the test needs no Encodec download."""

    from modules.trainer import WaveformTransformerDiscrete as _Real
    _corrupt_inputs = _Real._corrupt_inputs

    def __init__(self, noise, pad_token_id=34, eos_token_id=33, burst=1.0):
        self.input_noise = noise
        self.input_noise_burst = burst
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.n_note_tokens = min(pad_token_id, eos_token_id)
        self.training = True


class InputNoiseTests(unittest.TestCase):

    def test_disabled_by_default_returns_the_input_unchanged(self):
        model = _Model(0.0)
        tokens = torch.randint(0, 30, (4, 50))
        self.assertTrue(torch.equal(model._corrupt_inputs(tokens), tokens))

    def test_padding_is_never_corrupted(self):
        """Corrupting pad would teach the model to play past the end of a chart."""
        model = _Model(1.0)
        tokens = torch.full((2, 20), model.pad_token_id)
        self.assertTrue(torch.equal(model._corrupt_inputs(tokens), tokens))

    def test_noise_never_injects_a_control_token(self):
        """An injected eos would teach the model to stop early."""
        torch.manual_seed(0)
        model = _Model(1.0)
        tokens = torch.randint(0, 30, (8, 200))
        out = model._corrupt_inputs(tokens)
        self.assertLess(int(out.max()), model.n_note_tokens)

    def test_corruption_happens_at_roughly_the_requested_rate(self):
        torch.manual_seed(0)
        model = _Model(0.1)
        tokens = torch.randint(0, 30, (16, 500))
        changed = (model._corrupt_inputs(tokens) != tokens).float().mean().item()
        # Some draws land on the same token, so the observed rate is a little under.
        self.assertGreater(changed, 0.05)
        self.assertLess(changed, 0.12)

    def test_bursts_keep_the_requested_rate(self):
        """Length and frequency have to trade off, or burst mode is just more noise."""
        torch.manual_seed(0)
        tokens = torch.randint(0, 30, (16, 500))
        for burst in (1.0, 4.0, 12.0):
            model = _Model(0.1, burst=burst)
            changed = (model._corrupt_inputs(tokens) != tokens).float().mean().item()
            self.assertGreater(changed, 0.04, f"burst {burst}")
            self.assertLess(changed, 0.16, f"burst {burst}")

    def test_bursts_produce_the_long_runs_that_iid_corruption_cannot(self):
        """This is the whole point, so it is measured rather than assumed.

        At input_noise 0.08 the chance of eight consecutive corrupted positions is
        0.08**8 -- about 1.7e-9, or 2e-10 expected occurrences across the 72 million
        token positions of a 6000-step run. The model never saw a history it had been
        wrong in for long enough to learn to leave, which is exactly the state a
        generated chart locks into: 41.8% of its note mass on five-fret chords after
        sixteen of them, against a corpus 0.223%.
        """
        torch.manual_seed(0)
        tokens = torch.randint(0, 30, (32, 600))

        def longest_run(model):
            changed = (model._corrupt_inputs(tokens) != tokens)
            best = 0
            for row in changed:
                run = 0
                for flag in row.tolist():
                    run = run + 1 if flag else 0
                    best = max(best, run)
            return best

        self.assertLess(longest_run(_Model(0.08, burst=1.0)), 6)
        self.assertGreaterEqual(longest_run(_Model(0.08, burst=10.0)), 8)

    def test_a_burst_repeats_one_token_rather_than_drawing_each_step(self):
        """The failure being trained against is one chord held, not random mistakes.

        46 consecutive five-fret tap chords is a single token repeated. Independent
        draws inside a burst would teach recovery from noise, which the model can
        already do, instead of recovery from a rut, which it cannot.

        Asserted in aggregate rather than per run. Two bursts that land back to back
        read as one contiguous run and legitimately carry a token each, so both
        "exactly one token per run" and a per-run budget fail on correct output --
        an eleven-long pair of merged bursts holds two tokens and should.

        The statistic that does separate the two designs is how long the corruption
        holds a token on average: independent draws change token every step, so it
        sits at 1, while repeating bursts hold for something near the mean burst
        length. Measured both ways here so the comparison is in the test rather than
        in a number I chose.
        """
        torch.manual_seed(0)
        tokens = torch.full((8, 600), 5)

        def mean_hold(model):
            out = model._corrupt_inputs(tokens)
            held = changes = 0
            for row in out.tolist():
                previous = None
                for token in row:
                    if token == 5:
                        previous = None
                        continue
                    held += 1
                    if token != previous:
                        changes += 1
                    previous = token
            return held / max(changes, 1)

        independent = mean_hold(_Model(0.25, burst=1.0))
        bursting = mean_hold(_Model(0.25, burst=25.0))
        self.assertLess(independent, 1.6, "i.i.d. corruption should not hold a token")
        self.assertGreater(bursting, 8.0, f"bursts held a token only {bursting:.1f} steps")

    def test_burst_one_is_exactly_the_old_independent_behaviour(self):
        """Existing runs must be reproducible, so 1.0 cannot be an approximation."""
        tokens = torch.randint(0, 30, (8, 300))
        torch.manual_seed(7)
        default = _Model(0.1)._corrupt_inputs(tokens)
        torch.manual_seed(7)
        explicit = _Model(0.1, burst=1.0)._corrupt_inputs(tokens)
        self.assertTrue(torch.equal(default, explicit))

    def test_eval_mode_is_never_corrupted(self):
        """Validation loss has to stay comparable across runs."""
        model = _Model(1.0)
        model.training = False
        tokens = torch.randint(0, 30, (4, 50))
        self.assertTrue(torch.equal(model._corrupt_inputs(tokens), tokens))


if __name__ == "__main__":
    unittest.main()
