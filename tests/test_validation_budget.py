"""A run should say how much of itself is validation, before it spends the time.

configs/default.yaml has carried a comment about this for a while: Lightning's
val_check_interval counts BATCHES, ModelCheckpoint's every_n_train_steps counts
OPTIMIZER STEPS. The comment did not stop it happening again -- a run meant to validate
twice was set to val_check_interval=500 with accumulate_grad_batches=8 and
max_steps=1000, which is 8000 batches and sixteen validations, about four hours of
validation around one hour of training.
"""

import unittest

from omegaconf import OmegaConf

from modules.run_utils import validation_budget


def config(accumulate=8, max_steps=1000):
    return OmegaConf.create(
        {"trainer": {"accumulate_grad_batches": accumulate, "max_steps": max_steps}})


class ValidationBudgetTests(unittest.TestCase):

    def test_nothing_is_said_when_validation_is_not_scheduled(self):
        self.assertEqual(validation_budget(config(), None, 100), [])

    def test_the_batch_to_step_translation_is_reported(self):
        lines = validation_budget(config(accumulate=8), 800, None)
        self.assertIn("every 100 optimizer steps", lines[0])

    def test_the_run_that_went_wrong_is_warned_about(self):
        """val_check_interval=500, accumulate 8, 1000 steps, 100 val batches."""
        lines = validation_budget(config(), 500, 100)
        self.assertEqual(len(lines), 2)
        self.assertIn("16 validation run(s)", lines[0])
        self.assertIn("WARNING", lines[1])
        self.assertIn("N * 8", lines[1])

    def test_a_sensible_schedule_draws_no_warning(self):
        """Two validations of 50 batches against 8000 training batches."""
        lines = validation_budget(config(), 4000, 50)
        self.assertEqual(len(lines), 1)
        self.assertIn("2 validation run(s)", lines[0])

    def test_an_open_ended_run_reports_the_rate_but_not_a_share(self):
        """Without max_steps there is no total to take a fraction of."""
        lines = validation_budget(config(max_steps=-1), 500, 100)
        self.assertEqual(len(lines), 1)
        self.assertNotIn("validation run(s)", lines[0])

    def test_a_float_interval_is_a_fraction_of_an_epoch_not_a_batch_count(self):
        """Lightning treats a float as a proportion, so the arithmetic does not apply."""
        self.assertEqual(validation_budget(config(), 0.25, 100), [])


if __name__ == "__main__":
    unittest.main()
