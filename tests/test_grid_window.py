"""The grid and window must agree with the sequence limit and with each other.

discretize_time bins notes at grid_ms, and a window holding two notes in one bin is
discarded. At 20 ms only 18.2% of 30 s windows in tapping charts survived -- 22.8% of
note gaps there are shorter than 20 ms, and Robert's "yax03 - Down" reaches 7.3 ms. The
model was learning from a sixth of the overchart data, and the discarded part was the
fast tapping that defines the style.

10 ms at 15 s is the same 1502 tokens and recovers 97.2%. These pin that arithmetic,
because the failure is silent: a model asked for the wrong number of tokens produces
nonsense rather than an error.
"""

import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]


def load(name="audio"):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        return compose(config_name=name)


class GridWindowTests(unittest.TestCase):

    def setUp(self):
        self.cfg = load()

    def tokens(self):
        data = self.cfg.data
        return int(data.window_seconds * 1000 / data.grid_ms) + 2   # + bos/eos

    def test_a_window_fits_the_sequence_limit(self):
        self.assertLessEqual(self.tokens(), self.cfg.data.max_length)

    def test_the_grid_is_fine_enough_for_tapping(self):
        """22.8% of tapping-chart note gaps are under 20 ms; 10 ms covers 97.2%."""
        self.assertLessEqual(self.cfg.data.grid_ms, 10)

    def test_window_divides_evenly_into_grid_steps(self):
        """A fractional step would put the last bin off the grid."""
        steps = self.cfg.data.window_seconds * 1000 / self.cfg.data.grid_ms
        self.assertEqual(steps, int(steps))


class EngineAgreementTests(unittest.TestCase):
    """The engine must read the window from config, not assume it.

    chunk_sec was hardcoded to 30 in inference/engine.py while the trainer read
    window_seconds from config, so changing one silently broke the other: a model
    trained on 1502 tokens would be asked for 3002.
    """

    def test_engine_default_matches_nothing_by_accident(self):
        import inspect

        from inference.engine import Charter, TransformerConfig

        self.assertIn("window_seconds", TransformerConfig.__dataclass_fields__)
        source = inspect.getsource(Charter.generate)
        self.assertIn("self.config.window_seconds", source)
        self.assertNotIn("chunk_sec = 30", source)

    def test_exported_config_must_carry_the_timing(self):
        """Defaulting grid_ms or window_seconds on export hides the mismatch."""
        import export_checkpoint

        source = inspect.getsource(export_checkpoint.export_checkpoint)
        self.assertIn("window_seconds", source)
        self.assertIn("grid_ms", source)


import inspect  # noqa: E402  (used by EngineAgreementTests)

if __name__ == "__main__":
    unittest.main()
