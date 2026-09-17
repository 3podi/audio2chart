"""Every config key the entry points read must exist in the composed configs.

configs/audio.yaml once carried `seed: 42` as its only definition; rewriting that file
dropped it, and nothing failed until main.py reached set_seed_everything() at run time.
The key list is scraped from the sources so it cannot drift out of date.
"""

import re
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
MISSING = object()

# Plain `config.a.b` attribute access raises when the key is absent, so those keys are the
# mandatory ones. OmegaConf.select(config, "a.b", default=...) is how this codebase spells an
# optional setting, and those keys are allowed to be missing from the YAML.
ATTRIBUTE = re.compile(r"\bconfig\.((?:[a-z_][a-z0-9_]*)(?:\.[a-z_][a-z0-9_]*)*)")
SELECTED = re.compile(r"OmegaConf\.select\(\s*config\s*,\s*[\"']([a-z_.]+)[\"']([^)]*)\)")


def referenced_keys(*sources: Path) -> set[str]:
    """Config keys the given sources require to exist."""
    keys: set[str] = set()
    for source in sources:
        text = source.read_text(encoding="utf-8")
        keys |= set(ATTRIBUTE.findall(text))
        keys |= {key for key, rest in SELECTED.findall(text) if "default" not in rest}
    return keys


class ConfigSurfaceTest(unittest.TestCase):
    def _compose(self, name):
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            return compose(config_name=name)

    def _assert_resolves(self, config, keys, config_name):
        absent = sorted(k for k in keys if OmegaConf.select(config, k, default=MISSING) is MISSING)
        self.assertEqual([], absent, f"configs/{config_name}.yaml is missing: {absent}")

    def test_audio_config_covers_the_training_entry_point(self):
        keys = referenced_keys(ROOT / "main.py", ROOT / "modules" / "run_utils.py")
        self.assertIn("seed", keys)
        self._assert_resolves(self._compose("audio"), keys, "audio")

    def test_audio_config_matches_the_released_checkpoint_architecture(self):
        # load_pretrained_transformer refuses a partial load, so a drift here stops training.
        config = self._compose("audio")
        if not OmegaConf.select(config, "model.pretrained"):
            self.skipTest("no pretrained checkpoint configured")
        transformer = config.model.transformer
        self.assertEqual(1024, transformer.d_model)
        self.assertEqual(16, transformer.n_layers)
        self.assertEqual(8, transformer.num_kv_heads)
        self.assertEqual(2, transformer.compression)

    def test_window_and_grid_fit_inside_max_length(self):
        config = self._compose("audio")
        steps = config.data.window_seconds * 1000 / config.data.grid_ms
        self.assertEqual(steps, int(steps), "window must divide evenly into the grid")
        self.assertLessEqual(int(steps) + 2, config.data.max_length)


if __name__ == "__main__":
    unittest.main()


class OverridableKnobTests(unittest.TestCase):
    """A knob read with a default is not a knob you can set.

    modules/trainer reads input_noise with OmegaConf.select(..., default=0.0), which
    works for reading and does nothing for overriding: Hydra refuses
    `model.input_noise=0.08` unless the config declares the key. The staged config
    failed on exactly that, so the declaration is pinned here.
    """

    def test_input_noise_can_be_overridden(self):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="audio", overrides=["model.input_noise=0.08"])
        self.assertAlmostEqual(cfg.model.input_noise, 0.08)

    def test_input_noise_defaults_to_off(self):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="audio")
        self.assertEqual(cfg.model.input_noise, 0.0)


class ActiveModelConfigTests(unittest.TestCase):
    """configs/audio.yaml composes `model: audio_codec`, not audio_discrete.

    Editing the wrong one is silent: the file parses, the value looks set, and nothing
    reaches a run. It happened once -- a freeze_encoder change went into
    audio_discrete.yaml and was reported as a fix when audio_codec had already had the
    value for months.
    """

    def _model(self, *overrides):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            return compose(config_name="audio", overrides=list(overrides)).model

    def test_every_knob_reaches_the_composed_config(self):
        model = self._model()
        for key in ("freeze_encoder", "input_noise", "init_from", "freeze_layers"):
            self.assertIn(key, model, f"{key} is not in the composed model config")

    def test_init_from_can_be_overridden(self):
        self.assertEqual("/x.ckpt", self._model("model.init_from=/x.ckpt").init_from)

    def test_the_encoder_is_frozen_in_the_active_config(self):
        self.assertTrue(self._model().freeze_encoder)
