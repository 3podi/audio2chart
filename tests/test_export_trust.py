"""Exporting must not deserialize arbitrary Python unless asked.

torch.load with weights_only=False runs whatever the pickle says to run. Lightning's
save_hyperparameters() puts the whole Hydra config in the checkpoint, so a safe load
refuses our own files and the temptation is to flip the default -- which would also
silently execute a downloaded checkpoint. The escape hatch stays opt-in, and this
pins that default so it cannot drift.
"""

import unittest
from pathlib import Path
from unittest import mock

import export_checkpoint


class ExportTrustTests(unittest.TestCase):

    def _run(self, trust):
        """Call export_checkpoint far enough to capture the torch.load kwargs."""
        with mock.patch.object(export_checkpoint.torch, "load") as load, \
             mock.patch.object(export_checkpoint, "TransformerConfig"), \
             mock.patch.object(Path, "open", mock.mock_open(read_data="{}")), \
             mock.patch.object(export_checkpoint, "json") as js:
            # The exporter now refuses a config without the timing fields, since a
            # model exported with the wrong grid emits the wrong number of tokens.
            js.load.return_value = {"grid_ms": 10, "window_seconds": 15.0}
            load.return_value = {}
            try:
                export_checkpoint.export_checkpoint(
                    Path("ckpt"), Path("cfg"), Path("out"), trust=trust)
            except Exception:
                pass  # fails later on the empty state dict; we only need the load call
            if not load.call_args:
                self.fail("torch.load was never called")
            return load.call_args.kwargs

    def test_default_refuses_to_execute_pickled_code(self):
        self.assertTrue(self._run(trust=False)["weights_only"])

    def test_trust_opts_in(self):
        self.assertFalse(self._run(trust=True)["weights_only"])

    def test_the_flag_is_off_unless_passed(self):
        import inspect
        signature = inspect.signature(export_checkpoint.export_checkpoint)
        self.assertIs(signature.parameters["trust"].default, False)


if __name__ == "__main__":
    unittest.main()
