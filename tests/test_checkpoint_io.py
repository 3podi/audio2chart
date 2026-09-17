import tempfile
import unittest
from pathlib import Path

import torch

from modules.run_utils import DirectCheckpointIO


class DirectCheckpointIOTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.io = DirectCheckpointIO()
        self.payload = {"state_dict": {"w": torch.zeros(300_000)}, "epoch": 3}

    def tearDown(self):
        self.directory.cleanup()

    def test_round_trips(self):
        path = self.root / "last.ckpt"
        self.io.save_checkpoint(self.payload, path)
        self.assertTrue(path.is_file())
        loaded = torch.load(path, weights_only=False)
        self.assertEqual(3, loaded["epoch"])
        self.assertTrue(torch.equal(self.payload["state_dict"]["w"], loaded["state_dict"]["w"]))

    def test_creates_missing_directories(self):
        path = self.root / "a" / "b" / "last.ckpt"
        self.io.save_checkpoint(self.payload, path)
        self.assertTrue(path.is_file())

    def test_leaves_no_partial_file_behind(self):
        path = self.root / "last.ckpt"
        self.io.save_checkpoint(self.payload, path)
        self.assertEqual([], list(self.root.glob("*.partial")))

    def test_a_short_write_is_reported_and_cleaned_up(self):
        path = self.root / "tiny.ckpt"
        with self.assertRaises(RuntimeError):
            self.io.save_checkpoint({"x": 1}, path)      # far under MINIMUM_BYTES
        self.assertFalse(path.exists())
        self.assertEqual([], list(self.root.glob("*.partial")))

    def test_overwrites_an_existing_checkpoint(self):
        path = self.root / "last.ckpt"
        self.io.save_checkpoint(self.payload, path)
        self.io.save_checkpoint({"state_dict": {"w": torch.ones(300_000)}, "epoch": 9}, path)
        self.assertEqual(9, torch.load(path, weights_only=False)["epoch"])

    def test_rejects_storage_options(self):
        with self.assertRaises(TypeError):
            self.io.save_checkpoint(self.payload, self.root / "x.ckpt", storage_options={"a": 1})


if __name__ == "__main__":
    unittest.main()
