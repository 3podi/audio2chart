import json
import os
import tempfile
import unittest
from pathlib import Path

from modules.utils_train import validate_dataset


CHART = """[Song]
{
  Resolution = 192
  Offset = 0
}
[SyncTrack]
{
  0 = TS 4
  0 = B 120000
}
[ExpertSingle]
{
  0 = N 0 0
  192 = N 1 0
  384 = N 2 0
}
"""


class ValidationCacheTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.chart = self.root / "notes.chart"
        self.chart.write_text(CHART, encoding="utf-8")
        self.cache = self.root / "validation_cache.json"
        self.entries = [{
            "chart_path": str(self.chart),
            "difficulty": "ExpertSingle",
            "audio_path": "song.ogg",
            "raw_path": "song.raw",
            "length_samples": 100,
        }]
        self.kwargs = dict(
            difficulties=["Expert"], instruments=["Single"],
            grid_ms=20, error_policy="skip", window_seconds=30.0,
        )

    def tearDown(self):
        self.directory.cleanup()

    def test_first_pass_populates_the_cache(self):
        valid = validate_dataset(self.entries, cache_path=self.cache, **self.kwargs)
        self.assertEqual(1, len(valid))
        self.assertTrue(self.cache.is_file())
        blob = json.loads(self.cache.read_text(encoding="utf-8"))
        self.assertIn("signature", blob)
        self.assertEqual(1, len(blob["entries"]))

    def _corrupt_preserving_stamp(self):
        """Replace the chart with unparseable text of identical size and mtime."""
        info = self.chart.stat()
        self.chart.write_text("x" * info.st_size, encoding="utf-8")
        os.utime(self.chart, ns=(info.st_atime_ns, info.st_mtime_ns))
        self.assertEqual(info.st_size, self.chart.stat().st_size)
        self.assertEqual(info.st_mtime_ns, self.chart.stat().st_mtime_ns)

    def test_a_hit_does_not_reparse_the_chart(self):
        validate_dataset(self.entries, cache_path=self.cache, **self.kwargs)
        self._corrupt_preserving_stamp()
        # Same mtime and size, so the cached verdict stands and the garbage is never read.
        valid = validate_dataset(self.entries, cache_path=self.cache, **self.kwargs)
        self.assertEqual(1, len(valid))

    def test_a_changed_chart_is_revalidated(self):
        validate_dataset(self.entries, cache_path=self.cache, **self.kwargs)
        self.chart.write_text("garbage", encoding="utf-8")   # different size -> stamp differs
        with self.assertRaises(ValueError):
            validate_dataset(self.entries, cache_path=self.cache, **self.kwargs)

    def test_a_cached_failure_is_reused(self):
        bad = [dict(self.entries[0], difficulty="HardSingle")]
        with self.assertRaises(ValueError):
            validate_dataset(bad, cache_path=self.cache, **self.kwargs)
        entries = json.loads(self.cache.read_text(encoding="utf-8"))["entries"]
        self.assertEqual(1, len(entries))
        self.assertTrue(next(iter(entries.values()))["error"])

    def test_changed_settings_discard_the_cache(self):
        validate_dataset(self.entries, cache_path=self.cache, **self.kwargs)
        original = json.loads(self.cache.read_text(encoding="utf-8"))["signature"]
        other = dict(self.kwargs, grid_ms=40)
        validate_dataset(self.entries, cache_path=self.cache, **other)
        rewritten = json.loads(self.cache.read_text(encoding="utf-8"))["signature"]
        self.assertNotEqual(original, rewritten)
        self.assertEqual(40, rewritten["grid_ms"])

    def test_a_corrupt_cache_is_ignored_not_fatal(self):
        self.cache.write_text("{not json", encoding="utf-8")
        valid = validate_dataset(self.entries, cache_path=self.cache, **self.kwargs)
        self.assertEqual(1, len(valid))

    def test_works_without_a_cache_path(self):
        valid = validate_dataset(self.entries, **self.kwargs)
        self.assertEqual(1, len(valid))


if __name__ == "__main__":
    unittest.main()
