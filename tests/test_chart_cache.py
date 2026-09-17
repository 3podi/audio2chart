import pickle
import tempfile
import unittest
from pathlib import Path

from dataloader.audio_loader import ChunkedWaveformDataset
from chart.tokenizer import SimpleTokenizerGuitar

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


class ChartCacheTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        self.entries = []
        for i in range(8):
            chart = root / f"notes{i}.chart"
            chart.write_text(CHART, encoding="utf-8")
            raw = root / f"song{i}.raw"
            raw.write_bytes(b"\x00" * 4000)
            self.entries.append({
                "chart_path": str(chart), "difficulty": "ExpertSingle",
                "audio_path": str(root / f"song{i}.ogg"),
                "raw_path": str(raw), "length_samples": 2000,
            })
        tok = SimpleTokenizerGuitar()
        self.ds = ChunkedWaveformDataset(
            self.entries, bos_token=tok.bos_id, eos_token=tok.eos_id, pad_token=tok.pad_id,
            tokenizer=tok, window_seconds=30.0, sample_rate=24000, error_policy="skip",
            use_predecoded_raw=True, chart_cache_size=3, chunk_size=4,
        )

    def tearDown(self):
        self.dir.cleanup()

    def test_cache_starts_empty(self):
        # The old code parsed every chart in __init__, which is what blew up memory.
        self.assertEqual(0, len(self.ds.chart_cache))

    def test_entries_are_parsed_on_demand(self):
        key = (self.entries[0]["chart_path"], "ExpertSingle")
        entry = self.ds._chart_entry(key)
        self.assertIsNotNone(entry)
        notes, bpm_events, resolution, offset = entry
        self.assertEqual(192, resolution)
        self.assertTrue(notes)

    def test_cache_is_bounded(self):
        for item in self.entries:
            self.ds._chart_entry((item["chart_path"], "ExpertSingle"))
        self.assertLessEqual(len(self.ds.chart_cache), 3)

    def test_least_recently_used_is_evicted(self):
        keys = [(e["chart_path"], "ExpertSingle") for e in self.entries[:4]]
        for k in keys:
            self.ds._chart_entry(k)
        self.assertNotIn(keys[0], self.ds.chart_cache)
        self.assertIn(keys[-1], self.ds.chart_cache)

    def test_a_hit_refreshes_recency(self):
        keys = [(e["chart_path"], "ExpertSingle") for e in self.entries[:3]]
        for k in keys:
            self.ds._chart_entry(k)
        self.ds._chart_entry(keys[0])                      # touch the oldest
        self.ds._chart_entry((self.entries[3]["chart_path"], "ExpertSingle"))
        self.assertIn(keys[0], self.ds.chart_cache)        # survived because it was touched
        self.assertNotIn(keys[1], self.ds.chart_cache)

    def test_pickling_sends_an_empty_cache_to_workers(self):
        for item in self.entries[:3]:
            self.ds._chart_entry((item["chart_path"], "ExpertSingle"))
        self.assertGreater(len(self.ds.chart_cache), 0)
        revived = pickle.loads(pickle.dumps(self.ds))
        self.assertEqual(0, len(revived.chart_cache),
                         "workers must not inherit the parent's chart cache")

    def test_unreadable_chart_is_cached_as_none_under_skip(self):
        missing = (str(Path(self.dir.name) / "gone.chart"), "ExpertSingle")
        self.assertIsNone(self.ds._chart_entry(missing))
        self.assertIn(missing, self.ds.chart_cache)        # negative result cached too


if __name__ == "__main__":
    unittest.main()
