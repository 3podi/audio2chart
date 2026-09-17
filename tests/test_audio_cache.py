"""The audio cache has to hold entries and stay bounded.

It was read but never written, so every sample re-read the whole file. Filling it is
a real but small win -- at 15 s windows with one piece a read serves a quarter of what
it served at 30 s and two.

An earlier version of this docstring blamed the dead cache for the GPU starving, with
utilisation swinging 87% to 2%. That was wrong: profiled over 256 batches of the real
config, train_dataloader_next is 0.1% of a step, and filling the cache moved throughput
0.77 -> 0.82 it/s. The starvation came from checkpointing and validation, not here.

Bounded by count, not by the max_cache_gb budget -- that budget is per worker and sized
in gigabytes, which is how the chart cache once took the machine from 18 GB free to 0.5.
"""

import unittest
from collections import OrderedDict
from unittest import mock

import torch


class _Dataset:
    """The caching logic alone, without needing a corpus to construct against."""

    from dataloader.audio_loader import ChunkedWaveformDataset as _Real

    AUDIO_CACHE_ENTRIES = _Real.AUDIO_CACHE_ENTRIES
    _load_audio_file = _Real._load_audio_file

    def __init__(self):
        self._audio_cache = OrderedDict()
        self.use_predecoded_raw = True
        self.sample_rate = 24000

    def _get_worker_id(self):
        return 0


class AudioCacheTests(unittest.TestCase):

    def setUp(self):
        self.dataset = _Dataset()
        self.reads = []

    def _load(self, path):
        def fake(p, sr):
            self.reads.append(p)
            return torch.zeros(1, 10), sr
        with mock.patch("dataloader.audio_loader.load_raw_audio", side_effect=fake):
            return self.dataset._load_audio_file(path)

    def test_a_second_read_of_one_song_hits_the_cache(self):
        self._load("a.raw")
        self._load("a.raw")
        self.assertEqual(["a.raw"], self.reads)

    def test_the_cache_is_bounded_by_count(self):
        for index in range(_Dataset.AUDIO_CACHE_ENTRIES + 10):
            self._load(f"{index}.raw")
        self.assertLessEqual(len(self.dataset._audio_cache),
                             _Dataset.AUDIO_CACHE_ENTRIES)

    def test_eviction_is_least_recently_used(self):
        for index in range(_Dataset.AUDIO_CACHE_ENTRIES):
            self._load(f"{index}.raw")
        self._load("0.raw")                     # refresh the oldest
        self._load("new.raw")                   # forces one eviction
        self.assertIn("0.raw", self.dataset._audio_cache)
        self.assertNotIn("1.raw", self.dataset._audio_cache)

    def test_a_failed_read_is_not_cached(self):
        """Caching a failure would make one bad file poison every later attempt."""
        with mock.patch("dataloader.audio_loader.load_raw_audio",
                        side_effect=OSError("nope")):
            with self.assertRaises(Exception):
                self.dataset._load_audio_file("bad.raw")
        self.assertNotIn("bad.raw", self.dataset._audio_cache)


if __name__ == "__main__":
    unittest.main()
