import json
import pathlib
import tempfile
import unittest

from chart.chart_writer import fill_expert_single
from dataloader.merge_manifests import check_entry, merge


class CheckEntryTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.directory.name)
        self.chart = root / "notes.chart"
        self.chart.write_text("[Song]\n{\n}\n", encoding="utf-8")
        self.raw = root / "song.raw"
        self.raw.write_bytes(b"\x00" * 200)
        self.root = root
        self.entry = {
            "audio_path": "song.ogg",
            "chart_path": str(self.chart),
            "difficulty": "ExpertSingle",
            "raw_path": str(self.raw),
            "length_samples": 100,
        }

    def tearDown(self):
        self.directory.cleanup()

    def test_accepts_a_complete_entry(self):
        self.assertIsNone(check_entry(self.entry))

    def test_rejects_a_missing_field(self):
        self.assertIn("missing fields", check_entry(dict(self.entry, raw_path="")))

    def test_rejects_a_vanished_chart(self):
        entry = dict(self.entry, chart_path=str(self.root / "gone.chart"))
        self.assertEqual("chart file is gone", check_entry(entry))

    def test_rejects_a_missing_or_truncated_raw_file(self):
        self.assertEqual(
            "raw audio is missing",
            check_entry(dict(self.entry, raw_path=str(self.root / "gone.raw"))),
        )
        self.assertIn("100 samples", check_entry(dict(self.entry, length_samples=99)))

    def test_raw_check_can_be_skipped(self):
        entry = dict(self.entry, raw_path=str(self.root / "gone.raw"))
        self.assertIsNone(check_entry(entry, verify_raw=False))


class MergeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = pathlib.Path(self.directory.name)
        chart = root / "notes.chart"
        chart.write_text("[Song]\n{\n}\n", encoding="utf-8")
        raw = root / "song.raw"
        raw.write_bytes(b"\x00" * 200)
        self.root = root
        self.entry = {
            "audio_path": "song.ogg",
            "chart_path": str(chart),
            "difficulty": "ExpertSingle",
            "raw_path": str(raw),
            "length_samples": 100,
        }

    def tearDown(self):
        self.directory.cleanup()

    def _write(self, name, entries):
        path = self.root / name
        path.write_text(json.dumps(entries), encoding="utf-8")
        return path

    def test_deduplicates_on_chart_and_difficulty_and_drops_bad_entries(self):
        first = self._write("a.json", [self.entry, dict(self.entry, length_samples=1)])
        second = self._write("b.json", [self.entry, dict(self.entry, difficulty="HardSingle")])
        output = self.root / "merged.json"
        stats = merge([first, second], output, rejected_json=self.root / "rejected.json")

        self.assertEqual({"kept": 2, "duplicates": 1, "rejected": 1}, stats)
        merged = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual({"ExpertSingle", "HardSingle"}, {e["difficulty"] for e in merged})
        rejected = json.loads((self.root / "rejected.json").read_text(encoding="utf-8"))
        self.assertEqual(1, len(rejected))
        self.assertIn("reason", rejected[0])


class ChartWriterTest(unittest.TestCase):
    METADATA = {"name": "Song", "resolution": 192, "bpm": 120}

    def test_no_placeholder_survives(self):
        text = fill_expert_single([(0, "N", 0, 0)], metadata=dict(self.METADATA))
        self.assertNotIn("###", text)

    def test_defaults_are_applied_and_overridable(self):
        text = fill_expert_single([], metadata=dict(self.METADATA))
        self.assertIn('MusicStream = "song.ogg"', text)
        self.assertIn("Difficulty = 3", text)
        self.assertIn("Player2 = bass", text)

        custom = fill_expert_single([], metadata=dict(
            self.METADATA, musicstream="guitar.mp3", difficulty=6, player2="rhythm", year=1979
        ))
        self.assertIn('MusicStream = "guitar.mp3"', custom)
        self.assertIn("Difficulty = 6", custom)
        self.assertIn("Player2 = rhythm", custom)
        self.assertIn('Year = ", 1979"', custom)

    def test_empty_text_fields_fall_back_to_a_marker(self):
        text = fill_expert_single([], metadata=dict(self.METADATA, artist="", genre=None))
        self.assertIn('Name = "Song"', text)
        self.assertIn('Artist = "audio2chart"', text)
        self.assertIn('Genre = "audio2chart"', text)

    def test_notes_land_in_the_expert_single_block(self):
        text = fill_expert_single([(0, "N", 0, 0), (192, "N", 2, 96)], metadata=dict(self.METADATA))
        block = text.split("[ExpertSingle]")[1]
        self.assertIn("  0 = N 0 0", block)
        self.assertIn("  192 = N 2 96", block)


if __name__ == "__main__":
    unittest.main()
