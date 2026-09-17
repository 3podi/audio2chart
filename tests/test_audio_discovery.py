import tempfile
import unittest
from pathlib import Path

from dataloader.utils_dataloader import AUDIO_CANDIDATES, find_audio_for


class FindAudioTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def touch(self, *names):
        for name in names:
            (self.root / name).write_bytes(b"\x00")

    def test_prefers_song_over_guitar(self):
        self.touch("guitar.ogg", "song.ogg")
        self.assertEqual("song.ogg", find_audio_for(self.root).name)

    def test_falls_back_to_guitar(self):
        self.touch("guitar.ogg")
        self.assertEqual("guitar.ogg", find_audio_for(self.root).name)

    def test_recovers_an_arbitrarily_named_audio_file(self):
        # Three folders in the real library name the audio after the song.
        self.touch("falsefuneral.mp3")
        self.assertEqual("falsefuneral.mp3", find_audio_for(self.root).name)

    def test_never_picks_a_preview(self):
        # preview.* is a ~30s clip; pairing it with a full chart destroys the timing.
        self.touch("preview.ogg")
        self.assertIsNone(find_audio_for(self.root))

    def test_preview_is_ignored_even_beside_real_audio(self):
        self.touch("preview.ogg", "somesong.mp3")
        self.assertEqual("somesong.mp3", find_audio_for(self.root).name)

    def test_video_is_not_audio(self):
        self.touch("video.webm")
        self.assertIsNone(find_audio_for(self.root))

    def test_empty_folder(self):
        self.assertIsNone(find_audio_for(self.root))

    def test_non_audio_files_are_ignored(self):
        self.touch("album.jpg", "song.ini", "notes.chart")
        self.assertIsNone(find_audio_for(self.root))

    def test_stem_split_song_picks_guitar_by_candidate_order(self):
        self.touch("guitar.ogg", "bass.ogg", "drums.ogg", "vocals.ogg")
        self.assertEqual("guitar.ogg", find_audio_for(self.root).name)

    def test_fallback_is_deterministic(self):
        self.touch("b_track.mp3", "a_track.mp3")
        self.assertEqual("a_track.mp3", find_audio_for(self.root).name)

    def test_new_extensions_are_accepted(self):
        for ext in ("m4a", "aac", "flac", "oga"):
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / f"track.{ext}"
                p.write_bytes(b"\x00")
                self.assertEqual(p.name, find_audio_for(Path(d)).name, ext)

    def test_candidate_list_still_covers_the_common_names(self):
        for name in ("song.ogg", "song.opus", "song.mp3", "guitar.ogg"):
            self.assertIn(name, AUDIO_CANDIDATES)


if __name__ == "__main__":
    unittest.main()
