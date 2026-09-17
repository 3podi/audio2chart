"""In a folder that ships split, `song.*` is not the song.

Rock Band style rips supply the parts separately -- guitar.ogg, vocals.ogg, drums_1.ogg --
and `song.*` is the backing track, everything *except* those. Clone Hero plays all of them
together; the song is their sum.

`find_audio_for` preferred `song.*`, so for those folders the model was trained on a mix
with the guitar removed and asked to chart a part it could not hear. 73 songs in the
tapping split. The function's own docstring claimed the opposite -- "guitar.* wins, so the
model sees isolated guitar" -- and said 10 folders, so the comment was wrong about the
mechanism, the direction and the size.

Found by playing a generated chart and noticing the audio was wrong. No metric here could
have seen it: every one of them reads the chart, not what the model heard.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from dataloader.utils_dataloader import find_audio_for, stem_files


class StemDetectionTest(unittest.TestCase):

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)

    def _touch(self, *names: str) -> None:
        for name in names:
            (self.folder / name).write_bytes(b"\x00" * 64)

    def test_a_single_song_file_is_not_a_split(self):
        self._touch("song.ogg", "notes.chart")
        self.assertEqual([], stem_files(self.folder))

    def test_song_plus_named_parts_is_a_split(self):
        self._touch("song.ogg", "guitar.ogg", "vocals.ogg", "notes.chart")
        found = {p.name for p in stem_files(self.folder)}
        self.assertEqual({"song.ogg", "guitar.ogg", "vocals.ogg"}, found)

    def test_song_belongs_in_the_mix_rather_than_replacing_it(self):
        """The backing track is a part like any other, not the whole."""
        self._touch("song.ogg", "guitar.ogg")
        found = {p.name for p in stem_files(self.folder)}
        self.assertIn("song.ogg", found)

    def test_numbered_parts_count_as_one_named_part(self):
        """drums_1/2/3 beside song.* is still a split, but drums alone with song is too.

        The guard is on distinct *names*, so three numbered drum stems do not on their own
        make a folder look split when there is nothing else -- they do, because drums and
        song are two different names.
        """
        self._touch("song.ogg", "drums_1.ogg", "drums_2.ogg", "drums_3.ogg")
        self.assertGreaterEqual(len(stem_files(self.folder)), 4)

    def test_previews_are_never_treated_as_stems(self):
        self._touch("song.ogg", "guitar.ogg", "preview.ogg")
        found = {p.name for p in stem_files(self.folder)}
        self.assertNotIn("preview.ogg", found)

    def test_an_ordinary_folder_still_resolves_to_its_song_file(self):
        self._touch("song.ogg", "notes.chart")
        self.assertEqual("song.ogg", find_audio_for(self.folder).name)

    def test_a_folder_with_no_audio_resolves_to_nothing(self):
        self._touch("notes.chart")
        self.assertIsNone(find_audio_for(self.folder))


if __name__ == "__main__":
    unittest.main()
