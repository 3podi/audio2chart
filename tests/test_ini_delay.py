"""song.ini's `delay` is the second place a chart's audio offset can live.

Moonscraper's Config/clone_hero_ini_tags.txt lists it beside diff_guitar, and its
changelog puts the .chart `[Song] Offset` under "Advanced -> Legacy Options" -- so delay
is the newer field and Offset the legacy one. Nothing in this repository read delay until
2026-09-12.

Measured over the 1454-song tapping training split: 72 songs carry both and agree, 8
carry both and disagree, 128 carry only Offset, and 4 carry only delay. Those four were
training against audio they were shifted from, by up to two seconds.

Offset wins where both exist, because Offset is the field measured to improve alignment
against the audio onset envelope: as-written +0.197, ignored +0.140, inverted +0.089
over the 46 songs with |offset| >= 1.0 s, paired t = +3.78.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from chart.chart_processor import ChartProcessor

CHART = (
    "[Song]\n{{\n  Resolution = 192\n  Offset = \"{offset}\"\n}}\n"
    "[SyncTrack]\n{{\n  0 = B 120000\n}}\n"
    "[ExpertSingle]\n{{\n  0 = N 0 0\n  192 = N 1 0\n}}\n"
)


class IniDelayTest(unittest.TestCase):

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)

    def _build(self, offset: str, ini: str | None) -> Path:
        path = self.folder / "notes.chart"
        path.write_text(CHART.format(offset=offset), encoding="utf-8")
        if ini is not None:
            (self.folder / "song.ini").write_text(ini, encoding="utf-8")
        return path

    def _offset_of(self, path: Path) -> float:
        processor = ChartProcessor(["Expert"], ["Single"])
        processor.read_chart(str(path), target_sections="ExpertSingle")
        return float(processor.song_metadata["Offset"])

    def test_delay_fills_in_when_the_chart_has_no_offset(self):
        """The four songs that were silently misaligned."""
        path = self._build("0", "[song]\nname = x\ndelay = 2000\n")
        self.assertAlmostEqual(2.0, self._offset_of(path), places=6)

    def test_a_negative_delay_is_carried_through(self):
        path = self._build("0", "[song]\ndelay = -1500\n")
        self.assertAlmostEqual(-1.5, self._offset_of(path), places=6)

    def test_the_charts_own_offset_wins_over_a_disagreeing_delay(self):
        """Prevail carries Offset +0.870 against delay +0.001. Offset is the measured one."""
        path = self._build("0.870", "[song]\ndelay = 1\n")
        self.assertAlmostEqual(0.870, self._offset_of(path), places=6)

    def test_no_song_ini_is_not_an_error(self):
        path = self._build("0.25", None)
        self.assertAlmostEqual(0.25, self._offset_of(path), places=6)

    def test_a_missing_or_unparseable_delay_is_ignored(self):
        for ini in ("[song]\nname = x\n", "[song]\ndelay = \n", "[song]\ndelay = abc\n"):
            with self.subTest(ini=ini):
                path = self._build("0", ini)
                self.assertAlmostEqual(0.0, self._offset_of(path), places=6)

    def test_a_zero_delay_changes_nothing(self):
        path = self._build("0", "[song]\ndelay = 0\n")
        self.assertAlmostEqual(0.0, self._offset_of(path), places=6)

    def test_song_ini_oddities_do_not_raise(self):
        """15 of 1454 song.ini files in this corpus defeat configparser outright.

        Duplicate keys, no section header, stray bytes. A hand-rolled scan has to shrug
        at all of it rather than take a chart out of the corpus.
        """
        awkward = (
            "name = x\n"            # no [section] header at all
            "delay = 750\n"
            "delay = 900\n"          # duplicate key
            "= broken\n"
        )
        path = self._build("0", awkward)
        self.assertAlmostEqual(0.75, self._offset_of(path), places=6)

    def test_reading_from_text_without_a_path_still_works(self):
        """open_chart accepts raw text, in which case there is no song.ini to consult."""
        processor = ChartProcessor(["Expert"], ["Single"])
        processor.read_chart(None, chart_text=CHART.format(offset="0.4"),
                             target_sections="ExpertSingle")
        self.assertAlmostEqual(0.4, float(processor.song_metadata["Offset"]), places=6)


if __name__ == "__main__":
    unittest.main()
