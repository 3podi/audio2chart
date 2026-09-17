import os
import re


DIFFICULTIES = ["Expert", "Hard", "Medium", "Easy"]
# Clone Hero section suffixes.  Single and Drums remain the defaults used by the
# training pipeline; the additional names only broaden what the parser accepts.
INSTRUMENTS = [
    "Single", "DoubleGuitar", "DoubleBass", "Rhythm", "Keyboard", "Keys",
    "Drums", "GHLGuitar", "GHLBass", "GHLRhythm", "GHLCoop",
]

_SECTION = re.compile(
    r"(?ms)^\s*\[([^\]\r\n]+)\]\s*\{(.*?)^\s*\}"
)
_SYNC_EVENT = re.compile(r"(\d+)\s*=\s*B\s*(\d+)")
_NOTE_EVENT = re.compile(r"(\d+)\s*=\s*(N|S)\s*(\d+)\s*(\d+)")
_METADATA = re.compile(r'(Resolution|Offset|Genre)\s*=\s*"?([^"\r\n]+?)"?\s*$')


def _ini_delay_seconds(chart_path) -> float | None:
    """The `delay` beside a chart in song.ini, in seconds, or None.

    Clone Hero carries the audio offset in two places and Moonscraper writes both:
    `[Song] Offset` in the .chart, in seconds, and `delay` in song.ini, in milliseconds.
    Moonscraper's changelog puts the .chart field under "Advanced -> Legacy Options", so
    delay is the newer of the two.

    Measured over the 1454-song tapping training split: 72 songs carry both and agree,
    8 carry both and disagree, 128 carry only Offset, and 4 carry only delay. Those last
    four were silently misaligned -- by up to two seconds -- because nothing here read
    this field.

    Read with a hand-rolled scan rather than configparser: song.ini in the wild has
    duplicate keys, missing sections and mixed encodings, and 15 of 1454 files in this
    corpus defeat configparser outright.
    """
    if not chart_path:
        return None
    folder = os.path.dirname(str(chart_path))
    for name in ("song.ini", "Song.ini", "SONG.INI"):
        candidate = os.path.join(folder, name)
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8-sig", errors="replace") as stream:
                for line in stream:
                    key, separator, value = line.partition("=")
                    if separator and key.strip().lower() == "delay":
                        try:
                            return float(value.strip()) / 1000.0
                        except ValueError:
                            return None
        except OSError:
            return None
        return None
    return None


def _as_list(value):
    if isinstance(value, str):
        return [value]
    return list(value)


class ChartProcessor:
    def __init__(self, difficulties, instruments):
        difficulties = _as_list(difficulties)
        instruments = _as_list(instruments)
        if not all(value in DIFFICULTIES for value in difficulties):
            raise AssertionError(f"difficulties must be chosen from {DIFFICULTIES}")
        if not all(value in INSTRUMENTS for value in instruments):
            raise AssertionError(f"instruments must be chosen from {INSTRUMENTS}")

        self.difficulties = difficulties
        self.instruments = instruments
        self.sections = [
            difficulty + instrument
            for instrument in instruments
            for difficulty in difficulties
        ] + ["Song", "SyncTrack"]
        # Kept for callers that inspected these public attributes.
        self.regexes = {
            name: re.compile(rf"\[{re.escape(name)}\]\s*\{{(.*?)\}}", re.DOTALL)
            for name in self.sections
        }
        self.regex_metadata = _METADATA.pattern

    def open_chart(self, chart_path, chart_text=None):
        if chart_text is not None:
            self.chart_text = chart_text
        else:
            with open(chart_path, "r", encoding="utf-8-sig") as chart_file:
                self.chart_text = chart_file.read()
        self.synctrack = []
        self.notes = {}
        self.song_metadata = None

    def _scan_sections(self):
        """Scan the chart once using one reusable section pattern."""
        for match in _SECTION.finditer(self.chart_text):
            yield match.group(1).strip(), match.group(2).strip()

    def extract_sections(self):
        return self.extract_sections2()

    def extract_sections2(self, target_sections=None):
        selected = self.sections if target_sections is None else _as_list(target_sections)
        # Metadata and timing are required to interpret any requested note section.
        selected = set(selected) | {"Song", "SyncTrack"}
        return {
            name: content
            for name, content in self._scan_sections()
            if name in selected
        }

    @staticmethod
    def _parse_sync_track(content):
        events = []
        for line in content.splitlines():
            match = _SYNC_EVENT.match(line.strip())
            if match:
                events.append((int(match.group(1)), int(match.group(2))))
        return events

    @staticmethod
    def _parse_metadata(content):
        metadata = {}
        for line in content.splitlines():
            match = _METADATA.search(line.strip())
            if match:
                metadata[match.group(1)] = match.group(2).strip()
        return metadata

    @staticmethod
    def _parse_notes(content):
        notes = []
        for line in content.splitlines():
            match = _NOTE_EVENT.match(line.strip())
            if match:
                notes.append((int(match.group(1)), match.group(2),
                              int(match.group(3)), int(match.group(4))))
        return notes

    def read_chart(self, chart_path, chart_text=None, target_sections=None):
        self.open_chart(chart_path, chart_text=chart_text)
        sections = self.extract_sections2(target_sections=target_sections)
        if "SyncTrack" in sections:
            self.synctrack = self._parse_sync_track(sections["SyncTrack"])
        if "Song" in sections:
            self.song_metadata = self._parse_metadata(sections["Song"])
            # song.ini's `delay` fills in only when the chart carries no Offset of its
            # own. Where both exist and disagree -- 8 songs in the tapping split, one of
            # them Offset +0.870 s against delay +0.001 s -- Offset wins, because that is
            # the field measured to improve alignment against the audio onset envelope
            # (as-written +0.197, ignored +0.140, inverted +0.089, paired t = +3.78).
            if not float(self.song_metadata.get("Offset", 0.0) or 0.0):
                delay = _ini_delay_seconds(chart_path)
                if delay:
                    self.song_metadata["Offset"] = str(delay)
        self.notes = {
            name: self._parse_notes(content)
            for name, content in sections.items()
            if name not in {"Song", "SyncTrack"}
        }
