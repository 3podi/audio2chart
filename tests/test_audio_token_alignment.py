"""Does the audio the model sees line up with the tokens it is asked to predict?

This is the invariant everything else rests on. `_process_window` slices the waveform
at [start_sample, end_sample) and, separately, discretizes the chart's note times with
`start_time = start_sample / sample_rate`. If those two ever disagree the model is
trained to hear one moment and chart another, the loss looks completely normal, and
every generated chart is silently out of sync.

The check is direct: put a click in the audio at the same instant as a note in the
chart, pull a window, and require that the click's position inside the returned chunk
equals the note's grid index times the grid step. Both the window starting at zero and
a window starting mid-song are covered, because an alignment bug that cancels at t=0
is exactly the kind that survives casual testing.

The chart's Offset field is covered too. 14% of the tapping training split carries a
non-zero Offset, 199 of those songs by more than one grid step and one by five seconds,
so the sign of that term decides whether a seventh of the corpus is aligned.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from chart.tokenizer import SimpleTokenizerGuitar
from dataloader.audio_loader import ChunkedWaveformDataset

SAMPLE_RATE = 24000
WINDOW_SECONDS = 15
GRID_MS = 10
RESOLUTION = 192
BPM = 120                     # a quarter note is exactly 0.5 s, so ticks are easy to read
# Spread across 0-25 s so a window starting at 15 s has notes in it too.
NOTE_TICKS = [0, 192, 384, 576, 768, 1920, 3840, 5760, 6000, 6144, 7680, 9600]


def chart_text(offset: float) -> str:
    notes = "\n".join(f"  {tick} = N 0 0" for tick in NOTE_TICKS)
    return (
        "[Song]\n{\n"
        f"  Resolution = {RESOLUTION}\n"
        f'  Offset = "{offset}"\n'
        "}\n"
        "[SyncTrack]\n{\n"
        f"  0 = B {int(BPM * 1000)}\n"
        "}\n"
        "[ExpertSingle]\n{\n"
        f"{notes}\n"
        "}\n"
    )


class AudioTokenAlignmentTest(unittest.TestCase):

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)

    def _build(self, offset: float):
        """A song whose audio clicks exactly where its chart places notes."""
        chart_path = self.folder / "notes.chart"
        chart_path.write_text(chart_text(offset), encoding="utf-8")

        note_seconds = [offset + tick / RESOLUTION * (60.0 / BPM) for tick in NOTE_TICKS]
        total = int(40 * SAMPLE_RATE)
        audio = np.zeros(total, dtype=np.int16)
        for seconds in note_seconds:
            sample = int(round(seconds * SAMPLE_RATE))
            if sample < 0:
                continue        # a negative Offset can push tick 0 before the audio starts
            audio[sample] = 32767                                 # a one-sample click
        raw_path = self.folder / "song.raw"
        raw_path.write_bytes(audio.tobytes())

        item = {
            "chart_path": str(chart_path),
            "difficulty": "ExpertSingle",
            "audio_path": str(raw_path),
            "raw_path": str(raw_path),
            "length_samples": total,
        }
        tokenizer = SimpleTokenizerGuitar(expressive=True)
        dataset = ChunkedWaveformDataset(
            data=[item],
            bos_token=tokenizer.bos_id,
            eos_token=tokenizer.eos_id,
            pad_token=tokenizer.pad_id,
            tokenizer=tokenizer,
            window_seconds=WINDOW_SECONDS,
            sample_rate=SAMPLE_RATE,
            use_predecoded_raw=True,
            is_discrete=True,
            grid_ms=GRID_MS,
            chunk_size=1,
        )
        waveform, _ = dataset._load_audio_file(str(raw_path))
        return dataset, item, waveform, note_seconds

    def _clicks_and_notes(self, offset: float, start_seconds: float):
        dataset, item, waveform, note_seconds = self._build(offset)
        start_sample = int(round(start_seconds * SAMPLE_RATE))
        end_sample = start_sample + int(WINDOW_SECONDS * SAMPLE_RATE)
        result = dataset._process_window(waveform, item, start_sample, end_sample)

        chunk = result["audio"].squeeze(0).numpy()
        click_samples = sorted(int(i) for i in np.nonzero(np.abs(chunk) > 0.5)[0])

        pad = SimpleTokenizerGuitar(expressive=True).pad_id
        note_indices = [index for index, token in enumerate(result["note_values"])
                        if token != pad]
        return click_samples, note_indices, note_seconds, start_seconds

    def _assert_aligned(self, offset: float, start_seconds: float):
        clicks, notes, note_seconds, start = self._clicks_and_notes(offset, start_seconds)
        expected = [t for t in note_seconds
                    if start <= t < start + WINDOW_SECONDS and t >= 0]
        self.assertTrue(expected, "the window under test has to contain some notes")
        self.assertEqual(len(expected), len(clicks), "audio lost a click")
        self.assertEqual(len(expected), len(notes), "chart lost a note")

        # The claim: a note's grid index and its click's sample index describe the same
        # instant inside the window, to within half a grid step -- which is the whole
        # error the grid is allowed to introduce. NOTE_TICKS deliberately includes tick
        # 6000, which falls at exactly 62.5 grid steps and so sits on the bound: the tie
        # rounds down and the note lands a full half-step early. That is the worst case,
        # and it is what the +/- 5 ms in the tick-exactness tests refers to.
        bound = GRID_MS / 2000.0 + 1e-9
        for grid_index, click_sample in zip(notes, clicks):
            note_time = grid_index * (GRID_MS / 1000.0)
            click_time = click_sample / SAMPLE_RATE
            self.assertAlmostEqual(
                note_time, click_time, delta=bound,
                msg=f"grid index {grid_index} ({note_time:.4f}s) against "
                    f"click at {click_time:.4f}s")

    def test_a_window_starting_at_zero_is_aligned(self):
        self._assert_aligned(offset=0.0, start_seconds=0.0)

    def test_a_window_starting_mid_song_is_aligned(self):
        """The case where an off-by-start_time bug stops cancelling."""
        self._assert_aligned(offset=0.0, start_seconds=15.0)

    def test_a_positive_offset_shifts_audio_and_chart_together(self):
        self._assert_aligned(offset=0.25, start_seconds=0.0)

    def test_a_negative_offset_shifts_audio_and_chart_together(self):
        self._assert_aligned(offset=-0.15, start_seconds=15.0)


if __name__ == "__main__":
    unittest.main()
