"""Does a chart put notes where this song's notes go?

Everything in `chart.metrics` is computable from the chart file alone, so a model that
ignored the audio entirely and emitted a plausible generic pattern would pass all of it.
This module supplies the timing check that cannot be faked that way.

  precision -- of the notes placed, how many land on a real event
  recall    -- of the real events, how many got charted

Neither alone is enough: one perfectly-placed note has precision 1.0, and a note in every
grid slot has recall ~1.0.

MEASURED NEGATIVE RESULT -- do not use audio onsets as the reference.

The obvious reference is onsets detected in the waveform. It was tried and it does not
work. Over 10 songs, comparing a real chart against librosa onsets versus randomly
scattered notes against the same onsets:

    tol      real   scrambled     gap    real wins
    10 ms   0.110     0.098     0.012      3/10
    20 ms   0.194     0.189     0.006      5/10
    50 ms   0.527     0.428     0.099      7/10
    80 ms   0.672     0.593     0.079      9/10

At 20 ms a real chart beats random noise on half the songs -- chance. Tightening the
tolerance makes it worse, so this is not a tuning problem: full-mix onset detection fires
on drums and vocals, while the chart follows the guitar line, and at 7-10 NPS a generous
tolerance covers most of the timeline anyway. Making it work would need source separation
to isolate a guitar stem first.

USE THE REFERENCE CHART INSTEAD. Every song in the corpus already has a human chart, and
`agreement_with_reference` compares against that. Same arithmetic, a reference that
actually corresponds to what the model is asked to produce.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

DEFAULT_TOLERANCE = 0.050
"""Seconds. Wider than the 20 ms grid because onset detection is itself imprecise."""


@dataclass
class Alignment:
    precision: float
    recall: float
    f1: float
    notes: int
    onsets: int
    tolerance: float
    median_offset: float
    """Signed median seconds from note to nearest onset. A consistent non-zero value means
    the chart is systematically early or late rather than randomly misaligned."""


def load_raw_audio(path: str, sample_rate: int = 24000) -> np.ndarray:
    """Read the headerless 24 kHz mono s16 PCM the training pipeline already produced.

    Much cheaper than decoding the original ogg, and it is the exact signal the model saw.
    """
    data = np.fromfile(path, dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def detect_onsets(waveform: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    """Onset times in seconds."""
    import librosa

    if waveform.size == 0:
        return np.zeros(0)
    return librosa.onset.onset_detect(
        y=waveform, sr=sample_rate, units="time", backtrack=False
    )


def _nearest_offsets(sources: np.ndarray, targets: list[float]) -> np.ndarray:
    """Signed distance from each source to the nearest target."""
    if not targets or sources.size == 0:
        return np.zeros(0)
    offsets = np.empty(sources.size)
    for index, value in enumerate(sources):
        position = bisect.bisect_left(targets, value)
        candidates = []
        if position < len(targets):
            candidates.append(targets[position] - value)
        if position > 0:
            candidates.append(targets[position - 1] - value)
        offsets[index] = min(candidates, key=abs)
    return offsets


def alignment(
    note_times, onset_times, tolerance: float = DEFAULT_TOLERANCE
) -> Alignment:
    """Compare note times against onset times, both in seconds."""
    notes = np.asarray(sorted(note_times), dtype=float)
    onsets = sorted(float(value) for value in onset_times)

    if notes.size == 0 or not onsets:
        return Alignment(0.0, 0.0, 0.0, int(notes.size), len(onsets), tolerance, 0.0)

    note_offsets = _nearest_offsets(notes, onsets)
    matched_notes = int(np.sum(np.abs(note_offsets) <= tolerance))
    precision = matched_notes / notes.size

    onset_offsets = _nearest_offsets(np.asarray(onsets), list(notes))
    matched_onsets = int(np.sum(np.abs(onset_offsets) <= tolerance))
    recall = matched_onsets / len(onsets)

    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    median_offset = float(np.median(note_offsets)) if note_offsets.size else 0.0
    return Alignment(
        precision=precision,
        recall=recall,
        f1=f1,
        notes=int(notes.size),
        onsets=len(onsets),
        tolerance=tolerance,
        median_offset=median_offset,
    )


def chart_note_times(chart_path: str, section: str, processor, tokenizer) -> list[float]:
    """Note onset times in seconds for one chart section."""
    processor.read_chart(chart_path, target_sections=section)
    resolution = int(processor.song_metadata["Resolution"])
    offset = float(processor.song_metadata["Offset"])
    encoded = tokenizer.encode(processor.notes[section], resolution=resolution)
    timed = tokenizer.format_seconds(encoded, processor.synctrack, resolution, offset)
    return [item[0] for item in timed]


def agreement_with_reference(
    generated_times, reference_times, tolerance: float = DEFAULT_TOLERANCE
) -> Alignment:
    """How well a generated chart's note timing matches the human chart for the same song.

    This is the recommended timing metric. The reference is what the model is actually
    being asked to reproduce, unlike audio onsets -- see the module docstring for the
    measurement showing why those do not work.

    Interpretation is asymmetric and useful:
      low precision, high recall -- overcharting: it hit the real notes and added more
      high precision, low recall -- undercharting: what it placed was right, but sparse
    """
    return alignment(generated_times, reference_times, tolerance)


def alignment_against_onsets(entry: dict, processor, tokenizer,
                             tolerance: float = DEFAULT_TOLERANCE) -> Alignment:
    """Kept for reproducing the negative result in the module docstring. Not a quality metric."""
    note_times = chart_note_times(entry["chart_path"], entry["difficulty"], processor, tokenizer)
    onsets = detect_onsets(load_raw_audio(entry["raw_path"]))
    return alignment(note_times, onsets, tolerance)
