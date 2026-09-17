"""Which instrument is this passage of the chart following?

Robert's description of modern overcharts: they follow the vocals for crucial parts and
the instrumental where there are no vocals, and they do not try to play both at once.
That is a rule about *which line* the chart tracks, and it changes within a song.

Hand-coding it would mean guessing. Measured instead, it becomes a label: for each
window, score the chart's note onsets against each stem's onset envelope and report
which stem wins. Across a corpus that yields, per section, "this follows vocals" or
"this follows guitar" -- exactly the annotation a control track would need, derived
from what charters actually did rather than from a rule someone wrote down.

The scoring is deliberately the same machinery as chart.responsiveness, so the same
caution applies: a single window is noisy, and only the aggregate means anything. The
difference is that here the comparison is *between stems on the same audio*, which
cancels most of what makes the absolute number unreliable -- a busy passage is busy in
every stem.
"""

from __future__ import annotations

import numpy as np

from chart.responsiveness import note_density, onset_envelope, responsiveness

WINDOW_SECONDS = 10.0
"""Long enough for a phrase, short enough that a chart can switch lines within a song."""

MARGIN = 0.05
"""How far ahead a stem must score before the window is called for it rather than tied."""


def windows(duration: float, window: float = WINDOW_SECONDS):
    """(start, end) pairs covering the song."""
    count = max(1, int(duration // window))
    return [(i * window, min((i + 1) * window, duration)) for i in range(count)]


def follow_scores(note_times, stems: dict[str, np.ndarray], sample_rate: int,
                  duration: float, window: float = WINDOW_SECONDS
                  ) -> list[dict[str, float]]:
    """Per window, the chart's density correlation against each stem."""
    envelopes = {name: onset_envelope(audio, sample_rate)
                 for name, audio in stems.items()}
    density = note_density(list(note_times), duration, bin_seconds=1.0)

    out = []
    for start, end in windows(duration, window):
        lo, hi = int(start), int(end)
        slice_density = density[lo:hi]
        scores = {}
        for name, envelope in envelopes.items():
            scores[name] = responsiveness(slice_density, envelope[lo:hi])
        out.append(scores)
    return out


def label_windows(scores: list[dict[str, float]], margin: float = MARGIN) -> list[str]:
    """Name the stem each window follows, or "none" when nothing leads clearly.

    A window where two stems score alike is not evidence that the chart follows both --
    it is evidence the measurement cannot tell, most often because the stems are
    correlated there. Calling it "none" keeps that out of the training signal.
    """
    labels = []
    for window in scores:
        if not window:
            labels.append("none")
            continue
        ranked = sorted(window.items(), key=lambda kv: -kv[1])
        best, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
        labels.append(best if best_score > 0 and best_score - runner_up >= margin
                      else "none")
    return labels


def summarise(labels: list[str]) -> dict[str, float]:
    """Share of windows following each stem, for one chart."""
    if not labels:
        return {}
    total = len(labels)
    return {name: labels.count(name) / total for name in set(labels)}


def switches(labels: list[str]) -> int:
    """How often the chart changes which line it follows, ignoring unclear windows.

    Robert's overchart rule predicts this is well above zero: vocals during the verse,
    instrumental where the vocals stop. A chart that never switches is following one
    line throughout, which is a different style rather than a failed measurement.
    """
    named = [name for name in labels if name != "none"]
    return sum(1 for a, b in zip(named, named[1:]) if a != b)
