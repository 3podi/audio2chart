"""Difficulty features for grading an Expert chart on the community tier scale.

Robert's correction: Easy/Medium/Hard/Expert is the difficulty selector, not the scale
that matters. Expert charts span an enormous range -- Through The Fire & Flames sits in
single digits while Schmoo's World is around 24, and both are Expert. A model that can
be asked for "roughly a grade 16" needs tier as a number, and tier is a property of the
chart, so it can be measured.

The features here are chosen to separate charts that a density average cannot. The
difficulty ladder across Easy..Expert showed notes per second rising only 1.8x while
mean run length quadrupled, so sustained load and hand movement matter more than
average rate:

*Peak sustained density* -- the busiest ten seconds, since what breaks a run is the
hardest passage, not the mean.

*Endurance* -- the longest unbroken stretch above ten notes a second. Schmoo's World
and TTFAF differ here far more than in average density.

*Hand movement* -- mean fret jump and the share of jumps of three frets or more, which
is what forces a hand reposition rather than a finger roll.

*Technique mix* -- tap and chord rates, and the share of pattern instances that are
graded Expert markers.

No model is fitted here; this produces the inputs. `fit_tier.py` fits against the 801
charts whose setlist folders name a tier.
"""

from __future__ import annotations

import statistics

PEAK_WINDOW = 10.0
"""Seconds. Long enough to be sustained, short enough to catch one hard passage."""

FAST_SECONDS = 0.1
"""A gap this short or shorter counts as part of a fast stretch (>= 10 nps)."""

WIDE_JUMP = 3
"""Frets. At this distance the hand repositions rather than rolling fingers."""


def peak_density(times: list[float], window: float = PEAK_WINDOW) -> float:
    """Highest note rate sustained over any `window` seconds."""
    if len(times) < 2:
        return 0.0
    ordered = sorted(times)
    best, start = 0.0, 0
    for index, moment in enumerate(ordered):
        while ordered[start] < moment - window:
            start += 1
        best = max(best, (index - start + 1) / window)
    return best


def longest_fast_stretch(times: list[float], gap: float = FAST_SECONDS) -> float:
    """Longest unbroken run of notes spaced `gap` apart or closer, in seconds."""
    ordered = sorted(set(times))
    best, start = 0.0, None
    for index in range(len(ordered) - 1):
        if ordered[index + 1] - ordered[index] <= gap:
            if start is None:
                start = ordered[index]
        elif start is not None:
            best = max(best, ordered[index] - start)
            start = None
    if start is not None:
        best = max(best, ordered[-1] - start)
    return best


def hand_movement(runs: list[list[int]]) -> tuple[float, float]:
    """(mean fret jump, share of jumps >= WIDE_JUMP) over single-note runs."""
    jumps = [abs(b - a) for run in runs for a, b in zip(run, run[1:])]
    if not jumps:
        return 0.0, 0.0
    wide = sum(1 for jump in jumps if jump >= WIDE_JUMP) / len(jumps)
    return statistics.mean(jumps), wide


HAND_SPAN = 3
"""Frets reachable without moving the hand: index anchored, three fingers above.

So G R Y B is one hand position (span 3) and adding O forces a shift.
"""

ANCHOR_WINDOW = 16
"""Notes. How far apart the lowest fret may recur and still be worth holding."""


def anchorable(frets: list[int], window: int = ANCHOR_WINDOW,
               span: int = HAND_SPAN) -> bool:
    """Can the index finger hold the lowest fret through this passage?

    Robert: anchoring means holding the lowest note down and not letting it up until
    it is safe to. It is a technique applied across a passage, not a shape -- which is
    why the catalogue's twenty "anchor X-Y-X" entries are misnamed, since they describe
    three-note figures rather than anything about holding a fret.

    Unlike the slide-versus-anchor distinction, which is invisible in chart data
    because both produce identical notes, this is measurable: the lowest fret has to
    recur often enough to be worth keeping down, and the passage has to fit under one
    hand.
    """
    if len(frets) < 2:
        return False
    if max(frets) - min(frets) > span:
        return False
    lowest = min(frets)
    positions = [i for i, fret in enumerate(frets) if fret == lowest]
    if len(positions) < 2:
        return False
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    return max(gaps) <= window


def anchor_share(runs: list[list[int]], window: int = ANCHOR_WINDOW) -> float:
    """Share of notes sitting inside an anchorable passage.

    A better difficulty signal than raw fret movement, which counts distance without
    knowing whether the hand actually had to leave its position.
    """
    total = held = 0
    for run in runs:
        total += len(run)
        if anchorable(run, window):
            held += len(run)
    return held / total if total else 0.0


def features(profile, times: list[float], runs: list[list[int]],
             marker_share: float = 0.0) -> dict[str, float]:
    """Everything the tier model reads, from one chart."""
    mean_jump, wide_jumps = hand_movement(runs)
    return {
        "anchor_share": anchor_share(runs),
        "nps": profile.nps,
        "peak10": peak_density(times),
        "longest_fast": longest_fast_stretch(times),
        "pct_chord": 1.0 - profile.chord_hist.get(1, 0.0),
        "pct_tap": profile.pct_tap,
        "pct_sustain": profile.pct_sustain,
        "mean_jump": mean_jump,
        "wide_jumps": wide_jumps,
        "marker_share": marker_share,
        "nps_variation": profile.nps_variation,
    }


FEATURE_NAMES = ("nps", "peak10", "longest_fast", "pct_chord", "pct_tap",
                 "pct_sustain", "mean_jump", "wide_jumps", "marker_share",
                 "nps_variation", "anchor_share")
