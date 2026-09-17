"""Musical time on the highway, for segmenting a chart the way a charter reads it.

Robert's names for the white lines, which this module now follows exactly:

  measure line (bar line)  the thickest, brightest line. Start of a measure, the downbeat.
  beat line                thinner, evenly spaced between measure lines. Beats 2, 3, 4 in 4/4.
  half-beat line           fainter still, marking subdivisions -- eighths, sixteenths.

**Patterns are usually one to two BEAT lines long, not one to two measures.** That
correction matters more than it sounds: a 4/4 measure is four beats, so segmenting
patterns by measure groups roughly four times too much material and will happily merge
several distinct figures into one "unit". `group_by_beat` is the right segmentation for
pattern work; `group_by_bar` is the right one for drawing the highway, which is what
view_chart.py wants.

The reference's own grid-step rule -- "a pattern's rhythmic subdivision must strictly
dictate its note count per bar" -- reads naturally at either scale and so did not settle
it.

That matters because the alternative is a heuristic. `chart.patterns.single_note_runs`
splits on chords and on a fixed 250 ms gap, which has no musical meaning: it can join two
patterns that happen to be close together and split one that straddles a rest. Bars are
where the author actually placed the boundaries.

ChartProcessor keeps only the BPM events from [SyncTrack] and discards the TS lines, so
this module reads them from the file directly.
"""

from __future__ import annotations

import re
from pathlib import Path

TS_LINE = re.compile(r"^\s*(\d+)\s*=\s*TS\s+(\d+)(?:\s+(\d+))?\s*$")
DEFAULT_DENOMINATOR_EXPONENT = 2
"""A .chart TS line may omit the denominator; 2 means 2**2 = 4, i.e. x/4 time."""


def parse_time_signatures(chart_path: str | Path) -> list[tuple[int, int, int]]:
    """[(tick, numerator, denominator)], ascending. Always starts at tick 0."""
    events: list[tuple[int, int, int]] = []
    in_sync = False
    with Path(chart_path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("["):
                in_sync = stripped == "[SyncTrack]"
                continue
            if not in_sync:
                continue
            match = TS_LINE.match(line)
            if match:
                tick, numerator, exponent = match.groups()
                power = int(exponent) if exponent is not None else DEFAULT_DENOMINATOR_EXPONENT
                events.append((int(tick), int(numerator), 2 ** power))
    events.sort()
    if not events or events[0][0] != 0:
        events.insert(0, (0, 4, 4))          # Clone Hero's default is 4/4
    return events


def bar_length_ticks(numerator: int, denominator: int, resolution: int) -> int:
    """Ticks in one bar. `resolution` is ticks per quarter note."""
    return int(resolution * 4 * numerator / denominator)


def bar_boundaries(chart_path: str | Path, resolution: int, last_tick: int) -> list[int]:
    """Tick of every bar line from 0 through `last_tick`.

    Time signature changes restart the bar grid at the tick where they occur, which is what
    Moonscraper draws and therefore what the charter was looking at.
    """
    events = parse_time_signatures(chart_path)
    boundaries: list[int] = []
    for index, (tick, numerator, denominator) in enumerate(events):
        end = events[index + 1][0] if index + 1 < len(events) else last_tick + 1
        length = bar_length_ticks(numerator, denominator, resolution)
        if length <= 0:
            continue
        position = tick
        while position <= min(end - 1, last_tick):
            boundaries.append(position)
            position += length
    return boundaries


def beat_length_ticks(denominator: int, resolution: int) -> int:
    """Ticks in one beat. `resolution` is ticks per quarter note.

    The time signature's denominator says what note value gets the beat: 4 means a
    quarter (one resolution), 8 means an eighth (half of one). The numerator says how
    many of them fill a measure and is irrelevant here.
    """
    return int(resolution * 4 / denominator)


def beat_boundaries(chart_path: str | Path, resolution: int, last_tick: int) -> list[int]:
    """Tick of every beat line from 0 through `last_tick`, measure lines included.

    This is the segmentation to use for pattern work. Patterns run one to two beats, so
    measure-level grouping is about four times too coarse and merges separate figures.
    """
    events = parse_time_signatures(chart_path)
    boundaries: list[int] = []
    for index, (tick, _numerator, denominator) in enumerate(events):
        end = events[index + 1][0] if index + 1 < len(events) else last_tick + 1
        length = beat_length_ticks(denominator, resolution)
        if length <= 0:
            continue
        position = tick
        while position <= min(end - 1, last_tick):
            boundaries.append(position)
            position += length
    return boundaries


def group_by_bar(encoded, boundaries: list[int]) -> list[list]:
    """Split `tokenizer.encode(...)` output into one list per boundary interval.

    Named for bars but agnostic to what the boundaries are: hand it `beat_boundaries`
    and it groups by beat. Empty intervals are kept as empty lists so an index is an
    interval number.
    """
    if not boundaries:
        return [list(encoded)]
    bars: list[list] = [[] for _ in boundaries]
    index = 0
    for event in encoded:
        tick = event[0]
        while index + 1 < len(boundaries) and tick >= boundaries[index + 1]:
            index += 1
        while index > 0 and tick < boundaries[index]:
            index -= 1
        bars[index].append(event)
    return bars


def bars_for_chart(chart_path: str | Path, encoded, resolution: int) -> list[list]:
    """Convenience: boundaries from the file, then group."""
    if not encoded:
        return []
    last_tick = max(event[0] for event in encoded)
    return group_by_bar(encoded, bar_boundaries(chart_path, resolution, last_tick))


def beats_for_chart(chart_path: str | Path, encoded, resolution: int) -> list[list]:
    """One list per beat. The unit patterns actually live in."""
    if not encoded:
        return []
    last_tick = max(event[0] for event in encoded)
    return group_by_bar(encoded, beat_boundaries(chart_path, resolution, last_tick))


MAX_PERIOD = 6
"""Longest repeating unit to test for continuity across a bar line."""


def _continues(deltas: list[int], cut: int, window: int = 8) -> bool:
    """Does the same repeating motion carry on across position `cut`?

    Looks for a period that holds on both sides. Tremolo is period 1 and always continues;
    a zig that links into a different zig changes shape and does not.
    """
    if cut <= 0 or cut >= len(deltas):
        return False
    before = deltas[max(0, cut - window):cut]
    after = deltas[cut:cut + window]
    if not before or not after:
        return False
    for period in range(1, MAX_PERIOD + 1):
        if len(before) < period or len(after) < period:
            continue
        unit = before[-period:]
        if all(before[-(i + 1)] == unit[-(i % period) - 1] for i in range(len(before))) and \
           all(after[i] == unit[i % period] for i in range(min(len(after), period * 2))):
            return True
    return False


def soft_bar_cuts(ticks: list[int], deltas: list[int], boundaries: list[int]) -> list[int]:
    """Which bar lines inside a run are genuine boundaries.

    A bar line is a *candidate* boundary -- the place the author might have ended a figure.
    It becomes a real one only when the motion changes there. Cutting unconditionally
    shreds anything that legitimately runs through the line, and the corpus shows exactly
    that: strict per-bar segmentation turns one continuous tremolo into a separate "pattern"
    at every bar-multiple length.

    MEASURED: this does not work as a middle ground. Over 12,529 bar-line crossings inside
    runs, the continuity test fires on only 5.2% (6.6% at window 4, 6.3% at window 6), so it
    cuts 95% of lines and behaves like strict per-bar segmentation -- 47,669 segments against
    strict bars' 47,710, with slightly worse catalogue coverage (66.3% vs 67.3%) and *more*
    tremolo fragmentation. The requirement that a whole window be strictly periodic is too
    strong for real playing.

    Kept because the negative result is worth not rediscovering. For pattern identity, use
    gap-based runs; for rhythm, use the bar grid; and treat bar alignment as a per-pattern
    attribute rather than a segmentation rule -- known patterns start on a bar line 7.4% of
    the time against a 7.3% baseline for arbitrary positions, so it carries no signal in
    aggregate, but it ranges from 7% to 19% across individual patterns.

    Returns indices into `deltas` where a cut is justified.
    """
    if len(ticks) < 2:
        return []
    inside = set()
    for boundary in boundaries:
        if ticks[0] < boundary <= ticks[-1]:
            # The delta index whose span crosses this bar line.
            for index in range(len(deltas)):
                if ticks[index] < boundary <= ticks[index + 1]:
                    inside.add(index + 1)
                    break
    return sorted(index for index in inside if not _continues(deltas, index))
