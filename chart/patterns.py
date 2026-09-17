"""Find repeating fret-motion patterns in charts, without being told what to look for.

Robert's examples define the shape of the problem. Frets run low to high G R Y B O, so
playing Y R G is a descending triplet, and `O B R B O B R B O B R` is a "trip zig" -- a
triplet zigzag iterated about three times. Two properties follow from that:

  * A pattern is a shape, not a place. Patterns are transposable, so the same figure
    starting on a different fret is the same pattern. The transposition-invariant
    representation is therefore the sequence of *differences* between consecutive frets:
    Y R G and B Y R are both (-1, -1).

  * A pattern repeats. `O B R B` iterated is one named thing, not three unrelated ones.

So instead of hardcoding a catalogue -- which would mean guessing at definitions that are
still open questions in docs/section-conditioning.md -- this module finds *repeating delta
cycles* and reports them. Run it over the corpus and the common shapes rank themselves;
naming them is then a small labelling job over real, frequent structures rather than an
attempt to enumerate charting vocabulary from memory.

Beat space, not seconds: charters think in beats and bars (Robert, Q10), and the same shape
at 100 and 220 BPM is the same pattern at very different difficulty. Timing is carried
alongside so difficulty can be judged separately from identity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

OPEN_LANE = 7
MAX_CYCLE = 6
"""Longest repeating unit to look for. Trip zigs are 3-4; quads are 4-5."""
MIN_REPEATS = 2
"""A shape has to happen at least twice to be a repeating pattern."""


@dataclass(frozen=True)
class PatternInstance:
    signature: tuple[int, ...]
    """Transposition-invariant: consecutive fret differences of the repeating unit."""
    repeats: int
    start_time: float
    end_time: float
    start_fret: int
    notes: int

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def notes_per_second(self) -> float:
        return self.notes / self.duration if self.duration > 0 else 0.0


def single_note_runs(timed, tokenizer, max_gap: float = 0.25):
    """Split a chart into runs of consecutive single notes.

    Chords break a run: a fret *motion* pattern is about where one finger goes next, and a
    chord is a different kind of event. Sustains and flags do not break it -- a tapped
    trip zig is still a trip zig.
    """
    runs, current = [], []
    previous_time = None
    for entry in timed:
        time_value, token = entry[0], entry[1]
        chord_index = tokenizer.split(token)[0]
        lanes = [lane for lane in tokenizer.reverse_chord[chord_index] if lane != OPEN_LANE]
        gapped = previous_time is not None and time_value - previous_time > max_gap
        if len(lanes) != 1 or gapped:
            if len(current) >= 2:
                runs.append(current)
            current = []
            if len(lanes) == 1 and gapped:
                current = [(time_value, lanes[0])]
        else:
            current.append((time_value, lanes[0]))
        previous_time = time_value
    if len(current) >= 2:
        runs.append(current)
    return runs


def _cycle_length(deltas: list[int], period: int) -> int:
    """How many times the first `period` deltas repeat consecutively from the start."""
    if period > len(deltas):
        return 0
    unit = deltas[:period]
    repeats = 1
    index = period
    while index + period <= len(deltas) and deltas[index:index + period] == unit:
        repeats += 1
        index += period
    return repeats


def find_in_run(run, min_repeats: int = MIN_REPEATS, max_cycle: int = MAX_CYCLE):
    """Repeating delta cycles inside one run of single notes."""
    frets = [fret for _, fret in run]
    times = [time_value for time_value, _ in run]
    deltas = [b - a for a, b in zip(frets, frets[1:])]

    found, start = [], 0
    while start < len(deltas):
        best_period = best_repeats = 0
        for period in range(1, max_cycle + 1):
            repeats = _cycle_length(deltas[start:], period)
            # Prefer more notes covered; break ties toward the shorter unit, so an
            # alternation is reported as a 2-cycle rather than a 4-cycle of two of them.
            if repeats >= min_repeats and repeats * period > best_repeats * best_period:
                best_period, best_repeats = period, repeats
        if best_period == 0:
            start += 1
            continue
        covered = best_period * best_repeats
        found.append(PatternInstance(
            signature=tuple(deltas[start:start + best_period]),
            repeats=best_repeats,
            start_time=times[start],
            end_time=times[min(start + covered, len(times) - 1)],
            start_fret=frets[start],
            notes=covered + 1,
        ))
        start += covered
    return found


def find_patterns(timed, tokenizer, max_gap: float = 0.25) -> list[PatternInstance]:
    """Every repeating fret-motion cycle in a chart."""
    instances: list[PatternInstance] = []
    for run in single_note_runs(timed, tokenizer, max_gap):
        instances.extend(find_in_run(run))
    return instances


def signature_histogram(instances) -> Counter:
    """How often each shape occurs. This is what ranks the corpus's real vocabulary."""
    return Counter(instance.signature for instance in instances)


def describe_signature(signature: tuple[int, ...]) -> str:
    """A readable gloss. Deliberately descriptive, not a claim about charting names."""
    if not signature:
        return "empty"
    if all(delta == 0 for delta in signature):
        # 11% of instances in a 150-chart sample, averaging 5.4 repeats -- the same fret
        # struck repeatedly. Calling this "ascending by 0" would be nonsense.
        return "repeated note (tremolo)"
    if all(delta == signature[0] for delta in signature):
        direction = "ascending" if signature[0] > 0 else "descending"
        return f"{direction} by {abs(signature[0])}"
    if len(signature) == 2 and signature[0] == -signature[1]:
        return f"alternation of {abs(signature[0])}"
    if sum(signature) == 0:
        return f"closed cycle over {len(signature)} notes"
    return f"{len(signature)}-note cycle, net {sum(signature):+d}"
