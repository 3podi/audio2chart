"""Search each musical measure at several scales, and see what repeats.

Robert's method: "Try searching (from the start of the song to the end) each musical
measure (musical bar) for patterns, you find a bunch of patterns within one musical
measure, then you scan 3/4 beatlines in the same measure, then 2/4 in the same measure,
then 1/4 in the same measure and this way you can see if a pattern is repeating, or
transitioning into another pattern."

This is better than what discover_bar_patterns.py does, in two ways that matter. Windows
are aligned to the measure grid the charter was looking at rather than slid along the
note list, and the same span is read at several scales at once, so a figure and the
sub-figures inside it are both visible. Reading the same scale across consecutive
measures is what turns pattern detection into repetition and transition detection, which
is the part nothing here could do before.

Two amendments to his scheme, both measured rather than assumed:

SCALES GO BELOW ONE BEAT. His hierarchy stops at a quarter of a measure -- one beat in
4/4 -- but measure_pattern_spans.py puts the median catalogued pattern at 0.38 beats,
with every pattern's typical occurrence inside 1.1 beats. Stopping at one beat stops
above where most patterns live, so the ladder continues to a half and a quarter beat.

WINDOWS ARE ALSO READ AT A HALF-STEP OFFSET. Because patterns are short, many straddle a
boundary. A strictly aligned search misses those by construction -- the same failure
discover_bar_patterns.py has today, where a one-to-two-beat window cannot see the six
catalogued patterns that run past two beats at p90. Every scale is therefore read twice,
once on the grid and once offset by half its width, and the report says how much the
offset track actually found so the amendment can be judged rather than believed.
"""

from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chart.bars import bar_boundaries, parse_time_signatures
from chart.catalogue import CATALOGUE, chord_signature, positions
from chart.chart_processor import ChartProcessor
from chart.tokenizer import SimpleTokenizerGuitar

# Fractions of a measure. The first four are Robert's ladder -- the whole measure, then
# 3/4, 2/4 and 1/4 of it, which in 4/4 is four, three, two and one beat. The last two are
# the amendment: half a beat and a quarter beat, because that is where the median pattern
# actually lives.
SCALES: tuple[tuple[str, float], ...] = (
    ("measure", 1.0),
    ("3/4 measure", 0.75),
    ("half measure", 0.5),
    ("beat", 0.25),
    ("half beat", 0.125),
    ("quarter beat", 0.0625),
)

MIN_POSITIONS = 3
"""Fewest note positions that can be called a figure.

Two positions is one interval, which every chart is full of and which carries no shape.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path,
                        default=Path(r"G:\a2c_data\splits_tapping\train.json"))
    parser.add_argument("--charts", type=int, default=250)
    parser.add_argument("--top", type=int, default=12,
                        help="How many figures and transitions to list per scale.")
    parser.add_argument("--min-count", type=int, default=20,
                        help="Ignore figures seen fewer times than this.")
    return parser.parse_args()


def catalogue_names() -> dict[tuple, str]:
    """Signature -> catalogued name, using the same identity the search produces."""
    names: dict[tuple, str] = {}
    for name, text in CATALOGUE.items():
        # The catalogue writes lead-in notes before a '|'; the body is what repeats.
        # split_transition returns parsed frets rather than text, so the split is done
        # here to keep positions() as the single parser.
        body = text.split("|")[-1]
        signature = chord_signature(positions(body))
        if len(signature) >= MIN_POSITIONS:
            names.setdefault(signature, name)
    return names


def chart_positions(item: dict, tokenizer: SimpleTokenizerGuitar):
    """(ticks, fret tuples, resolution) for one chart, or None if unreadable."""
    processor = ChartProcessor(["Expert"], ["Single"])
    processor.read_chart(item["chart_path"], target_sections="ExpertSingle")
    notes = processor.notes.get("ExpertSingle")
    if not notes:
        return None
    resolution = int(processor.song_metadata["Resolution"])
    encoded = tokenizer.encode(notes, resolution=resolution)
    if not encoded:
        return None

    ticks, frets = [], []
    for event in encoded:
        lanes = tokenizer.reverse_chord.get(
            event[1] // (tokenizer.N_FLAG * tokenizer.N_SUSTAIN), ())
        playable = tuple(lane for lane in lanes if lane <= 4)
        if not playable:
            continue
        ticks.append(event[0])
        frets.append(playable)
    return (ticks, frets, resolution) if len(ticks) >= MIN_POSITIONS else None


def measure_length(chart_path: str, resolution: int) -> int:
    """Ticks in a measure, from the chart's first time signature."""
    events = parse_time_signatures(chart_path)
    numerator, denominator = (events[0][1], events[0][2]) if events else (4, 4)
    return int(resolution * 4 * numerator / denominator)


def figures_in(ticks, frets, start: int, stop: int, width: int, offset: int):
    """Signatures of every window of `width` between start and stop, at `offset`.

    Yields (window_start, signature). Windows are half-open so a note on a boundary
    belongs to the window that begins there, the way a charter reads a bar line.
    """
    # bisect rather than a scan per window. The obvious version walks every tick for
    # every window, which is windows x notes -- on 150 charts at six scales and two
    # tracks that did not finish in ten minutes.
    position = start + offset
    while position < stop:
        end = position + width
        low = bisect_left(ticks, position)
        high = bisect_left(ticks, end)
        if high - low >= MIN_POSITIONS:
            yield position, chord_signature(frets[low:high])
        position = end


def main() -> None:
    args = parse_args()
    entries = json.loads(args.split.read_text(encoding="utf-8"))
    tokenizer = SimpleTokenizerGuitar(expressive=True)
    known = catalogue_names()

    # Per scale: how often each figure is seen, on-grid against offset, how often a
    # figure is immediately followed by itself, and what it turns into when it is not.
    seen: dict[str, Counter] = defaultdict(Counter)
    found_by_track: dict[str, Counter] = defaultdict(Counter)
    repeats: dict[str, Counter] = defaultdict(Counter)
    transitions: dict[str, Counter] = defaultdict(Counter)
    offset_only: dict[str, set] = defaultdict(set)
    aligned_seen: dict[str, set] = defaultdict(set)

    scanned = 0
    for item in entries:
        if scanned >= args.charts:
            break
        try:
            read = chart_positions(item, tokenizer)
            if read is None:
                continue
            ticks, frets, resolution = read
            bars = bar_boundaries(item["chart_path"], resolution, ticks[-1])
            span = measure_length(item["chart_path"], resolution)
        except Exception:
            continue
        if not bars or span <= 0:
            continue
        scanned += 1

        for label, fraction in SCALES:
            width = max(1, int(round(span * fraction)))
            for track, offset in (("aligned", 0), ("offset", width // 2)):
                # One sequence per track across the whole song, so "what follows what"
                # is asked of windows that are genuinely adjacent.
                sequence = []
                for index, bar in enumerate(bars):
                    stop = bars[index + 1] if index + 1 < len(bars) else ticks[-1] + 1
                    for _at, signature in figures_in(
                            ticks, frets, bar, stop, width, offset):
                        sequence.append(signature)
                        seen[label][signature] += 1
                        found_by_track[label][track] += 1
                        if track == "aligned":
                            aligned_seen[label].add(signature)
                        else:
                            offset_only[label].add(signature)
                for first, second in zip(sequence, sequence[1:]):
                    if first == second:
                        repeats[label][first] += 1
                    else:
                        transitions[label][(first, second)] += 1

    print(f"{scanned} charts scanned, measure-aligned, {len(SCALES)} scales x 2 tracks\n")

    for label, fraction in SCALES:
        counts = seen[label]
        if not counts:
            continue
        solid = [(sig, n) for sig, n in counts.most_common() if n >= args.min_count]
        total = sum(counts.values())
        repeated = sum(repeats[label].values())
        pairs = repeated + sum(transitions[label].values())
        print(f"=== {label}  ({fraction:g} of a measure) ===")
        print(f"  {total:,} figures, {len(counts):,} distinct, "
              f"{len(solid):,} seen {args.min_count}+ times")
        if pairs:
            print(f"  immediately repeats: {repeated:,}/{pairs:,} "
                  f"({100 * repeated / pairs:.1f}% of adjacent pairs)")

        for signature, count in solid[:args.top]:
            name = known.get(signature, "")
            shapes = "".join("C" if len(shape) > 1 else "." for shape, _ in signature)
            moves = ",".join(str(move) for _shape, move in signature[1:])
            label_text = f"{name}" if name else f"[{shapes}] moves {moves}"
            print(f"    {count:>7,}  {label_text[:58]}")

        common = [(pair, n) for pair, n in transitions[label].most_common(3)
                  if n >= args.min_count]
        for (first, second), count in common:
            one = known.get(first, f"[{len(first)} pos]")
            two = known.get(second, f"[{len(second)} pos]")
            print(f"    {count:>7,}  {one[:26]} -> {two[:26]}")
        print()

    print("Does reading at a half-step offset find anything the aligned grid misses?")
    print(f"{'scale':<16}{'aligned':>10}{'offset':>10}{'offset-only shapes':>20}")
    for label, _fraction in SCALES:
        if not found_by_track[label]:
            continue
        extra = offset_only[label] - aligned_seen[label]
        print(f"  {label:<14}{found_by_track[label]['aligned']:>10,}"
              f"{found_by_track[label]['offset']:>10,}{len(extra):>20,}")
    print("\n  Shapes in the last column exist only between boundaries, so a strictly")
    print("  aligned search cannot see them at all.")


if __name__ == "__main__":
    main()
