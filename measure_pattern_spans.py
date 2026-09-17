"""How many beat lines does each pattern actually span?

Robert: "I said most patterns live in 1-2 beat lines but that isn't always the case, and
having a way to exactly know how many beatlines a single pattern is contained in would be
significantly better than guess work."

He is right, and the guesswork has already cost something. `discover_bar_patterns.py`
searches one- and two-beat windows because of the original estimate, so any figure longer
than two beats is invisible to it by construction -- the tool cannot find what it does not
look for, and it would report the estimate back as if it were a finding.

This measures it instead. For every catalogue entry, find its occurrences in real charts
and record how many beats each occurrence spans, using the chart's own tempo map and time
signature so a beat is whatever that song says it is.

The output is per pattern, because "most patterns" is exactly the kind of aggregate that
hides the cases worth knowing about: a trill and a fourteen-note cake have no business
sharing a window size.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chart.bars import beat_boundaries, parse_time_signatures
from chart.catalogue import CATALOGUE, entry_signature, is_chorded_entry
from chart.chart_processor import ChartProcessor
from chart.tokenizer import SimpleTokenizerGuitar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path,
                        default=Path(r"G:\a2c_data\splits_tapping\train.json"))
    parser.add_argument("--charts", type=int, default=250)
    parser.add_argument("--min-occurrences", type=int, default=20,
                        help="Patterns seen fewer times than this are reported but not "
                             "trusted; a span from three sightings is not a distribution.")
    return parser.parse_args()


def beat_length(chart_path: str, resolution: int) -> float:
    """Ticks per beat, from the chart's own time signature denominator."""
    events = parse_time_signatures(chart_path)
    denominator = events[0][2] if events else 4
    return resolution * 4 / denominator


def main() -> None:
    args = parse_args()
    entries = json.loads(args.split.read_text(encoding="utf-8"))
    tokenizer = SimpleTokenizerGuitar(expressive=True)

    # Single-note signatures only. A chorded signature has one entry per position rather
    # than per transition, so mixing the two would compare different things.
    wanted = {}
    for name, text in CATALOGUE.items():
        if is_chorded_entry(text):
            continue
        signature = entry_signature(text)
        if len(signature) >= 2:
            wanted.setdefault(signature, []).append(name)

    spans: dict[tuple, list[float]] = defaultdict(list)
    scanned = 0

    for item in entries:
        if scanned >= args.charts:
            break
        try:
            processor = ChartProcessor(["Expert"], ["Single"])
            processor.read_chart(item["chart_path"], target_sections="ExpertSingle")
            notes = processor.notes.get("ExpertSingle")
            if not notes:
                continue
            resolution = int(processor.song_metadata["Resolution"])
            encoded = tokenizer.encode(notes, resolution=resolution)
            per_beat = beat_length(item["chart_path"], resolution)
        except Exception:
            continue
        if not encoded or per_beat <= 0:
            continue
        scanned += 1

        ticks, frets = [], []
        for event in encoded:
            lanes = tokenizer.reverse_chord.get(
                event[1] // (tokenizer.N_FLAG * tokenizer.N_SUSTAIN), ())
            playable = tuple(lane for lane in lanes if lane <= 4)
            ticks.append(event[0])
            frets.append(playable[0] if len(playable) == 1 else None)

        deltas = [b - a if a is not None and b is not None else None
                  for a, b in zip(frets, frets[1:])]

        for signature in wanted:
            width = len(signature)
            for start in range(len(deltas) - width + 1):
                chunk = deltas[start:start + width]
                if None in chunk or tuple(chunk) != signature:
                    continue
                # The occurrence covers width+1 notes, so width+1 tick positions.
                span_ticks = ticks[start + width] - ticks[start]
                spans[signature].append(span_ticks / per_beat)

    print(f"{scanned} charts scanned\n")
    rows = []
    for signature, names in wanted.items():
        values = spans.get(signature, [])
        if not values:
            continue
        rows.append((statistics.median(values), len(values), names[0], values))

    rows.sort()
    print(f"{'pattern':<38}{'beats (median)':>15}{'p10':>7}{'p90':>7}{'seen':>8}")
    for median, count, name, values in rows:
        ordered = sorted(values)
        p10 = ordered[int(0.10 * len(ordered))]
        p90 = ordered[int(0.90 * len(ordered))]
        mark = "" if count >= args.min_occurrences else "  (thin)"
        print(f"  {name[:36]:<36}{median:>15.2f}{p10:>7.2f}{p90:>7.2f}{count:>8}{mark}")

    solid = [r for r in rows if r[1] >= args.min_occurrences]
    if solid:
        medians = [r[0] for r in solid]
        print(f"\n  {len(solid)} patterns seen at least {args.min_occurrences} times")
        print(f"  median span across them: {statistics.median(medians):.2f} beats")

        # Reported on both statistics, because they disagree and the disagreement is
        # the answer. A first version counted only medians, said "within two beats:
        # 69/69 (100%)", and printed advice to size windows by p90 immediately below --
        # where six of those same patterns run past two beats, one to 7.17. Quoting the
        # median alone would have confirmed the 1-2 beat estimate and hidden the cases
        # a search window has to be built for.
        by_median = sum(1 for m in medians if m <= 2.0)
        p90s = [(sorted(values)[int(0.90 * len(values))], name)
                for _median, _count, name, values in solid]
        by_p90 = sum(1 for value, _name in p90s if value <= 2.0)
        print(f"  within two beats, typical occurrence: {by_median}/{len(solid)}"
              f"  ({100 * by_median / len(solid):.0f}%)")
        print(f"  within two beats, 9 in 10 occurrences: {by_p90}/{len(solid)}"
              f"  ({100 * by_p90 / len(solid):.0f}%)")

        widest = max(solid, key=lambda r: r[0])
        print(f"  widest median: {widest[2]} at {widest[0]:.2f} beats")
        for value, name in sorted(p90s, reverse=True)[:5]:
            if value > 2.0:
                print(f"    past two beats at p90: {name} at {value:.2f}")
        print("\n  A window sized for the median will truncate everything above it. The p90")
        print("  column is the one to size a search window against, per pattern.")


if __name__ == "__main__":
    main()
