"""Find repeating figures that fit in one or two beats, the way a charter writes them.

Robert: most patterns are usually one or two *beat lines* long for one complete rep --
not measures, which are four beats at 4/4 -- and variety charts
are a mix of everything -- so they are where unlisted vocabulary is most likely to show
up.

This is a different search from chart/discover.py. That one takes gap-based runs and
reports whatever the catalogue fails to claim, which finds fragments as readily as
figures. This one takes the bar grid as the unit, because a rep that repeats is a rep
that fits the bar, and asks which bar-length shapes recur across many charts.

Bar segmentation was recorded as a failure yesterday, and that finding stands: cutting
runs at every bar line shreds anything that legitimately crosses one. The difference is
what the bars are used for. As a segmentation rule they destroy continuous figures; as
a *window* for finding a repeating unit they are exactly the charter's own frame.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chart.bars import beat_boundaries, group_by_bar
from chart.catalogue import (CATALOGUE, body_signature, chord_signature,
                             is_chorded, parse)
from chart.chart_processor import ChartProcessor
from chart.discover import render
from chart.tokenizer import SimpleTokenizerGuitar

MIN_NOTES = 4
"""Fewer than this in a bar is not a figure, it is a couple of notes."""

MAX_NOTES = 32
"""More than this and the "bar" is a dense tremolo, not a rep worth naming."""


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--styles", type=Path,
                        default=Path(r"G:\a2c_data\chart_styles.json"))
    parser.add_argument("--style", default="variety")
    parser.add_argument("--beats", type=int, default=1, choices=(1, 2, 4),
                        help="Beats per candidate rep. Robert's unit is the beat, not "
                             "the measure -- one complete rep is a beat or two. "
                             "Measures are four times too wide and return several "
                             "figures concatenated.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--chords", action="store_true",
                        help="Include figures containing chords. Off by default because "
                             "the catalogue is written in single notes, so chorded "
                             "figures are all reported as new.")
    parser.add_argument("--only-chords", action="store_true",
                        help="Only figures that contain a chord.")
    parser.add_argument("--positions", type=int, default=0,
                        help="Instead of matching whole bars, slide a window of this "
                             "many positions within each bar. Whole-bar matching only "
                             "finds figures that fill a bar, so a short figure sitting "
                             "inside a busier bar -- the H pattern is three hits -- is "
                             "invisible to it.")
    parser.add_argument("--min-notes", type=int, default=MIN_NOTES,
                        help="Positions per figure. The default of 4 excludes the H "
                             "pattern, which is three hits (RB -> RYB -> RB), so chord "
                             "searches want 3.")
    return parser.parse_args()


def known_bodies() -> set[tuple[int, ...]]:
    return {body_signature(text) for text in CATALOGUE.values()}


def base_unit(signature: tuple[int, ...]) -> tuple[int, ...]:
    """The shortest repeating unit of a signature.

    A bar holding four reps of a trill reads as a 7-delta signature that matches no
    catalogue entry, and calling that a new pattern is wrong -- it is a known pattern
    played more times. Robert's own framing is one *complete rep*, so reduce to the rep
    before asking whether it is known.
    """
    length = len(signature)
    for period in range(1, length // 2 + 1):
        if length % period and length - period > period:
            pass
        if all(signature[i] == signature[i % period] for i in range(length)):
            return signature[:period]
    return signature


def is_known(signature: tuple[int, ...], known: set[tuple[int, ...]]) -> bool:
    """Known outright, or an exact repetition of something known.

    Deliberately strict. An earlier version also accepted a signature whose leading
    half was catalogued, which dismissed anything merely *starting* with a common
    shape -- including the chained descent O B Y R | B Y R G, which Robert confirms is
    a pattern and which begins with a descending quad. Suppressing real figures is a
    worse failure here than listing a few known ones twice.
    """
    return signature in known or base_unit(signature) in known


# Four notes on one fret is not a pattern -- Robert's catalogue has no notion of it,
# and treating repeated same-fret notes as a figure imports a guitar term that does
# not apply to charting.
def is_static(signature: tuple[int, ...]) -> bool:
    """Mostly repeated notes on one fret, with at most an occasional move.

    Robert: repeated same-fret notes are not a pattern. All-zero catches the pure
    case, but "eight reds then eight greens" is the same thing with one step in the
    middle, so anything more than half static is excluded too.
    """
    if not signature:
        return True
    return sum(1 for delta in signature if delta == 0) * 2 >= len(signature)


def bar_signatures(chart_path: str, tokenizer, span: int, chords: bool = False,
                   min_notes: int = MIN_NOTES, width: int = 0):
    """Signatures of each `span`-beat window holding a figure.

    Without `chords` a chord ends the figure, which is how the catalogue is written.
    With it, positions are described by shape and root movement instead, so tap-chord
    figures like the H pattern become visible.
    """
    processor = ChartProcessor(["Expert"], ["Single"])
    processor.read_chart(chart_path, target_sections="ExpertSingle")
    notes = processor.notes.get("ExpertSingle")
    if not notes:
        return []
    resolution = int(processor.song_metadata["Resolution"])
    encoded = tokenizer.encode(notes, resolution=resolution)
    if not encoded:
        return []
    last = max(event[0] for event in encoded)
    # Beat lines, not measure lines -- Robert: "most patterns are USUALLY 1-2 beat lines
    # in length", and a 4/4 measure is four of those, so measure grouping merges separate
    # figures.
    #
    # `resolution * span` was the old rule and it is only right when the time signature
    # counts quarter notes. 15.5% of songs in this corpus (93 of 600 sampled) contain a
    # denominator other than 4 -- /8 in 48 of them, plus /1, /2, /16, /32 and /64 -- and
    # in those the beat is not `resolution` ticks. beat_boundaries reads the denominator
    # from [SyncTrack], where .chart stores it as a power-of-two exponent.
    every_beat = beat_boundaries(chart_path, resolution, last)
    boundaries = every_beat[::span] if span > 1 else every_beat
    bars = group_by_bar(encoded, boundaries)

    out = []
    for index in range(len(bars)):
        window = list(bars[index])
        if not (min_notes <= len(window) <= MAX_NOTES):
            continue
        positions = []
        broken = False
        for event in window:
            lanes = tokenizer.reverse_chord.get(
                event[1] // (tokenizer.N_FLAG * tokenizer.N_SUSTAIN), ())
            if not lanes or any(lane > 4 for lane in lanes):
                broken = True           # an open note ends the figure either way
                break
            if len(lanes) > 1 and not chords:
                broken = True
                break
            positions.append(tuple(lanes))
        if broken or len(positions) < min_notes:
            continue
        groups = ([positions[i:i + width] for i in range(len(positions) - width + 1)]
                  if width else [positions])
        for group in groups:
            if len(group) < min_notes:
                continue
            if chords:
                out.append(chord_signature(group))
            else:
                frets = [p[0] for p in group]
                out.append(tuple(b - a for a, b in zip(frets, frets[1:])))
    return out


def main():
    args = parse_args()
    styles = json.loads(args.styles.read_text(encoding="utf-8"))
    charts = [path for path, tags in styles.items() if args.style in tags]
    if args.limit:
        charts = charts[:args.limit]
    print(f"{len(charts)} charts tagged '{args.style}'")

    tokenizer = SimpleTokenizerGuitar(expressive=True)
    known = known_bodies()
    counts = collections.Counter()
    charts_with = collections.Counter()
    scanned = 0
    for path in charts:
        try:
            signatures = bar_signatures(path, tokenizer, args.beats,
                                        chords=args.chords or args.only_chords,
                                        min_notes=args.min_notes,
                                        width=args.positions)
        except Exception:
            continue
        scanned += 1
        for signature in set(signatures):
            charts_with[signature] += 1
        for signature in signatures:
            counts[signature] += 1

    print(f"scanned {scanned}; {len(counts)} distinct {args.beats}-beat shapes\n")
    print(f"{'count':>7}{'charts':>8}  {'known':<7}{'notes':<26}signature")
    shown = 0
    chorded = args.chords or args.only_chords
    for signature, count in counts.most_common():
        if charts_with[signature] < 3:
            continue
        if args.only_chords and not is_chorded(signature):
            continue
        if not chorded and is_static(signature):
            continue
        if chorded:
            shown_text = " ".join(
                "".join("GRYBO"[f] for f in shape) + (f"({move:+d})" if move else "")
                for shape, move in signature)
            print(f"{count:>7}{charts_with[signature]:>8}  {'':<7}{shown_text[:70]}")
        else:
            mark = "yes" if is_known(signature, known) else "NEW"
            print(f"{count:>7}{charts_with[signature]:>8}  {mark:<7}"
                  f"{render(signature)[:24]:<26}{signature}")
        shown += 1
        if shown >= args.top:
            break


if __name__ == "__main__":
    main()
