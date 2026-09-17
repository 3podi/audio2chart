"""Which alternating figures also exist in a chorded form?

Robert on the skip: the unchorded figure O G B G Y G B G Y G R G became the chorded
GO GB GY GB GY GR -- the same descent, with the moving note played together with the
anchor instead of alternating with it. If chording is a general transformation rather
than a one-off, other alternating figures should have chorded counterparts too.

The transform: a figure where every other note is the same fret A, and the notes
between it are X1 X2 X3, becomes chords AX1 AX2 AX3. So for each alternating figure
found in the corpus, look for its chorded image, and report both counts. A figure with
a common unchorded form and no chorded form is evidence the transform is not general.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chart.catalogue import chord_signature
from chart.chart_processor import ChartProcessor
from chart.tokenizer import SimpleTokenizerGuitar

MIN_ALTERNATIONS = 3
"""Fewer than three and it is not an alternating figure, just two notes."""


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--styles", type=Path,
                        default=Path(r"G:\a2c_data\chart_styles.json"))
    parser.add_argument("--style", default="tapping")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top", type=int, default=14)
    return parser.parse_args()


def read_positions(path: str, tokenizer):
    processor = ChartProcessor(["Expert"], ["Single"])
    processor.read_chart(path, target_sections="ExpertSingle")
    notes = processor.notes.get("ExpertSingle")
    if not notes:
        return []
    resolution = int(processor.song_metadata["Resolution"])
    out = []
    for event in tokenizer.encode(notes, resolution=resolution):
        lanes = tokenizer.reverse_chord.get(
            event[1] // (tokenizer.N_FLAG * tokenizer.N_SUSTAIN), ())
        out.append(tuple(lanes) if lanes and max(lanes) <= 4 else None)
    return out


def alternating_runs(positions):
    """Maximal stretches of single notes where every other note is the same fret."""
    singles = [p[0] if p and len(p) == 1 else None for p in positions]
    index = 0
    while index < len(singles) - 3:
        anchor = singles[index + 1]
        if anchor is None or singles[index] is None:
            index += 1
            continue
        end = index
        while (end + 1 < len(singles) and singles[end + 1] == anchor
               and end + 2 < len(singles) and singles[end + 2] is not None
               and singles[end + 2] != anchor):
            end += 2
        movers = tuple(singles[i] for i in range(index, end + 1, 2))
        if len(movers) >= MIN_ALTERNATIONS and None not in movers:
            yield anchor, movers
            index = end + 1
        else:
            index += 1


def chorded_runs(positions):
    """Stretches of two-note chords that share a fret -- the chorded image."""
    index = 0
    while index < len(positions) - 2:
        window = []
        while (index + len(window) < len(positions)
               and positions[index + len(window)] is not None
               and len(positions[index + len(window)]) == 2):
            window.append(positions[index + len(window)])
        if len(window) >= MIN_ALTERNATIONS:
            shared = set(window[0])
            for chord in window[1:]:
                shared &= set(chord)
            if len(shared) == 1:
                anchor = shared.pop()
                movers = tuple(next(f for f in c if f != anchor) for c in window)
                yield anchor, movers
            index += len(window)
        else:
            index += max(1, len(window))


def main():
    args = parse_args()
    styles = json.loads(args.styles.read_text(encoding="utf-8"))
    charts = [p for p, tags in styles.items() if args.style in tags]
    if args.limit:
        charts = charts[:args.limit]

    tokenizer = SimpleTokenizerGuitar(expressive=True)
    plain, chorded = collections.Counter(), collections.Counter()
    plain_charts, chorded_charts = collections.Counter(), collections.Counter()
    scanned = 0
    for path in charts:
        try:
            positions = read_positions(path, tokenizer)
        except Exception:
            continue
        scanned += 1
        here_plain, here_chord = set(), set()
        for anchor, movers in alternating_runs(positions):
            key = tuple(m - anchor for m in movers)     # transposition-invariant
            plain[key] += 1
            here_plain.add(key)
        for anchor, movers in chorded_runs(positions):
            key = tuple(m - anchor for m in movers)
            chorded[key] += 1
            here_chord.add(key)
        for key in here_plain:
            plain_charts[key] += 1
        for key in here_chord:
            chorded_charts[key] += 1

    print(f"scanned {scanned} charts tagged '{args.style}'\n")
    # Rank by how well a shape is attested in BOTH forms -- sorting by plain count
    # answers "what is common", not "what gets chorded".
    def render(key):
        return " ".join(f"{m:+d}" for m in key[:8])

    both_forms = sorted(
        (k for k in plain if chorded_charts[k]),
        key=lambda k: -min(plain_charts[k], chorded_charts[k]))

    print(f"{'movers (from anchor)':<26}{'plain':>8}{'charts':>8}"
          f"{'chorded':>9}{'charts':>8}")
    for key in both_forms[:args.top]:
        print(f"{render(key):<26}{plain[key]:>8}{plain_charts[key]:>8}"
              f"{chorded[key]:>9}{chorded_charts[key]:>8}")

    both = sum(1 for k in plain if chorded[k])
    print(f"\n{both} of {len(plain)} alternating shapes also occur chorded")


if __name__ == "__main__":
    main()
