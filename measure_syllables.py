"""Do human charters put a note on each syllable?

Robert's end goal, in his words: "you would chart to the syllabols of the lyrics with
adequate patterns that match the number of actions (to execute the pattern) to
syllabals." He also said an AI would have to work out the syllable count.

Mostly it would not. Charts that carry lyrics carry them one event per syllable, with the
hyphen convention marking continuation -- 'The', 'A-', 'ris-', 'to-', 'crats,' -- and each
event has a tick. About 22% of the library has them, median 250 per chart, on the order of
a million events in total. That is syllable timing a human already aligned to the audio.

Before building anything on top of that, this asks whether the premise holds: when a chart
has lyrics, does its guitar line actually track them? Two things are measured per chart.

COVERAGE -- the share of syllables with a note within a sixteenth. High means the guitar
lands where the voice does.

LIFT -- the same share after sliding the whole note track by a random offset, divided out.
Coverage on its own is not evidence: a chart dense enough has a note near everything. Lift
asks whether notes land on syllables more than they land anywhere.

The distribution is the point. A converted Rock Band chart follows the guitar and should
show no lift; an overchart following the vocal should. If a subset does, that subset is
the training set for what Robert actually wants.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chart.chart_processor import ChartProcessor
from chart.tokenizer import SimpleTokenizerGuitar

LYRIC = re.compile(r'^\s*(\d+)\s*=\s*E\s+"?lyric\s+([^"\n]*)"?\s*$', re.M | re.I)

# A syllable counts as charted if a note is within this many beats of it.
#
# 0.25 (a sixteenth) was the first choice and it does not work: these charts carry a
# median 4.24 notes per syllable, so at a quarter beat a dense chart covers every
# syllable by accident. Coverage came out at 0.82-1.00 against a chance floor of
# 0.60-0.80, which caps the achievable lift near 1.33 -- "You're a Mean One, Mr. Grinch"
# scored coverage 1.00, so it could not have scored higher however well it tracked the
# voice. A ceiling-limited metric cannot answer the question.
#
# 0.06 of a beat is about a 64th note, tight enough that a note has to be placed on the
# syllable rather than near it.
TOLERANCE_BEATS = 0.06

SHUFFLES = 5
"""Random offsets per chart for the chance baseline. The offset is large and wraps, so
the note track keeps its own rhythm and only its relationship to the voice is destroyed.
"""


def _use_utf8_stdout() -> None:
    """Chart folder names carry characters cp1252 cannot encode, such as the Omega in
    one of these song titles. Windows consoles default to cp1252, so printing a results
    table crashes after all the work is done -- generate.py already hit this and solves
    it the same way."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=Path(r"G:\a2c_data\audio_dataset_with_raw.json"))
    parser.add_argument("--charts", type=int, default=120,
                        help="How many lyric-bearing charts to measure.")
    parser.add_argument("--min-syllables", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_BEATS,
                        help="Beats within which a note counts as landing on a "
                             "syllable. Too loose and a dense chart covers everything "
                             "by accident, which caps the lift a chart can score.")
    return parser.parse_args()


def syllable_ticks(text: str) -> list[int]:
    """Tick of every syllable event, in order.

    Markup like '<i>' and section-style asides are dropped: they carry no syllable.
    """
    out = []
    for tick, word in LYRIC.findall(text):
        cleaned = word.strip()
        if not cleaned or cleaned.startswith("<"):
            continue
        out.append(int(tick))
    return sorted(out)


def coverage(note_ticks: list[int], syllables: list[int], tolerance: float) -> float:
    """Share of syllables with a note within `tolerance` ticks."""
    if not syllables or not note_ticks:
        return 0.0
    from bisect import bisect_left

    hit = 0
    for tick in syllables:
        index = bisect_left(note_ticks, tick)
        nearest = min(
            (abs(note_ticks[i] - tick)
             for i in (index - 1, index) if 0 <= i < len(note_ticks)),
            default=None)
        if nearest is not None and nearest <= tolerance:
            hit += 1
    return hit / len(syllables)


def main() -> None:
    _use_utf8_stdout()
    args = parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    random.Random(args.seed).shuffle(entries)
    tokenizer = SimpleTokenizerGuitar(expressive=True)
    rng = random.Random(args.seed)

    rows = []
    seen: set[str] = set()
    for item in entries:
        if len(rows) >= args.charts:
            break
        path = item["chart_path"]
        if path in seen:
            continue
        seen.add(path)
        try:
            text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        syllables = syllable_ticks(text)
        if len(syllables) < args.min_syllables:
            continue
        try:
            processor = ChartProcessor(["Expert"], ["Single"])
            processor.read_chart(path, target_sections="ExpertSingle")
            resolution = int(processor.song_metadata["Resolution"])
            encoded = tokenizer.encode(
                processor.notes["ExpertSingle"], resolution=resolution)
        except Exception:
            continue
        note_ticks = sorted({event[0] for event in encoded})
        if len(note_ticks) < args.min_syllables:
            continue

        # Only the stretch the voice actually sings over; a long instrumental outro
        # would otherwise count as syllables the guitar failed to cover.
        low, high = syllables[0], syllables[-1]
        inside = [t for t in note_ticks if low <= t <= high]
        if len(inside) < args.min_syllables:
            continue

        tolerance = args.tolerance * resolution
        actual = coverage(inside, syllables, tolerance)

        span = high - low or 1
        chance = []
        for _ in range(SHUFFLES):
            shift = rng.randrange(span)
            moved = sorted(((t - low + shift) % span) + low for t in inside)
            chance.append(coverage(moved, syllables, tolerance))
        expected = statistics.mean(chance) or 1e-9

        rows.append({
            "name": Path(path).parent.name[:38],
            "syllables": len(syllables),
            "notes": len(inside),
            "ratio": len(inside) / len(syllables),
            "coverage": actual,
            "chance": expected,
            "lift": actual / expected,
        })

    if not rows:
        raise SystemExit("no lyric-bearing charts matched")

    rows.sort(key=lambda r: -r["lift"])
    print(f"{len(rows)} lyric-bearing charts, syllable coverage within "
          f"{args.tolerance} of a beat\n")
    print(f"{'chart':<40}{'syll':>6}{'notes':>7}{'n/syl':>7}"
          f"{'cover':>8}{'chance':>8}{'lift':>7}")
    for row in rows[:15]:
        print(f"  {row['name']:<38}{row['syllables']:>6}{row['notes']:>7}"
              f"{row['ratio']:>7.2f}{row['coverage']:>8.2f}{row['chance']:>8.2f}"
              f"{row['lift']:>7.2f}")
    if len(rows) > 15:
        print(f"  ... and {len(rows) - 15} more")

    lifts = sorted(r["lift"] for r in rows)
    ratios = sorted(r["ratio"] for r in rows)
    print(f"\n  lift      median {statistics.median(lifts):.2f}   "
          f"p10 {lifts[len(lifts)//10]:.2f}   p90 {lifts[-max(1,len(lifts)//10)]:.2f}")
    print(f"  notes/syl median {statistics.median(ratios):.2f}   "
          f"p10 {ratios[len(ratios)//10]:.2f}   p90 {ratios[-max(1,len(ratios)//10)]:.2f}")

    strong = [r for r in rows if r["lift"] >= 1.25]
    print(f"\n  charts tracking the voice (lift >= 1.25): {len(strong)}/{len(rows)}"
          f"  ({100*len(strong)/len(rows):.0f}%)")
    print("  Those are the ones that chart to the vocal rather than to the guitar, and")
    print("  they are the training set for charting to syllables. A lift near 1.00 means")
    print("  the notes land on syllables no more than they land anywhere.")


if __name__ == "__main__":
    main()
