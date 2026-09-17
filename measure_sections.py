"""Do section markers predict how busy a chart is?

79% of the library labels its own structure in [Events] -- 'section Verse 1',
'section Chorus', 'solo' -- a median 16 markers a chart. We load none of it.

It matters because of a limitation the generator cannot currently work around. Chunks
decode independently from BOS, so a generated chart has no song-level structure: the
model cannot know it is in the second chorus rather than the first, and nothing can carry
over. Robert, on the generated charts: the patterns "don't go with the music".

Section labels would be a cheap way in, but only if they carry signal. Two questions, both
answered against a null rather than by eye:

DOES STRUCTURE EXIST AT THE MARKED PLACES? Density is measured per section and compared
with density measured between the same number of randomly placed boundaries. If real
boundaries separate busy stretches from quiet ones better than arbitrary ones do, the
markers are on something.

DOES THE NAME CARRY IT? Sections are bucketed by what the charter called them, and each
bucket's density is compared with its own chart's average. A chorus that is reliably
busier than its verse is a label the model could be conditioned on; labels that all sit
at 1.0 would mean the words are decoration.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chart.chart_processor import ChartProcessor
from chart.tokenizer import SimpleTokenizerGuitar

SECTION = re.compile(r'^\s*(\d+)\s*=\s*E\s+"?section\s+([^"\n]*)"?\s*$', re.M | re.I)

# Charters name sections freely, so the words are bucketed rather than taken literally.
# Ordered: the first pattern that matches wins, so "pre-chorus" is not read as "chorus".
BUCKETS: tuple[tuple[str, str], ...] = (
    ("pre-chorus", r"pre.?chorus"),
    ("intro", r"\bintro"),
    ("outro", r"\boutro|\bending\b"),
    ("verse", r"\bverse"),
    ("chorus", r"\bchorus|\bhook\b"),
    ("bridge", r"\bbridge"),
    ("solo", r"\bsolo|\blead\b"),
    ("breakdown", r"break.?down|\bbreak\b"),
    ("interlude", r"inter.?lude"),
)

MIN_SECTIONS = 4
SHUFFLES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=Path(r"G:\a2c_data\audio_dataset_with_raw.json"))
    parser.add_argument("--charts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _use_utf8_stdout() -> None:
    """Section names carry characters cp1252 cannot encode."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def bucket_of(name: str) -> str | None:
    lowered = name.strip().lower()
    for label, pattern in BUCKETS:
        if re.search(pattern, lowered):
            return label
    return None


def densities(note_ticks: list[int], boundaries: list[int]) -> list[float]:
    """Notes per tick in each span between consecutive boundaries."""
    from bisect import bisect_left

    out = []
    for start, stop in zip(boundaries, boundaries[1:]):
        width = stop - start
        if width <= 0:
            continue
        count = bisect_left(note_ticks, stop) - bisect_left(note_ticks, start)
        out.append(count / width)
    return out


def contrast(values: list[float]) -> float:
    """Spread of density across sections, normalised so charts compare.

    Coefficient of variation rather than raw standard deviation: a dense chart would
    otherwise look structured merely for being dense.
    """
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def main() -> None:
    _use_utf8_stdout()
    args = parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    random.Random(args.seed).shuffle(entries)
    tokenizer = SimpleTokenizerGuitar(expressive=True)
    rng = random.Random(args.seed)

    real_contrast, fake_contrast = [], []
    by_bucket: dict[str, list[float]] = defaultdict(list)
    seen: set[str] = set()
    read = 0

    for item in entries:
        if read >= args.charts:
            break
        path = item["chart_path"]
        if path in seen:
            continue
        seen.add(path)
        try:
            text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
        except OSError:
            continue
        markers = [(int(tick), name) for tick, name in SECTION.findall(text)]
        if len(markers) < MIN_SECTIONS:
            continue
        try:
            processor = ChartProcessor(["Expert"], ["Single"])
            processor.read_chart(path, target_sections="ExpertSingle")
            resolution = int(processor.song_metadata["Resolution"])
            encoded = tokenizer.encode(
                processor.notes["ExpertSingle"], resolution=resolution)
        except Exception:
            continue
        note_ticks = sorted(event[0] for event in encoded)
        if len(note_ticks) < 100:
            continue

        markers.sort()
        low, high = note_ticks[0], note_ticks[-1]
        boundaries = [low] + [t for t, _ in markers if low < t < high] + [high]
        if len(boundaries) < MIN_SECTIONS:
            continue
        read += 1

        real = densities(note_ticks, boundaries)
        real_contrast.append(contrast(real))

        # The null: the same number of boundaries, placed anywhere. If the markers are
        # meaningful, cutting the song at them separates busy from quiet better than
        # cutting it arbitrarily.
        for _ in range(SHUFFLES):
            cuts = sorted(rng.uniform(low, high) for _ in range(len(boundaries) - 2))
            fake_contrast.append(contrast(densities(note_ticks, [low] + cuts + [high])))

        chart_mean = statistics.mean(real) if real else 0.0
        if not chart_mean:
            continue
        for (start, name), stop in zip(markers, [t for t, _ in markers[1:]] + [high]):
            label = bucket_of(name)
            if label is None or stop <= start:
                continue
            from bisect import bisect_left
            count = bisect_left(note_ticks, stop) - bisect_left(note_ticks, start)
            by_bucket[label].append((count / (stop - start)) / chart_mean)

    if not read:
        raise SystemExit("no charts with section markers matched")

    print(f"{read} charts with at least {MIN_SECTIONS} section markers\n")
    print("Does density change at the marked places, or anywhere?")
    print(f"  contrast at real boundaries   {statistics.median(real_contrast):.3f}")
    print(f"  contrast at random boundaries {statistics.median(fake_contrast):.3f}")
    ratio = (statistics.median(real_contrast) /
             (statistics.median(fake_contrast) or 1e-9))
    print(f"  ratio                         {ratio:.2f}x")

    print("\nDoes the name predict it? Density relative to the chart's own average.")
    print(f"{'section':<14}{'sections':>10}{'median':>9}{'p10':>8}{'p90':>8}")
    for label, _pattern in BUCKETS:
        values = sorted(by_bucket.get(label, []))
        if len(values) < 20:
            continue
        print(f"  {label:<12}{len(values):>10}{statistics.median(values):>9.2f}"
              f"{values[len(values)//10]:>8.2f}{values[-max(1,len(values)//10)]:>8.2f}")
    print("\n  1.00 means a section as busy as the rest of its own chart. Labels that all")
    print("  sit near 1.00 would mean the words are decoration and not worth loading.")


if __name__ == "__main__":
    main()
