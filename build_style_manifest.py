"""Emit a training manifest holding one charting style.

Robert's taxonomy: CSC charts follow the guitar only; overcharts are mostly tapping and
follow the vocals, falling back to the instrumental when the vocals stop; wii charts are
overcharts except that a guitar solo is charted CSC-style and much easier; chording,
technical and strumming are out of scope for now.

Training on all 13,720 charts mixes every one of those. A CSC chart and an overchart of
the same song look completely different -- one tracks guitar throughout, the other
switches -- and the model sees both with no signal telling them apart. That is a
plausible cause of the output inconsistency measured yesterday, where eight generated
charts ranged from 0.000 to 0.655 sustains and -0.172 to +0.383 pattern lift.

Three sources, and the behavioural one is the best of them. Measured over a 700-chart
sample, asking what fraction of a chart's notes are taps:

    genre-tagged      median tap rate 0.964   100% are >= 80% taps
    folder-labelled   median tap rate 0.643    31% are >= 80% taps
    unlabelled        median tap rate 0.085     8% are >= 80% taps

So the genre tag is accurate and rare, the folder labels are about two thirds dilution
-- Robert said those packs are "mostly" overcharts and that is what mostly looks like --
and metadata misses real tapping charts entirely, since 8% of unlabelled ones qualify.
Filtering on the tap rate itself finds roughly 1,886 charts library-wide: seven times
the tagged set, without the dilution, including the ones no label would catch.

Label sources:

*song.ini genre* -- 750 charts carry an explicit style ("Tapping", "Variety",
"Technical", "Chording"). High confidence, small slice, and the only source that names
the style directly.

*folder names* -- 2,877 charts from overchart folders and the Fatty Hero and OOS packs,
which Robert says are mostly overcharts. Lower confidence, much larger.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

DEFAULT_STYLES = ("tapping",)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=Path(r"G:\a2c_data\audio_dataset_with_raw.json"))
    parser.add_argument("--styles", type=Path,
                        default=Path(r"G:\a2c_data\chart_styles.json"),
                        help="Chart path -> style tags, from song.ini genre")
    parser.add_argument("--folders", type=Path,
                        default=Path(r"G:\a2c_data\overchart_list.json"),
                        help="Chart paths identified by folder name instead")
    parser.add_argument("--style", nargs="+", default=list(DEFAULT_STYLES),
                        help="Styles to keep from the genre tags")
    parser.add_argument("--min-tap-rate", type=float, default=None,
                        help="Keep charts whose tap rate is at least this, whatever "
                             "their labels say. 0.8 is Robert's threshold for an "
                             "overchart. Requires reading every chart, so it is slow, "
                             "but it measures the style instead of trusting a folder.")
    parser.add_argument("--tap-cache", type=Path,
                        default=Path("G:/a2c_data/tap_rates.json"))
    parser.add_argument("--include-folders", action="store_true",
                        help="Also keep charts identified only by folder name. Larger "
                             "and less certain: Robert says those packs are *mostly* "
                             "overcharts, not entirely.")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def measure_tap_rates(entries, cache_path: Path) -> dict[str, float]:
    """Tap rate per chart, cached -- reading 13,720 charts is not quick."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from chart.chart_processor import ChartProcessor
    from chart.metrics import profile_from_timed
    from chart.tokenizer import SimpleTokenizerGuitar

    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    tokenizer = SimpleTokenizerGuitar(expressive=True)
    fresh = 0
    for index, entry in enumerate(entries, 1):
        path = entry["chart_path"]
        if path in cache:
            continue
        try:
            processor = ChartProcessor(["Expert"], ["Single"])
            processor.read_chart(path, target_sections="ExpertSingle")
            notes = processor.notes.get("ExpertSingle")
            if not notes:
                cache[path] = None
                continue
            resolution = int(processor.song_metadata["Resolution"])
            timed = tokenizer.format_seconds(
                tokenizer.encode(notes, resolution=resolution), processor.synctrack,
                resolution, float(processor.song_metadata["Offset"]))
            profile = profile_from_timed(timed, tokenizer)
            cache[path] = profile.pct_tap if profile.positions >= 100 else None
        except Exception:
            cache[path] = None
        fresh += 1
        if fresh % 250 == 0:
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  measured {index}/{len(entries)}", flush=True)
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return {k: v for k, v in cache.items() if v is not None}


def main():
    args = parse_args()
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    tagged = json.loads(args.styles.read_text(encoding="utf-8"))
    wanted = set(args.style)

    by_folder = set()
    if args.include_folders and args.folders.exists():
        by_folder = set(json.loads(args.folders.read_text(encoding="utf-8")))

    tap_rates = {}
    if args.min_tap_rate is not None:
        tap_rates = measure_tap_rates(entries, args.tap_cache)

    sources = Counter()
    kept = []
    for entry in entries:
        path = entry["chart_path"]
        tags = set(tagged.get(path, ()))
        rate = tap_rates.get(path)
        if args.min_tap_rate is not None:
            # The behavioural filter overrides the labels in both directions: a chart
            # that plays like an overchart is one whatever its folder says, and a chart
            # in an overchart folder that does not is dilution.
            if rate is None or rate < args.min_tap_rate:
                continue
            sources["tap rate" if not (tags & wanted) else "tap rate + genre tag"] += 1
        elif tags & wanted:
            sources["genre tag"] += 1
        elif path in by_folder:
            sources["folder name"] += 1
        else:
            continue
        row = dict(entry)
        row["style_source"] = ("tap_rate" if args.min_tap_rate is not None
                               else "genre" if tags & wanted else "folder")
        if rate is not None:
            row["tap_rate"] = round(rate, 4)
        kept.append(row)

    out = args.out or args.manifest.with_name(
        args.manifest.stem + "_" + "_".join(sorted(wanted)) + ".json")
    out.write_text(json.dumps(kept, indent=1), encoding="utf-8")

    print(f"wrote {out}  ({len(kept)} of {len(entries)} charts)")
    for name, count in sources.most_common():
        print(f"  {name:<14}{count:>6}")
    if len(kept) < 500:
        print(f"\nNOTE: {len(kept)} charts is small for a 245 M model. Fine-tune from an")
        print("      existing checkpoint rather than training from scratch.")


if __name__ == "__main__":
    main()
