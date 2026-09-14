"""How much of the library changes tempo, and what does one BPM cost there?

`validate_tempo.py` measures detect_tempo against charts that carry a single [SyncTrack]
event, and on those it is very accurate -- median error 0.000 BPM, 34 of 36 within 1.0
BPM. It excludes multi-tempo charts on purpose, because a song that changes tempo has no
single right answer to compare a single number against.

That exclusion hides the case Robert reported. Playing the generated iCarly chart: "Leave
it all to me is out of sync a small bit and the beat lines are out of place". That chart
carries thirteen tempo events; we detected 104.900 BPM against a first event of 128.226.

WHAT A WRONG MAP ACTUALLY BREAKS. Not note timing on its own. The model emits absolute
times, convert_notes_to_ticks turns them into ticks through the map we generate, and
Clone Hero turns them back through the same map -- the errors cancel and the note plays
when it should. Two things do break:

  * the beat lines, which are drawn from the map directly, so they sit wherever the
    wrong tempo puts them; and
  * `--snap auto`, which quantises every note onto that grid. A grid at the wrong rate
    shoves each note by a different amount, which reads as jitter rather than as
    lateness.

Robert reported both halves, which is what the mechanism predicts.

This measures how much of the corpus we cannot represent: how many charts carry more
than one tempo event, and for those, how far a single-BPM grid has drifted from the
chart's own map by the time the last tempo event arrives.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chart.tempo import parse_sync_track
from chart.time_conversion import preprocess_bpm_segments, tick_to_seconds

# The token grid. A grid error smaller than this cannot move a note to another slot,
# so it is the threshold that decides whether a single tempo is good enough.
GRID_MS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path,
                        default=Path(r"G:\a2c_data\splits_tapping"))
    parser.add_argument("--charts", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def drift_of(events: list[tuple[int, int]], resolution: int) -> float | None:
    """Seconds between the chart's own map and a flat one, at the last tempo event.

    Measured at the last event rather than at the end of the song because that is the
    last point both maps are defined for without guessing a song length.
    """
    last_tick = events[-1][0]
    if not last_tick:
        return None
    segments = preprocess_bpm_segments(events, resolution)
    true_seconds = tick_to_seconds(
        last_tick, segments, resolution, [s[2] for s in segments])
    flat = preprocess_bpm_segments([(0, events[0][1])], resolution)
    flat_seconds = tick_to_seconds(last_tick, flat, resolution, [s[2] for s in flat])
    return abs(true_seconds - flat_seconds)


def main() -> None:
    args = parse_args()
    entries: list[dict] = []
    for name in ("train.json", "val.json"):
        path = args.split / name
        if path.is_file():
            entries += json.loads(path.read_text(encoding="utf-8"))
    random.Random(args.seed).shuffle(entries)

    single = 0
    drifts: list[float] = []
    spreads: list[float] = []
    counts: list[int] = []
    read = 0

    for entry in entries:
        if read >= args.charts:
            break
        try:
            events, _ts, resolution = parse_sync_track(entry["chart_path"])
        except Exception:
            continue
        if not events:
            continue
        read += 1
        counts.append(len(events))
        if len(events) == 1:
            single += 1
            continue
        tempos = [value / 1000.0 for _tick, value in events]
        spreads.append(max(tempos) / min(tempos))
        drift = drift_of(events, resolution)
        if drift is not None:
            drifts.append(drift)

    if not read:
        raise SystemExit("no charts read")

    multi = read - single
    print(f"{read} charts read\n")
    print(f"  one tempo event       {single:>5}  {100 * single / read:>5.1f}%")
    print(f"  more than one         {multi:>5}  {100 * multi / read:>5.1f}%")
    if multi:
        many = [c for c in counts if c > 1]
        print(f"  median events, of those {statistics.median(many):.0f}"
              f"   most in one chart {max(many)}")

    if spreads:
        spreads.sort()
        print("\n  fastest / slowest tempo inside one chart")
        print(f"    median {statistics.median(spreads):.3f}x"
              f"   p90 {spreads[int(0.9 * len(spreads))]:.3f}x"
              f"   max {max(spreads):.3f}x")

    if drifts:
        drifts.sort()
        threshold = GRID_MS / 1000.0
        past = sum(1 for d in drifts if d > threshold)
        print("\n  drift at the last tempo event, one BPM against the chart's own map")
        print(f"    median {statistics.median(drifts) * 1000:>10.1f} ms")
        print(f"    p90    {drifts[int(0.9 * len(drifts))] * 1000:>10.1f} ms")
        print(f"    worst  {max(drifts) * 1000:>10.1f} ms")
        print(f"    past the {GRID_MS:.0f} ms grid: {past}/{len(drifts)}")

        representable = single + (len(drifts) - past)
        print(f"\n  charts a single tempo can represent within the grid: "
              f"{representable}/{read}  ({100 * representable / read:.1f}%)")
        print("  The rest get beat lines in the wrong place, and --snap auto quantises")
        print("  their notes onto that wrong grid.")


if __name__ == "__main__":
    main()
