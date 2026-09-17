"""Show a slice of a chart the way Moonscraper shows it.

Robert's idea: he opens a chart in Moonscraper, names a tick range, and I read the same
notes -- so when he explains what a passage is doing, we are looking at the same thing
rather than at his description of it.

Ticks are the shared coordinate because Moonscraper displays them directly. Beat and
bar lines are drawn in because a charter reads position relative to the grid, not as an
absolute number, and because Robert's patterns are described in bars.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chart.bars import bar_boundaries, parse_time_signatures

NOTE_LINE = re.compile(r"^\s*(\d+)\s*=\s*N\s+(\d+)\s+(\d+)")
LANES = "GRYBO"
OPEN_LANE = 7
FORCE_FLAG = 5
TAP_FLAG = 6


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chart", type=Path)
    parser.add_argument("--from", dest="start", type=int, required=True,
                        help="First tick to show (as Moonscraper reports it)")
    parser.add_argument("--to", dest="end", type=int, required=True)
    parser.add_argument("--section", default="ExpertSingle")
    parser.add_argument("--width", type=int, default=0,
                        help="Truncate after this many positions (0 = no limit)")
    return parser.parse_args()


def read_section(path: Path, section: str):
    """{tick: {"lanes": set, "tap": bool, "force": bool, "sustain": int}}"""
    positions: dict[int, dict] = {}
    inside = False
    with path.open(encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("["):
                inside = stripped == f"[{section}]"
                continue
            if not inside:
                continue
            match = NOTE_LINE.match(line)
            if not match:
                continue
            tick, lane, length = (int(g) for g in match.groups())
            slot = positions.setdefault(tick, {"lanes": set(), "tap": False,
                                               "force": False, "sustain": 0})
            if lane == TAP_FLAG:
                slot["tap"] = True
            elif lane == FORCE_FLAG:
                slot["force"] = True
            else:
                slot["lanes"].add(lane)
                slot["sustain"] = max(slot["sustain"], length)
    return positions


def resolution_of(path: Path) -> int:
    match = re.search(r"^\s*Resolution\s*=\s*(\d+)",
                      path.read_text(encoding="utf-8-sig", errors="replace"), re.M)
    return int(match.group(1)) if match else 192


def main():
    args = parse_args()
    positions = read_section(args.chart, args.section)
    resolution = resolution_of(args.chart)
    in_range = sorted(t for t in positions if args.start <= t <= args.end)
    if not in_range:
        print(f"no notes between ticks {args.start} and {args.end} in [{args.section}]")
        return

    boundaries = set(bar_boundaries(args.chart, resolution, args.end + 1))
    signatures = parse_time_signatures(args.chart)
    print(f"{args.chart.name}  [{args.section}]  resolution {resolution}  "
          f"ticks {args.start}..{args.end}")
    print(f"time signatures: " + ", ".join(f"{t}: {n}/{d}" for t, n, d in signatures[:4]))
    print(f"{len(in_range)} positions\n")
    print(f"{'tick':>8} {'bar':>5} {'beat':>6}  {'G R Y B O':<11} flags      sustain")

    shown = 0
    previous = None
    for tick in in_range:
        if args.width and shown >= args.width:
            print(f"  ... {len(in_range) - shown} more positions")
            break
        slot = positions[tick]
        # A bar line between this note and the last one is where a charter sees the
        # figure restart, so it is drawn rather than left to arithmetic.
        if previous is not None:
            crossed = [b for b in boundaries if previous < b <= tick]
            for boundary in sorted(crossed):
                print(f"{boundary:>8} {'':>5} {'':>6}  " + "-" * 11 + "  bar line")
        beat = tick / resolution
        bar = sum(1 for b in sorted(boundaries) if b <= tick)
        lane_text = " ".join(LANES[i] if i in slot["lanes"] else "."
                             for i in range(5))
        if OPEN_LANE in slot["lanes"]:
            lane_text = "( open )   "
        flags = []
        if slot["tap"]:
            flags.append("tap")
        if slot["force"]:
            flags.append("hopo")
        sustain = f"{slot['sustain']}" if slot["sustain"] else ""
        print(f"{tick:>8} {bar:>5} {beat:>6.2f}  {lane_text:<11} "
              f"{','.join(flags) or '-':<10} {sustain}")
        previous = tick
        shown += 1


if __name__ == "__main__":
    main()
