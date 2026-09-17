"""Cut the library-wide stem plan down to the songs one experiment actually needs.

stem_plan.json covers the whole library: 11,158 songs need separating, which is tens of
GPU-hours. The question that has to be answered first is much smaller -- does training on
stems improve how well the chart follows the music at all? Robert agreed the shape of
that test: separate a few hundred songs, train twice on the same songs (mixed against
stems), and compare conditioning. Only if it helps is the full run worth starting.

This writes a plan in the same format separate_stems.py already reads, containing only
songs that are in the training split, so no GPU time is spent separating audio the
experiment will never load.

Songs already shipping with stems are excluded twice over: they are not in
needs_separation to begin with, and the split's audio_path for them now points at the
cached mix rather than at a folder this would look at.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=Path(r"G:\a2c_data\stem_plan.json"))
    parser.add_argument("--split", type=Path,
                        default=Path(r"G:\a2c_data\splits_tapping"))
    parser.add_argument("--out", type=Path,
                        default=Path(r"G:\a2c_data\stem_plan_experiment.json"))
    parser.add_argument("--songs", type=int, default=300,
                        help="How many to separate. 300 is the experiment size; the "
                             "full set is what --songs 0 gives.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    candidates = plan["needs_separation"]

    # Chart paths are per-difficulty and per-song; the plan is per-folder.
    wanted: set[str] = set()
    for name in ("train.json", "val.json"):
        path = args.split / name
        if not path.is_file():
            continue
        for entry in json.loads(path.read_text(encoding="utf-8")):
            wanted.add(str(Path(entry["chart_path"]).parent).lower())

    in_split = [folder for folder in candidates if folder.lower() in wanted]
    print(f"{len(candidates):,} songs need separation library-wide")
    print(f"{len(in_split):,} of them are in {args.split.name}")

    chosen = in_split
    if args.songs and len(in_split) > args.songs:
        chosen = random.Random(args.seed).sample(in_split, args.songs)
        chosen.sort()

    args.out.write_text(json.dumps({"needs_separation": chosen}, indent=1),
                        encoding="utf-8")
    print(f"wrote {len(chosen):,} to {args.out}")
    print(f"\n  separate_stems.py --plan {args.out} --two-stems vocals")
    print("  Two stems rather than four: the experiment asks whether separating the "
          "charted line helps at all,")
    print("  and vocals is what a modern overchart follows. Four costs more for a "
          "question this does not answer.")


if __name__ == "__main__":
    main()
