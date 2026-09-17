"""Emit a training manifest whose audio is a stem rather than the full mix.

The model reads whatever the manifest names, so conditioning on an isolated instrument
needs no change to the dataloader or the architecture -- only a manifest that points
somewhere else. That keeps the experiment cheap and reversible: train once on the mix,
once on the guitar stem, and compare with evaluate_conditioning.py.

Three sources of audio, in preference order:

1. A stem the song already ships with (`guitar.ogg` next to `notes.chart`). Ground
   truth, no separation artifacts -- but concentrated in official game rips, which are
   older and simpler than the modern customs worth learning from.
2. A stem produced by separate_stems.py.
3. The original mix, when neither exists.

The manifest records which source each entry used, because a model trained on a mixture
of ground-truth stems, separated stems and full mixes is three experiments at once, and
the report has to be able to say so.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

AUDIO_EXTENSIONS = (".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac")
# htdemucs names the non-vocal, non-drum, non-bass residue "other"; for guitar-led
# music that is where the guitar ends up.
SEPARATED_EQUIVALENT = {"guitar": "other", "vocals": "vocals",
                        "drums": "drums", "bass": "bass"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=Path(r"G:\a2c_data\audio_dataset_with_raw.json"))
    parser.add_argument("--stems-root", type=Path, default=Path(r"G:\a2c_data\stems"))
    parser.add_argument("--stem", default="guitar",
                        choices=sorted(SEPARATED_EQUIVALENT),
                        help="Which instrument the chart is assumed to follow")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--require-stem", action="store_true",
                        help="Drop songs with no stem instead of falling back to the "
                             "mix, so the run is one experiment rather than three")
    return parser.parse_args()


def shipped_stem(folder: str, stem: str) -> str | None:
    """A stem the song already ships with, if it is separable from other audio."""
    try:
        names = os.listdir(folder)
    except OSError:
        return None
    audio = [n for n in names if os.path.splitext(n)[1].lower() in AUDIO_EXTENSIONS]
    others = {os.path.splitext(n)[0].lower() for n in audio}
    others -= {"preview", "crowd", "video"}
    if len(others) < 2:
        return None                      # a lone "guitar.ogg" is the whole mix renamed
    for name in audio:
        if os.path.splitext(name)[0].lower().startswith(stem):
            return os.path.join(folder, name)
    return None


def separated_stem(folder: str, stem: str, root: Path) -> str | None:
    """A stem produced by separate_stems.py, if that song has been processed."""
    import hashlib

    digest = hashlib.sha1(folder.encode("utf-8", "replace")).hexdigest()[:10]
    key = f"{Path(folder).name[:60].strip() or 'song'}_{digest}"
    candidate = root / key / f"{SEPARATED_EQUIVALENT[stem]}.ogg"
    return str(candidate) if candidate.exists() else None


def main():
    args = parse_args()
    out = args.out or args.manifest.with_name(
        args.manifest.stem + f"_{args.stem}.json")
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))

    sources = Counter()
    result = []
    for entry in entries:
        folder = os.path.dirname(entry["audio_path"])
        path = shipped_stem(folder, args.stem)
        source = "shipped"
        if path is None:
            path = separated_stem(folder, args.stem, args.stems_root)
            source = "separated"
        if path is None:
            if args.require_stem:
                sources["dropped"] += 1
                continue
            path, source = entry["audio_path"], "mix"
        sources[source] += 1
        row = dict(entry)
        row["audio_path"] = path
        row["audio_source"] = source
        # The pre-decoded raw belongs to the mix, so it no longer applies.
        row.pop("raw_path", None)
        result.append(row)

    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    total = sum(sources.values())
    print(f"wrote {out}  ({len(result)} entries)")
    for name, count in sources.most_common():
        print(f"  {name:<10}{count:>7}  ({100*count/total:.1f}%)")
    if sources["mix"] and not args.require_stem:
        print("\nNOTE: entries still on the full mix make this a mixed experiment. Use")
        print("      --require-stem to train on stems alone.")


if __name__ == "__main__":
    main()
