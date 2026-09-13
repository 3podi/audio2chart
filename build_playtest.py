"""Package generated charts as Clone Hero songs, so a human can actually play one.

Every measurement in this project is an aggregate statistic, and a chart can match all of
them and still be unplayable. Nothing here can see whether patterns are placed sensibly,
whether the transitions are reachable, or whether the thing is in sync in-game -- and
generation uses detect_tempo on the audio, so an internally-perfect chart can still be
audibly wrong.

Thirty seconds of someone playing one answers all of that.

Songs come from the validation split, which the model never trained on. The audio is
copied beside the chart so what plays is exactly what the chart was generated from.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

VAL_SPLIT = Path(r"G:\a2c_data\splits_tapping\val.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="Exported model directory")
    parser.add_argument("--out", type=Path,
                        default=Path(r"G:\Clone Hero Songs\_audio2chart playtest"),
                        help="Where to write the song folders. Defaults into the Clone "
                             "Hero library under a name that sorts to the top, so it is "
                             "obvious what it is and easy to delete.")
    parser.add_argument("--songs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance", type=float, default=1.5,
                        help="1.5 moves density to the human figure (18.1 -> 21.4 nps "
                             "against a human 21.01), replicated on three clips and eight. "
                             "It does not reliably improve how well the chart follows the "
                             "music; that claim did not survive a wider sample.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def read_ini(folder: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for name in ("song.ini", "Song.ini", "SONG.INI"):
        path = folder / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                fields[key.strip().lower()] = value.strip()
        break
    return fields


def safe_name(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in " -_&'" else "_" for c in text)
    return keep.strip() or "untitled"


def main() -> None:
    args = parse_args()
    entries = json.loads(VAL_SPLIT.read_text(encoding="utf-8"))
    random.Random(args.seed).shuffle(entries)
    args.out.mkdir(parents=True, exist_ok=True)

    built = 0
    for entry in entries:
        if built >= args.songs:
            break
        source = Path(entry["chart_path"]).parent
        audio = Path(entry["audio_path"])
        if not audio.is_file():
            continue

        fields = read_ini(source)
        name = fields.get("name", source.name)
        artist = fields.get("artist", "unknown")
        folder = args.out / safe_name(f"{artist} - {name} [AI]")
        folder.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {artist} - {name}")
        generated = folder / "notes.chart"
        result = subprocess.run([
            sys.executable, str(ROOT / "generate.py"), str(audio),
            "--model_name", str(args.model),
            "--detect-tempo", "--snap", "auto",
            "--temperature", str(args.temperature),
            "--guidance", str(args.guidance),
            "--name", name, "--artist", artist,
            "--charter", "audio2chart (AI)",
            "--output", str(folder),
        ])
        if result.returncode or not generated.is_file():
            print("  generation failed; skipping")
            continue

        # Clone Hero plays whatever `MusicStream` names, and generate.py sets that to the
        # basename of the audio it was given -- so the copy has to keep that exact name,
        # not a tidied one. These files happen to be called song.ogg/mp3/opus, which would
        # have hidden the mismatch until the first source named anything else.
        #
        # Copy rather than link: the library is the user's, and a broken link there is
        # worse than a duplicated file.
        target_audio = folder / audio.name
        if not target_audio.exists():
            shutil.copy2(audio, target_audio)

        (folder / "song.ini").write_text(
            "[song]\n"
            f"name = {name} [AI]\n"
            f"artist = {artist}\n"
            "album = audio2chart\n"
            "charter = audio2chart (AI)\n"
            "genre = Tapping\n"
            "diff_guitar = 0\n"
            "delay = 0\n",
            encoding="utf-8")
        print(f"  wrote {folder}")
        built += 1

    print(f"\nBuilt {built} song folder(s) in {args.out}")
    print("Open Clone Hero, rescan the song library, and look for the [AI] entries.")
    print("\nWhat to look for, because the metrics cannot see any of it:")
    print("  - does it play in sync, or does it drift against the music")
    print("  - are the patterns in sensible places, or just plausible shapes anywhere")
    print("  - is it physically playable -- reachable transitions, workable anchors")
    print("  - does it follow the song's structure at all")


if __name__ == "__main__":
    main()
