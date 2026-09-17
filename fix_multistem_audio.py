"""Repoint multi-stem songs at a proper mix, and rebuild their decoded audio.

In a folder that ships split -- guitar.ogg, vocals.ogg, drums_1.ogg and so on -- `song.*`
is not the song. It is the backing track, everything except the parts supplied separately.
`find_audio_for` preferred `song.*`, so for those folders the model was trained on the mix
with the guitar removed and asked to chart a part it could not hear.

73 songs in the tapping split, 4%. Robert found it by playing a generated chart and
noticing the audio was wrong.

Training reads the pre-decoded `raw_path`, not `audio_path`, so repointing the manifest is
only half of it -- the raw file has to be rebuilt from the mix as well, or nothing changes.

Writes a .bak beside each split it touches.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dataloader.utils_dataloader import mixed_audio_for, stem_files

SAMPLE_RATE = 24000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="*", type=Path, default=[
        Path(r"G:\a2c_data\splits_tapping\train.json"),
        Path(r"G:\a2c_data\splits_tapping\val.json"),
    ])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rebuild_raw(source: Path, destination: Path) -> bool:
    """Decode to the headerless int16 mono the loader expects."""
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
               "-f", "s16le", "-acodec", "pcm_s16le",
               "-ar", str(SAMPLE_RATE), "-ac", "1", str(destination)]
    try:
        return subprocess.run(command, timeout=600).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def main() -> None:
    args = parse_args()
    total_fixed = 0

    for split in args.splits:
        if not split.is_file():
            print(f"missing: {split}")
            continue
        entries = json.loads(split.read_text(encoding="utf-8"))
        affected = []

        for entry in entries:
            folder = Path(entry["chart_path"]).parent
            if len(stem_files(folder)) < 2:
                continue
            affected.append(entry)

        print(f"\n{split.name}: {len(affected)} of {len(entries)} are multi-stem")
        if args.dry_run:
            for entry in affected[:10]:
                print(f"  would fix {Path(entry['chart_path']).parent.name[:60]}")
            continue

        fixed = 0
        for index, entry in enumerate(affected, 1):
            folder = Path(entry["chart_path"]).parent
            mixed = mixed_audio_for(folder)
            if mixed is None:
                print(f"  [{index}/{len(affected)}] mix failed: {folder.name[:50]}")
                continue

            raw = Path(entry["raw_path"]) if entry.get("raw_path") else None
            if raw is None:
                print(f"  [{index}/{len(affected)}] no raw_path: {folder.name[:50]}")
                continue
            if not rebuild_raw(mixed, raw):
                print(f"  [{index}/{len(affected)}] raw rebuild failed: {folder.name[:50]}")
                continue

            entry["audio_path"] = str(mixed)
            entry["length_samples"] = raw.stat().st_size // 2
            fixed += 1
            if index % 10 == 0 or index == len(affected):
                print(f"  [{index}/{len(affected)}] rebuilt")

        if fixed:
            backup = split.with_suffix(split.suffix + ".bak")
            if not backup.exists():
                shutil.copy2(split, backup)
            split.write_text(json.dumps(entries, indent=1), encoding="utf-8")
            print(f"  wrote {split.name} ({fixed} fixed, backup at {backup.name})")
        total_fixed += fixed

    print(f"\n{total_fixed} song(s) repointed at a proper mix.")
    if total_fixed:
        print("The validation cache keys on chart mtime and size, not audio, so it does")
        print("not need clearing -- the charts themselves did not change.")


if __name__ == "__main__":
    main()
