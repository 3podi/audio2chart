"""Separate song audio into stems, for songs that do not already ship with them.

Robert's point: a chart follows one line -- the guitar, or the vocals in a modern
overchart -- and the model currently sees a mono mix through Encodec at 3 kbps. That
would explain why swapping in another song's audio costs only +0.1891 nats: the model
can hear that it is loud rock music but cannot pick out the line being charted.

18.7% of the library already ships separated, but those are overwhelmingly official
game rips -- old, comparatively simple charts. Modern Clone Hero customs, which are the
style worth learning, are the ones that need separating.

Resumable by construction: a song whose output already exists is skipped, so this can
be stopped and restarted freely across a run measured in tens of hours.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "htdemucs"
DEFAULT_STEMS = ("vocals", "drums", "bass", "other")
MARKER = ".stems_done"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=Path(r"G:\a2c_data\stem_plan.json"),
                        help="JSON from the stem inventory; uses its needs_separation list")
    parser.add_argument("--out", type=Path, default=Path(r"G:\a2c_data\stems"),
                        help="Destination root; mirrors one folder per song")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0, help="Stop after N songs (0 = all)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--two-stems", default=None,
                        help="Only split this stem from the rest, e.g. 'vocals'. Much "
                             "smaller output when the other stems are not needed.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be processed and exit.")
    return parser.parse_args()


def song_key(folder: str) -> str:
    """A stable directory name for a song folder, collision-free across the library."""
    import hashlib

    digest = hashlib.sha1(folder.encode("utf-8", "replace")).hexdigest()[:10]
    return f"{Path(folder).name[:60].strip() or 'song'}_{digest}"


def audio_in(folder: str) -> Path | None:
    """The single mixed file for a song that has not been separated."""
    audio = (".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac")
    skip = {"preview", "crowd", "video"}
    found = []
    try:
        for name in os.listdir(folder):
            stem, dot, ext = name.rpartition(".")
            if dot and f".{ext.lower()}" in audio and stem.lower() not in skip:
                found.append(Path(folder) / name)
    except OSError:
        return None
    return found[0] if len(found) == 1 else None


def main():
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    folders = plan["needs_separation"]
    args.out.mkdir(parents=True, exist_ok=True)

    todo, done, unreadable = [], 0, []
    for folder in folders:
        destination = args.out / song_key(folder)
        if (destination / MARKER).exists():
            done += 1
            continue
        source = audio_in(folder)
        if source is None:
            # More than one audio file, or none readable -- the inventory said this
            # song was unseparated, so a surprise here is worth reporting rather than
            # silently counting as complete.
            unreadable.append(folder)
            continue
        todo.append((folder, source, destination))

    print(f"{len(folders)} songs need separation: {done} done, {len(todo)} to go, "
          f"{len(unreadable)} with no single readable audio file")
    if unreadable[:3]:
        for folder in unreadable[:3]:
            print(f"  skipped: {folder}")
    if args.dry_run or not todo:
        for folder, source, _ in todo[:5]:
            print(f"  would process {source}")
        return

    # Imported here so --dry-run works without loading torch.
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.audio import AudioFile
    from demucs.pretrained import get_model

    model = get_model(args.model).to(args.device).eval()
    wanted = [args.two_stems] if args.two_stems else list(model.sources)
    print(f"model {args.model}  stems {wanted}  device {args.device}")

    limit = args.limit or len(todo)
    started = time.time()
    for index, (folder, source, destination) in enumerate(todo[:limit], 1):
        try:
            wav = AudioFile(source).read(streams=0, samplerate=model.samplerate,
                                         channels=model.audio_channels)
            reference = wav.mean(0)
            wav = (wav - reference.mean()) / (reference.std() + 1e-8)
            with torch.no_grad():
                sources = apply_model(model, wav[None], device=args.device,
                                      progress=False)[0]
            sources = sources * reference.std() + reference.mean()

            destination.mkdir(parents=True, exist_ok=True)
            for name, audio in zip(model.sources, sources):
                if name not in wanted:
                    continue
                torchaudio.save(str(destination / f"{name}.ogg"), audio.cpu(),
                                model.samplerate, format="ogg")
            # Written last, so an interrupted song is retried rather than half-trusted.
            (destination / MARKER).write_text(folder, encoding="utf-8")
            rate = (time.time() - started) / index
            print(f"[{index}/{min(limit, len(todo))}] {source.parent.name[:48]}  "
                  f"{rate:.1f}s/song  eta {rate*(min(limit,len(todo))-index)/3600:.1f}h",
                  flush=True)
        except Exception as error:
            print(f"[{index}] FAILED {source.parent.name[:48]}: {str(error)[:70]}",
                  flush=True)


if __name__ == "__main__":
    main()
