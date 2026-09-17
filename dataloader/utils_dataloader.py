"""Dataset discovery and deterministic grouped splitting utilities."""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from chart.chart_processor import ChartProcessor
from chart.tokenizer import SimpleTokenizerGuitar


logger = logging.getLogger(__name__)
# Upper bound on note events in one chart section. Note that lane 5 (forced) and lane 6
# (tap) are separate events, and 47% of positions in this corpus carry a tap, so a dense
# chart reaches this ceiling at roughly half as many playable notes as the number suggests.
#
# Measured over G:\Clone Hero Songs: at 5000 this filter silently dropped 2,971 of the
# 11,621 charts that have audio -- 99.2% of everything missing from the manifest, and
# precisely the densest, most informative charts. Their sizes run min 5,010 / median 8,192
# / p95 29,872 / max 683,801; 50,000 recovers 2,907 of them (97.8%) while still excluding
# the joke charts in the tail, which are not human-playable and would only teach noise.
MAX_NOTES = 50000
# Preferred audio filenames, most-specific first. Surveyed over G:\Clone Hero Songs:
# 11,624 of 11,641 chart folders carry audio and these match 11,621 of them.
AUDIO_CANDIDATES = ("song.opus", "song.ogg", "song.mp3", "song.wav", "song.flac",
                    "song.m4a", "song.aac", "song.oga",
                    "guitar.opus", "guitar.ogg", "guitar.mp3", "guitar.wav", "guitar.flac")

AUDIO_EXTENSIONS = (".ogg", ".opus", ".mp3", ".wav", ".flac", ".m4a", ".aac", ".wma", ".oga")
"""Extensions the fallback will accept. Deliberately excludes .webm -- the 105 in this
library are all video.webm, not audio."""

AUDIO_EXCLUDED_STEMS = ("preview", "video", "crowd")
"""Never treat these as the song, whatever their extension.

preview.* is a short clip -- 64 of them here. Picking one as the song audio would pair a
full-length chart with ~30 seconds of sound and silently destroy the timing for that entry,
which is exactly the class of error that is invisible until a model trains on it.
"""


STEM_PREFIXES = ("guitar", "rhythm", "bass", "vocals", "drums", "keys", "song")
"""Filenames Clone Hero mixes together when a song ships split.

`song.*` is in here deliberately: in a multi-stem folder it is not the whole song, it is
the leftover backing track, and it belongs in the sum like any other stem.
"""

STEM_CACHE = Path(r"G:\a2c_data\mixed_audio")


def stem_files(directory: Path) -> list[Path]:
    """The separate stems in a folder, or [] if it is an ordinary single-audio song."""
    try:
        entries = [p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
                   and not p.stem.lower().startswith(AUDIO_EXCLUDED_STEMS)]
    except OSError:
        return []
    stems = [p for p in entries
             if any(p.stem.lower().startswith(prefix) for prefix in STEM_PREFIXES)]
    # One file called song.* is just a song. Two or more named parts is a split.
    named = {p.stem.lower().rstrip("0123456789_") for p in stems}
    return sorted(stems) if len(named) > 1 else []


def mixed_audio_for(directory: Path) -> Path | None:
    """Sum a folder's stems into one cached file, the way Clone Hero plays them.

    Cached under a hash of the input paths and sizes, so a folder is mixed once and a
    changed folder is mixed again. Returns None if ffmpeg is unavailable or fails, and
    callers fall back to the ordered candidates.
    """
    stems = stem_files(directory)
    if len(stems) < 2:
        return None

    import hashlib
    import subprocess

    signature = hashlib.sha1(
        "|".join(f"{p.name}:{p.stat().st_size}" for p in stems).encode()
    ).hexdigest()[:16]
    STEM_CACHE.mkdir(parents=True, exist_ok=True)
    destination = STEM_CACHE / f"{directory.name[:60]}_{signature}.ogg"
    if destination.is_file():
        return destination

    # normalize=0 keeps each stem at its authored level -- amix's default divides by the
    # input count, which buries a guitar that shares a folder with three drum stems.
    #
    # But a plain sum overshoots: seven stems of one song peaked at 2.146, and anything
    # over 1.0 clips. A limiter was tried and was not enough either -- capping at 0.95
    # still decoded at 1.149, because Vorbis overshoots on decode. So measure the sum's
    # real peak and apply the exact gain, which needs two passes and is worth it for a
    # file that is written once and cached.
    inputs: list[str] = []
    for stem in stems:
        inputs += ["-i", str(stem)]
    mix = f"amix=inputs={len(stems)}:duration=longest:normalize=0"

    # astats, not volumedetect. volumedetect converts to 16-bit integer before measuring,
    # so it clamps: on a sum that genuinely peaked near 2.0 it reported 0.0 dB, the gain
    # computed from it did nothing, and the mix still clipped. Vorbis stores float and
    # happily holds samples above 1.0, so the measurement has to be a float one.
    try:
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", *inputs, "-filter_complex",
             f"{mix},astats=measure_overall=Peak_level:measure_perchannel=none",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None

    peak_db = None
    for line in probe.stderr.splitlines():
        if "Peak level dB:" in line:
            try:
                peak_db = float(line.split("Peak level dB:")[1].strip().split()[0])
            except (IndexError, ValueError):
                peak_db = None
    if peak_db is None:
        return None

    # Land at -1 dBFS, leaving room for the decoder's overshoot.
    gain_db = -1.0 - peak_db
    command = ["ffmpeg", "-y", "-loglevel", "error", *inputs,
               "-filter_complex", f"{mix},volume={gain_db:.2f}dB",
               "-c:a", "libvorbis", "-q:a", "6", str(destination)]
    try:
        if subprocess.run(command, timeout=600).returncode != 0:
            destination.unlink(missing_ok=True)
            return None
    except (OSError, subprocess.SubprocessError):
        destination.unlink(missing_ok=True)
        return None
    return destination if destination.is_file() else None


def find_audio_for(directory: Path) -> Path | None:
    """The audio file for a chart folder, or None.

    Tries the known names first, then falls back to any audio file that is not a preview or
    a video. The fallback recovers folders that name their audio after the song itself
    (three in this library) without which those charts are simply dropped.

    CORRECTION, and it ran the wrong way round. This used to say that for stem-split
    songs "guitar.* wins, so the model sees isolated guitar", and that only 10 folders
    were affected. Both were wrong. `song.*` comes first in AUDIO_CANDIDATES, so `song.*`
    won -- and in a multi-stem folder `song.*` is the *backing track*, everything except
    the instruments supplied separately. The model was therefore trained on the mix with
    the guitar removed, and asked to chart a part it could not hear. 73 songs in the
    tapping split, 4%, not 10.

    Clone Hero plays every stem together, so the song is their sum. `mixed_audio_for`
    builds that sum once and caches it, and it is what this returns for such folders.
    """
    mixed = mixed_audio_for(directory)
    if mixed is not None:
        return mixed

    for name in AUDIO_CANDIDATES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    try:
        loose = sorted(
            path for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in AUDIO_EXTENSIONS
            and not path.stem.lower().startswith(AUDIO_EXCLUDED_STEMS)
        )
    except OSError:
        return None
    return loose[0] if loose else None


def find_chart_files(root_folder: str | Path) -> list[str]:
    root = Path(root_folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Chart root does not exist: {root}")
    return sorted(str(path) for path in root.rglob("*.chart"))


def find_audio_files(
    root: str | Path,
    difficulties: Iterable[str],
    instruments: Iterable[str],
    output_json: str | Path = "results/audio_dataset.json",
    skipped_json: str | Path = "results/audio_skipped.json",
    max_notes: int = MAX_NOTES,
) -> tuple[str, str]:
    """Discover valid chart/audio pairs and write dataset manifests."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    processor = ChartProcessor(list(difficulties), list(instruments))
    tokenizer = SimpleTokenizerGuitar()
    entries: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    empty_sections = 0
    oversized_sections = 0

    for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
        chart_path = directory / "notes.chart"
        if not chart_path.is_file():
            continue
        audio_path = find_audio_for(directory)
        if audio_path is None:
            skipped.append({"path": str(directory), "reason": "missing_audio"})
            continue
        try:
            processor.read_chart(str(chart_path))
            if not processor.song_metadata or not processor.synctrack:
                raise ValueError("Chart is missing Song metadata or SyncTrack events")
            resolution = int(processor.song_metadata["Resolution"])
            offset = float(processor.song_metadata["Offset"])
            for section, notes in processor.notes.items():
                if not notes:
                    empty_sections += 1
                    continue
                if len(notes) >= max_notes:
                    oversized_sections += 1
                    skipped.append({
                        "path": str(directory),
                        "reason": "note_count_over_max",
                        "error": f"{section} has {len(notes)} note events (max_notes={max_notes})",
                    })
                    continue
                encoded = tokenizer.encode(notes, resolution=resolution)
                tokenizer.format_seconds(encoded, processor.synctrack, resolution, offset)
                entries.append({
                    "audio_path": str(audio_path),
                    "chart_path": str(chart_path),
                    "difficulty": section,
                })
        except (KeyError, OSError, TypeError, ValueError) as error:
            skipped.append({
                "path": str(directory),
                "reason": "chart_validation_error",
                "error": str(error),
            })

    _write_json(output_json, entries)
    _write_json(skipped_json, skipped)
    logger.info("Discovered %d entries; skipped %d directories", len(entries), len(skipped))
    logger.info(
        "Section filter dropped %d empty and %d oversized sections (max_notes=%d)",
        empty_sections, oversized_sections, max_notes,
    )
    return str(output_json), str(skipped_json)


def _write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def _grouped_split(
    entries: list[dict[str, Any]],
    group_key: str,
    val_ratio: float,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], set[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between zero and one")
    if not entries:
        raise ValueError("Cannot split an empty dataset")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, entry in enumerate(entries):
        group = entry.get(group_key)
        difficulty = entry.get("difficulty")
        if not isinstance(group, str) or not group:
            raise ValueError(f"Entry {index} has no valid {group_key!r}")
        if not isinstance(difficulty, str) or not difficulty:
            raise ValueError(f"Entry {index} has no valid 'difficulty'")
        groups[group].append(entry)

    group_names = sorted(groups)
    if len(group_names) < 2:
        raise ValueError("At least two distinct audio groups are required for a split")
    random.Random(random_seed).shuffle(group_names)
    val_count = min(len(group_names) - 1, max(1, round(len(group_names) * val_ratio)))
    val_groups = set(group_names[:val_count])
    train_groups = set(group_names[val_count:])
    train = [entry for group in sorted(train_groups) for entry in groups[group]]
    val = [entry for group in sorted(val_groups) for entry in groups[group]]
    return train, val, train_groups, val_groups


def _split_manifest(
    input_json: str | Path,
    train_json: str | Path,
    val_json: str | Path,
    group_key: str,
    val_ratio: float,
    random_seed: int,
) -> tuple[set[str], set[str]]:
    with Path(input_json).open(encoding="utf-8") as stream:
        entries = json.load(stream)
    if not isinstance(entries, list):
        raise ValueError("Dataset manifest must contain a JSON list")
    train, val, train_groups, val_groups = _grouped_split(
        entries, group_key, val_ratio, random_seed
    )
    _write_json(train_json, train)
    _write_json(val_json, val)
    logger.info(
        "Split %d groups into %d train and %d validation groups",
        len(train_groups) + len(val_groups), len(train_groups), len(val_groups),
    )
    return train_groups, val_groups


def split_json_entries_by_audio(
    input_json: str,
    train_json: str,
    val_json: str,
    val_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[set[str], set[str]]:
    return _split_manifest(
        input_json, train_json, val_json, "audio_path", val_ratio, random_seed
    )


def split_json_entries_by_audio_raw(
    input_json: str,
    train_json: str,
    val_json: str,
    val_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[set[str], set[str]]:
    return _split_manifest(
        input_json, train_json, val_json, "raw_path", val_ratio, random_seed
    )
