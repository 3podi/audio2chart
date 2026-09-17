"""Discover chart/audio pairs and convert audio to headerless PCM files."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from dataloader.utils_dataloader import find_audio_files


logger = logging.getLogger(__name__)
# Must match config.model.sample_rate. load_raw_audio reads headerless PCM and trusts the
# rate it is told, so a 16 kHz .raw fed to the 24 kHz Encodec plays 1.5x fast with the
# chart timing silently wrong -- and training still looks healthy.
SAMPLE_RATE = 24000


def convert_single_audio(
    audio_path: str,
    raw_dir: str,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[str, int, str | None]:
    """Convert one audio file and return its path, sample count, and optional error."""
    source = Path(audio_path)
    destination_dir = Path(raw_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    if not source.parent.name:
        return audio_path, -1, f"Invalid parent directory for {audio_path}"
    _tag = hashlib.sha1(str(source).encode("utf-8", "surrogatepass")).hexdigest()[:10]
    destination = destination_dir / f"{source.parent.name[:80]}_{_tag}.raw"
    if destination.is_file() and destination.stat().st_size > 0:
        return str(destination), destination.stat().st_size // 2, None

    command = [
        "ffmpeg", "-i", str(source), "-f", "s16le", "-ar", str(sample_rate),
        "-ac", "1", "-y", str(destination),
    ]
    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
            timeout=300,
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode(errors="replace").strip()
        return audio_path, -1, f"FFmpeg failed: {message}"
    except subprocess.TimeoutExpired:
        return audio_path, -1, "FFmpeg timed out after 300 seconds"
    except OSError as error:
        return audio_path, -1, f"Cannot run FFmpeg: {error}"
    return str(destination), destination.stat().st_size // 2, None


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def convert_all_to_raw(
    input_json: str,
    raw_dir: str = "raw_audio",
    sample_rate: int = SAMPLE_RATE,
    max_workers: int = 8,
) -> str:
    """Convert manifest audio files and write a manifest containing raw paths."""
    if sample_rate <= 0 or max_workers <= 0:
        raise ValueError("sample_rate and max_workers must be positive")
    input_path = Path(input_json)
    with input_path.open(encoding="utf-8") as stream:
        entries = json.load(stream)
    if not isinstance(entries, list):
        raise ValueError("Input manifest must contain a JSON list")

    audio_paths = sorted({entry["audio_path"] for entry in entries})
    converted: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(convert_single_audio, path, raw_dir, sample_rate): path
            for path in audio_paths
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting audio"):
            source = futures[future]
            try:
                raw_path, length_samples, error = future.result()
            except Exception as error:  # Process boundaries may propagate arbitrary worker errors.
                errors.append({"audio_path": source, "error": f"Worker failed: {error}"})
                continue
            if error:
                errors.append({"audio_path": source, "error": error})
            else:
                converted[source] = {
                    "raw_path": raw_path,
                    "length_samples": length_samples,
                }

    updated = [dict(entry, **converted[entry["audio_path"]])
               for entry in entries if entry["audio_path"] in converted]
    failed = [entry for entry in entries if entry["audio_path"] not in converted]
    output_path = input_path.with_name(f"{input_path.stem}_with_raw.json")
    _write_json(output_path, updated)
    if failed:
        _write_json(input_path.with_name(f"{input_path.stem}_conversion_failed.json"), failed)
    if errors:
        _write_json(input_path.with_name(f"{input_path.stem}_conversion_errors.json"), errors)
    logger.info("Converted %d/%d audio files", len(converted), len(audio_paths))
    return str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--difficulties", nargs="+", default=["Expert"])
    parser.add_argument("--instruments", nargs="+", default=["Single"])
    parser.add_argument("--raw-dir", default="raw_audio")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-json", default="results/audio_dataset.json")
    parser.add_argument("--skipped-json", default="results/audio_skipped.json")
    parser.add_argument("--max-notes", type=int, default=None,
                        help="Override the per-section note-event ceiling")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    find_audio_files(
        root=arguments.root,
        difficulties=arguments.difficulties,
        instruments=arguments.instruments,
        output_json=arguments.output_json,
        skipped_json=arguments.skipped_json,
        **({"max_notes": arguments.max_notes} if arguments.max_notes else {}),
    )
    output = convert_all_to_raw(
        input_json=arguments.output_json,
        raw_dir=arguments.raw_dir,
        sample_rate=arguments.sample_rate,
        max_workers=arguments.workers,
    )
    logger.info("Wrote converted dataset manifest to %s", output)


if __name__ == "__main__":
    main()
