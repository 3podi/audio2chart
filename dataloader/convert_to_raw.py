import os
import json
import timeit
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chart.chart_processor import ChartProcessor
from chart.tokenizer import SimpleTokenizerGuitar

# --- Constant
MAX_NOTES = 5000

def find_audio_files(
    root,
    difficulties,
    instruments,
    output_json="results/audio_dataset.json",
    skipped_json="results/audio_skipped.json"
):
    start_time = timeit.default_timer()

    processor = ChartProcessor(difficulties, instruments)
    tokenizer = SimpleTokenizerGuitar()

    entries = []
    skipped = []

    for dirpath, _, filenames in tqdm(os.walk(root), desc="Processing folders"):
        try:
            # --- check for chart file
            if "notes.chart" not in filenames:
                if "notes.mid" in filenames:
                    skipped.append({"path": dirpath, "reason": ".mid chart"})
                else:
                    skipped.append({"path": dirpath, "reason": "missing_chart"})
                continue

            # --- check for audio file
            audio_file = None
            for candidate in ["song.opus", "song.ogg", "guitar.opus", "guitar.ogg"]:
                if candidate in filenames:
                    audio_file = candidate
                    break

            if audio_file is None:
                skipped.append({"path": dirpath, "reason": "missing_audio"})
                continue

            chart_path = Path(dirpath) / "notes.chart"
            audio_path = Path(dirpath) / audio_file

            # --- process chart
            processor.read_chart(str(chart_path))

            for section, notes in processor.notes.items():
                # --- validate charts
                try:
                    if len(notes) > MAX_NOTES:
                        raise("Skipping chart for too many notes.")
                    bpm_events = processor.synctrack
                    resolution = int(processor.song_metadata['Resolution'])
                    offset = float(processor.song_metadata['Offset'])

                    tokenized_chart = tokenizer.encode(note_list=notes)
                    tokenized_chart = tokenizer.format_seconds(
                        tokenized_chart, bpm_events, resolution=resolution, offset=offset
                    )

                    if len(notes) >0:
                        entries.append({
                            "audio_path": str(audio_path),
                            "chart_path": str(chart_path),
                            "difficulty": section,
                            # "synctrack": processor.synctrack,
                            # "notes": notes,
                            # "song_metadata": processor.song_metadata,
                        })
                    else:
                        raise("Empty note section.")
                except Exception as e:
                    skipped.append({
                        "path": dirpath,
                        "reason": "note_reading_error",
                        "error": str(e) 
                    })

        except Exception as e:
            skipped.append({
                "path": dirpath,
                "reason": "processing_error",
                "error": str(e)
            })

    # Save processed entries (single JSON array)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    # Save skipped folders
    with open(skipped_json, "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2)

    elapsed = timeit.default_timer() - start_time
    print(f"✅ Saved {len(entries)} entries to {output_json}")
    print(f"⚠️  Skipped {len(skipped)} folders -> {skipped_json}")
    print(f"⏱  Processing time: {elapsed:.2f} seconds")

    return output_json, skipped_json

# --------------------
# Conversion Logic
# --------------------

def convert_single_audio(opus_path: str, raw_dir: str, sr: int = 16000) -> tuple[str, int, Optional[str]]:
    """
    Convert one .opus/.ogg file to .raw.
    Returns: (raw_path, length_samples, error_message)
    """
    try:
        # Ensure raw_dir exists
        os.makedirs(raw_dir, exist_ok=True)

        opus_parent = Path(opus_path).parent.name  # e.g., "abc123" from "./abc123/song.opus"
        if not opus_parent.strip():
            raise ValueError(f"Invalid parent folder name for {opus_path}")

        raw_path = os.path.join(raw_dir, opus_parent + ".raw")

        # Skip if already exists
        if os.path.exists(raw_path):
            # Verify it's non-empty
            if os.path.getsize(raw_path) > 0:
                size_bytes = os.path.getsize(raw_path)
                length_samples = size_bytes // 2  # 16-bit = 2 bytes/sample
                return raw_path, length_samples, None

        # Run ffmpeg
        cmd = [
            'ffmpeg',
            '-i', opus_path,
            '-f', 's16le',
            '-ar', str(sr),
            '-ac', '1',
            '-y',  # overwrite if exists
            raw_path
        ]

        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

        # Get file size and compute samples
        size_bytes = os.path.getsize(raw_path)
        length_samples = size_bytes // 2  # 16-bit signed integer = 2 bytes per sample

        return raw_path, length_samples, None

    except subprocess.CalledProcessError as e:
        return opus_path, -1, f"FFMPEG failed: {e.stderr.decode().strip()}"
    except Exception as e:
        return opus_path, -1, f"Unexpected error: {str(e)}"


def convert_all_to_raw(
    input_json: str,
    raw_dir: str = "raw_audio",
    sr: int = 16000,
    max_workers: int = 8
):
    """
    Read entries from input_json, convert all audio files to .raw,
    then save an updated JSON with "raw_path" and "length_samples".
    """
    print(f"🔄 Loading dataset from {input_json}...")
    with open(input_json, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # Extract unique audio paths
    unique_audio_paths = list(set(entry["audio_path"] for entry in entries))
    total = len(unique_audio_paths)

    print(f"🔍 Found {total} unique audio files to convert...")

    # Use ProcessPoolExecutor for true parallelism (ffmpeg is CPU-bound)
    converted = {}
    errors = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all conversions
        future_to_path = {
            executor.submit(convert_single_audio, path, raw_dir, sr): path
            for path in unique_audio_paths
        }

        # Collect results as they complete
        for future in tqdm(as_completed(future_to_path), total=total, desc="Converting to .raw"):
            audio_path = future_to_path[future]
            try:
                raw_path, length_samples, error = future.result()
                if error:
                    errors.append({"audio_path": audio_path, "error": error})
                    continue
                converted[audio_path] = {"raw_path": raw_path, "length_samples": length_samples}
            except Exception as e:
                errors.append({"audio_path": audio_path, "error": f"Worker crashed: {str(e)}"})

    # Update entries with raw_path and length_samples
    updated_entries = []
    failed_entries = []

    for entry in entries:
        audio_path = entry["audio_path"]
        if audio_path in converted:
            meta = converted[audio_path]
            new_entry = entry.copy()
            new_entry["raw_path"] = meta["raw_path"]
            new_entry["length_samples"] = meta["length_samples"]
            updated_entries.append(new_entry)
        else:
            failed_entries.append(entry)

    # Log failures
    if failed_entries:
        fail_json = input_json.replace(".json", "_conversion_failed.json")
        with open(fail_json, "w", encoding="utf-8") as f:
            json.dump(failed_entries, f, indent=2)
        print(f"⚠️  {len(failed_entries)} entries failed to convert -> saved to {fail_json}")

    # Save only successful ones
    output_json = input_json.replace(".json", "_with_raw.json")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(updated_entries, f, indent=2)

    # Save errors
    error_json = input_json.replace(".json", "_conversion_errors.json")
    if errors:
        with open(error_json, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
        print(f"⚠️  {len(errors)} conversion errors saved to {error_json}")

    print(f"✅ Successfully converted {len(converted)} files.")
    print(f"📊 Saved enhanced dataset to: {output_json}")
    print(f"ℹ️  Total valid entries: {len(updated_entries)}")

    return output_json


# --------------------
# Main Entry Point
# --------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert all audio files in dataset to .raw format for fast loading.")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing folders with song.opus and notes.chart")
    parser.add_argument("--difficulties", nargs="+", default=["Expert"], help="List of difficulties to include (e.g., Expert Hard Medium)")
    parser.add_argument("--instruments", nargs="+", default=["Single"], help="List of instruments to include (e.g., Single Double)")
    parser.add_argument("--raw-dir", type=str, default="raw_audio", help="Directory to store .raw files (default: raw_audio)")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate for conversion (default: 16000)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel conversion processes (default: 8)")
    parser.add_argument("--output-json", type=str, default="results/audio_dataset.json", help="Output JSON for found entries")
    parser.add_argument("--skipped-json", type=str, default="results/audio_skipped.json", help="Output JSON for skipped folders")

    args = parser.parse_args()

    # Step 1: Find all audio/chart pairs
    print("🚀 Step 1: Scanning for audio and chart files...")
    find_audio_files(
        root=args.root,
        difficulties=args.difficulties,
        instruments=args.instruments,
        output_json=args.output_json,
        skipped_json=args.skipped_json
    )

    # Step 2: Convert all audio files to .raw
    print("\n🚀 Step 2: Converting audio files to .raw format...")
    convert_all_to_raw(
        input_json=args.output_json,
        raw_dir=args.raw_dir,
        sr=args.sr,
        max_workers=args.workers
    )

    print("\n🎉 All done! You can now use:")
    print(f"   use_predecoded_raw=True, precomputed_windows=False")
    print(f"   in your DataLoader with the new file: {args.output_json.replace('.json', '_with_raw.json')}")
