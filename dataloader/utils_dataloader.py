import os
import json
from pathlib import Path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chart.chart_processor import ChartProcessor
from chart.tokenizer import SimpleTokenizerGuitar
import timeit
from tqdm import tqdm

from collections import defaultdict
from sklearn.model_selection import train_test_split


# --- Constant
MAX_NOTES = 5000


def find_chart_files(root_folder):
    """
    Recursively finds all .chart files in root_folder and its subdirectories.

    Args:
        root_folder (str): Path to the root folder to search.

    Returns:
        list: List of absolute paths to all .chart files found.
    """
    chart_files = []
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.chart'):
                chart_files.append(os.path.join(root, file))
    return chart_files

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
                    bpm_events = processor.synctrack
                    resolution = int(processor.song_metadata['Resolution'])
                    offset = float(processor.song_metadata['Offset'])

                    tokenized_chart = tokenizer.encode(note_list=notes)
                    tokenized_chart = tokenizer.format_seconds(
                        tokenized_chart, bpm_events, resolution=resolution, offset=offset
                    )

                    if len(notes) > 0 and len(notes) < MAX_NOTES:
                        entries.append({
                            "audio_path": str(audio_path),
                            "chart_path": str(chart_path),
                            "difficulty": section,
                            # "synctrack": processor.synctrack,
                            # "notes": notes,
                            # "song_metadata": processor.song_metadata,
                        })
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


from collections import defaultdict
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def split_json_entries_by_audio(
    input_json: str,
    train_json: str,
    val_json: str,
    val_ratio: float = 0.1,
    random_seed: int = 42
):
    """
    Split dataset entries into train/validation by audio_path while
    approximating stratification by difficulty. All entries from the same
    audio_path go into the same split.
    """

    # --- load dataset (JSON array)
    with open(input_json, "r", encoding="utf-8") as f:
        entries = json.load(f)

    audio_groups = defaultdict(list)  # audio_path -> list of entries
    difficulties_per_audio = defaultdict(set)

    for entry in entries:
        audio_path = entry["audio_path"]
        audio_groups[audio_path].append(entry)
        difficulties_per_audio[audio_path].add(entry["difficulty"])

    audio_paths = list(audio_groups.keys())

    # --- map difficulty -> audio paths
    difficulty_to_paths = defaultdict(list)
    for path, diffs in difficulties_per_audio.items():
        for d in diffs:
            difficulty_to_paths[d].append(path)

    train_paths, val_paths, assigned_paths = set(), set(), set()

    # --- stratified split (approximate)
    for d, paths in difficulty_to_paths.items():
        available = list(set(paths) - assigned_paths)
        if not available:
            continue

        if len(available) == 1:
            # handle edge case: only 1 sample
            only_path = available[0]
            if len(val_paths) / max(1, len(audio_paths)) < val_ratio:
                val_paths.add(only_path)
            else:
                train_paths.add(only_path)
            assigned_paths.add(only_path)
            continue

        val_count = max(1, int(len(available) * val_ratio))
        # Ensure val_count < len(available)
        val_count = min(val_count, len(available) - 1)

        train_split, val_split = train_test_split(
            available, test_size=val_count, random_state=random_seed
        )
        train_paths.update(train_split)
        val_paths.update(val_split)
        assigned_paths.update(available)

    # --- assign any remaining unassigned paths
    remaining = set(audio_paths) - train_paths - val_paths
    for path in remaining:
        if len(val_paths) / max(1, len(audio_paths)) < val_ratio:
            val_paths.add(path)
        else:
            train_paths.add(path)

    # --- collect entries
    train_entries = [e for p in train_paths for e in audio_groups[p]]
    val_entries = [e for p in val_paths for e in audio_groups[p]]

    # --- write outputs
    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_entries, f, indent=2)

    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_entries, f, indent=2)

    actual_ratio = len(val_paths) / max(1, len(audio_paths))
    print(f"✅ Train: {len(train_paths)} audio files, Val: {len(val_paths)} audio files")
    print(f"✅ Train JSON: {train_json}, Val JSON: {val_json}")
    print(f"📊 Desired val ratio: {val_ratio:.2f}, Actual: {actual_ratio:.2f}")

    return train_paths, val_paths


import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from typing import List, Dict, Set, DefaultDict


def split_json_entries_by_audio_raw(
    input_json: str,
    train_json: str,
    val_json: str,
    val_ratio: float = 0.1,
    random_seed: int = 42
):
    """
    Split dataset entries into train/validation by raw_path while
    approximating stratification by difficulty. All entries from the same
    raw_path go into the same split.

    Assumes input JSON has entries with keys: "raw_path", "difficulty"
    (as generated by convert_to_raw.py)
    """

    # --- Load dataset (JSON array of entries)
    with open(input_json, "r", encoding="utf-8") as f:
        entries = json.load(f)

    # --- Group entries by raw_path
    raw_groups: DefaultDict[str, List[Dict]] = defaultdict(list)
    difficulties_per_raw: DefaultDict[str, Set[str]] = defaultdict(set)

    for entry in entries:
        raw_path = entry.get("raw_path")
        if not raw_path:
            raise ValueError(f"Entry missing 'raw_path': {entry}")
        raw_groups[raw_path].append(entry)
        difficulties_per_raw[raw_path].add(entry["difficulty"])

    # --- Extract list of unique raw_paths
    raw_paths = list(raw_groups.keys())

    # --- Map difficulty -> list of raw_paths (for stratification)
    difficulty_to_raws: DefaultDict[str, List[str]] = defaultdict(list)
    for raw_path, diffs in difficulties_per_raw.items():
        for d in diffs:
            difficulty_to_raws[d].append(raw_path)

    train_raws: Set[str] = set()
    val_raws: Set[str] = set()
    assigned_raws: Set[str] = set()

    # --- Stratified splitting by difficulty
    for difficulty, raw_list in difficulty_to_raws.items():
        # Only consider unassigned paths
        available = list(set(raw_list) - assigned_raws)
        if not available:
            continue

        if len(available) == 1:
            # Edge case: only one audio file for this difficulty
            only_raw = available[0]
            current_val_ratio = len(val_raws) / max(1, len(raw_paths))
            if current_val_ratio < val_ratio:
                val_raws.add(only_raw)
            else:
                train_raws.add(only_raw)
            assigned_raws.add(only_raw)
            continue

        # Calculate how many to put in val
        val_count = max(1, int(len(available) * val_ratio))
        val_count = min(val_count, len(available) - 1)  # ensure at least one in train

        # Split
        train_split, val_split = train_test_split(
            available,
            test_size=val_count,
            random_state=random_seed
        )
        train_raws.update(train_split)
        val_raws.update(val_split)
        assigned_raws.update(available)

    # --- Handle any remaining unassigned raw_paths
    remaining = set(raw_paths) - train_raws - val_raws
    for raw_path in remaining:
        current_val_ratio = len(val_raws) / max(1, len(raw_paths))
        if current_val_ratio < val_ratio:
            val_raws.add(raw_path)
        else:
            train_raws.add(raw_path)

    # --- Collect full entries for train and val
    train_entries = [entry for raw in train_raws for entry in raw_groups[raw]]
    val_entries = [entry for raw in val_raws for entry in raw_groups[raw]]

    # --- Write outputs
    with open(train_json, "w", encoding="utf-8") as f:
        json.dump(train_entries, f, indent=2)

    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(val_entries, f, indent=2)

    total_raws = len(raw_paths)
    actual_val_ratio = len(val_raws) / max(1, total_raws)

    print(f"✅ Train: {len(train_raws)} unique .raw files, Val: {len(val_raws)} unique .raw files")
    print(f"✅ Total entries: {len(entries)} → Train: {len(train_entries)}, Val: {len(val_entries)}")
    print(f"✅ Train JSON: {train_json}")
    print(f"✅ Val JSON: {val_json}")
    print(f"📊 Desired val ratio: {val_ratio:.2f}, Actual: {actual_val_ratio:.2f}")

    return train_raws, val_raws