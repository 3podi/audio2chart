import os
import json
from pathlib import Path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chart.chart_processor import ChartProcessor
import timeit
from tqdm import tqdm

from collections import defaultdict
from sklearn.model_selection import train_test_split


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
    entries = []
    skipped = []

    for dirpath, _, filenames in tqdm(os.walk(root), desc="Processing folders"):
        try:
            # --- check for chart file
            if "notes.chart" not in filenames:
                if "notes.mid" in filenames:
                    skipped.append({
                        "path": dirpath,
                        "reason": ".mid chart"
                    })
                else:
                    skipped.append({
                        "path": dirpath,
                        "reason": "missing_chart"
                    })
                continue

            # --- check for audio file
            audio_file = None
            if "song.opus" in filenames:
                audio_file = "song.opus"
            elif "song.ogg" in filenames:
                audio_file = "song.ogg"
            elif "guitar.opus" in filenames:
                audio_file = "guitar.opus"
            elif "guitar.ogg" in filenames:
                audio_file = "guitar.ogg"

            if audio_file is None:
                skipped.append({
                    "path": dirpath,
                    "reason": "missing_audio"
                })
                continue

            chart_path = Path(dirpath) / "notes.chart"
            audio_path = Path(dirpath) / audio_file

            # --- process chart
            processor.read_chart(str(chart_path))

            for section, notes in processor.notes.items():
                entries.append({
                    "audio_path": str(audio_path),
                    "chart_path": str(chart_path),
                    "difficulty": section,
                    #"synctrack": processor.synctrack,
                    #"notes": notes,
                    #"song_metadata": processor.song_metadata
                })

        except Exception as e:
            skipped.append({
                "path": dirpath,
                "reason": "processing_error",
                "error": str(e)
            })

    # Save processed entries
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    # Save skipped folders
    with open(skipped_json, "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2)

    elapsed = timeit.default_timer() - start_time
    print(f"✅ Saved {len(entries)} entries to {output_json}")
    print(f"⚠️  Skipped {len(skipped)} folders -> {skipped_json}")
    print(f"⏱  Processing time: {elapsed:.2f} seconds")
    
    return entries, skipped


def split_json_entries_by_audio(
    input_jsonl: str,
    train_jsonl: str,
    val_jsonl: str,
    val_ratio: float = 0.1,
    random_seed: int = 42
):
    """
    Split dataset entries into train/validation by audio_path while 
    approximating stratification by difficulty. All entries from the same 
    audio_path go into the same split.
    """

    # --- load all entries
    audio_groups = defaultdict(list)  # audio_path -> list of entries
    difficulties_per_audio = defaultdict(set)

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            audio_path = entry["audio_path"]
            audio_groups[audio_path].append(entry)
            difficulties_per_audio[audio_path].add(entry["difficulty"])

    audio_paths = list(audio_groups.keys())

    # --- map difficulty -> audio paths
    difficulty_to_paths = defaultdict(list)
    for path, diffs in difficulties_per_audio.items():
        for d in diffs:
            difficulty_to_paths[d].append(path)

    # --- approximate stratified split by audio_path
    train_paths = set()
    val_paths = set()
    assigned_paths = set()

    for d, paths in difficulty_to_paths.items():
        available = list(set(paths) - assigned_paths)
        if not available:
            continue

        val_count = max(1, int(len(available) * val_ratio))
        train_split, val_split = train_test_split(
            available, test_size=val_count, random_state=random_seed
        )
        train_paths.update(train_split)
        val_paths.update(val_split)
        assigned_paths.update(available)

    # assign any remaining unassigned paths
    remaining = set(audio_paths) - train_paths - val_paths
    for path in remaining:
        if len(val_paths) / len(audio_paths) < val_ratio:
            val_paths.add(path)
        else:
            train_paths.add(path)

    # --- write all entries to JSONL
    def write_entries(paths, out_file):
        with open(out_file, "w", encoding="utf-8") as f:
            for path in tqdm(paths, desc=f"Writing {out_file}"):
                for entry in audio_groups[path]:
                    f.write(json.dumps(entry) + "\n")

    write_entries(train_paths, train_jsonl)
    write_entries(val_paths, val_jsonl)

    print(f"✅ Train: {len(train_paths)} audio files, Val: {len(val_paths)} audio files")
    print(f"✅ Train JSONL: {train_jsonl}, Val JSONL: {val_jsonl}")

    return train_paths, val_paths
