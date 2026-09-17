"""Merge chart/audio manifests into the single file `main.py` expects.

`load_data_splits` reads `<data.root_folder>/audio_dataset_with_raw.json` and nothing else,
so the .chart-derived and .mid-derived halves of the corpus have to be combined into one
file under that exact name. Entries are keyed on (chart_path, difficulty); the raw audio is
checked to exist and to match the recorded sample count, because a truncated or half-written
.raw is indistinguishable from a good one at train time — `load_raw_audio` reads headerless
PCM and trusts whatever it is handed.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

BYTES_PER_SAMPLE = 2
REQUIRED_FIELDS = ("audio_path", "chart_path", "difficulty", "raw_path", "length_samples")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        entries = json.load(stream)
    if not isinstance(entries, list):
        raise ValueError(f"Manifest must contain a JSON list: {path}")
    return entries


def check_entry(entry: dict[str, Any], verify_raw: bool = True) -> str | None:
    """Return a reason to drop this entry, or None when it is usable."""
    missing = [field for field in REQUIRED_FIELDS if not entry.get(field)]
    if missing:
        return f"missing fields: {','.join(missing)}"
    if not Path(entry["chart_path"]).is_file():
        return "chart file is gone"
    if not verify_raw:
        return None
    raw = Path(entry["raw_path"])
    if not raw.is_file():
        return "raw audio is missing"
    samples = raw.stat().st_size // BYTES_PER_SAMPLE
    if samples == 0:
        return "raw audio is empty"
    if samples != entry["length_samples"]:
        return f"raw audio is {samples} samples, manifest says {entry['length_samples']}"
    return None


def merge(
    sources: list[Path],
    output: Path,
    verify_raw: bool = True,
    rejected_json: Path | None = None,
) -> dict[str, int]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    duplicates = 0

    for source in sources:
        entries = read_manifest(source)
        logger.info("%s: %d entries", source, len(entries))
        for entry in entries:
            reason = check_entry(entry, verify_raw)
            if reason:
                reasons[reason.split(":")[0]] += 1
                rejected.append(dict(entry, reason=reason, source=str(source)))
                continue
            key = (entry["chart_path"], entry["difficulty"])
            if key in merged:
                duplicates += 1
                continue
            merged[key] = entry

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(list(merged.values()), stream, indent=2)
        stream.write("\n")
    if rejected and rejected_json is not None:
        with rejected_json.open("w", encoding="utf-8") as stream:
            json.dump(rejected, stream, indent=2)
            stream.write("\n")

    logger.info("Wrote %d entries to %s", len(merged), output)
    logger.info("Dropped %d duplicates", duplicates)
    for reason, count in reasons.most_common():
        logger.info("Rejected %d entries: %s", count, reason)
    return {"kept": len(merged), "duplicates": duplicates, "rejected": len(rejected)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="*_with_raw.json manifests")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination, normally <root_folder>/audio_dataset_with_raw.json")
    parser.add_argument("--rejected-json", type=Path, default=None)
    parser.add_argument("--no-verify-raw", action="store_true",
                        help="Skip the raw-audio existence and length check")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rejected = arguments.rejected_json
    if rejected is None:
        rejected = arguments.output.with_name(f"{arguments.output.stem}_rejected.json")
    merge(arguments.sources, arguments.output, not arguments.no_verify_raw, rejected)


if __name__ == "__main__":
    main()
