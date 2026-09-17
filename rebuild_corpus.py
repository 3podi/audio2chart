"""Revalidate the corpus after a grid or window change.

`validate_dataset` decides which songs have a usable window, and its cache keys on the
grid and window among other things -- so changing `grid_ms` from 20 to 10 correctly
invalidates every verdict. Training would rebuild it on first startup anyway; doing it
here means the run starts immediately and any surprise shows up now rather than an hour
into a training job.

Reports the before and after so the effect of a grid change is visible rather than
assumed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from main import load_data_splits
from modules.utils_train import validate_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-name", default="audio")
    parser.add_argument("--drop-cache", action="store_true",
                        help="Delete the cache first. Not normally needed -- the "
                             "signature already includes the grid and window -- but "
                             "useful if a verdict is suspected stale for another reason.")
    return parser.parse_args()


def main():
    args = parse_args()
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        config = compose(config_name=args.config_name)

    cache_path = Path(config.data.root_folder) / "validation_cache.json"
    if args.drop_cache and cache_path.exists():
        cache_path.unlink()
        print(f"deleted {cache_path}")

    print(f"grid {config.data.grid_ms} ms, window {config.data.window_seconds} s, "
          f"tokens {int(config.data.window_seconds * 1000 / config.data.grid_ms) + 2}")

    train_files, val_files = load_data_splits(config)
    before = len(train_files) + len(val_files)
    print(f"manifest: {before} entries ({len(train_files)} train, {len(val_files)} val)")

    validation = dict(
        difficulties=list(config.data.difficulties),
        instruments=list(config.data.instruments),
        grid_ms=config.data.grid_ms,
        error_policy=config.data.error_policy,
        window_seconds=config.data.window_seconds,
        cache_path=cache_path,
    )
    started = time.time()
    train_ok = validate_dataset(train_files, **validation)
    val_ok = validate_dataset(val_files, **validation)
    elapsed = time.time() - started

    after = len(train_ok) + len(val_ok)
    print(f"\nusable: {after} of {before} ({100 * after / max(before, 1):.1f}%)")
    print(f"  train {len(train_ok)}/{len(train_files)}   val {len(val_ok)}/{len(val_files)}")
    print(f"  took {elapsed / 60:.1f} min; cache at {cache_path}")


if __name__ == "__main__":
    main()
