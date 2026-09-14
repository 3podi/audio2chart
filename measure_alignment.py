"""Does the model use the audio in time, or only as a global impression?

Three independent attacks on conditioning all landed in the same place. Measured on the
same 364 held-out songs, the cost of handing the model a different song's audio:

  Encodec, 1,454 tapping charts   +0.0177 (1.5%)
  Encodec, 13,356 charts          +0.0106 (1.0%)
  MERT,    1,454 tapping charts   +0.0163 (1.4%)

Nine times the data did not help. A music-understanding encoder did not help. That points
away from what the audio is encoded as and toward how it reaches the decoder.

THE HYPOTHESIS. Cross-attention applies RoPE to both sides, and the two sequences are on
different time scales: a token step is 10 ms, an audio frame after compression 2 is
26.7 ms. A time-aligned pair is therefore (token i, audio frame 0.375 i), and the
relative offset RoPE sees for it grows linearly across the window -- 0 at the start, 469
in the middle, 937 at the end. RoPE assumes a fixed relationship shows up at a fixed
offset. Here "the same moment in time" shows up at a different offset at every position,
so "attend to the audio at my own timestamp" is not a rule the model can express.

If that is right, the model cannot be reading the audio in time. It would be extracting
roughly a bag of features -- what kind of song this is -- which is worth about a point of
loss and no more.

THE TEST. Shift the audio against the chart instead of swapping it. A model reading the
audio in time should be hurt by a shift of a fraction of a second, because the onsets no
longer line up. A model reading a global impression should barely notice, because a
shifted song is still the same song.

  shift ~ swap    the model is time-aligned, and the ceiling is elsewhere
  shift ~ 0       the model is not reading the audio in time at all

The second outcome would make the RoPE mismatch the first thing to fix, and would explain
why neither more data nor a better encoder moved anything.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

MIN_BATCH = 2

# Seconds to slide the audio by. The smallest is well under a beat at any tempo; the
# largest is long enough that no onset could still line up.
SHIFTS = (0.1, 0.25, 0.5, 1.0, 2.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--config-name", default="audio")
    parser.add_argument("--override", "-o", action="append", default=[])
    parser.add_argument("--split-folder", default=None)
    parser.add_argument("--songs", type=int, default=250)
    parser.add_argument("--batches", type=int, default=60)
    parser.add_argument("--trust", action="store_true",
                        help="Unpickle with weights_only=False. Only for a checkpoint "
                             "you produced yourself, never a downloaded one.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, root)
    from chart.tokenizer import SimpleTokenizerGuitar
    from main import build_dataloaders, load_data_splits
    from modules.trainer import WaveformTransformerDiscrete

    config_dir = args.config_dir or os.path.join(root, "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=args.config_name, overrides=list(args.override))
    with open_dict(cfg):
        cfg.loader.num_workers = 0
        cfg.loader.val_batch_size = MIN_BATCH
        cfg.loader.val_num_pieces = MIN_BATCH
        if args.split_folder:
            cfg.data.split_folder = args.split_folder

    tokenizer = SimpleTokenizerGuitar()
    train_files, val_files = load_data_splits(cfg)
    random.Random(0).shuffle(val_files)
    _, val_loader, _ = build_dataloaders(cfg, tokenizer, train_files[:args.songs],
                                         val_files[:args.songs])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WaveformTransformerDiscrete(
        vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_id,
        eos_token_id=tokenizer.eos_id, cfg_model=cfg.model,
        cfg_optimizer=cfg.optimizer)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu",
                           weights_only=not args.trust)
        model.load_state_dict(state["state_dict"], strict=False)
    model = model.to(device).eval()

    sample_rate = int(cfg.model.sample_rate)
    pad_id = tokenizer.pad_id
    weights = model.class_weights.to(device)

    # own, each shift, swapped
    columns = ["own"] + [f"{s:g}s" for s in SHIFTS] + ["swap"]
    collected: dict[str, list[float]] = {name: [] for name in columns}

    scored = 0
    with torch.no_grad():
        while scored < args.batches:
            progressed = False
            for batch in val_loader:
                if scored >= args.batches:
                    break
                audio = batch["input_values"].to(device)
                mask = batch["padding_mask"].to(device)
                notes = batch["note_values"].to(device)
                if audio.size(0) < MIN_BATCH:
                    continue
                progressed = True
                scored += 1

                inputs = notes[:, :-1].contiguous()
                targets = notes[:, 1:].contiguous()
                flat = targets.reshape(-1)
                is_note = flat != pad_id

                variants = {"own": audio}
                for seconds in SHIFTS:
                    # Roll the waveform, so the audio stays the same song at the same
                    # loudness and only its alignment to the chart changes. Anything that
                    # survives this is not being read in time.
                    variants[f"{seconds:g}s"] = torch.roll(
                        audio, int(seconds * sample_rate), dims=-1)
                variants["swap"] = torch.roll(audio, 1, dims=0)

                for name, variant in variants.items():
                    embedding = model._encode_audio(variant, mask)
                    logits = model.transformer(inputs, embedding, attention_mask=None,
                                               class_ids=None)
                    per_token = F.cross_entropy(
                        logits.reshape(-1, model.vocab_size), flat,
                        weight=weights, reduction="none")
                    collected[name].append(per_token[is_note].mean().item())
            if not progressed:
                break

    if not collected["own"]:
        raise SystemExit("measured nothing")

    own = statistics.mean(collected["own"])
    swap = statistics.mean(collected["swap"])
    print(f"{len(collected['own'])} batches, note positions only\n")
    print(f"{'audio given to the model':<28}{'loss':>9}{'vs own':>10}{'% of swap':>11}")
    print(f"  {'its own':<26}{own:>9.4f}{'':>10}{'':>11}")
    for name in columns[1:]:
        value = statistics.mean(collected[name])
        delta = value - own
        share = 100 * delta / (swap - own) if swap != own else float("nan")
        label = "a different song" if name == "swap" else f"its own, shifted {name}"
        print(f"  {label:<26}{value:>9.4f}{delta:>+10.4f}{share:>10.0f}%")

    quarter = statistics.mean(collected["0.25s"]) - own
    full = swap - own
    print(f"\n  A 0.25 s shift recovers {100 * quarter / full:.0f}% of the penalty for a")
    print("  completely different song.")
    if quarter < 0.25 * full:
        print("\n  NOT READING THE AUDIO IN TIME. Sliding the same song by a quarter of a")
        print("  second barely registers, so what the model takes from the audio is close")
        print("  to a global impression -- which is why neither more data nor a better")
        print("  encoder moved conditioning. The cross-attention RoPE mismatch is the")
        print("  thing to fix: see the module docstring.")
    elif quarter > 0.75 * full:
        print("\n  READING THE AUDIO IN TIME. A quarter-second slip costs nearly as much")
        print("  as the wrong song, so alignment is working and the low conditioning")
        print("  number means something else.")
    else:
        print("\n  PARTIALLY TIME-ALIGNED. Some of the signal is positional and some is")
        print("  not; worth repeating per shift before concluding.")


if __name__ == "__main__":
    main()
