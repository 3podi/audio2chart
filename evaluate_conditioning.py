"""Measure how much a checkpoint actually uses the audio.

Note accuracy cannot answer this. It is dominated by padding -- roughly 86% of slots
in a window are pad, an all-pad baseline already scores 85.6%, and a model that
ignored the audio entirely would still look respectable. The question "is this chart
about this song?" needs a different measurement.

The test here is an ablation: score a batch twice, once with each sequence's own
audio and once with a neighbour's, and report what the swap costs. That difference
is the part of the prediction the audio is responsible for. Everything else -- the
note history, the pattern vocabulary, the rhythm prior -- cancels out.

Three things this has to get right, each of which produced a confidently wrong
number first:

*Batches need at least two sequences.* The swap is a permutation within a batch, so
a batch of one silently measures nothing and reports a delta of exactly zero.

*The permutation must be a derangement.* torch.randperm on a small batch returns the
identity a large fraction of the time -- half of all batches at size two -- and each
of those compares the audio against itself.

*Padding must be scored separately.* Pooled cross-entropy is ~86% padding, which the
audio has no bearing on, and it dilutes the effect about sevenfold. Reported here
both ways: the note-position number is the one that means anything.
"""

from __future__ import annotations

import argparse
import random

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

MIN_BATCH = 2
"""Below this the swap has nothing to permute between."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None,
                        help="Lightning .ckpt to score. Omit to score the pretrained "
                             "checkpoint the config names, i.e. before fine-tuning.")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--config-name", default="audio")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--songs", type=int, default=40)
    parser.add_argument("--trust", action="store_true",
                        help="Unpickle the checkpoint with weights_only=False. Only for "
                             "a checkpoint you produced yourself, never a downloaded one.")
    return parser.parse_args()


def load_config(config_dir, config_name):
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    with open_dict(cfg):
        cfg.loader.num_workers = 0
        cfg.loader.val_batch_size = MIN_BATCH
        cfg.loader.val_num_pieces = MIN_BATCH
    return cfg


def score(model, batch, pad_id, device):
    """(pooled, note-only) loss with the right audio and with a neighbour's."""
    audio = batch["input_values"].to(device)
    mask = batch["padding_mask"].to(device)
    notes = batch["note_values"].to(device)
    if audio.size(0) < MIN_BATCH:
        return None

    emb = model._encode_audio(audio, mask)
    inputs, targets = notes[:, :-1].contiguous(), notes[:, 1:].contiguous()
    flat = targets.reshape(-1)
    is_note = flat != pad_id
    weights = model.class_weights.to(device)

    out = []
    # roll is a derangement for any batch size; randperm is not.
    for embedding in (emb, torch.roll(emb, 1, dims=0)):
        logits = model.transformer(inputs, embedding, attention_mask=None, class_ids=None)
        per_token = F.cross_entropy(logits.reshape(-1, model.vocab_size), flat,
                                    weight=weights, reduction="none")
        out.append((per_token.mean().item(), per_token[is_note].mean().item()))
    return out


def report(tag, rows):
    if not rows:
        print(f"{tag}: measured nothing -- every batch was smaller than {MIN_BATCH}")
        return
    n = len(rows)
    for name, index in (("all positions", 0), ("note positions", 1)):
        correct = sum(r[0][index] for r in rows) / n
        swapped = sum(r[1][index] for r in rows) / n
        deltas = sorted(r[1][index] - r[0][index] for r in rows)
        hurt = sum(1 for d in deltas if d > 0)
        print(f"{tag:<26} {name:<15} own {correct:7.4f}  swapped {swapped:7.4f}  "
              f"delta {swapped - correct:+.4f} ({100 * (swapped - correct) / correct:+5.1f}%)  "
              f"median {deltas[n // 2]:+.4f}  hurt {hurt}/{n}")


def main():
    args = parse_args()
    import os
    import sys

    root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, root)
    from chart.tokenizer import SimpleTokenizerGuitar
    from main import build_dataloaders, load_data_splits
    from modules.pretrained import load_pretrained_transformer
    from modules.trainer import WaveformTransformerDiscrete

    cfg = load_config(args.config_dir or os.path.join(root, "configs"), args.config_name)
    tokenizer = SimpleTokenizerGuitar()
    train_files, val_files = load_data_splits(cfg)
    random.Random(0).shuffle(val_files)
    _, val_loader, _ = build_dataloaders(cfg, tokenizer, train_files[:args.songs],
                                         val_files[:args.songs])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WaveformTransformerDiscrete(
        vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_id,
        eos_token_id=tokenizer.eos_id, cfg_model=cfg.model, cfg_optimizer=cfg.optimizer)

    if args.checkpoint:
        if not args.trust:
            raise SystemExit(
                "Loading a Lightning checkpoint needs weights_only=False, which "
                "executes pickled code. Pass --trust if you produced this file.")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(dict(state["state_dict"]), strict=False)
        tag = os.path.basename(args.checkpoint)
    else:
        load_pretrained_transformer(model, cfg.model.pretrained, tokenizer, seed=cfg.seed)
        tag = "pretrained (no fine-tune)"

    model = model.to(device).eval()
    # Keep drawing until `--batches` scored batches exist, re-iterating the loader if it
    # runs dry. Asking for 100 used to yield 20: the validation loader exhausts well
    # before that, `report` prints len(rows), and every run came out "hurt 15/20"
    # whatever was requested -- so the metric could not be tightened, and the spread
    # across checkpoints (+0.8%, +1.0%, +1.5%) could not be told from noise.
    #
    # Each pass draws different windows from the same songs, so extra passes are fresh
    # samples rather than repeats.
    rows = []
    passes = 0
    with torch.no_grad():
        while len(rows) < args.batches and passes < 20:
            passes += 1
            drawn = 0
            for batch in val_loader:
                if len(rows) >= args.batches:
                    break
                drawn += 1
                result = score(model, batch, tokenizer.pad_id, device)
                if result is not None:
                    rows.append(result)
            if drawn == 0:
                break
    if passes > 1:
        print(f"  (loader exhausted after {len(rows) // passes} scored batches; "
              f"made {passes} passes to reach {len(rows)})")
    report(tag, rows)


if __name__ == "__main__":
    main()
