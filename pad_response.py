"""Will this model ever stop playing?

The fastest and most decisive check on a charting checkpoint, and the one that found the
density fault on 2026-09-11. It needs one forward pass per row and no generation at all.

Feed the model a history and read P(pad) straight off the logits. A healthy model should
keep a meaningful probability of silence whatever it has just played; both runs of
2026-09-11 did not:

    history                  model    corpus
    just bos                 0.904       --
    2 notes                  0.092    0.751
    4 alternating            0.003    0.398

Two consecutive notes collapsed silence tenfold and it never recovered, so generation
saturated and the charts never rested -- 0.35-0.85 rests a minute against a human 15.1.
The suspected cause is `model.pad_class_weight`, which down-weights pad to 0.1 in the
loss while pad is 86% of all targets.

Read the alternating row first. It is what an ordinary sparse chart's history looks like,
and if P(pad) is near zero there the model cannot produce one, whatever the sampler does.
No chart metric tells you that as directly, and generating six charts to find out costs
forty minutes instead of thirty seconds.

The `in data` column is the corpus answer to the same question, measured over 400 tapping
charts and 8.9 million grid slots. Against it the 2026-09-11 runs are not slightly
miscalibrated, they are inverted:

    history                  data     model
    2 consecutive notes     0.7510   0.0915
    4 consecutive notes     0.4076   0.0036
    32 consecutive notes    0.0078   0.0041
    4 alternating           0.3981   0.0032

Note the 32-note row, where model and data agree. The model has collapsed the whole
conditional distribution onto its dense-run mode: after two notes it answers as though it
had seen thirty-two. That is the shape of the fault, and a fix has to move the short
histories without disturbing the long one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from inference.engine import Charter

LENGTHS = (0, 1, 2, 4, 8, 16, 32)

# What the corpus itself says, measured over 400 tapping charts and 8.9M grid slots at
# grid_ms 10 -- the same binning the model is trained on. Overall P(pad) is 0.834.
#
# Printed beside the model's numbers because without it the table invites guesswork. The
# 2026-09-11 runs looked merely "low" until this column existed; against it they are
# giving the opposite answer, most starkly on the alternating row where the data says
# silence is 93% likely and the model says 0%.
DATA_AFTER_NOTES = {1: 0.9645, 2: 0.7510, 4: 0.4076, 8: 0.1728, 16: 0.0888, 32: 0.0078}

# Alternating references, measured at the SAME length the probe uses. A single figure
# will not do: P(pad) after strict alternation falls steeply with how long the
# alternation has run, because a long regular passage is a dense one.
#
# CORRECTION. This was one number, 0.9261, printed beside every alternating row. It was
# measured on a six-slot lookback whose last slot was a NOTE, while the probe builds
# histories whose last slot is a PAD. Those are opposite questions -- "does the
# alternation continue" (yes, 93% of the time) against "does this gap extend" (5-40%) --
# and the mismatch made every model tested look roughly a hundred times worse on these
# rows than it is.
DATA_ALTERNATING = {4: 0.3981, 8: 0.1986, 16: 0.1103, 32: 0.0499}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, help="Exported model directory")
    parser.add_argument("audio", type=Path, help="A clip at least one window long")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Reported alongside the raw probability, because sampling "
                             "sharpens: 0.58 raw becomes 0.90 at 0.5.")
    parser.add_argument("--note", type=int, default=100,
                        help="Token id to use as 'a note' in the synthetic history.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Charter.from_pretrained(str(args.model)).to(device).eval()
    pad_id = model.config.pad_token_id
    bos_id = model.config.bos_token_id

    values, mask = model._read_audio(str(args.audio), device)
    window = int(float(model.config.window_seconds) * model.config.sample_rate
                 if hasattr(model.config, "sample_rate")
                 else float(model.config.window_seconds) * 24000)
    if values.size(-1) < window:
        raise SystemExit(f"Audio must be at least {window / 24000:.0f}s long")
    values, mask = values[..., :window], mask[..., :window]

    with torch.no_grad():
        encoded = model.encoder.encode(values, mask, bandwidth=3.0)
        codes = encoded.audio_codes.squeeze(0)
        audio = sum(model.transformer.codes_embedding[i](codes[:, i]) for i in range(4))
        audio = model.transformer.norm_audio(audio)
        if model.transformer.compression:
            audio = model.transformer.audio_compression(audio)

    def probability(history: list[int]) -> tuple[float, float]:
        ids = torch.tensor([[bos_id] + history], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model.transformer(ids, audio)[:, -1, :].float()
        raw = torch.softmax(logits, dim=-1)[0, pad_id].item()
        hot = torch.softmax(logits / args.temperature, dim=-1)[0, pad_id].item()
        return raw, hot

    print(f"{args.model}\n")
    print(f"{'history':<30}{'P(pad)':>10}{f'  at t={args.temperature}':>12}"
          f"{'in data':>10}")

    for count in LENGTHS:
        raw, hot = probability([args.note] * count)
        reference = DATA_AFTER_NOTES.get(count)
        shown = f"{reference:>10.4f}" if reference is not None else " " * 10
        print(f"  {f'{count} notes' if count else 'just bos':<28}"
              f"{raw:>10.4f}{hot:>12.4f}{shown}")

    print()
    for count in LENGTHS[1:]:
        raw, hot = probability([pad_id] * count)
        print(f"  {f'{count} pads':<28}{raw:>10.4f}{hot:>12.4f}")

    print()
    alternating_raw = {}
    for count in sorted(DATA_ALTERNATING):
        history = [args.note if index % 2 == 0 else pad_id for index in range(count)]
        raw, hot = probability(history)
        alternating_raw[count] = raw
        print(f"  {f'{count} alternating':<28}{raw:>10.4f}{hot:>12.4f}"
              f"{DATA_ALTERNATING[count]:>10.4f}")

    # Judge the RAW column against the corpus, not the temperature-scaled one.
    #
    # An earlier version compared the t=0.5 value against a corpus frequency, which is
    # apples to oranges: the corpus number is a plain rate, and temperature reshapes the
    # distribution before anything is sampled. It made a well-calibrated checkpoint read
    # as WEAK -- its raw alternating-32 was 0.066 against a corpus 0.050, i.e. slightly
    # *over*, while the t=0.5 value of 0.0074 looked like a 7x shortfall.
    #
    # Calibration and sampling are separate questions and this tool answers the first.
    # What the sampler then does with a correct distribution is reported below it, because
    # sharpening a small probability makes it very small: at pad_class_weight=1.0 the
    # model rests 5.0 times a minute at temperature 1.0 and not once at 0.5.
    raw_two, hot_two = probability([args.note] * 2)
    ratios = {count: alternating_raw[count] / DATA_ALTERNATING[count]
              for count in alternating_raw}
    worst = min(ratios.values())
    two_ratio = raw_two / DATA_AFTER_NOTES[2]
    print()
    print("  calibration against the corpus (raw, before any sampling):")
    print(f"    after two notes {two_ratio:.2f}x, alternating "
          + ", ".join(f"{count}:{ratio:.2f}x" for count, ratio in sorted(ratios.items())))
    if worst < 0.1 or two_ratio < 0.1:
        print("  FAIL: silence is an order of magnitude less likely than the corpus says.")
        print("        Expect saturated output that never rests.")
    elif worst < 0.5 or two_ratio < 0.5:
        print("  WEAK: silence is suppressed but not absent. Expect over-dense output,")
        print("        worse the longer the model has been playing.")
    else:
        print("  OK: silence tracks the corpus. Density is a sampling question now --")
        print(f"       note temperature 0.5 turns P(pad)={raw_two:.3f} into {hot_two:.3f}"
              " after two notes,")
        print("       so a sharpened sampler can still suppress rests the model would take.")


if __name__ == "__main__":
    main()
