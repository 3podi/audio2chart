"""Does emitting a chord make the same chord more likely again?

The pad collapse was found this way: feed the model a synthetic history, read the next-token
distribution straight off the logits, and compare against what the corpus does. Two notes in
its own context dropped P(pad) from 0.90 to 0.09 and it could never rest again.

A generated chart now shows the same shape in a different token. Ticks 0-1739 of a
regenerated LESBIAN VAMPIRE KILLERS are 46 consecutive GRYBO tap chords, one every 30 ms --
five frets at once, thirty-three times a second. Robert: "position 401 - 2189 is impossible".
Human tapping charts use five-fret chords 0.223% of the time; that chart used them 3.12%.

A temperature sweep could not separate cause from luck: 0.90 produced 10.75% four-note
chords where 1.00 produced 0.74%, which is not a curve, it is noise on a process that
sometimes locks. This asks the model directly instead.

If a run of one chord drives its own probability up, no sampling setting fixes it -- the
fault is in training, exactly as the pad weight was.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chart.tokenizer import SimpleTokenizerGuitar
from inference.engine import Charter

LENGTHS = (0, 1, 2, 4, 8, 16)

# Measured over 120 human tapping charts, 453,989 note positions.
HUMAN_CHORD_SHARE = {1: 0.94439, 2: 0.03589, 3: 0.01258, 4: 0.00492, 5: 0.00223}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Charter.from_pretrained(str(args.model)).to(device).eval()
    tok = SimpleTokenizerGuitar(expressive=True)

    # The tokens for a GRYBO tap chord and a plain green tap, via the same composition the
    # tokenizer uses: (chord * N_FLAG + flag) * N_SUSTAIN + sustain, flag bit 1 = tap.
    chord_index = {frets: index for index, frets in tok.reverse_chord.items()}
    grybo = chord_index.get((0, 1, 2, 3, 4))
    green = chord_index.get((0,))
    if grybo is None or green is None:
        raise SystemExit("could not locate the chord tokens in this vocabulary")
    grybo_tap = tok.compose(grybo, flag=2, sustain=0)
    green_tap = tok.compose(green, flag=2, sustain=0)

    values, mask = model._read_audio(str(args.audio), device)
    window = int(float(model.config.window_seconds) * 24000)
    values, mask = values[..., :window], mask[..., :window]
    with torch.no_grad():
        encoded = model.encoder.encode(values, mask, bandwidth=3.0)
        codes = encoded.audio_codes.squeeze(0)
        audio = sum(model.transformer.codes_embedding[i](codes[:, i]) for i in range(4))
        audio = model.transformer.norm_audio(audio)
        if model.transformer.compression:
            audio = model.transformer.audio_compression(audio)

    bos = model.config.bos_token_id

    def distribution(history: list[int]) -> torch.Tensor:
        ids = torch.tensor([[bos] + history], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model.transformer(ids, audio)[:, -1, :].float()
        return torch.softmax(logits / args.temperature, dim=-1)[0]

    def chord_mass(probabilities: torch.Tensor) -> dict[int, float]:
        """Total probability on each chord size, so rare-token mass is visible."""
        sizes: dict[int, float] = {}
        for token in range(tok.n_notes):
            index = token // (tok.N_FLAG * tok.N_SUSTAIN)
            lanes = tok.reverse_chord.get(index, ())
            playable = [lane for lane in lanes if lane <= 4]
            if playable:
                sizes[len(playable)] = sizes.get(len(playable), 0.0) + \
                    float(probabilities[token])
        total = sum(sizes.values()) or 1.0
        return {size: value / total for size, value in sizes.items()}

    print(f"{args.model}   temperature {args.temperature}\n")
    print("P(this exact GRYBO tap again), after a history of them")
    print(f"{'history':<26}{'P(same)':>10}{'5-note mass':>14}")
    for count in LENGTHS:
        probabilities = distribution([grybo_tap] * count)
        mass = chord_mass(probabilities)
        label = f"{count} GRYBO taps" if count else "just bos"
        print(f"  {label:<24}{float(probabilities[grybo_tap]):>10.4f}"
              f"{mass.get(5, 0.0):>13.3%}")

    print(f"\n  human 5-note share for comparison: {HUMAN_CHORD_SHARE[5]:.3%}")

    print("\nfor contrast, the same after plain green taps")
    for count in LENGTHS:
        probabilities = distribution([green_tap] * count)
        mass = chord_mass(probabilities)
        label = f"{count} green taps" if count else "just bos"
        print(f"  {label:<24}{float(probabilities[green_tap]):>10.4f}"
              f"{mass.get(5, 0.0):>13.3%}")

    # Judge on the chord-size mass, not on the ratio of one token's probability.
    #
    # A first attempt printed "two of them raise the odds 18033369244x", which is a
    # meaningless number: the denominator is a probability the model rounds to zero from a
    # clean context, so the ratio reports floating-point noise rather than an effect. The
    # mass on 5-note chords is bounded, comparable to the corpus, and says the same thing
    # without inventing precision.
    clean = chord_mass(distribution([]))
    locked = chord_mass(distribution([grybo_tap] * 16))
    human = HUMAN_CHORD_SHARE[5]
    print()
    print(f"  5-note mass: {clean.get(5, 0.0):.3%} from a clean context, "
          f"{locked.get(5, 0.0):.3%} after sixteen, human {human:.3%}")
    if locked.get(5, 0.0) > 10 * human:
        print("  SELF-REINFORCING. A run of these drives their own probability far past")
        print("  anything the corpus does, so no sampling setting fixes it -- the same")
        print("  shape as the pad collapse, where the fix was in training.")
    elif locked.get(5, 0.0) > 2 * human:
        print("  Mildly self-reinforcing: a run raises their odds, but within reach of")
        print("  top_k or a lower temperature.")
    else:
        print("  Not self-reinforcing. Long runs would be sampling luck, not a trap.")


if __name__ == "__main__":
    main()
