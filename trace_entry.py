"""How does a chart get INTO a chord run?

chord_response answered the second half of the question -- once sixteen GRYBO taps are in
the history, 41.8% of the note mass is five-fret chords and the run sustains itself. It
did not answer the first half. From a clean context P(GRYBO tap) is 0.000000, and it stays
0.000000 under guidance 1.5 and 3.0, so neither the model nor the amplifier opens a chart
with one on purpose.

Yet the regenerated LESBIAN VAMPIRE KILLERS opens with one and holds it for 46 positions.

This traces a real generation of that song's first chunk with the settings build_playtest
uses -- temperature 1.0, top_k 32, guidance 1.5 -- and records, at the step where each
chord size is first emitted, the probability and rank the sampler gave it. If entry is a
rare draw from the tail, the number will be small and the rank high; that separates
"the model wants this" from "the model allowed it once and then could not get out".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

import torch
import torch.nn.functional as F

from chart.tokenizer import SimpleTokenizerGuitar
from inference.engine import Charter

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("model", type=Path)
parser.add_argument("audio", type=Path)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top-k", type=int, default=32)
parser.add_argument("--guidance", type=float, default=1.5)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

AUDIO = args.audio
TEMPERATURE, TOP_K, GUIDANCE, SEED = (
    args.temperature, args.top_k, args.guidance, args.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Charter.from_pretrained(str(args.model)).to(device).eval()
tok = SimpleTokenizerGuitar(expressive=True)
pad_id = model.config.pad_token_id

size_of = {}
for token in range(tok.n_notes):
    lanes = tok.reverse_chord.get(token // (tok.N_FLAG * tok.N_SUSTAIN), ())
    playable = [lane for lane in lanes if lane <= 4]
    if playable:
        size_of[token] = len(playable)

values, mask = model._read_audio(str(AUDIO), device)
window = int(float(model.config.window_seconds) * 24000)
values, mask = values[..., :window], mask[..., :window]
with torch.no_grad():
    encoded = model.encoder.encode(values, mask, bandwidth=3.0)
    codes = encoded.audio_codes.squeeze(0)
    audio = sum(model.transformer.codes_embedding[i](codes[:, i]) for i in range(4))
    audio = model.transformer.norm_audio(audio)
    if model.transformer.compression:
        audio = model.transformer.audio_compression(audio)
negative = model._contrast_audio(audio)

steps = int(float(model.config.window_seconds) * 1000 / model.config.grid_ms)
layers = model.transformer.n_layers
self_cache = [None] * layers
cross_cache = [None] * layers
negative_self = [None] * layers
negative_cross = [None] * layers

torch.manual_seed(SEED)
current = torch.tensor([[model.config.bos_token_id]], dtype=torch.long, device=device)
first_seen = {}
emitted = []

for step in range(steps):
    with torch.no_grad():
        logits, self_cache, cross_cache = model.transformer(
            current, audio, attention_mask=None, class_ids=None, step=step,
            use_cache=True, self_cache=self_cache, cross_cache=cross_cache)
        negative_logits, negative_self, negative_cross = model.transformer(
            current, negative, attention_mask=None, class_ids=None, step=step,
            use_cache=True, self_cache=negative_self, cross_cache=negative_cross)
    guided = logits[:, -1] + GUIDANCE * (logits[:, -1] - negative_logits[:, -1])

    # The sampler's own view, so the probability reported is the one it drew against.
    scaled = guided / TEMPERATURE
    top_values, _ = torch.topk(scaled, min(TOP_K, scaled.size(-1)))
    masked = scaled.masked_fill(scaled < top_values[..., -1:], -float("inf"))
    probabilities = F.softmax(masked, dim=-1)
    token = int(torch.multinomial(probabilities, num_samples=1))

    size = size_of.get(token)
    if size is not None:
        emitted.append((step, size))
        if size not in first_seen:
            order = torch.argsort(guided[0], descending=True)
            rank = int((order == token).nonzero()[0])
            first_seen[size] = (step, float(probabilities[0, token]), rank)

    current = torch.tensor([[token]], dtype=torch.long, device=device)

print(f"first chunk of {AUDIO.parent.name}")
print(f"temperature {TEMPERATURE}, top_k {TOP_K}, guidance {GUIDANCE}, seed {SEED}\n")
print(f"{'chord size':<12}{'first at step':>14}{'P at that step':>16}{'rank':>7}")
for size in sorted(first_seen):
    step, probability, rank = first_seen[size]
    print(f"  {size} note(s){'':<2}{step:>14}{probability:>16.5f}{rank:>7}")

counts = {}
for _, size in emitted:
    counts[size] = counts.get(size, 0) + 1
total = sum(counts.values()) or 1
print(f"\n{total} notes in this chunk")
for size in sorted(counts):
    print(f"  {size} note(s)  {counts[size]:>5}  {100 * counts[size] / total:>6.2f}%")

big = [step for step, size in emitted if size >= 4]
if big:
    print(f"\n  4+ note chords at steps {big[0]}..{big[-1]}, {len(big)} of them")
    print("  A high rank and a small probability at first sight means the sampler let one")
    print("  through from the tail; the run after it is the model refusing to leave.")

# RESULT, on the chunk that Robert reported as impossible to play:
#
#   chord size   first at step  P at that step   rank
#     1 note(s)            126         0.00469      3
#     2 note(s)             12         0.00652     19
#     3 note(s)             74         0.03102      5
#     5 note(s)             24         0.18352      1
#
# The two-note chord at step 12 came in at rank 19 on probability 0.0065 -- the sampler
# reaching into the tail, which is what top_k is for and what it is expected to do
# occasionally. Twelve steps later a five-fret chord is the model's *rank-1* token at
# probability 0.184, against a corpus that uses five-fret chords 0.223% of the time.
#
# So no sampling setting fixes this. One unlikely token puts the history somewhere the
# model was never trained, and the predictions there are unrelated to the audio. The fix
# is in modules/trainer.py:_corrupt_inputs -- see the docstring there.
