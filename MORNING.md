# Status

Branch `expressive-vocab-finetune`. Nothing pushed, `configs/audio.yaml` now edited
(deliberately, see below). 286 tests passing.

## Evening: the grid was throwing away the data we care about

**The 20 ms grid could not represent tapping charts.** `discretize_time` bins notes at
`grid_ms` and discards any window holding two in one bin. In tapping charts 22.8% of
note gaps are shorter than 20 ms, 99% of charts contain at least one, and only 18.2% of
30 s windows survived. Robert's own `yax03 - Down` reaches 7.3 ms.

Fixed to **10 ms at a 15 s window**, which is the same 1502 tokens and costs nothing:

    Hot N Cold     20ms/30s   1 of 7 windows clean,   484 notes kept, 6112 lost
                   10ms/15s  14 of 14 clean,         6596 kept,          0 lost
    corpus         20ms/30s  13,198 of 13,720 songs usable
                   10ms/15s  13,701 of 13,720

Also a sync fault on its own terms: +/-10 ms of quantisation where notes sat 18 ms
apart. Now +/-5 ms. This was the direct answer to Robert's question about whether the
model can line notes up.

A bug came with it: `inference/engine.py` hardcoded `chunk_sec = 30` while the trainer
read the window from config, so changing the grid would have had generation ask a
1502-token model for 3002 -- nonsense output, no error. The engine reads config now and
export refuses a config.json missing either timing field.

## Selecting training data by measuring it

Robert's idea: find charts where at least 80% of notes are taps, since that is what an
overchart is. It beats both metadata sources, measured over 700 charts:

    genre-tagged      median tap rate 0.964   100% are >= 80% taps
    folder-labelled   median tap rate 0.643    31%
    unlabelled        median tap rate 0.085     8%

The genre tag is accurate and rare. The folder labels are two thirds dilution -- the
2,908-chart manifest I was about to train on would have been mostly charts in another
style. And 8% of unlabelled charts qualify, which is several hundred no label catches.

## Catalogue, corrected from Robert's walkthrough

123 -> 100 entries, and the audit went from 5 absent / 9 rare to 1 absent / 2 rare.

- **Cakes were wrong.** Written as two trills (`G R G R O B O B`); they are a trill
  against the anchor fret repeated, then a descent back to it. Robert gave me the
  sequences yesterday and the file still had the wrong ones. Three of four started
  matching immediately once corrected.
- **Raked chimneys removed** -- Robert says they are the same thing as cakes.
- **Castles cut to one.** Only the green anchor is attested.
- **The twenty "anchor" entries moved to UNCONFIRMED_SHAPES.** Anchoring is holding the
  lowest fret through a passage, a technique, not a three-note shape -- and asked what
  those figures are called, Robert said he does not know what pattern that is.
- **H pattern corrected** to `RB RYB RB`, and a variant added from H-ell.
- **The skip added** -- the base pair is in 82% of tapping charts and was absent from a
  124-entry catalogue entirely.

## What the discovery tooling learned today

Segmenting on **beats, not measures**. Robert's "a pattern is a bar or two" meant a
beat or two; measures are four times too wide and returned several figures concatenated.
Figures that appeared in 3-11 charts under measure windows appear in 100-160 of 262
tapping charts under beat windows.

Chord patterns are visible now. The H surfaced unprompted as `GO GYO GO` in 23 charts,
and the chorded descent Robert named is in 70. Chording is not a general transformation
-- 70 of 4,618 alternating shapes have a chorded counterpart -- but where it applies it
dominates: an eight-note anchored descent appears in 7 charts alternating and 49
chorded.

---

Branch `expressive-vocab-finetune`. 13 commits over the night and day, nothing pushed,
no PRs opened, `configs/audio.yaml` untouched. 242 tests passing.

## Three faults, separated

The Paparazzi chart was bad for three independent reasons. Each needs a different fix,
and treating them as one problem is why the diagnosis took so long.

**1. It is missing most of the chart vocabulary.** Zero chords, zero taps, 0.1%
sustains, against human medians of 0.128, 0.194 and 0.049. Not the model's fault --
teacher-forced it predicts all three at near-human rates (taps 0.181 against a true
0.197), and the decode/writer path is bit-exact. The cause is exposure bias in the
decoding loop: the tap rate runs away 0.224 -> 0.723 across one 30s chunk as the model
conditions on its own output, and collapses to a flat zero at a lower temperature. No
sampler setting fixes it. *Fix staged: `model.input_noise`, off by default.*

**2. It does not follow the music.** Audio conditioning is real and consistent but
weak: swapping in another song's audio costs +0.1891 nats on note positions and hurts
on 20 of 20 batches, while the note-history prior carries the rest. *Fix staged:
`freeze_layers: 4` gives 45% more trainable parameters in the blocks that read audio.
Classifier-free guidance is being measured now.*

**3. It does not phrase, and it overspends the hard patterns.** A human Expert chart
breathes constantly -- 34 rests a minute, typically 0.38s, a third of the song at rest,
and only 1 of 120 charts has none. The generated chart rests 1.1 times a minute for a
median 5.82s: 31x fewer, 15x longer, about the same total silence arranged completely
differently. It plays continuously and then stops dead. And Expert-marker patterns make up 14.9% of its pattern
instances against a human 2.7% (+3.23 sd), where a charter spends one for emphasis.
Real Expert charts are ordinary vocabulary that becomes hard in places. *No fix
staged.*

Two things I claimed here first and had to retract, both from judging a number without
a baseline: the chart's density is *more* variable than 93% of human charts (0.606
against a human mean 0.409), not flat; and its 235-note mean run is normal for Expert,
whose own mean is 232 -- I cited it as a defect against a measurement I had taken that
morning. The long runs are a consequence of the phrasing, not a separate problem.

I also first wrote fault 3 as "no gap exceeds 250 ms anywhere, not rare, none". There
are about three; they printed as 0.00 in a two-decimal histogram. Three overstatements
in one fault, each caught only by measuring the human baseline -- worth remembering
before acting on any "it never does X" in these notes.

## Guidance does not work (36 songs, settled)

Classifier-free guidance with a wrong-song negative gives +0.042 +/- 0.045 on
responsiveness -- 95% CI [-0.046, +0.129], sign test p = 0.405, helped 21 of 36 -- and
doubles note density as a side effect (3575 -> 7087 notes). It is not a fix for fault 2.

That is consistent with the conditioning ablation: you cannot amplify a signal the
model barely uses. The audio has to be strengthened in training, not at inference.

**The metric is weaker than its validation suggested.** Human charts beat generated
ones by only +0.081 +/- 0.047, 95% CI [-0.012, +0.174] -- marginally non-significant at
n=36, despite the ablation independently showing weak audio use. Separating generated
from human reliably would need ~130 songs. I said at n=7 that "generated charts score
near zero while humans track the music"; at full sample that is directional, not
established.

## CORRECTION: most of the above came from one chart, and one chart lied

Eight charts generated from eight different songs, scored the same way:

    metric        mean   median    human
    chord        0.060    0.043    0.128
    tap          0.195    0.224    0.194
    sustain      0.082    0.000    0.049
    nps         25.301   26.297    7.813
    lift         0.072    0.007    0.209
    rests       11.171    1.192   34.000

**Taps are normal.** 0.224 median against a human 0.194. "Zero taps" was true of the
Paparazzi chart alone. Withdrawn.

**Pattern vocabulary is NOT human-level.** Median lift +0.007 -- at chance -- against a
human +0.209. The Paparazzi chart's +0.230 was an outlier, and "the model has learned
to write real guitar, every fault is about placement" was wrong. Withdrawn.

**Density is 3.2x too high** (25.3 nps against 7.8), which I had backwards: the
Paparazzi chart was too sparse at 5.5, so I recorded sparseness as the defect.

What survives and is now properly established: sustains collapse (median 0.000, though
one chart reached 0.655, so it is bimodal), phrasing is badly broken (median 1.19 rests
a minute against 34), and chords run about a third of human.

The real lesson is the spread. Across eight charts, sustain ran 0.000 to 0.655, pattern
lift -0.172 to +0.383, rests 0.0 to 80.3 a minute. Output quality varies enormously by
song, so no single chart characterises this model -- including the one that started
this investigation.

## The good news, reduced

The model reaches human-level pattern vocabulary on its best charts -- one scored
+0.383 and another +0.340, above the human mean of +0.209 -- so the capability exists.
It just is not reliable: the median chart sits at chance. That is a weaker and more
useful claim than the one it replaces.

## What I would run next, and why it is one experiment

Faults 1 and 3 may be the same defect. Both are the model behaving differently on its
own output than on real history: the vocabulary collapse (zero taps, zero sustains)
and the phrasing failure (1.1 rests a minute against 34) are both the free-running
model failing to make a decision it makes correctly when teacher-forced. The pad-vs-note
decision that suppresses rests is the same one that suppresses chords and taps.

If that is right, `input_noise` improves both together, and that is a falsifiable
prediction worth testing before building anything more elaborate. If taps come back and
phrasing does not, they are separate problems and fault 3 needs its own fix.

One run tests it, and `evaluate_chart.py` reads the answer off in one command.

## Ready for the next run

    python main.py --config-name audio       loader.train_num_pieces=1 trainer.accumulate_grad_batches=8       model.freeze_layers=4 model.input_noise=0.08

Smoke-tested end to end: 0.449 s/batch, 3.6s per optimizer step against the current
config's 30.4s, loss falling 3.82 -> 2.99, no OOM, and it writes nothing near the good
checkpoint. `model.input_noise` did not work the first time -- Hydra refuses to
override a key the config does not declare, and reading it with a default is not the
same as declaring it. Now declared, with tests.

An epoch drops from ~22 hours to under 3. `evaluate_conditioning.py` scores fault 2 on
a checkpoint, `evaluate_chart.py` scores faults 1 and 3 on a chart.

## Answering the difficulty question

Difficulty is qualitative, not density. Over 120 songs charted at all four levels by
the same charter, notes per second rises only 1.8x across the whole ladder, while mean
run length is flat through Hard and then quadruples at Expert, where the tap rate also
jumps 72%. Chords peak at HARD. Sustains run backwards.

Catalogue patterns are now graded from the corpus rather than by hand -- 165,792
occurrences, in `chart/grades.py`. Read against the null of 2.76, not 2.5: Expert
sections are denser, so every pattern carries an Expert bias for free. Corrected, only
13 of 67 patterns mark difficulty at all; the rest is universal vocabulary.

Together those say difficulty comes from *arrangement*, not from exotic pattern
selection -- which argues for JM's annotated-section registry over a scalar knob.

**One question for Robert:** `trip zig split G-Y-O` grades +1.02 above null and `trip
zig split G-B-O` grades -0.29 below it. Same family, opposite ends. G-Y-O splits evenly
(deltas 2, 2), G-B-O does not (3, 1). Is split regularity what makes these hard, or are
47 occurrences of G-B-O too few to trust?

---


# Detailed record


Branch `expressive-vocab-finetune`. Nothing pushed, no PRs opened. Six local commits.

## The headline: two suspects cleared, one real cause found

**`freeze_layers: 8` did not break the audio conditioning.** Fine-tuning *increased*
it. Swapping a batch's audio for a neighbour's costs, on note positions only:

| | own audio | swapped | delta | hurt |
|---|---|---|---|---|
| released, pre-finetune | 4.9165 | 5.0350 | +0.1184 (+2.4%) | 20/20 |
| ours, b6000 | 1.4639 | 1.6530 | +0.1891 (+12.9%) | 20/20 |

**There is no train/inference audio mismatch.** The pre-decoded `.raw` training path
and the engine's librosa+processor path produce the same waveform (cosine ~1.0, gain
1.0, residual 0.3-3% from decoder differences alone).

So the model does use the audio, consistently. It is just weak next to the note-history
prior: per-note perplexity 4.32 with the right audio, 5.22 with the wrong song's. A
model that would rather continue a plausible pattern than follow the music produces
exactly the mechanical charts you saw.

**What was actually broken: we were holding 3.4 GiB of VRAM for a gradient that cannot
exist.** `configs/model/audio_discrete.yaml` had `freeze_encoder: false`, which reads
like it enables encoder training. It cannot -- the discrete path returns quantiser
indices (`int64`, `grad_fn None`), and gradient does not flow through an integer. So
every forward built and retained a full Encodec activation graph nothing backpropagated
through. Measured 4.173 GiB against 0.807 GiB under `no_grad`, byte-identical codes.

That is the VRAM pressure that made me introduce `freeze_layers: 8` in the first place.
It cost decoder capacity to buy nothing.

## Your tempo rule, done and verified

`chart/tempo.py`. Detected 140.006 BPM on the Paparazzi file; first audio note at
0.546s; anchor beat at 0.574s = tick 480, exactly one beat at resolution 480; lead-in
BPM 104.611 so the lead-in lands on the audio's first note. First generated note sits
at tick 480, on the beat line carrying the tempo event.

The anchor is the *tracked beat* nearest the first onset, not the onset itself, so a
note played slightly ahead of the beat does not drag the grid. Beat tracking never
identifies downbeats, so a beat line is the strongest claim the analysis supports --
which is what you specified.

Exact sync is impossible: `.chart` stores BPM as integer milli-BPM. Residual is ~1.5
microseconds, four orders of magnitude under the 20 ms grid.

## New tools

- `evaluate_conditioning.py` -- repeatable audio-ablation, so any future run can be
  checked against the numbers above.
- `chart/responsiveness.py` -- does a chart follow the music? Correlates per-second
  note density against the onset envelope.
- `--guidance` on `generate.py` -- classifier-free guidance, amplifying the one term
  the ablation says exists. Negative branch is real audio, not a zero vector: this
  model never saw audio dropout, so a null input is off-distribution.

## Three mistakes I made and caught

1. First ablation printed `delta +0.0000` for both models. `val_batch_size = 1`, and
   the swap needs two sequences to permute between, so it skipped every batch and
   reported zeros rather than saying it measured nothing.
2. Second version used `torch.randperm`, which returns the identity half the time at
   batch size two -- half the batches compared the audio against itself.
3. I scored loss over all positions. ~86% are padding, which diluted the effect ~7x
   and made me report "fine-tuning changed nothing" when it had roughly doubled it.

Also: I briefly thought the training and inference waveforms were uncorrelated
(cosine -0.001). That was me feeding an MP3 to a raw-PCM reader.

## Metric honesty

`chart/responsiveness` is validated over 60 full human charts: +0.248 against own
audio, +0.054 against a stranger's, winning 46/60. But per-song sd is 0.284, wider
than the effect -- on one song in four a real human chart scores worse against its own
audio than a stranger's. A single song's score is noise. `minimum_songs(0.1) == 33`.

I had started a three-song guidance sweep. By that analysis it could not have concluded
anything, so I stopped it and replaced it with a 36-song paired version.

## Running overnight

1. VRAM sweep -- a real training step at `freeze_layers` 8 / 4 / 0, batch 1 and 2, to
   see what the freed 3.4 GiB buys. Result in `G:\a2c_data\` task output.
2. 36-song paired guidance sweep, resumable, appending to
   `G:\a2c_data\guidance_sweep.jsonl`, log at `G:\a2c_data\guidance_sweep.log`.

## Retracted: precomputing Encodec codes is not the fix

I had this queued as the big throughput win, carried from an old estimate that the
step was ~9.8s against ~0.2s of model math. Profiled properly at freeze_layers 0:

    dataloader fetch (decode + window)   0.292s    2.0%
    encodec encode                       0.246s    1.6%
    transformer fwd+bwd+step            14.436s   96.4%
    total                               14.973s

Precomputing the codes would save 1.6%. Data loading is already fine -- the lazy LRU
cache and four workers did their job. The whole cost is the transformer's own
forward+backward, which is absurd for a 230M model at batch 1.

The config sets no precision at all, so Lightning runs 32-true and TF32 is never
enabled. Benchmarking fp32 / TF32 / bf16 now; that is where the time is.

## Confirmed since the first draft

- `freeze_layers: 0` runs in the *real* pipeline, not just synthetically: 48 batches,
  no OOM, 7938 MiB steady, loss stepping normally. The good b6000 checkpoint is
  untouched.
- `--snap` now works. It rounds against the song's beat grid, and until the tempo fix
  there was no grid to round against. At 140 BPM a generated eighth-note run came out
  246/247/224 ticks against a true 240; snapping fixes that without moving the first
  note off its beat line.
- Second effect of the `freeze_encoder` bug: `audio_encoder.eval()` was never called,
  so Encodec ran in *train* mode for the whole 24-hour run. It is deterministic now.

## The throughput answer: we have been paging VRAM all along

`train_num_pieces: 2` puts two 30-second windows in every batch. Measured:

    seqs 1  freeze 8    0.315s/step   peak  5.905 GiB
    seqs 2  freeze 8    6.194s/step   peak  9.476 GiB
    seqs 2  freeze 0   10.424s/step   peak 11.040 GiB

The card has 8 GiB. Two sequences need 9.5 GiB, and Windows' WDDM driver does not
refuse -- it oversubscribes into host RAM, so instead of an out-of-memory error the
step runs 19.7x slower. That is why every component measured fine in isolation: each
one is fine, until the working set crosses 8 GiB and every access starts crossing
PCIe.

Gradient accumulation gives the same effective batch without the cliff:

    now       pieces 2 x accum 4   ->  4 x 6.194s = 24.8s per optimizer step
    proposed  pieces 1 x accum 8   ->  8 x 0.315s =  2.5s per optimizer step

Same eight sequences per update, about 10x faster. An epoch goes from roughly 22
hours to roughly 2.2.

Two genuine throughput bugs fixed on the way, both cheap operations made expensive by
how they touch the GPU, and neither visible in the source:

- The audio encoder ran under bf16 autocast, where cuDNN has no fused LSTM kernel, so
  SEANet's 2250-step recurrence fell back to an unfused path: 0.130s -> 0.712s. Worth
  about 2.3s/batch in practice.
- LogGradientNorm summed `.item()` per parameter, one GPU->CPU sync each, ~470 stalls
  per optimizer step to log a single number: 1.09s, 2.5% of training time.

Things I chased that turned out not to matter, so nobody repeats them: precomputing
Encodec codes (1.6% of a step), the dataloader (0.034%), per-window chart
re-processing (0.004s), and the optimizer (19ms -- the profiler's 8.57s
`optimizer_step` double-counts the closure's forward and backward).

The dead `_audio_cache` is still a real bug -- it is read but never written, so
`max_cache_gb` and the whole chunk-eviction system control nothing. It is not a
throughput problem, and switching it on would risk 6 GiB per worker across 4 workers,
which is the host-RAM blowup from earlier in this project. It should be deleted, not
activated.

## 11 Sep: the charts are missing most of the vocabulary, and it is not the model

Separate fault from the audio-conditioning one, and probably the bigger part of why
the Paparazzi chart looked the way it did.

Profiled against the corpus, the generated chart had **zero chords, zero tap notes and
0.1% sustains** where human charts have 12.7%, 47% and 22.9%, plus 6x too many opens,
2x too many greens, a mean run of 235 notes and 99.7% of its note spacings in two
adjacent buckets. That is "mechanical" made measurable.

The model is not the problem. Teacher-forced it is well calibrated:

    chord    true 0.205   model mass 0.171
    tap      true 0.197   model mass 0.181
    sustain  true 0.071   model mass 0.091

And the output path is lossless -- a human chart round-tripped through
convert_notes_to_ticks -> decode -> fill_expert_single comes back bit-identical
(tap 0.993 -> 0.993).

**It is exposure bias in the decoding loop.** Across one 30s chunk at temperature 1.0
the tap rate climbs 0.224 -> 0.454 -> 0.671 -> 0.723 as the model conditions on more of
its own output. At temperature 0.5 it is a flat 0.000 with no sustains at all:
sharpening suppresses the runaway and the vocabulary together. A pad-logit bias, which
should have separated density from vocabulary, collapses output 22-fold over one unit
and suppresses chords and taps preferentially -- the signature of a self-reinforcing
loop, where what the model emits becomes the history it reads.

So no sampler setting fixes this, and the three I tried are three knobs on one loop.
The fix is in training -- scheduled sampling, or simply more of it now that a run is
~10x faster. `--pad-bias` is kept as a no-op-by-default diagnostic with the numbers in
its docstring.

Also settled: universal rule 1 (no note overlapping) is real -- zero violations in
836,327 human notes across 300 charts. Our output satisfies it, but vacuously, because
it emits almost no sustains.

## Decisions waiting for you

- **The next training run's config.** Measured end to end in the real pipeline:

        loader.train_num_pieces: 1          (was 2)
        trainer.accumulate_grad_batches: 8  (was 4)
        model.freeze_layers: 4              (was 8)

  Same eight sequences per optimizer step, about 9.4x faster (3.2s against 30.4s),
  and 45% more trainable parameters -- layers 4-7, the lower blocks feeding the audio
  cross-attention. Epoch time goes from roughly 22 hours to under 3.

  | pieces 1 x accum 8 | per batch | per update | trainable | VRAM |
  |---|---|---|---|---|
  | freeze 8 | 0.380s | 3.0s  | 121.1M | 6259 MiB |
  | freeze 4 | 0.404s | 3.2s  | 175.7M | ~7950 MiB |
  | freeze 0 | 2.0s   | 16.8s | 230.2M | 7947 MiB, paging |
  | *current* | *7.6s* | *30.4s* | *121.1M* | *7950 MiB, paging* |

  freeze 4 held 1.79 -> 2.00 it/s rising over 25 optimizer steps with flat VRAM, so
  it is not fragmenting. freeze 0 has the most capacity and is 5x slower: past the
  8 GiB ceiling, capacity is paid for in PCIe round-trips rather than parameters.
  If a long run ever does degrade, freeze 8 is the safe fallback with 1.7 GB spare.

  I have not edited configs/audio.yaml. The numbers make the case but committing you
  to a multi-day run is your call.
- **Audio dropout for real CFG.** Current guidance uses a wrong-song negative because
  the model has no null-audio concept. Training with ~10% audio dropout would give it
  one and make guidance considerably stronger. Cheap to add, needs a retrain.
- Still open from before: PR #3 is far behind; the pattern questions (tremolo naming,
  `ladder descending split`, what unqualified "trip zig" means).
