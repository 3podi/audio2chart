# audio2chart — session handoff

## Where this stands (2026-09-13)

**The model now charts like a human on every aggregate measured.** One line in the loss
was the whole problem.

`modules/trainer.py` weighted the pad token at 0.1 while pad is 86% of every target
sequence. Its share of the objective was 0.86 x 0.1 = 0.086 against 0.14 for real notes,
so **notes dominated the loss 62/38 while being 14% of the data**, and the model was
trained for *which note* and barely at all for *note or silence*. It is now
`model.pad_class_weight`, and at 1.0:

| | w=0.1 | w=1.0, 1500 | **w=1.0, 3000** | human |
|---|---|---|---|---|
| nps, median of 6 | 59.97 | 23.39 | **23.12** | **21.01** |
| rests/min, median | 0.85 | 15.35 | **17.25** | **15.10** |
| pattern lift, median | 0.61 | 0.45 | **0.41** | **0.42** |
| responsiveness, median | — | — | **+0.133** | **+0.249** |
| audio ablation, notes | +0.8%, 14/20 | +1.0%, 16/20 | +1.5%, 14/20 | — |

Per-chart spread sits inside the human range too: rests 7.4-24.1 against a human
0.53-55.3, nps 12.4-34.9 against 3.97-33.58.

**It follows the music.** Responsiveness -- note density against the audio onset envelope
-- is positive on every one of six charts: -0.003, +0.018, +0.145, +0.132, +0.302, +0.133.
Randomly scattered notes score -0.041 on this corpus and human charts +0.249, so the model
sits about half way to human and clearly above chance. That is the one measure here a
model ignoring the audio could not fake.

A single chart from the 1500-step model scored -0.400 on the same measure and looked like
evidence of anti-correlation. It was not: the human range across ten charts is -0.547 to
+0.717, so one chart could never have shown it, and six charts from the next checkpoint
came out positive. Do not read this metric off one chart.

**More training helped, and the loss change cost nothing.** Doubling to 3000 steps put
pattern lift on the human figure exactly (0.41 against 0.416) and took alternating silence
to 0.96x the corpus. `val/acc_nonpad` also recovered from 0.607 at 1500 steps to 0.663 --
above what the broken weight-0.1 model reached -- so the apparent accuracy cost of
reweighting pad was undertraining, not a real trade.

**It now slightly over-rests**: 17.25 rests a minute against a human 15.10, and P(pad)
after two notes is 0.863 against a corpus 0.751. Mild over-resting is a much better
failure than the original, but a pad weight a little under 1.0 is the knob if it matters.

### What is still not right

- **Audio conditioning is real but modest.** The ablation gives +1.5% on note positions
  hurting 14 of 20 batches, which is marginal on its own, and responsiveness is +0.133
  against a human +0.249. The model does attend to the music; it attends about half as
  much as a human charter does. This is the largest remaining gap and the next thing
  worth attacking -- `model.transformer.audio_drop` is 0.0, and raising it would train
  the model to handle missing audio, which is what classifier-free guidance needs and the
  engine's `--guidance` flag already implements.
- Slight over-resting, 17.25 a minute against 15.10.
- 8 ms at a 12 s window is still free data (97.0% -> 98.8% usable windows at the same 1502
  tokens) and has not been taken.

### Guidance moves density, not conditioning

`--guidance` runs a second decoder branch against a neighbouring chunk as the negative
(`_contrast_audio`), which works without training because the model has no notion of "no
audio" to contrast against. Note that `model.transformer.audio_drop` is *not* the lever it
looks like -- it is `nn.Dropout` on the audio embedding, feature regularisation, not the
sample-level conditioning dropout classifier-free guidance would need.

What guidance 1.5 does, measured twice:

| | guidance 0 | guidance 1.5 | human |
|---|---|---|---|
| nps | 18.1 | **21.4** | 21.01 |
| responsiveness, median of 8 | +0.040 | +0.072 | +0.249 |

**Density improves and replicates. Conditioning does not.** A first sweep over three clips
put responsiveness at +0.479 and looked like conditioning solved; eight paired clips give a
median of +0.072, better on 5 of 8, paired t = +0.65, p = 0.36. That is nothing.

The reason the three-clip result was so wrong is worth keeping: **responsiveness is
extremely noisy per clip.** At guidance 0 alone the eight clips span -0.682 to +0.191, and
human charts span -0.547 to +0.717. Three samples of a quantity with that spread will show
almost anything. Six is not obviously enough either -- the +0.133 median from the six-chart
scoring and the +0.040 here are the same model on different clips.

Conditioning remains the open problem. The ablation (+1.5% on note positions, 14 of 20
batches) is the more stable measure and says the same thing: real, weak.

### Run it like this

    model.pad_class_weight=1.0        # not 0.1
    generate --temperature 1.0        # not 0.5
    loader.val_num_pieces=4           # 1 made checkpoint selection noise-driven
    trainer.log_dir=G:/a2c_runs       # C: filled to 7 MB free

Check `diag/pad_alternating_ratio` in metrics.csv at the first validation. If silence is
an order of magnitude below the corpus the run will not produce a usable chart and is not
worth finishing -- that was visible at 300 steps and previously took two full runs and
eighty minutes of scoring to notice.

---


Updated 2026-09-11 evening. **Supersedes everything in this file before today**, which
was written 2026-09-05 and said "Nothing is running. No checkpoint exists." Both are now
false. `MORNING.md` covers the grid change in more depth and is still accurate; this file
is the current state.

Everything below was measured on this machine.

---

## The throughput problem is solved, and the earlier diagnosis of it was wrong

Two days went into the dataloader — filling the dead audio cache, raising worker counts,
writing a hand-rolled loader profiler. Lightning's own profiler over 256 batches of the
real training config:

```
run_training_epoch      91.7 s   (0.358 s/batch, 2.79 it/s)
training_step           0.176 s  49%
backward                0.130 s  36%
optimizer_step          0.311 s  11%
train_dataloader_next   0.0005 s  0.1%
```

**The loader has never been the bottleneck.** Filling the audio cache moved throughput
0.77 → 0.82 it/s, not the 87% the old comment in `dataloader/audio_loader.py` claimed;
that claim is retracted in `31ddf89`.

The real cause was a unit mismatch between two config keys that sit next to each other
and look symmetrical:

- `val_check_interval` counts **batches**
- `ModelCheckpoint`'s `every_n_train_steps` counts **optimizer steps**

Both were launched at 100 with `accumulate_grad_batches=8`, so the run validated once per
training batch and wrote a 2.42 GB checkpoint every 100 batches — onto G:, which measures
**108 MB/s** and is the same drive streaming the songs. C: measures 704 MB/s.

| | before | after |
|---|---|---|
| throughput | 0.82 it/s | **2.79 it/s** |
| GPU utilisation | 16% | **87%** |

`trainer.log_dir` now moves checkpoints and CSV metrics off the corpus drive. **An idle
GPU waiting on disk is indistinguishable from a starved loader in `nvidia-smi`** — that
is why the wrong diagnosis survived so long. Check the profiler before believing either.

---

## Sync: how exact are the ticks?

Robert's requirement was "ticks need to be exact. I dont want any sync issues."

**Exact, with a known boundary.** `tests/test_tick_exactness.py` runs real tick positions
through the whole pipeline — ticks → seconds → `discretize_time` → seconds →
`choose_snap` → ticks — and requires identity, not closeness. Swept over 120–400 BPM and
16th through 96th notes:

- every subdivision whose spacing clears the 10 ms grid step recovers **exactly, 0.0 ms
  error**, down to a 12.5 ms gap (96ths at 200 BPM, 64ths at 300)
- triplets and mid-song tempo changes included; a BPM rounded to `###.#` stays exact
- below the grid step (8.3 ms) discretization **raises and the window is dropped** — it
  refuses rather than shipping a desynced chart
- the one case that drifts is an irregular burst on no common subdivision (ten ticks at
  200 BPM, 15.6 ms), where `choose_snap` has nothing to snap to; residual stays inside
  half a grid step

`tests/test_audio_token_alignment.py` covers the invariant underneath all of that: the
audio slice and the token grid describe the same instant. It puts a click in the audio
where a note sits in the chart and requires the click's position in the returned chunk to
equal the note's grid index times the grid step, for a window at t=0 and one mid-song.

**The `Offset` field matters more than it looks.** 208 of the 1454 songs in the tapping
training split carry a non-zero one, 199 by more than a grid step, median 0.150 s and up
to 5.0 s. Scored against the audio onset envelope over the 46 songs with |offset| ≥ 1.0 s:
as-written **+0.197** > ignored **+0.140** > inverted **+0.089**, paired t = +3.78. That
ordering is what a correct sign looks like. It holds.

Also checked: only 2 of 1454 charts outlast their audio, both by 0.02 s. Silence-padded
windows with real note targets are not a problem in this corpus.

**The grid could still go finer for free.** Measured over the whole 1454-song tapping
split rather than the 120 hand-picked charts the original decision used:

| grid | window | tokens | windows usable | songs 100% | songs dropped |
|---|---|---|---|---|---|
| 20 ms | 30 s | 1502 | 59.9% | 36% | 293 |
| 10 ms | 15 s | 1502 | 97.0% | 80% | 7 |
| **8 ms** | **12 s** | **1502** | **98.8%** | **90%** | **3** |
| 6 ms | 12 s | 2002 | 99.5% | 95% | 1 |

8 ms at 12 s costs the identical 1502 tokens and is strictly better data. Not taken
because changing the grid invalidates the validation cache and wastes any run in
flight — it is a start-of-next-run change. Note also that the "18.2% usable at 20 ms"
figure quoted everywhere came from those 120 charts; on the training split it is 59.9%.
The change was still right, but the smaller number does not describe this corpus.

The loader retries a window that collides and draws another: 197 retries over ~5000
batches in Run A, 189 resolved on the second attempt, zero exhausted. Working as
designed, but it means the 3% of unusable windows are silently skipped rather than
reported, and they concentrate in the fastest 25 songs.

---

## Training runs in flight

Two runs, chained, on the tapping subset (1447 train / 363 val), scored afterwards with
`score_run.py`:

- **Run A** — `model.init_from` the b6000 checkpoint (carries chart knowledge, but was
  trained entirely at 20 ms with the sync error baked in)
- **Run B** — from the released checkpoint, no init

Both: `loader.train_num_pieces=1 trainer.accumulate_grad_batches=8 model.freeze_layers=4
model.input_noise=0.08 trainer.max_steps=1500 trainer.val_check_interval=1200
trainer.limit_val_batches=100 trainer.checkpoint_every_n_steps=200
trainer.log_dir=C:/a2c_runs`

Note `val_check_interval=1200` is 150 optimizer steps, not 1200 of them. Read the warning
in `configs/default.yaml` before changing it.

The falsifiable question for scoring: the model learned note spacing in position units at
20 ms, so at 10 ms every rhythm it knew is wrong by exactly 2×. That should wash out in
fine-tuning. If it did not, note density comes out around double and the inter-onset
histogram clusters at half the right intervals.

**RETRACTED: Run A is not overfitting.** An earlier version of this section said it
was, from three validation points after a peak. The next point made a new high and the
claim did not survive:

| step | val | train | gap |
|---|---|---|---|
| 692 | 0.660 | 0.660 | +0.000 |
| 873 | 0.650 | 0.668 | +0.018 |
| 1054 | 0.639 | 0.672 | +0.033 |
| **1235** | **0.666** | 0.672 | **+0.006** |

The train/val gap widened to +0.033 and then collapsed back to +0.006 as validation hit a
new high. That is a noisy metric, not divergence.

The noise has a cause worth fixing: `limit_val_batches=100` with `val_batch_size=1` and
`val_num_pieces=1` computes the monitored metric over **100 windows**. A swing of ±0.013
around 0.653 is what that sample size produces, and the monitored checkpoint is selected
on it — so which weights get saved as "best" is partly luck. Raising `limit_val_batches`,
or `val_num_pieces`, would cost little (validation is a small share of the run now that
checkpoints are on a fast drive) and would make checkpoint selection mean something.

Score `best-checkpoint.ckpt` rather than `last.ckpt` regardless: it is monitored on
val/acc_nonpad_epoch, so it tracks the peak wherever the peak lands.

---

## What the model actually produces (probe, one song)

Run A's best checkpoint, generating over a 60 s validation clip, scored against the
tapping baseline. Three sampling settings, to separate a sampling artifact from a
property of the model:

| setting | nps | rests/min | pattern lift | coverage |
|---|---|---|---|---|
| default (t 0.5, top_k 32) | 71.26 | **0.0** | +0.579 | 0.840 |
| pad-bias 0.4 | 69.30 | **0.0** | +0.684 | 0.960 |
| t 1.0, top_k 128, pad-bias 2 | 46.66 | **0.0** | +0.538 | 0.804 |
| **human tapping** | **21.01** | **15.1** | +0.416 | ~0.68 |

Two things, and they point in opposite directions.

**The pattern vocabulary is good.** Catalogue coverage 0.80-0.96 against a chance floor
around 0.27, lift at or above the human median in all three settings. Whatever else is
wrong, the model is producing real charting shapes, not noise. That is the thing the
catalogue work was for and it appears to have landed.

**It does not know when to stop.** Density runs 2.2x to 3.4x human and never comes down
to it; rests are *exactly zero* at every setting, against a human 15.1 a minute. The
model fills about 71% of all grid slots at the shipped defaults. Sampling moves density
(71 -> 47) but not phrasing, so this is structural, not a knob that was set wrong.

Two separate faults, measured apart. Teacher-forced on real history the model predicts
pad 79.3% of the time against the data's ~86% (`val/pad_pred_rate` 0.793, pad true
positives 0.936). Generating, it emits pad about 29% of the time. So most of the damage
happens at generation, but not all of it.

Density of the generated chart by position inside a 15 s chunk, 100 ms bins, as a share
of the 10 grid slots each bin contains:

    first 300 ms   4.9/10
    300-1000 ms    6.5/10
    1-2 s          7.3/10
    plateau        7-8/10
    human          ~1.4/10

**The prior is wrong before drift can start.** At the first token the context is nothing
but bos and the audio, and it already fills 54% of slots against a human 14% -- roughly
four times too dense with no opportunity to have drifted yet. Class weighting or a pad
prior is the lever there, not scheduled sampling.

**Then it drifts, and locks in fast.** Density climbs from 54% to about 75% over roughly
130 tokens and stays. That is self-reinforcing: a dense history tells the model it is in
a dense passage. `input_noise=0.08` was aimed at this and is clearly not enough.

A caution about how this was measured, because the first attempt got it backwards. Binned
at 5 s, the profile reads 65.8 nps in the first third of a chunk against 52.0 in the last
-- ratio 0.79, which looks like *no* accumulation and briefly seemed to refute the drift
reading. Saturation completes inside the first bin, so that test could not see it. Use
100 ms bins over the first two seconds, not thirds of a chunk.

The rests figure is the starkest number here: the longest gap anywhere in 60 s of output
is 0.040 s, and the median gap is 0.0099 s -- one grid step. The model does not pause at
all, ever, while human tapping charts rest for a quarter second or more about fifteen
times a minute.

Note what this is *not*: evaluate_chart's docstring predicted the opposite failure -- zero
chords, zero taps, almost no sustains at the shipped defaults. That is not what this
checkpoint does. The docstring describes an older checkpoint and should not be trusted
for this one.

**One song.** The rests=0 result is consistent across three settings, but all three are
the same clip. The 6-chart scoring run is what decides whether it generalises, and no
conclusion here should be repeated until it does.

---

## Scored results, six charts each

Both runs finished 1500 steps. Scored against the tapping baseline.

| | Run A (from b6000) | Run B (from released) | human |
|---|---|---|---|
| val non-pad peak | **0.6657** | 0.6514 | — |
| nps, median of 6 | 59.97 | 75.72 | **21.01** |
| rests/min, median | 0.85 | 0.35 | **15.10** |
| pattern lift, median | +0.61 | +0.65 | +0.42 |
| pct_tap, median | 1.00 | 1.00 | 0.96 |
| audio ablation, note positions | **+0.8%**, hurt 14/20 | +0.3%, hurt 11/20 | — |

**Run A is the better model on every axis that matters.** Higher validation accuracy,
density closer to human (60 against 76), and it is the only one of the two that
demonstrably uses the audio at all: swapping a batch's audio for a neighbour's costs it
0.8% on note positions and hurts 14 of 20 batches, against 0.3% and 11 of 20 for Run B.
Eleven of twenty is chance. So initialising from b6000 was worth doing, which is the
question the A/B was set up to answer.

**Both share the fatal fault, and it generalises.** Across all twelve charts, density runs
1.8x to 4.2x human and rests never exceed 1.8 a minute against a human 15.1. The
one-song probe was not a fluke.

**Pattern lift above human is not the good news it looks like.** Both runs exceed the
human median (+0.42), and Run B's range of +0.62 to +0.71 sits above the highest of the
twelve human charts measured (+0.614). Combined with saturated density that reads as
over-regularity -- the model spamming catalogue shapes continuously -- rather than
musicality. Catalogue coverage is worth having, but a chart that is *more* patterned than
any human one is not obviously closer to a human chart.

The single Run B chart at 22.54 nps is worth noting: one chart in twelve landed at human
density, so the model is not incapable of it, just overwhelmingly unlikely to get there.

---

## Why it never stops: pad is down-weighted 10x in the loss

Measured directly, not inferred. Feeding the model a history and reading P(pad) off one
forward pass, no sampling involved:

| history | P(pad) raw | after temperature 0.5 |
|---|---|---|
| just bos | 0.580 | **0.904** |
| 1 note | 0.599 | 0.901 |
| **2 notes** | **0.132** | **0.092** |
| 4 notes | 0.027 | 0.004 |
| 8 notes | 0.009 | 0.0002 |
| 32 pads | 0.807 | 0.998 |
| **32 alternating note/pad** | **0.002** | **0.0000** |

**Two consecutive notes collapse P(pad) tenfold and it never recovers.** An *alternating*
note/pad history -- an ordinary sparse chart -- drives pad to zero outright. The all-pad
state is the only one in which this model will stay silent.

So generation starts 90% likely to be quiet, one unlucky pair of notes locks it into
permanent saturation, and it plays without pause for the rest of the song. That is the
whole story of the 71 nps and the zero rests, and it is why `--pad-bias` cannot rescue
it: countering a 0.90 -> 0.0002 collapse needs about 8 logits, and the bias applies
uniformly, so by +3 it has annihilated output everywhere else (the sweep in
engine.py `_make_sampler` shows 0.66 nps at pad+3).

### The cause is in `modules/trainer.py:430`

```python
self.class_weights = torch.ones(self.vocab_size)
self.class_weights[self.pad_token_id] = 0.1
```

Pad is 86% of all targets and is weighted 0.1. Its share of the loss is therefore
0.86 x 0.1 = 0.086 against 0.14 x 1.0 = 0.14 for real notes: **notes dominate the
objective roughly 62/38 while being 14% of the data.** The model is optimised for *which
note* and barely at all for *note or silence*.

That predicts exactly the two things measured. Pattern vocabulary is above human -- lift
+0.58 against +0.42, coverage up to 0.96 -- and density is 3.4x human with no rests at
all. The model is good at what the loss emphasises and bad at what it discounts.

The down-weighting was presumably there to stop a collapse to all-pad, since predicting
pad everywhere scores 86% accuracy. That is a real hazard and the reason to keep *some*
down-weighting. 0.1 has simply overshot into the opposite failure.

### What to try next, in order

1. **Raise the pad weight** to somewhere around 0.3-0.5 and retrain. This is the cheapest
   test of the whole diagnosis and it changes one number.
2. Watch `val/pad_pred_rate` against the data's ~0.86 as the primary signal; it sat at
   0.79 teacher-forced this run, which looked fine and hid the generation collapse.
3. Re-run the P(pad)-versus-history table above. It takes one forward pass per row and is
   far more diagnostic than any chart metric -- if two notes still collapse pad, nothing
   downstream will be right.

Do not tune `--pad-bias` to compensate. It treats a training fault at inference, it is
not orthogonal (suppressing notes suppresses chords and taps preferentially), and the
engine docstring already records that it does not land anywhere near a human density.

---

## Two tools were silently broken

**`fit_tier.py` had never run.** `main()` read `args.min_tier`; `parse_args` never declared
it, so every invocation died with `AttributeError` before touching a chart. That is why it
kept reappearing on the to-do list as "never run". `tests/test_cli_arguments.py` now parses
every top-level tool and requires each `args.<name>` to be declared; the rest are clean.

**`audit_catalogue.py` could not see chord patterns.** `parse` resolves one letter at a
time, so `'RB'` matches no fret and `'RB RYB RB'` signs as the empty tuple. All four
chorded entries collided into one reported "duplicate", then fell below the `len(sig) >= 2`
filter and were never scanned against a single chart — while the report said "ABSENT:
none". Distinct bodies went 70 → 74 once fixed.

**`tests/test_tempo.py` imported pytest**, which is installed in neither environment here,
so the whole file raised ImportError under unittest discovery and none of its nine tests
had ever executed — including the BPM-rounding one. Its rounding test also asserted that
the string `"round("` appeared in `detect_tempo`'s source, which would pass with the
rounding commented out. Both fixed.

---

## The tier grader does not work, and more features will not fix it

`fit_tier.py`, first real run:

```
two human label sources disagreeing with each other   5.47 tiers
this model, held out                                  5.43 tiers
predicting a constant                                 6.26 tiers
```

17% of charts within two tiers on a 1–26 scale. **The model is already at the label noise
floor.** Median signed difference between the two sources is 0, so there is nothing to
calibrate away — charters do not agree with each other, and a constant prediction (5.30)
beats one source predicting the other.

The features are fine. Rank correlation *inside one author's setlist* has median
best-feature |rho| **0.50** — 0.86 (nps, Shobas_ Charts), 0.69 (peak10, Community Track
Packs) — against **0.26** pooled across packs. Difficulty is tracked; the tier numbers are
not on one scale.

What would work: a per-pack offset fitted alongside, or pairwise judgements from one
person. Robert has a consistent scale in his head — TTFAF in single digits, Schmoo's World
around 24 — and that is the label source this needs.

---

## What Moonscraper told us (2026-09-12)

Robert suggested reading the installed editor at
`C:\Program Files (x86)\Moonscraper Chart Editor`. It ships as a compiled Unity app --
the logic is in `Assembly-CSharp.dll`, not source -- but its config and changelog are
plain text and answered one real question.

**`song.ini` has a `delay` field, and nothing here reads it.**
`Config/clone_hero_ini_tags.txt` lists `delay = 0` alongside `diff_guitar`. It is a
second timing offset, independent of the `.chart` `[Song] Offset` that the pipeline does
read. Measured across the 1454-song tapping training split:

| | songs |
|---|---|
| both non-zero and agreeing | 72 |
| both non-zero, **disagreeing** by >2 ms | 8 |
| Offset only (handled correctly) | 128 |
| **`delay` only — silently misaligned** | **4** |
| neither | 1242 |

So `delay` is usually the same quantity in milliseconds -- Requiem dies irae is
`Offset 5.000` and `delay 5000`, Everlong `-2.230` and `-2230` -- but not always. Twelve
songs (0.8%) are either misaligned or ambiguous. The disagreements are strange rather
than small: `Prevail` carries `Offset +0.870` against `delay +0.001`.

Not fixed. Reading it means threading a song.ini lookup through ChartProcessor, which
currently works from chart text alone and has no path when handed text directly, so every
call site changes. That is a poor trade for 0.3% of the corpus while the model is
emitting three times too many notes. Worth doing once the density fault is closed.

Supporting evidence for which field is canonical: the changelog records *"Moved Offset
property option from general Settings to be under Advanced->Legacy Options"*. Moonscraper
treats `.chart` Offset as legacy, which is consistent with `delay` being the modern path
and with both being written for compatibility. It does not by itself say which Clone Hero
honours at playback -- the empirical onset-alignment test does, and it says Offset as
written is correct for the songs that carry it.

**What it could not usefully tell us.** The obvious next targets -- the HOPO threshold
that the `forced` flag inverts, and sustain cutoff rules -- turn out not to matter for
this corpus. Tapping charts barely use either: human `pct_forced` is 0.0009 and
`pct_sustain` 0.0067, and the generated charts sit at 0.000 for both. Chasing the exact
HOPO tick threshold would have been precision applied to a feature that is absent from
the data we train on.

One representational note for later, if the project moves beyond tapping. The `.chart`
`forced` flag does not record "this is a HOPO" -- it *inverts* whatever the note would
naturally be, and natural HOPO depends on the tick gap to the previous note. So the same
musical intent maps to different tokens at different spacings, and the model has to learn
that equivalence rather than being handed it. Storing a derived `is_hopo` instead would
be a cleaner representation. Irrelevant while forced flags are ~0.1% of notes.

---

## The cause, and what moves it (2026-09-12)

`modules/trainer.py` weighted the pad token at 0.1 in the cross-entropy loss while pad is
86% of all targets. Its share of the objective was therefore 0.86 x 0.1 = 0.086 against
0.14 for real notes: **notes dominated the loss 62/38 while being 14% of the data**, and
the model was trained for *which note* and barely at all for *note or silence*.

That predicts exactly the two things measured — pattern vocabulary above human, density
3.4x human with no rests. It is now `model.pad_class_weight`.

Dose-response, P(pad) at temperature 0.5 after a note history:

| history | w=0.1 | w=0.35 | w=1.0 | corpus |
|---|---|---|---|---|
| 2 notes | 0.092 | 0.340 | **0.816** | 0.751 |
| 4 notes | 0.004 | 0.011 | 0.100 | 0.408 |
| alternating-4 | 0.003 | 0.038 | **0.280** | 0.398 |
| 32 notes | 0.004 | 0.245 | 0.636 | 0.008 |

Monotonic, which settles the mechanism. w=1.0 lands the two-note case and still
overshoots the dense regime badly — after 32 consecutive notes it wants to stop, where the
corpus says 0.008 — but that reading is from a run that died at step 150 and may not be
converged. A full-length w=1.0 run is in flight.

### A measurement error worth not repeating

The alternating reference was 0.9261 for a while, and it was wrong. It had been measured
on histories ending in a **note** while the probe builds histories ending in a **pad** —
"does the alternation continue" (93%) against "does this gap extend" (5-40%), opposite
questions. It made every checkpoint look about ten times worse on those rows and produced
a flat FAIL verdict. Correct values, matched to the probe's own lengths: 0.398 at four
slots, 0.199 at eight, 0.110 at sixteen, 0.050 at thirty-two. P(pad) falls steeply with
how long the alternation has run, because a long regular passage is a dense one, so no
single figure could ever have been right.

It then took a second commit to fix, because the same wrong constant had been copied into
`WatchPadCollapse`. Check for duplicates of a constant before believing it is fixed.

### Watch this during training, not after

`WatchPadCollapse` logs `diag/pad_after_k_notes` and `diag/pad_alternating_ratio` at every
validation. It exists because **every other metric here is teacher-forced** — accuracy,
loss, and `pad_pred_rate` are all conditioned on a correct history, and this fault is
entirely about what happens when the history is the model's own. `val/pad_pred_rate` read
a healthy 0.79 through two complete runs in which the model could not rest.

The fault was visible at 300 steps. It cost two full runs and eighty minutes of chart
scoring to find.

---

## Operational notes from the same day

**Do not run corpus scans while training.** Measured on the same run, same minute: 6.0
optimizer steps per minute with analysis scripts running against 10.0 without. The six
dataloader workers need the CPU.

**`nvidia-smi` utilization lies.** It reports whether any kernel ran in a sampling window,
not how full the device is. During the slow stretch it read 100% while
`clocks_throttle_reasons.active` was `0x1` (GpuIdle), power draw 74 W of 265 W and the SM
clock 780 MHz of 2100. Query power, clocks and throttle reasons -- utilization alone has
now been misleading twice.

**C: filled to 7 MB free** because checkpoints were pointed there. Everything project-owned
now lives on G:, `save_checkpoint` refuses to start without three checkpoints of headroom,
and its cleanup no longer raises over the original exception.

---

## Environment traps

- The training interpreter is **`.venv313`** (torch 2.9.1+cu128, RTX 3060 Ti, 8 GB).
  `.venv` is a different, CPU-only install whose `pyvenv.cfg` points at a `C:\Users\rober`
  path that does not exist on this machine. `python` on PATH is a third install with no
  torch. Use `.venv313/Scripts/python.exe`.
- **pytest is not installed anywhere.** Tests run under `python -m unittest discover -s
  tests -t .` — 301 passing.
- `pkill -f` does not reach native Windows processes. Use
  `Get-CimInstance Win32_Process` and `Stop-Process`.
- Writing Python through a bash heredoc mangles backslash escapes (`\a`, `\t`, `\U`).
  Patch from a file, or edit directly.

---

## Open, in rough priority order

1. **Score both runs** with `score_run.py` once they finish, over several charts. One
   chart characterises nothing — across eight from this model sustains ran 0.000–0.655
   and pattern lift −0.172 to +0.383, and a day of conclusions from a single chart had to
   be retracted.
2. **`triangle slide 8-note` and `sweep 8-note` hold the identical sequence**
   `G R Y B O B Y R` under two names, which double-counts in catalogue coverage. Which
   name is right is Robert's call; left alone rather than merged.
3. **Stem separation** — 2,706 overchart songs, ~9 GPU-hours, ~37 GB. Built and ready,
   waiting on a scope decision.
4. **Cross-chunk context** — architectural, and more pressing since the 15 s window
   doubled the number of chunk boundaries in a song.
5. **The frozen Encodec encoder is ~18% of every step** (0.065 s of a 0.358 s batch) and
   recomputes features that cannot change while `freeze_encoder: true`. Precomputing codes
   per window would buy roughly 1.2×, at the cost of foreclosing encoder fine-tuning and
   waveform augmentation. Not done.
6. **`tests/test_grid_window.py::test_engine_default_matches_nothing_by_accident`
   asserts on source text** rather than behaviour — it would pass if someone wrote
   `chunk_sec = 30.0`. Making it behavioural means touching the hash-pinned
   `inference/engine.py`, so it was left for a deliberate pass.
7. Sampling defaults need a multi-clip comparison (GPU, so not while training).
8. Nothing has been pushed. The branch is far ahead of PR #3.
