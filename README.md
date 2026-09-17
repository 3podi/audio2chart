# audio2chart

[![arXiv](https://img.shields.io/badge/arXiv-2511.03337-b31b1b.svg)](https://arxiv.org/abs/2511.03337)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3podi/audio2chart/blob/main/notebooks/audio2chart_charting.ipynb)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Collection-orange)](https://huggingface.co/collections/3podi/audio2chart-v10)

**audio2chart** is an open-source deep learning framework for **audio to chart generation**, converting raw audio into structured `.chart` files used in Guitar Hero style rhythm games.

Input: a `.wav` or `.mp3` audio file (minimum 30 seconds)
Output: a playable `.chart` file compatible with Clone Hero

A complete description of the methodology, architecture, and experiments is in the [arXiv publication](https://arxiv.org/abs/2511.03337).

---

## Quick start on Google Colab

Try it in your browser with no local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3podi/audio2chart/blob/main/notebooks/audio2chart_charting.ipynb)

The notebook installs dependencies, downloads a pretrained model from Hugging Face, and transcribes your own `.mp3` or `.wav` into a `.chart`.

---

## Requirements

**Python 3.13 is required** for the pinned `requirements.txt`. This is narrower than it looks, and installing on anything else will fail:

| pin | constraint |
|---|---|
| `audioop-lts==0.2.2` | Python **>= 3.13** (backports the `audioop` module removed in 3.13) |
| `networkx==3.5` | Python **>= 3.11** |
| `numba==0.62.1` | ships wheels for cp310-cp313; its sdist hard-refuses **>= 3.14** |

To run on Python 3.11 or 3.12, drop the `audioop-lts` line first. There is no supported path on 3.10 or 3.14.

## Installation

```bash
git clone https://github.com/3podi/audio2chart.git
cd audio2chart
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere), then:

```bash
pip install -r requirements.txt
```

### GPU support

The `torch` wheels on PyPI are **CPU-only on Windows**. `pip install -r requirements.txt` therefore gives you a CPU build, and `torch.cuda.is_available()` returns `False`. For CUDA, reinstall the torch trio from the PyTorch index:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 --force-reinstall torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
```

Inference works on CPU; a 3-minute song takes roughly two minutes. Training on CPU is not practical.

---

## Inference

```bash
python generate.py path/to/audio.wav
```

This downloads the default model (`3podi/charter-v1.0-40-M-best-acc`) plus the `facebook/encodec_24khz` audio encoder (~89 MB total, cached after the first run) and writes `./<song_name>/notes.chart`.

Note that `--output` takes a **directory**, not a filename. The file inside is always named `notes.chart`.

### The beat grid

By default the chart is written with a **fixed 200 BPM** `[SyncTrack]`. Notes still line up with the audio, because they are placed by absolute time, but the bar lines and beat grid are musically meaningless: the chart will look wrong in an editor and will not snap to anything.

Three ways to get a real grid, best first:

**1. Reuse an existing chart's tempo map** (exact, no estimation):

```bash
python generate.py song.wav --sync-from path/to/existing/notes.chart --snap 32
```

Copies `Resolution` and the full `[SyncTrack]` (including tempo changes and time signatures) from a chart you already have. Useful for re-charting a song, or generating an extra difficulty.

**2. Detect the tempo from the audio** (automatic, accurate for steady tempo):

```bash
python generate.py song.wav --detect-tempo --snap 32
```

Runs a phase-locked search over the onset envelope. This is deliberately *not* `librosa.beat.beat_track`, whose tempo is octave-unstable and too imprecise to hold sync over a full song. On a 193 s test track the search landed within 0.0024% of the tempo a human charter had entered by hand (about 5 ms of drift end to end) in roughly 6 seconds.

**3. Enter the tempo yourself:**

```bash
python generate.py song.wav --bpm 168 --snap 32
```

### Snapping

`--snap N` rounds every note to the nearest 1/N note. It is only meaningful once the grid is correct, so pair it with one of the options above.

The model emits tokens on a fixed millisecond grid (40 ms for the `40-` checkpoints), so raw output lands on arbitrary ticks. At 168 BPM a 1/32 note is 44.6 ms, which makes `--snap 32` a near-lossless fit: no note moves more than half a grid step (22.3 ms at that tempo). The result reads like a hand-authored chart in Moonscraper instead of scattered off-grid notes.

Snapping can round two adjacent tokens on the same lane onto the same tick; duplicates are dropped, which slightly reduces the note count.

### Options

| flag | default | meaning |
|---|---|---|
| `audio_path` | *(required)* | input audio, must be >= 30 s |
| `--model_name` | `3podi/charter-v1.0-40-M-best-acc` | Hugging Face model id or local path |
| `--temperature` | `0.5` | sampling temperature |
| `--top_k` | `32` | top-k sampling |
| `--sync-from` | none | copy `Resolution` + `[SyncTrack]` from an existing `.chart` |
| `--detect-tempo` | off | estimate tempo from the audio |
| `--snap` | `0` (off) | snap notes to the nearest 1/N note |
| `--guidance` | `0` (off) | classifier-free guidance; costs a second forward pass per step |
| `--pad-bias` | `0` | added to the pad token's logit, shifting note density |
| `--max-parallel-chunks` | `4` | 30 s chunks decoded at once; lower if generation is unexpectedly slow |
| `--bpm` | `200` | fixed BPM, used only without `--sync-from` / `--detect-tempo` |
| `--resolution` | `480` | ticks per quarter note (overridden by `--sync-from`) |
| `--name` `--artist` `--album` `--genre` `--charter` | derived | chart metadata |
| `--output` | `./<song_name>/` | destination **directory** |

Full example:

```bash
python generate.py song.wav --name "Song Title" --artist "Artist" --album "Album" --genre "Metal" --detect-tempo --snap 32 --output ./out
```

### Scoring a generated chart

`evaluate_chart.py` compares a chart against what human charters actually do, because
"is this any good?" is several questions and note accuracy answers none of them:

```bash
python evaluate_chart.py out/notes.chart
```

It reports what the chart is made of (chord, tap, forced and sustain rates against the
per-chart corpus median), whether the shapes are real guitar patterns (catalogue
coverage above that chart's own chance floor), how it is paced (rests per minute), and
whether it breaks universal rule 1. Pass `--audio` to also score how well note density
tracks the music -- meaningful in aggregate, noisy on a single chart.

`evaluate_conditioning.py` answers a different question about a *checkpoint*: does the
model use the audio at all? It scores each batch twice, once with its own audio and
once with a neighbour's, and reports what the swap costs. Note accuracy cannot tell you
this -- roughly 86% of slots are padding, and an all-pad baseline already scores 85.6%.

```bash
python evaluate_conditioning.py --checkpoint path/to/best-checkpoint.ckpt --trust
```

### Using the model directly

```python
from inference.engine import Charter

model = Charter.from_pretrained("3podi/charter-v1.0-40-M-best-acc")
seqs = model.generate("path/to/song.wav")
```

---

## Known issues

### Some audio files fail to decode

The loader is `librosa.load`, which tries libsndfile and falls back to `audioread`. Two ways this bites:

- libsndfile's bundled **Ogg Opus** support is incomplete. It will happily read the header and report a correct duration, then fail partway through decoding with `LibsndfileError: Supported file format but file is malformed` on a file that is not actually malformed.
- The `audioread` fallback then needs an external decoder (ffmpeg or gstreamer). Without one you get `audioread.exceptions.NoBackendError` and the run dies.

Also note that a file's **extension is not checked**. A `song.mp3` that is really an Ogg Opus stream is common in Clone Hero song folders and hits exactly this path.

Workaround: transcode to WAV first. With ffmpeg installed:

```bash
ffmpeg -i song.mp3 -ac 1 -ar 24000 song24k.wav
```

Or without installing anything system-wide, in a throwaway virtualenv (PyAV bundles its own FFmpeg):

```bash
pip install av numpy
```

To confirm what a file actually is:

```bash
python -c "import soundfile; print(soundfile.info('song.mp3'))"
```

### Unicode crashes on Windows consoles

`generate.py` contains emoji in the argparse description and in the final success message. On a Windows console using the cp1252 code page both raise `UnicodeEncodeError`:

- `python generate.py --help` fails outright.
- A successful run prints a traceback **after** the chart has already been written. The chart is fine; only the confirmation message failed.

Workaround:

```bash
PYTHONIOENCODING=utf-8 python generate.py song.wav
```

### Star power is not modelled

Sustains and the forced/tap flags travel inside the token, but star power does not. It is a phrase-level property spanning many notes and would need run-length modelling, so generated charts contain no SP phrases.

### Time signatures are not detected

`--detect-tempo` estimates tempo only. The written time signature is always `4/4`. Only `--sync-from` carries real `TS` events across.

### Tempo detection assumes a roughly constant tempo

`--detect-tempo` fits a single BPM plus phase, searching **100-200 BPM** by default. Songs with genuine mid-song tempo changes, or a true tempo outside that range, need `--sync-from` or a manual `--bpm`. The lead-in before the first beat is absorbed into a short first tempo segment so that tick 0 still maps to t=0 and `Offset` stays 0.

### Regenerating the frozen inference hashes

`tests/test_architecture_surface.py` pins a SHA-1 of each inference-path file, hashed with line
endings normalised to LF so the pins hold on Windows checkouts too. If you deliberately change
one of those files, re-pin it:

```bash
python -c "import hashlib,pathlib;print(hashlib.sha1(pathlib.Path('generate.py').read_bytes().replace(b'

',b'
')).hexdigest())"
```

---

## Hugging Face models

Pretrained models are published at [huggingface.co/3podi](https://huggingface.co/3podi). Variants differ by:

- **Time resolution:** `20` ms or `40` ms
- **Model size:** `S` (~25 M params) or `M` (~225 M params)
- **Checkpoint type:** `best-acc` / `best-acc-nonpad`

Browse them in the [Charter v1.0 collection](https://huggingface.co/collections/3podi/audio2chart-v10).

The repository default is `3podi/charter-v1.0-40-M-best-acc`.

---

## Training

### Baseline (no audio conditioning)

`baseline.py` is a Transformer decoder that predicts chart tokens symbolically. Dataset:

```bash
huggingface-cli download 3podi/audio2chart-charts charts.zip --repo-type dataset -d <data_path>
```

```bash
unzip <data_path>/charts.zip -d <data_path>/charts
```

```bash
python baseline.py
```

Configuration is handled by **Hydra** from the `configs/` directory. Override any parameter on the command line:

```bash
python baseline.py root_folder=<data_path> is_discrete=True window_seconds=30 grid_ms=40
```

### Audio-conditioned model

`main.py` trains the model that maps encoded audio features to chart tokens. The default uses frozen pretrained Encodec features; a trainable SEANet encoder is selectable through Hydra:

```bash
python main.py model=audio_discrete
```

Both encoder choices use the same discrete charting Lightning module.

### Fine-tuning a released checkpoint on your own library

`main.py` builds a fresh model unless `model.pretrained` names a checkpoint, in which case
`modules/pretrained.py` loads it first. It accepts a Hugging Face repo id, a directory holding a
`pytorch_model.bin`, or a Lightning `.ckpt`, and it widens the token embedding and output
projection when the training vocabulary is the expressive one and the checkpoint was trained on
the legacy 35-token one.

**The model config must match the checkpoint's `config.json`.** The released `charter-v1.0-*`
models are 1024-wide with 16 layers, `num_kv_heads: 8` and `compression: 2` (20 ms) or `3`
(40 ms); the repo default is 512-wide with 8 layers. Loading refuses rather than proceeding
partially, because a mismatch would otherwise leave most of the decoder randomly initialised
while every metric still looked healthy. `configs/audio.yaml` is already pinned to the 20 ms
model.

Building the dataset for a Clone Hero library:

```bash
# 1. discover chart/audio pairs and transcode the audio to headerless 24 kHz PCM
python -m dataloader.convert_to_raw --root "/path/to/Clone Hero Songs"     --difficulties Expert --instruments Single     --raw-dir /data/raw_audio --output-json /data/audio_dataset.json

# 2. merge any additional manifests into the one main.py reads
python -m dataloader.merge_manifests /data/audio_dataset_with_raw.json /data/other_with_raw.json     --output /data/audio_dataset_with_raw.json

# 3. train
python main.py data.root_folder=/data
```

The sample rate must match `model.sample_rate` (24000 for Encodec). `load_raw_audio` reads
headerless PCM and trusts the rate it is told, so a mismatch plays every song at the wrong speed
with the chart misaligned, and training still looks healthy.

Two settings routinely surprise people on a small GPU:

- The tensor the model sees holds `batch_size * num_pieces` sequences, not `batch_size` — the
  collate function flattens `num_pieces` windows out of every song into the batch.
- At `grid_ms: 20` a 30-second window is 1502 tokens, roughly four times the attention cost of
  the 40 ms grid.

---

## Tests and benchmarks

The dependency-light chart, tokenizer, and timing suite needs no ML stack:

```bash
python -m unittest tests.test_chart_pipeline
```

Full suite (see the CRLF caveat under Known issues):

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Dependency-heavy tests skip with a clear reason when the training stack is unavailable. Performance comparisons are kept separate:

```bash
python tests/benchmark_chart_pipeline.py
```

```bash
python tests/benchmark_chart_pipeline.py --max-ratio 1.25
```

The benchmark verifies reference/refactored output equivalence before reporting median timing ratios for small, typical, and large synthetic charts. `--max-ratio` makes it fail when any ratio exceeds the threshold.

---

## Repository structure

```
audio2chart/
├── inference/
│   ├── model_inference.py      # Inference model with KV-cache support
│   ├── layers.py               # Self-contained inference model primitives
│   └── engine.py               # Inference engine
├── chart/
│   ├── tokenizer.py            # Tokenization utilities
│   ├── chart_processor.py      # Chart parsing / preprocessing
│   ├── time_conversion.py      # Note times <-> tick values
│   ├── tempo.py                # SyncTrack parsing and tempo detection
│   └── chart_writer.py         # Writes decoded charts to .chart format
├── dataloader/
│   ├── convert_to_raw.py       # Convert audio to raw format
│   ├── audio_io.py             # Audio reading helpers
│   ├── audio_loader.py         # Dataloader for audio-conditioned training
│   ├── notes_loader.py         # Dataloader for notes-only training
│   └── utils_dataloader.py     # Dataset utils
├── modules/
│   ├── models.py               # torch.nn models
│   ├── transformer_layers.py   # Training model primitives
│   ├── trainer.py              # Lightning training modules
│   ├── training_transformer.py # Audio-conditioned training model
│   ├── scheduler.py            # LR schedulers
│   ├── utils_train.py          # Training utils
│   └── run_utils.py            # Training run setup and lifecycle
├── notebooks/
│   └── audio2chart_charting.ipynb  # Colab notebook for charting
├── baseline.py                 # Baseline training script
├── main.py                     # Audio-conditioned training script
├── generate.py                 # Main inference entry point
└── requirements.txt
```

---

## Citation

```
@misc{tripodi2025audio2chartendendaudio,
      title={audio2chart: End to End Audio Transcription into playable Guitar Hero charts}, 
      author={Riccardo Tripodi},
      year={2025},
      eprint={2511.03337},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2511.03337}, 
}
```

---

## Contact

For questions, discussions, or contributions, open an Issue on GitHub.

## Acknowledgments

This project was supported by **AMD** and the **AMD Developer Cloud**, whose compute resources made training and experimentation possible.

---

⭐ If you find this project useful, please give it a star!
