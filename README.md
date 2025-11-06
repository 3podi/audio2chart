# audio2chart

[![arXiv](https://img.shields.io/badge/arXiv-2511.03337-b31b1b.svg)](https://arxiv.org/abs/2511.03337)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3podi/audio2chart/blob/main/notebooks/audio2chart-charting.ipynb)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Collection-orange)](https://huggingface.co/collections/3podi/audio2chart-v10)


**audio2chart** is an open-source deep learning framework for **audio to chart generation**, converting raw audio into structured `.chart` files used in Guitar Hero style rhythm games.  
A complete description of the methodology, architecture, and experiments can be found in our [arXiv publication](https://arxiv.org/abs/2511.03337).

The repository provides a full codebase for both **training** and **inference**, including data processing pipelines, neural network architectures, and ready to use scripts for generating playable charts from real songs.


---

## Overview

The repository contains everything needed to:
- Run pretrained models to generate `.chart` files from audio
- Train or reproduce the baseline sequence model

The main use case is simple:

Input: an `.wav`, or `.mp3` audio file  
Output: a playable `.chart` file compatible with Clone Hero

---

## Quick Start on Google Colab

You can try audio2chart instantly in your browser without setup.

1. Click on the following botton:
     
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3podi/audio2chart/blob/main/notebooks/audio2chart-charting.ipynb)

3. The notebook will:
   - Install dependencies  
   - Download a pretrained model from Hugging Face  
   - Transcribe your own `.mp3` or `.wav` file into a `.chart` file  
---

## Local Installation

To run locally (with GPU support):

```
git clone https://github.com/3podi/audio2chart.git
cd audio2chart
pip install -r requirements.txt
```

You’ll also need PyTorch (with CUDA if available).  
Install it from: https://pytorch.org/get-started/locally/

---

## Inference

`generate.py` allows you to transcribe an audio file into a `.chart` file using a pretrained model from [Hugging Face](https://huggingface.co/3podi).

```bash
python generate.py path/to/audio.mp3
```

This will automatically download the default model (`3podi/charter-v1.0-40-M-best-acc-nonpad`), generate chart tokens from the input audio, and save the resulting `.chart` file in the current directory.


Full example with custom parameters:

```bash
python generate.py path/to/audio.mp3 \
  --model_name 3podi/charter-v1.0-40-M-best-acc-nonpad \
  --temperature <float_temperature> \
  --top_k <int_topk> \
  --name "<song_title>" \
  --artist "<artist_name>" \
  --album "<album_name>" \
  --genre "<genre_name>" \
  --charter "<charter_name>" \
  --output <output_path>.chart
```

The script will:
1. Load the specified pretrained model  
2. Encode the input audio  
3. Generate dense token sequences conditioned on audio  
4. Decode them into time-aligned notes  
5. Write the final `.chart` file to the output path

---

## Baseline Model

`baseline.py` implements a simple Transformer decoder that predicts chart tokens without any audio conditioning.

The baseline can be trained and evaluated on publicly available tokenized chart datasets and serves as a reference model for pure symbolic note prediction.
The dataset is available at:  
👉 [**3podi/audio2chart-charts**](https://huggingface.co/datasets/3podi/audio2chart-charts)

It contains a `charts.zip` archive with preprocessed chart data ready for training.

To download and extract the dataset:

```bash
huggingface-cli download 3podi/audio2chart-charts charts.zip --repo-type dataset -d <data_path>
unzip <data_path>/charts.zip -d <data_path>/charts
```

### Running the Baseline

```bash
python baseline.py
```

Training and evaluation are configured via **Hydra**, which automatically loads the default configuration files located in the `configs/` directory.  
All parameters such as learning rate, batch size, epochs, or save directory, are defined in the YAML files.

To override a parameter at runtime, simply append it to the command line, for example:

```bash
python baseline.py root_folder=<data_path> is_discrete=True window_seconds=30 grid_ms=40
```

---

## Audio-Conditioned Model

`main.py` is used to train or evaluate the audio-conditioned Transformer model that maps encoded audio features to chart tokens.

The pretrained weights are provided on Hugging Face (see below), and inference is fully supported through `generate.py`.

---

## 🤗 Hugging Face Models

You can load and run the pretrained **Charter** model directly from [Hugging Face](https://huggingface.co/3podi).

```python
from inference.engine import Charter

# Load the pretrained audio-to-chart model
model = Charter.from_pretrained("3podi/charter-v1.0-40-M-best-acc-nonpad")

# Generate chart tokens from an audio file
seqs = model.generate("path/to/song.mp3")
```

The default model `3podi/charter-v1.0-40-M-best-acc-nonpad` offers a strong balance between quality and speed.  

Other available models vary by:
- **Time resolution:** `20` ms or `40` ms (controls temporal precision)
- **Model size:** `S` (~25 M params) or `M` (~225 M params)
- **Checkpoint type:** `best-acc` / `best-acc-nonpad`

You can explore all model variants in the  
👉 [Charter v1.0 collection on Hugging Face](https://huggingface.co/collections/3podi/audio2chart-v10).

---

## Repository Structure

```
audio2chart/
├── inference/
│   ├── model_inference.py     # Inference model with KV-cache support
│   └── engine.py              # Inference engine
├── chart/
│   ├── tokenizer.py           # Tokenization utilities
│   ├── time_conversion.py     # Convert note times to tick values and viceversa
│   └── chart_writer.py        # Writes decoded charts to .chart format
├── dataloader/
│   ├── convert_to_raw.py      # Script to convert audio to raw format
│   ├── audio_loader.py        # Dataloader for audio conditioned training
│   ├── notes_loader.py        # Dataloader for notes only training
│   └── utils_dataloader.py    # Dataset utils
├── modules/
│   ├── models.py              # torch.nn models
│   ├── trainer.py             # Lightning training modules
│   ├── transformer_layers.py  # torch.nn transformer layers
│   ├── transformer2.py        # torch.nn layers and final training model 
│   ├── utils_trian.py         # Training utils
│   └── scheduler.py           # Learning rate scheduler used during training
├── notebooks/
│   └── generating.py          # Colab notebook for charting
├── baseline.py                # Baseline training script
├── main.py                    # Audio-conditioned training script
├── generate.py                # Main inference entry point
└── requirements.txt
```

---

## Citation

If you use audio2chart or its models in your work, please consider citing:

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

For questions, discussions, or contributions, open an Issue on GitHub or contact the maintainer via Hugging Face:  
https://huggingface.co/3podi

---

⭐ If you find this project useful, please give it a star!
