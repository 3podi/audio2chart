# audio2chart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3podi/audio2chart/blob/main/notebooks/audio2chart-charting.ipynb)

audio2chart is an open-source project that automatically transcribes audio files into Guitar Hero / Clone Hero `.chart` files using neural nets.  

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

`generate.py` lets you transcribe an audio file into a `.chart` file using a pretrained model from Hugging Face.

### Example usage

```
python generate.py path/to/song
```

### Arguments

audio_path — Path to input audio (must be ≥30 s)  
--model_name — Hugging Face model ID or local path (default: 3podi/charter-v1.0-40-M-best-acc)  
--temperature — Sampling temperature (default: 1.0)  
--top_k — Top-K sampling (default: 5)  
--name — Song title (default: file basename)  
--artist — Artist name (default: "audio2chart")  
--album — Album name (default: "audio2chart")  
--genre — Genre (default: "audio2chart")  
--charter — Charter name (default: audio2chart/<model>-<temp>-<topk>)  
--output — Output path (.chart) (default: ./<basename>.chart)

The script will:
1. Load the pretrained model  
2. Generate dense token sequences conditioned on audio  
3. Decode them into time-aligned notes  
4. Write a `.chart` file to the specified output path

---

## Baseline Model

`baseline.py` implements a simple sequence-to-sequence baseline that predicts chart tokens directly (without audio conditioning).

This model can be trained and evaluated using publicly available tokenized chart data.

### Running the baseline

```
python baseline.py 
```

Typical configuration options:
- --epochs: number of training epochs  
- --batch_size: training batch size  
- --lr: learning rate  
- --save_dir: where to save checkpoints

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
model = Charter.from_pretrained("3podi/charter-v1.0-40-M-best-acc")

# Generate chart tokens from an audio file
seqs = model.generate("path/to/song.mp3")
```

The default model `3podi/charter-v1.0-40-M-best-acc` offers a strong balance between quality and speed.  

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
│   ├── model_inference.py # Inference model with KV-cache support
│   └── engine.py          # Inference engine
├── chart/
│   ├── tokenizer.py       # Tokenization utilities
│   ├── time_conversion.py # Convert note times to tick values and viceversa
│   └── chart_writer.py    # Writes decoded charts to .chart format
├── baseline.py            # Baseline training script
├── main.py                # Audio-conditioned training script
├── generate.py            # Main inference entry point
└── requirements.txt
```

---

## Citation

If you use audio2chart or its models in your work, please consider citing:

```

```

---

## License

See LICENSE for details.

---

## Contact

For questions, discussions, or contributions, open an Issue on GitHub or contact the maintainer via Hugging Face:  
https://huggingface.co/3podi

---

⭐ If you find this project useful, please give it a star!
