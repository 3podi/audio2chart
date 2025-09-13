import os
import warnings
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
import librosa

import random
from typing import Dict, Union, List, Optional
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chart.chart_processor import ChartProcessor
#torchaudio.set_audio_backend("sox_io")

DIFFICULTIES = ['Expert', 'Hard', 'Medium', 'Easy']
INSTRUMENTS = ['Single']
DIFF_MAPPING = {
    'Expert': 0,
    'Hard': 1,
    'Medium': 2,
    'Easy': 3
}


from pydub import AudioSegment
import numpy as np
import torch

def load_audio_librosa(path, target_sr=16000):
    """
    Load audio and return a tensor [1, T] in float32.
    """
    # librosa loads mono by default
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        waveform_np, sr = librosa.load(path, sr=target_sr, mono=True, duration=20*60)  # waveform_np is (T,)
    
    # convert to tensor and add channel dim
    waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()  # [1, T]
    
    return waveform, sr


def load_audio_pydub(audio_path, target_sr=None):
    """
    Load audio from any format supported by ffmpeg via pydub.
    Returns: waveform (torch.FloatTensor [channels, samples]), sample_rate
    """

    # Let ffmpeg handle format automatically
    audio = AudioSegment.from_file(audio_path)

    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples())

    # Reshape according to number of channels
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).T  # [channels, samples]
    else:
        samples = samples[np.newaxis, :]  # [1, samples]

    # Convert to float32 in [-1, 1]
    samples = samples.astype(np.float32) / (1 << (8 * audio.sample_width - 1))

    # Convert to torch tensor
    waveform = torch.from_numpy(samples)

    # Optional resampling
    if target_sr is not None and sr != target_sr:
        #import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr



class WaveformDataset(Dataset):
    def __init__(
        self,
        data,
        bos_token: int,
        eos_token: int,
        pad_token: int,
        max_length: int = 256,
        difficulties=['Expert'],
        instruments=['Single'],
        window_seconds: float = 10.0,
        sample_rate: int = 16000,
        audio_processor=None,
        tokenizer=None,
        conditional=False,
        augment=False,
        num_pieces: int = 1,  # number of random windows per audio
    ):
        self.data = data
        self.difficulties = difficulties
        self.instruments = instruments
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * window_seconds)
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.conditional = conditional
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.augment = augment
        self.num_pieces = num_pieces
        self.chart_processor = ChartProcessor(difficulties, instruments)

    def __len__(self):
        return len(self.data)

    def _augment(self, waveform):
        if random.random() < 0.5:
            gain_db = random.uniform(-6, 6)
            waveform = waveform * (10 ** (gain_db / 20))
        if random.random() < 0.5:
            noise_amp = 0.005 * waveform.abs().max() * random.random()
            waveform = waveform + noise_amp * torch.randn_like(waveform)
        if random.random() < 0.3:
            waveform = -waveform
        return waveform

    def _process_window(self, waveform, item):
        window_samples = self.num_samples
        max_start = waveform.shape[-1] - window_samples
        start_sample = random.randint(0, max_start) if max_start > 0 else 0
        end_sample = start_sample + window_samples
        chunk = waveform[:, start_sample:end_sample]

        if self.augment:
            chunk = self._augment(chunk)

        # --- chart + tokens ---
        #print(item["chart_path"])
        #print(item["difficulty"])
        self.chart_processor.read_chart(chart_path=item["chart_path"], target_sections=item["difficulty"])
        notes = self.chart_processor.notes
        #print('Notes: ', notes)
        notes = notes[item["difficulty"]]
        bpm_events = self.chart_processor.synctrack
        resolution = int(self.chart_processor.song_metadata['Resolution'])
        offset = float(self.chart_processor.song_metadata['Offset'])

        start_seconds = start_sample / self.sample_rate
        end_seconds = end_sample / self.sample_rate

        tokenized_chart = self.tokenizer.encode(note_list=notes)
        tokenized_chart = self.tokenizer.format_seconds(
            tokenized_chart, bpm_events, resolution=resolution, offset=offset
        )

        filtered = [(t, v, d) for (t, v, d, _) in tokenized_chart
                    if start_seconds <= t < end_seconds]

        if filtered:
            note_times, note_values, note_durations = map(list, zip(*filtered))
            note_times = [(t - start_seconds) / self.window_seconds for t in note_times]
        else:
            note_times, note_values, note_durations = [], [], []

        diff = [-1] if not self.conditional else [
            mapped_diff for diff, mapped_diff in DIFF_MAPPING.items() if diff in item["difficulty"]
        ]

        return {
            "audio": chunk,
            "note_times": note_times,
            "note_values": note_values,
            "note_durations": note_durations,
            "cond_diff": diff,
        }

    def __getitem__(self, idx):
        for attempt in range(10):
            try:
                item = self.data[idx] if attempt == 0 else random.choice(self.data)
                #waveform, sr = load_audio_pydub(item["audio_path"], target_sr=self.sample_rate)
                #waveform, sr = torchaudio.load(item["audio_path"]) 
                waveform, sr = load_audio_librosa(item["audio_path"])
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != self.sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                if waveform.shape[1] < self.sample_rate * self.window_seconds:
                    continue

                results = [self._process_window(waveform.clone(), item)
                            for _ in range(self.num_pieces)]
                return results
            except Exception as e:
                print(f"[Warning] Failed to load {item['audio_path']}: {e}")
                continue
        #raise RuntimeError(f"Failed to load a valid sample after 10 attempts")


class AudioChartCollator:
    def __init__(self, bos_token, eos_token, pad_token=-100, max_length=512, conditional=False):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.conditional = conditional

    def __call__(self, batch):
        # flatten batch: list of lists -> list
        flat_batch = [sample for sublist in batch for sample in sublist]

        if not flat_batch:
            return {}

        max_batch_len = max(len(sample["note_values"]) for sample in flat_batch) + 2
        max_batch_len = min(max_batch_len, self.max_length)

        batch_audio, batch_note_values, batch_note_times, batch_note_durations = [], [], [], []
        attention_masks, batch_diff = [], []

        for sample in flat_batch:
            audio = sample['audio']
            note_times = [0.0] + sample["note_times"] + [1.0]
            note_durations = [0.0] + sample["note_durations"] + [0.0]
            note_values = [self.bos_token] + sample["note_values"] + [self.eos_token]

            note_times = note_times[:max_batch_len]
            note_durations = note_durations[:max_batch_len]
            note_values = note_values[:max_batch_len]

            seq_len = len(note_values)
            pad_len = max_batch_len - seq_len

            padded_values = note_values + [self.pad_token] * pad_len
            padded_times = note_times + [0.0] * pad_len
            padded_durations = note_durations + [0.0] * pad_len

            attn_len = seq_len - 1
            attention_mask = [1] * attn_len + [0] * (max_batch_len - attn_len)

            batch_audio.append(audio)
            batch_note_values.append(padded_values)
            batch_note_times.append(padded_times)
            batch_note_durations.append(padded_durations)
            attention_masks.append(attention_mask)
            batch_diff.append(sample["cond_diff"])

        return {
            "audio": torch.stack(batch_audio, dim=0).float(),
            "note_values": torch.tensor(batch_note_values, dtype=torch.long),
            "note_times": torch.tensor(batch_note_times, dtype=torch.float),
            "note_durations": torch.tensor(batch_note_durations, dtype=torch.float),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "cond_diff": torch.tensor(batch_diff, dtype=torch.long) if self.conditional else None,
        }


def create_audio_chart_dataloader(
        data,
        audio_processor,
        window_seconds,
        tokenizer,
        difficulties: List[str] = ['Expert'],
        instruments: List[str] = ['Single'],
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 8,
        shuffle: bool = True,
        conditional: bool = False,
        vocab=None,
        num_pieces: int = 16
):
    if vocab is None:
        vocab = tokenizer.mapping_noteseqs2int
        bos_token_id = len(vocab.keys())
        eos_token_id = bos_token_id + 1
        pad_token_id = eos_token_id + 1
        vocab['<bos>'] = bos_token_id
        vocab['<eos>'] = eos_token_id
        vocab['<PAD>'] = pad_token_id

    collator = AudioChartCollator(
        bos_token=vocab['<bos>'],
        eos_token=vocab['<eos>'],
        pad_token=vocab['<PAD>'],
        max_length=max_length,
        conditional=conditional
    )

    dataset = WaveformDataset(
        data=data,
        difficulties=difficulties,
        instruments=instruments,
        window_seconds=window_seconds,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        conditional=conditional,
        bos_token=vocab['<bos>'],
        eos_token=vocab['<eos>'],
        pad_token=vocab['<PAD>'],
        max_length=max_length,
        num_pieces=num_pieces
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator
    )

    return dataloader, vocab



