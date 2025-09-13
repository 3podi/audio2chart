# chunked_audio_chart_dataset.py

import os
import random
import warnings
import threading
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from concurrent.futures import ProcessPoolExecutor
import torchaudio
import librosa

from chart.chart_processor import ChartProcessor

# --------------------
# Constants
# --------------------
DIFF_MAPPING = {
    'Expert': 0,
    'Hard': 1,
    'Medium': 2,
    'Easy': 3
}


# --------------------
# Audio loading
# --------------------
def load_audio_librosa(path, target_sr=16000, max_duration=20*60):
    """Load audio and return tensor [1, T]"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        waveform_np, sr = librosa.load(path, sr=target_sr, mono=True, duration=max_duration)
    return torch.from_numpy(waveform_np).unsqueeze(0).float(), sr



def load_audio_librosa_safe(path, target_sr=16000, timeout_seconds=10):
    """
    Load audio with librosa, but cancel if it takes too long.
    Returns (waveform, sr) or raises TimeoutError.
    """
    def _load():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FutureWarning)
            waveform_np, sr = librosa.load(
                path, sr=target_sr, mono=True, duration=20 * 60
            )
        return waveform_np, sr

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_load)
        try:
            waveform_np, sr = future.result(timeout=timeout_seconds)
            waveform = torch.from_numpy(waveform_np).unsqueeze(0).float()
            return waveform, sr
        except TimeoutError:
            raise TimeoutError(f"Audio loading timed out for {path}")


# --------------------
# Dataset
# --------------------
class ChunkedWaveformDataset(Dataset):
    def __init__(
        self,
        data,
        bos_token,
        eos_token,
        pad_token,
        tokenizer,
        max_length: int = 256,
        difficulties=['Expert'],
        instruments=['Single'],
        window_seconds: float = 10.0,
        sample_rate: int = 16000,
        num_pieces: int = 1,
        max_cache_gb: float = 2.0,
        chunk_size: int = None,
        chunk_repeat: int = 1,
        shuffle_chunks: bool = True,
        conditional: bool = False,
        augment: bool = False,
    ):
        self.data = data
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.difficulties = difficulties
        self.instruments = instruments
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * window_seconds)
        self.num_pieces = num_pieces
        self.max_cache_gb = max_cache_gb
        self.chunk_repeat = chunk_repeat
        self.shuffle_chunks = shuffle_chunks
        self.conditional = conditional
        self.augment = augment

        self.chart_processor = ChartProcessor(difficulties, instruments)
        self._audio_cache = {}
        self._cache_lock = threading.RLock()

        # Estimate chunk size if not given
        if chunk_size is None:
            self.chunk_size = self._estimate_chunk_size()
        else:
            self.chunk_size = chunk_size

        # Prepare chunk order
        self.num_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size
        self.chunk_order = list(range(self.num_chunks))
        print('NUM chunks: ', self.num_chunks)
        print('CHUNK order: ', self.chunk_order)
        if self.shuffle_chunks:
            random.shuffle(self.chunk_order)

        self.current_chunk_idx = -1
        self.current_chunk = []

    def _estimate_chunk_size(self, sample_n=3):
        """Estimate how many files can fit into max_cache_gb."""
        sample_files = random.sample(self.data, min(sample_n, len(self.data)))
        sizes = []
        for item in sample_files:
            waveform, _ = load_audio_librosa(item["audio_path"], self.sample_rate, max_duration=30)
            sizes.append(waveform.element_size() * waveform.numel())
        avg_size = np.mean(sizes) if sizes else 1
        max_bytes = self.max_cache_gb * (1024 ** 3)
        max_files = int(max_bytes // avg_size)
        return max(1, min(max_files, len(self.data)))

    def _load_audio_file(self, audio_path):
        """Load audio, use cache if available."""
        with self._cache_lock:
            if audio_path in self._audio_cache:
                return self._audio_cache[audio_path]
        
        #print(f"[DEBUG] Loading audio: {audio_path}")
        waveform, sr = load_audio_librosa_safe(audio_path, self.sample_rate)
        with self._cache_lock:
            self._audio_cache[audio_path] = (waveform, sr)
        #print(f"[DEBUG] Loaded audio: {audio_path}")
        return waveform, sr

    def _load_chunk(self, chunk_idx):
        """Load a chunk into memory and clear previous cache."""
        #print('Chunk idx: ', chunk_idx)
        self.current_chunk = self.data[chunk_idx * self.chunk_size : (chunk_idx + 1) * self.chunk_size]
        #print('Items in chunk: ', len(self.current_chunk))
        self._audio_cache.clear()
        self.current_chunk_idx = chunk_idx

    def _augment(self, waveform):
        if not self.augment:
            return waveform
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

        # chart + tokens
        self.chart_processor.read_chart(chart_path=item["chart_path"], target_sections=item["difficulty"])
        notes = self.chart_processor.notes[item["difficulty"]]
        bpm_events = self.chart_processor.synctrack
        resolution = int(self.chart_processor.song_metadata['Resolution'])
        offset = float(self.chart_processor.song_metadata['Offset'])

        start_seconds = start_sample / self.sample_rate
        end_seconds = end_sample / self.sample_rate

        tokenized_chart = self.tokenizer.encode(note_list=notes)
        tokenized_chart = self.tokenizer.format_seconds(
            tokenized_chart, bpm_events, resolution=resolution, offset=offset
        )

        filtered = [(t, v, d) for (t, v, d, _) in tokenized_chart if start_seconds <= t < end_seconds]

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

    def __len__(self):
        return len(self.data) * self.chunk_repeat

    def __getitem__(self, idx):
        #print(f"[DEBUG] __getitem__ start idx={idx}")
        relative_idx = idx % len(self.data)
        chunk_idx = relative_idx // self.chunk_size
        real_chunk_idx = self.chunk_order[chunk_idx]

        if real_chunk_idx != self.current_chunk_idx:
            self._load_chunk(real_chunk_idx)

        item_idx_in_chunk = relative_idx % self.chunk_size
        if item_idx_in_chunk >= len(self.current_chunk):
            item_idx_in_chunk = len(self.current_chunk) - 1
        item = self.current_chunk[item_idx_in_chunk]

        waveform, sr = self._load_audio_file(item["audio_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        results = [self._process_window(waveform.clone(), item) for _ in range(self.num_pieces)]
        #print(f"[DEBUG] __getitem__ end idx={idx}")
        return results


# --------------------
# Collator
# --------------------
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

        max_batch_len = min(
            max(len(sample["note_values"]) for sample in flat_batch) + 2,
            self.max_length
        )

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


# --------------------
# Factory
# --------------------
def create_chunked_audio_chart_dataloader(
        data,
        tokenizer,
        window_seconds = 30,
        difficulties=['Expert'],
        instruments=['Single'],
        batch_size=32,
        max_length=512,
        num_workers=8,
        shuffle_chunks=True,
        conditional=False,
        num_pieces=16,
        max_cache_gb=100,
        chunk_size=None,
        chunk_repeat=10,
):
    # add special tokens if missing
    vocab = tokenizer.mapping_noteseqs2int
    if '<bos>' not in vocab:
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

    dataset = ChunkedWaveformDataset(
        data=data,
        bos_token=vocab['<bos>'],
        eos_token=vocab['<eos>'],
        pad_token=vocab['<PAD>'],
        tokenizer=tokenizer,
        difficulties=difficulties,
        instruments=instruments,
        window_seconds=window_seconds,
        num_pieces=num_pieces,
        max_cache_gb=max_cache_gb,
        chunk_size=chunk_size,
        chunk_repeat=chunk_repeat,
        shuffle_chunks=shuffle_chunks,
        conditional=conditional,
    )

    print(f"[INFO] Using chunk_size={dataset.chunk_size} (auto from max_cache_gb={max_cache_gb} GB)")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # chunk shuffle handled internally
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True
    )
    return dataloader, vocab

