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
        
        # Pre-build item-to-data mapping to avoid chunking logic
        if chunk_size is None:
            chunk_size = self._estimate_chunk_size()
        
        self.chunk_size = chunk_size
        
        # Create a flat list of (data_item, chunk_id) tuples
        # Each worker will know exactly what to load
        self.items = []
        num_chunks = (len(data) + chunk_size - 1) // chunk_size
        
        for repeat in range(chunk_repeat):
            chunk_order = list(range(num_chunks))
            if shuffle_chunks:
                random.shuffle(chunk_order)
            
            for chunk_idx in chunk_order:
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(data))
                chunk_data = data[start_idx:end_idx]
                
                # Add each item in chunk with its chunk_id
                for item in chunk_data:
                    self.items.append((item, chunk_idx))
        
        # Worker-specific state (will be different per worker process)
        self._worker_id = None
        self._audio_cache = {}
        self._current_chunk_id = None

    def _estimate_chunk_size(self, sample_n=3):
        """Estimate chunk size"""
        sample_files = random.sample(self.data, min(sample_n, len(self.data)))
        sizes = []
        for item in sample_files:
            try:
                waveform, _ = load_audio_librosa(item["audio_path"], self.sample_rate, max_duration=30)
                sizes.append(waveform.element_size() * waveform.numel())
            except:
                sizes.append(100 * 1024 * 1024)  # 100MB fallback
        
        avg_size = np.mean(sizes) if sizes else 100 * 1024 * 1024
        max_bytes = self.max_cache_gb * (1024 ** 3)
        max_files = int(max_bytes // avg_size)
        return max(1, min(max_files, len(self.data) // 4))  # Conservative estimate

    def _get_worker_id(self):
        """Get current worker ID"""
        worker_info = torch.utils.data.get_worker_info()
        return worker_info.id if worker_info else 0

    def _should_clear_cache(self, chunk_id):
        """Check if we need to clear cache for new chunk"""
        worker_id = self._get_worker_id()
        
        # Initialize worker state
        if self._worker_id != worker_id:
            self._worker_id = worker_id
            self._audio_cache.clear()
            self._current_chunk_id = None
            return True
        
        # Clear cache if switching chunks
        if self._current_chunk_id != chunk_id:
            self._audio_cache.clear()
            self._current_chunk_id = chunk_id
            return True
            
        return False

    def _load_audio_file(self, audio_path):
        """Load audio with worker-local caching"""
        # Check cache first
        if audio_path in self._audio_cache:
            return self._audio_cache[audio_path]
        
        try:
            waveform, sr = load_audio_librosa_safe(audio_path, self.sample_rate, timeout_seconds=20)
            
            # Only cache if we have reasonable memory usage
            if len(self._audio_cache) < self.chunk_size * 2:
                self._audio_cache[audio_path] = (waveform, sr)
            
            return waveform, sr
            
        except Exception as e:
            print(f"Worker {self._get_worker_id()}: Failed to load {audio_path}: {e}")
            # Return silence instead of crashing
            empty_audio = torch.zeros(1, self.num_samples)
            return empty_audio, self.sample_rate

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
        
        # Handle short audio
        if waveform.shape[-1] < window_samples:
            padding = window_samples - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        max_start = waveform.shape[-1] - window_samples
        start_sample = random.randint(0, max_start) if max_start > 0 else 0
        end_sample = start_sample + window_samples
        chunk = waveform[:, start_sample:end_sample]

        if self.augment:
            chunk = self._augment(chunk)

        try:
            # Process chart
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
            
        except Exception as e:
            print(f"Worker {self._get_worker_id()}: Chart processing failed: {e}")
            # Return empty chart data
            return {
                "audio": chunk,
                "note_times": [],
                "note_values": [],
                "note_durations": [],
                "cond_diff": [-1],
            }

    def __len__(self):
        return len(self.items)


    def __getitem__(self, idx):
        max_attempts = 5
        
        for attempt in range(max_attempts):
            try:
                # Use original idx on first attempt, then sample random ones
                current_idx = idx if attempt == 0 else random.randint(0, len(self.items) - 1)
                item, chunk_id = self.items[current_idx]
                
                # Clear cache if we're switching chunks (per worker)
                self._should_clear_cache(chunk_id)
                
                # Load audio
                waveform, sr = self._load_audio_file(item["audio_path"])
                
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Test chart processing first before generating windows
                try:
                    self.chart_processor.read_chart(chart_path=item["chart_path"], target_sections=item["difficulty"])
                    notes = self.chart_processor.notes[item["difficulty"]]
                    
                    # If notes are empty, skip this item entirely
                    if not notes:
                        if attempt < max_attempts - 1:
                            continue
                        else:
                            raise ValueError("Empty notes in chart")
                            
                except Exception as e:
                    if attempt == 0:
                        print(f"Worker {self._get_worker_id()}: Chart processing failed for {item['chart_path']}: {e}")
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        raise e

                # Generate multiple windows - chart is known to be good
                results = []
                for _ in range(self.num_pieces):
                    try:
                        result = self._process_window(waveform.clone(), item)
                        results.append(result)
                    except Exception as e:
                        print(f"Worker {self._get_worker_id()}: Window processing failed: {e}")
                        # This should be rare now since chart was pre-validated
                        results.append({
                            "audio": torch.zeros(1, self.num_samples),
                            "note_times": [],
                            "note_values": [],
                            "note_durations": [],
                            "cond_diff": [-1],
                        })
                
                return results
                    
            except Exception as e:
                if attempt == 0:
                    print(f"Worker {self._get_worker_id()}: __getitem__({current_idx if 'current_idx' in locals() else idx}) failed: {e}")
                
                # Try next sample if not last attempt
                if attempt < max_attempts - 1:
                    continue
        
        # Final fallback
        print(f"Worker {self._get_worker_id()}: All {max_attempts} attempts failed for idx {idx}")
        empty_result = {
            "audio": torch.zeros(1, self.num_samples),
            "note_times": [],
            "note_values": [],
            "note_durations": [],
            "cond_diff": [-1],
        }
        return [empty_result] * self.num_pieces


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
        num_pieces=6,
        max_cache_gb=10,  # Reduced default for per-worker memory
        chunk_size=None,
        chunk_repeat=1,
):
    # Adjust cache per worker
    if num_workers > 0:
        max_cache_gb = max_cache_gb / num_workers
    
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

    print(f"[INFO] Total items: {len(dataset)}, chunk_size={dataset.chunk_size}, workers={num_workers}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Now safe to shuffle since items are pre-built
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return dataloader, vocab
