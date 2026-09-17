import logging
import os
from collections import OrderedDict
import random
import subprocess
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import platform


from transformers import AutoProcessor

from chart.chart_processor import ChartProcessor
from dataloader.audio_io import MusicAugmenter, load_opus_ffmpeg, load_raw_audio


logger = logging.getLogger(__name__)
ERROR_POLICIES = {"strict", "skip"}

DIFF_MAPPING = {"Expert": 0, "Hard": 1, "Medium": 2, "Easy": 3}


# --------------------
# Optimized Dataset
# --------------------

class ChunkedWaveformDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        bos_token: int,
        eos_token: int,
        pad_token: int,
        tokenizer,
        max_length: int = 256,
        difficulties: Optional[List[str]] = None,
        instruments: Optional[List[str]] = None,
        window_seconds: float = 10.0,
        sample_rate: int = 16000,
        num_pieces: int = 1,
        max_cache_gb: float = 2.0,
        chunk_size: Optional[int] = None,
        chunk_repeat: int = 1,
        shuffle_chunks: bool = True,
        conditional: bool = False,
        augment: bool = False,
        use_predecoded_raw: bool = False,        
        precomputed_windows: bool = False,       
        decode_to_raw_on_init: bool = False,     
        raw_dir: str = "raw_audio",              # where to store when converting
        is_discrete: bool = False,
        grid_ms: int = 20,
        error_policy: str = "strict",
        max_sample_retries: int = 5,
        chart_cache_size: int = 512,
    ):
        self.data = data
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.tokenizer = tokenizer
        self.max_length = max_length
        if not data:
            raise ValueError("Audio dataset cannot be empty")
        if error_policy not in ERROR_POLICIES:
            raise ValueError(f"error_policy must be one of {sorted(ERROR_POLICIES)}")
        if max_sample_retries < 0:
            raise ValueError("max_sample_retries must be non-negative")
        self.difficulties = difficulties or ["Expert"]
        self.instruments = instruments or ["Single"]
        self.error_policy = error_policy
        self.max_sample_retries = max_sample_retries
        self.window_seconds = window_seconds
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * window_seconds)
        self.num_pieces = num_pieces
        self.max_cache_gb = max_cache_gb
        self.chunk_repeat = chunk_repeat
        self.shuffle_chunks = shuffle_chunks
        self.conditional = conditional
        self.augment = augment
        self.use_predecoded_raw = use_predecoded_raw
        self.precomputed_windows = precomputed_windows
        self.decode_to_raw_on_init = decode_to_raw_on_init
        self.raw_dir = raw_dir
        self.is_discrete = is_discrete
        self.grid_ms = grid_ms

        self.chart_processor = ChartProcessor(self.difficulties, self.instruments)
        self.music_augmenter = MusicAugmenter(enabled=augment, sample_rate=self.sample_rate)

        # Chart metadata is cached lazily and bounded. Caching all of it up front cost about
        # 3.5 GB in this process -- and Windows spawns dataloader workers rather than forking,
        # so every worker received a pickled copy: measured at ~18 GB consumed in the two
        # minutes around worker spawn, leaving 0.5 GB free on a 32 GB machine. It also bought
        # very little, because each song is visited roughly once per epoch, so a full cache
        # only ever helps across epochs.
        self.chart_cache_size = max(1, int(chart_cache_size))
        self.chart_cache: "OrderedDict[Tuple[str, str], Optional[Tuple[List, List, int, float]]]" = OrderedDict()

        # If auto-converting .opus to .raw
        if self.decode_to_raw_on_init and not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir, exist_ok=True)
            logger.info("Converting Opus files to raw audio in %s", self.raw_dir)
            self._convert_all_to_raw()

        # Precompute windows if enabled
        if self.precomputed_windows:
            logger.info("Precomputing fixed audio windows")
            self.audio_windows = []  # list of (item_idx, start_sample, end_sample)
            for idx, item in enumerate(data):
                audio_path = item["audio_path"]
                if self.use_predecoded_raw:
                    # We need length_samples from metadata
                    if "length_samples" not in item:
                        raise ValueError(f"Item {idx} missing 'length_samples' for precomputed_windows with raw mode")
                    total_samples = item["length_samples"]
                else:
                    # Estimate from audio (slow, but only once)
                    try:
                        waveform, _ = load_opus_ffmpeg(audio_path, sample_rate, timeout_seconds=30)
                        total_samples = waveform.shape[-1]
                    except (RuntimeError, OSError) as error:
                        if self.error_policy == "strict":
                            raise
                        logger.warning("Cannot inspect %s: %s", audio_path, error)
                        continue
                num_windows = max(1, total_samples // self.num_samples)
                for i in range(num_windows):
                    start = i * self.num_samples
                    end = min(start + self.num_samples, total_samples)
                    window_key = item["raw_path"] if self.use_predecoded_raw else item["audio_path"]
                    self.audio_windows.append((window_key, start, end))
            logger.info("Precomputed %d windows", len(self.audio_windows))

        # Build flat item list (chunked + repeated)
        if chunk_size is None:
            chunk_size = self._estimate_chunk_size()

        self.chunk_size = chunk_size
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
                for item in chunk_data:
                    self.items.append((item, chunk_idx))

        # Worker-local state
        self._worker_id = None
        # Ordered so the oldest entry can be evicted; see _load_audio_file.
        self._audio_cache = OrderedDict()
        self._current_chunk_id = None

        logger.info("Dataset initialized with %d items and chunk size %d", len(self.items), chunk_size)

    def __getstate__(self):
        """Send workers an empty cache; each fills its own, bounded.

        Windows spawns dataloader workers rather than forking, so whatever this object
        holds is pickled into every one of them. A populated chart cache therefore
        multiplies by num_workers + 1.
        """
        state = self.__dict__.copy()
        state["chart_cache"] = OrderedDict()
        return state

    def _chart_entry(self, key: Tuple[str, str]):
        """Parsed chart for `key`, or None if it cannot be read. LRU-bounded."""
        if key in self.chart_cache:
            self.chart_cache.move_to_end(key)
            return self.chart_cache[key]

        chart_path, section = key
        entry = None
        try:
            self.chart_processor.read_chart(chart_path=chart_path, target_sections=section)
            entry = (
                self.chart_processor.notes[section],
                self.chart_processor.synctrack,
                int(self.chart_processor.song_metadata["Resolution"]),
                float(self.chart_processor.song_metadata["Offset"]),
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            if self.error_policy == "strict":
                raise RuntimeError(f"Failed to read chart {key}") from error
            logger.warning("Skipping invalid chart %s: %s", key, error)

        self.chart_cache[key] = entry
        while len(self.chart_cache) > self.chart_cache_size:
            self.chart_cache.popitem(last=False)
        return entry

    def _convert_all_to_raw(self):
        """Convert all .opus files to .raw in background (one-time setup)"""
        from concurrent.futures import ThreadPoolExecutor
        def convert_single(item):
            opus_path = item["audio_path"]
            if not opus_path.endswith(".opus"):
                return
            raw_path = os.path.join(self.raw_dir, os.path.basename(opus_path).replace(".opus", ".raw"))
            if os.path.exists(raw_path):
                return
            cmd = [
                'ffmpeg',
                '-i', opus_path,
                '-f', 's16le',
                '-ar', str(self.sample_rate),
                '-ac', '1',
                raw_path
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                # Update item to point to raw
                item["audio_path"] = raw_path
                item["length_samples"] = int(self.sample_rate * 120)  # safe estimate
            except (OSError, subprocess.SubprocessError) as e:
                logger.error("Failed to convert %s to raw audio: %s", opus_path, e)

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor:
            list(executor.map(convert_single, self.data))

    def _estimate_chunk_size(self, sample_n=3):
        sample_files = random.sample(self.data, min(sample_n, len(self.data)))
        sizes = []
        for item in sample_files:
            try:
                if self.use_predecoded_raw:
                    size_bytes = item.get("length_samples", self.sample_rate*60) * 2  # 16-bit = 2 bytes/sample
                else:
                    waveform, _ = load_opus_ffmpeg(item["audio_path"], self.sample_rate, timeout_seconds=30)
                    size_bytes = waveform.element_size() * waveform.numel()
                sizes.append(size_bytes)
            except (RuntimeError, OSError):
                sizes.append(100 * 1024 * 1024)  # 100MB fallback
        avg_size = np.mean(sizes) if sizes else 100 * 1024 * 1024
        max_bytes = self.max_cache_gb * (1024 ** 3)
        max_files = int(max_bytes // avg_size)
        return max(1, min(max_files, len(self.data) // 4))

    def _get_worker_id(self):
        worker_info = torch.utils.data.get_worker_info()
        return worker_info.id if worker_info else 0

    def _should_clear_cache(self, chunk_id):
        worker_id = self._get_worker_id()
        if self._worker_id != worker_id:
            self._worker_id = worker_id
            self._audio_cache.clear()
            self._current_chunk_id = None
            return True
        if self._current_chunk_id != chunk_id:
            self._audio_cache.clear()
            self._current_chunk_id = chunk_id
            return True
        return False

    AUDIO_CACHE_ENTRIES = 32
    """Decoded songs held per worker. At ~10.8 MB each that is ~350 MB per worker,
    which four workers can afford; the `max_cache_gb` budget could not be trusted for
    this because it is per worker and sized in gigabytes."""

    def _load_audio_file(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Decoded audio for one song, reusing a recent read where possible.

        This cache was read but never written, so every sample re-read the whole file.
        Populating it is worth doing -- at 15 s windows with one piece a read serves a
        quarter of what it served at 30 s and two, so four times the I/O -- but it is a
        small win, and an earlier version of this note claimed far more than that.

        CORRECTION. That version said the dead cache was why "the GPU starves:
        utilisation measured swinging 87% -> 2%". It was not. Lightning's profiler over
        256 batches of the real training config puts train_dataloader_next at 0.00048 s
        against a 0.358 s batch -- 0.1%. The loader has never been the bottleneck at
        either window size, and filling this cache moved throughput 0.77 -> 0.82 it/s.

        The 87% -> 2% swing was real but had another cause: val_check_interval counts
        batches while ModelCheckpoint's every_n_train_steps counts optimizer steps, so a
        run launched with both at 100 under accumulate_grad_batches=8 validated once per
        training batch and wrote a 2.42 GB checkpoint to the 108 MB/s song drive every
        100 batches. Fixing that took the same config from 0.82 to 2.79 it/s. The GPU was
        idle waiting on disk, which looks identical from nvidia-smi.

        Bounded by count rather than by the `max_cache_gb` budget, because that budget
        is what made populating this risky: 6 GB per worker across four workers is the
        host-RAM exhaustion the chart cache already caused once. A fixed number of
        entries is predictable -- 32 songs at ~10.8 MB is ~350 MB per worker.
        """
        cached = self._audio_cache.get(audio_path)
        if cached is not None:
            self._audio_cache.move_to_end(audio_path)
            return cached

        try:
            if self.use_predecoded_raw:
                # Must have 'length_samples' in item metadata
                waveform, sr = load_raw_audio(audio_path, self.sample_rate)  # load longer buffer
            else:
                waveform, sr = load_opus_ffmpeg(audio_path, self.sample_rate, timeout_seconds=20)
            self._audio_cache[audio_path] = (waveform, sr)
            while len(self._audio_cache) > self.AUDIO_CACHE_ENTRIES:
                self._audio_cache.popitem(last=False)
            return waveform, sr

        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                f"Worker {self._get_worker_id()} failed to load {audio_path}"
            ) from error


    def _augment(self, waveform):
        return self.music_augmenter(waveform)

    def _process_window(self, waveform: torch.Tensor, item: Dict, start_sample: int, end_sample: int) -> Dict:

        #worker_id = self._get_worker_id()
        #logger.info(f"[WORKER {worker_id}] Starting _process_window for {item['raw_path']} @ {start_sample}:{end_sample}", extra={'worker_id': worker_id})

        # Extract window (may be shorter than self.num_samples if file is short)
        chunk = waveform[:, start_sample:end_sample]  # Shape: [1, T], T <= self.num_samples

        actual_len = chunk.shape[-1]
        if actual_len < self.num_samples:
            # Pad with silence at the end
            pad_amount = self.num_samples - actual_len
            chunk = torch.nn.functional.pad(chunk, (0, pad_amount), mode='constant', value=0.0)
        elif actual_len > self.num_samples:
            # Enforce the public fixed-window invariant.
            chunk = chunk[:, :self.num_samples]

        # Chunk is guaranteed to be [1, self.num_samples]
        if chunk.shape[-1] != self.num_samples:
            raise RuntimeError(f"Chunk shape {chunk.shape} does not match {self.num_samples}")

        if self.augment:
            chunk = chunk.clone()  # only clone when augmenting
            chunk = self._augment(chunk)

        #logger.info(f"[WORKER {worker_id}] Window extracted, now processing chart...", extra={'worker_id': worker_id})


        try:
            key = (item["chart_path"], item["difficulty"])
            entry = self._chart_entry(key)
            if entry is None:
                raise ValueError("Chart was marked as bad")
            notes, bpm_events, resolution, offset = entry

            #logger.info(f"[WORKER {worker_id}] Got chart cache, encoding notes...", extra={'worker_id': worker_id})

            start_seconds = start_sample / self.sample_rate
            end_seconds = end_sample / self.sample_rate

            tokenized_chart = self.tokenizer.encode(note_list=notes, resolution=resolution)
            #logger.info(f"[WORKER {worker_id}] Tokenized {len(tokenized_chart)} notes", extra={'worker_id': worker_id})
            tokenized_chart = self.tokenizer.format_seconds(
                tokenized_chart, bpm_events, resolution=resolution, offset=offset
            )
            #logger.info(f"[WORKER {worker_id}] Formatted seconds, got {len(tokenized_chart)} events", extra={'worker_id': worker_id})
            filtered = [(t, v, d) for (t, v, d, _) in tokenized_chart if start_seconds <= t < end_seconds]
            
            if self.is_discrete:

                #Extract time and token lists
                time_list = [note[0] for note in filtered]
                tokens_list = [note[1] for note in filtered]
                
                
                grid = self.tokenizer.discretize_time(
                            time_list, 
                            tokens_list, 
                            self.pad_token,
                            grid_ms=self.grid_ms, 
                            window_seconds=self.window_seconds,
                            start_time=start_seconds
                        )
                
                diff = self._difficulty(item["difficulty"])

                return {
                    "audio": chunk,
                    "note_values": grid,
                    "cond_diff": diff,
                }

            else:

                #logger.info(f"[WORKER {worker_id}] Filtered {len(filtered)} events in window", extra={'worker_id': worker_id})

                if filtered:
                    note_times, note_values, note_durations = map(list, zip(*filtered))
                    note_times = [(t - start_seconds) / self.window_seconds for t in note_times]
                    note_times = [note_times[0]] + [
                        note_times[i] - note_times[i - 1] for i in range(1, len(note_times))
                    ]
                else:
                    note_times, note_values, note_durations = [], [], []

                diff = self._difficulty(item["difficulty"])

                return {
                    "audio": chunk,
                    "note_times": note_times,
                    "note_values": note_values,
                    "note_durations": note_durations,
                    "cond_diff": diff,
                }

        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Chart window processing failed") from error

    def _difficulty(self, section):
        if not self.conditional:
            return [-1]
        for difficulty, mapped in DIFF_MAPPING.items():
            if section.startswith(difficulty):
                return [mapped]
        raise ValueError(f"Unknown difficulty section: {section}")


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> List[Dict]:
        max_attempts = 1 if self.error_policy == "strict" else self.max_sample_retries + 1
        for attempt in range(max_attempts):
            try:
                current_idx = idx if attempt == 0 else random.randint(0, len(self.items) - 1)
                item, chunk_id = self.items[current_idx]

                self._should_clear_cache(chunk_id)

                # Load audio
                audio_path = item["raw_path"] if self.use_predecoded_raw else item["audio_path"]
                waveform, sr = self._load_audio_file(audio_path)

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Validate the chart is readable before doing any window work.
                key = (item["chart_path"], item["difficulty"])
                if self._chart_entry(key) is None:
                    if attempt < max_attempts - 1:
                        continue
                    else:
                        raise ValueError(f"Chart {key} failed to load")

                # Generate multiple windows
                results = []
                for _ in range(self.num_pieces):
                    try:
                        if self.precomputed_windows:
                            # Pick a random precomputed window belonging to this item
                            item_windows = [
                                window for window in self.audio_windows
                                if window[0] == audio_path
                            ]
                            if not item_windows:
                                raise RuntimeError(f"No windows found for item {item}")
                            _, start_sample, end_sample = random.choice(item_windows)
                        else:
                            # Random window within audio
                            max_start = max(0, waveform.shape[-1] - self.num_samples)
                            start_sample = random.randint(0, max_start) if max_start > 0 else 0
                            end_sample = start_sample + self.num_samples
                            if end_sample > waveform.shape[-1]:
                                end_sample = waveform.shape[-1]
                                start_sample = end_sample - self.num_samples
                                if start_sample < 0:
                                    start_sample = 0

                        result = self._process_window(waveform, item, start_sample, end_sample)
                        results.append(result)
                    except (KeyError, RuntimeError, TypeError, ValueError) as error:
                        raise RuntimeError(
                            f"Worker {self._get_worker_id()} failed to process a window"
                        ) from error
                return results

            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                logger.warning(
                    "Worker %d failed dataset item %d (attempt %d/%d): %s",
                    self._get_worker_id(), current_idx, attempt + 1, max_attempts, error,
                )
                if attempt < max_attempts - 1:
                    continue
                raise RuntimeError(
                    f"Dataset item {idx} failed after {max_attempts} attempt(s)"
                ) from error


# --------------------
# Collator
# --------------------

def _collate_batch_impl(batch: List[List[Dict]], bos_token: int, eos_token: int, pad_token: int, max_length: int, conditional: bool, processor: Optional[object] = None) -> Dict:
    """
    Collate a batch of samples into a dictionary for training.
    Args:
        batch: List of lists of dictionaries, each containing 'audio', 'note_values', 'note_times', 'note_durations', 'cond_diff'.
        bos_token: Beginning-of-sequence token ID.
        eos_token: End-of-sequence token ID.
        pad_token: Padding token ID for note values.
        max_length: Maximum sequence length for padding.
        conditional: Whether to include 'cond_diff' in output.
        processor: Audio processor (facebook/Encodec). If None, use raw audio directly.
    Returns:
        Dict with 'input_values', 'padding_mask', 'note_values', 'note_times', 'note_durations', 'attention_mask', and optionally 'cond_diff'.
    """
    flat_batch = [sample for sublist in batch for sample in sublist]
    if not flat_batch:
        return {}

    max_batch_len = min(
        max(len(sample["note_values"]) for sample in flat_batch) + 2,  # +2 for BOS/EOS
        max_length
    )

    batch_input_values, batch_padding_mask, batch_note_values = [], [], []
    batch_note_times, batch_note_durations, attention_masks, batch_diff = [], [], [], []

    for sample in flat_batch:
        audio = sample['audio']  # Expected: [1, T] float tensor
        note_times = [0.0] + sample["note_times"] + [1.0]
        note_durations = [0.0] + sample["note_durations"] + [0.0]
        note_values = [bos_token] + sample["note_values"] + [eos_token]

        # Truncate to max_batch_len
        note_times = note_times[:max_batch_len]
        note_durations = note_durations[:max_batch_len]
        note_values = note_values[:max_batch_len]

        seq_len = len(note_values)
        pad_len = max_batch_len - seq_len

        # Pad sequences
        padded_values = note_values + [pad_token] * pad_len
        padded_times = note_times + [0.0] * pad_len
        padded_durations = note_durations + [0.0] * pad_len
        attention_mask = [1] * (seq_len - 1) + [0] * (max_batch_len - seq_len + 1)

        if processor is not None:
            # Process audio with processor
            inputs = processor(
                raw_audio=audio.squeeze(),  # Remove channel dim if [1, T]
                sampling_rate=processor.sampling_rate,
                return_tensors='pt'
            )
            input_values = inputs["input_values"].squeeze(0)  # [1, channels=1, seq_len] -> [seq_len]
            padding_mask = inputs["padding_mask"].squeeze(0)  # [1, seq_len] -> [seq_len]
        else:
            # Use raw audio directly
            input_values = audio.squeeze(0)  # [1, T] -> [T]
            padding_mask = torch.ones_like(input_values, dtype=torch.long)  # Assume no padding

        batch_input_values.append(input_values)
        batch_padding_mask.append(padding_mask)
        batch_note_values.append(padded_values)
        batch_note_times.append(padded_times)
        batch_note_durations.append(padded_durations)
        attention_masks.append(attention_mask)
        if conditional:
            batch_diff.append(sample["cond_diff"])

    # Stack and ensure correct types
    output = {
        "input_values": torch.stack(batch_input_values, dim=0).float(),
        "padding_mask": torch.stack(batch_padding_mask, dim=0).long(),
        "note_values": torch.tensor(batch_note_values, dtype=torch.long),
        "note_times": torch.tensor(batch_note_times, dtype=torch.float),
        "note_durations": torch.tensor(batch_note_durations, dtype=torch.float),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }
    if conditional:
        output["cond_diff"] = torch.tensor(batch_diff, dtype=torch.long)

    return output



def _collate_batch_impl_discrete(batch: List[List[Dict]], bos_token: int, eos_token: int, conditional: bool, processor: Optional[object] = None) -> Dict:
    """
    Collate a batch of samples into a dictionary for training.
    Args:
        batch: List of lists of dictionaries, each containing 'audio', 'note_values', 'cond_diff'.
        bos_token: Beginning-of-sequence token ID.
        eos_token: End-of-sequence token ID.
        conditional: Whether to include 'cond_diff' in output.
        processor: Audio processor (facebook/Encodec). If None, use raw audio directly.
    Returns:
        Dict with 'input_values', 'padding_mask', 'note_values', and optionally 'cond_diff'.
    """
    flat_batch = [sample for sublist in batch for sample in sublist]
    batch_audio, batch_note_values, batch_diff, batch_padding_mask = [], [], [], []

    for sample in flat_batch:
        audio = sample['audio']  # Expected: [1, T] float tensor
        note_values = [bos_token] + sample["note_values"] + [eos_token]
        
        if processor is not None:
            # Process audio with processor
            inputs = processor(
                raw_audio=audio.squeeze(),  # Remove channel dim if [1, T]
                sampling_rate=processor.sampling_rate,
                return_tensors='pt'
            )
            input_values = inputs["input_values"].squeeze(0)  # [channels=1, sequence_length] -> [sequence_length]
            padding_mask = inputs["padding_mask"].squeeze(0)  # [1, sequence_length] -> [sequence_length]
        else:
            # Use raw audio directly
            input_values = audio.squeeze(0)  # [1, T] -> [T]
            padding_mask = torch.ones_like(input_values, dtype=torch.long)  # Assume no padding

        batch_audio.append(input_values)
        batch_padding_mask.append(padding_mask)
        batch_note_values.append(note_values)
        if conditional:
            batch_diff.append(sample["cond_diff"])

    # Stack and ensure correct types
    output = {
        "input_values": torch.stack(batch_audio, dim=0).float(),
        "padding_mask": torch.stack(batch_padding_mask, dim=0).long(),
        "note_values": torch.tensor(batch_note_values, dtype=torch.long),
    }
    if conditional:
        output["cond_diff"] = torch.tensor(batch_diff, dtype=torch.long)

    return output

class AudioChartCollator:
    def __init__(
            self, 
            bos_token: int, 
            eos_token: int, 
            pad_token: int = -100, 
            max_length: int = 512, 
            conditional: bool = False, 
            is_discrete: bool = False,
            use_processor: bool = False,
            processor_checkpoint: str = "facebook/encodec_24khz",
            sample_rate: Optional[int] = None,
        ):

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.conditional = conditional
        if use_processor:
            self.processor = AutoProcessor.from_pretrained(processor_checkpoint)
            if sample_rate is not None and self.processor.sampling_rate != sample_rate:
                raise ValueError(
                    f"Processor {processor_checkpoint} requires {self.processor.sampling_rate} Hz, "
                    f"but the dataloader uses {sample_rate}"
                )
        else:
            self.processor = None

        if is_discrete:
            self._collate_fn = _collate_batch_impl_discrete
            self._args = (bos_token, eos_token, conditional, self.processor)
        else:
            self._collate_fn = _collate_batch_impl
            self._args = (bos_token, eos_token, pad_token, max_length, conditional, self.processor)


    def __call__(self, batch: List[List[Dict]]) -> Dict:
        return self._collate_fn(batch, *self._args)
    
# --------------------
# Dataloader Factory
# --------------------

def create_chunked_audio_chart_dataloader(
    data: List[Dict],
    tokenizer,
    window_seconds: float = 10.0,
    sample_rate: int = 16000,
    difficulties: Optional[List[str]] = None,
    instruments: Optional[List[str]] = None,
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 8,
    shuffle_chunks: bool = True,
    conditional: bool = False,
    num_pieces: int = 6,
    max_cache_gb: float = 20.0,
    chunk_size: Optional[int] = None,
    chunk_repeat: int = 1,
    use_predecoded_raw: bool = False,
    precomputed_windows: bool = False,
    decode_to_raw_on_init: bool = False,
    raw_dir: str = "raw_audio",
    augment: bool = False,
    is_discrete: bool = False,
    grid_ms: int =  20,
    use_processor: bool = False,
    processor_checkpoint: str = "facebook/encodec_24khz",
    error_policy: str = "strict",
    max_sample_retries: int = 5,
    chart_cache_size: int = 512,
) -> Tuple[DataLoader, Dict]:
    """
    Set use_predecoded_raw=True and predecode files with ffmpeg beforehand.
    """

    # Adjust cache per worker. Note this budget is per dataloader, and persistent_workers
    # keeps the training loader's workers alive while validation spawns its own set, so
    # the machine must hold roughly twice this figure at once. The 20 GB default therefore
    # means 40 GB of audio cache during validation, which on a 32 GB host kills the run at
    # the most memory-hungry moment -- serializing a checkpoint right after validation.
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
        conditional=conditional,
        is_discrete=is_discrete,
        use_processor=use_processor,
        processor_checkpoint=processor_checkpoint,
        sample_rate=sample_rate,
    )

    dataset = ChunkedWaveformDataset(
        data=data,
        bos_token=vocab['<bos>'],
        eos_token=vocab['<eos>'],
        pad_token=vocab['<PAD>'],
        tokenizer=tokenizer,
        difficulties=difficulties or ["Expert"],
        instruments=instruments or ["Single"],
        window_seconds=window_seconds,
        sample_rate=sample_rate,
        num_pieces=num_pieces,
        max_cache_gb=max_cache_gb,
        chunk_size=chunk_size,
        chunk_repeat=chunk_repeat,
        shuffle_chunks=shuffle_chunks,
        conditional=conditional,
        augment=augment,  # Enable in training if needed
        use_predecoded_raw=use_predecoded_raw,
        precomputed_windows=precomputed_windows,
        decode_to_raw_on_init=decode_to_raw_on_init,
        raw_dir=raw_dir,
        is_discrete=is_discrete,
        grid_ms=grid_ms,
        error_policy=error_policy,
        max_sample_retries=max_sample_retries,
        chart_cache_size=chart_cache_size,
    )

    logger.info("Created dataset with %d items", len(dataset))

    available_cpus = os.cpu_count() or 1
    effective_workers = min(num_workers, max(1, available_cpus // 2))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_chunks,
        num_workers=effective_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=shuffle_chunks,
        persistent_workers=effective_workers > 0,
        prefetch_factor=4 if effective_workers > 0 else None,
        timeout=60 if effective_workers > 0 else 0,
        multiprocessing_context=(
            "spawn" if platform.system() == "Windows" and effective_workers > 0 else None
        ),
    )
    return dataloader, vocab
