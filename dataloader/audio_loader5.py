import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import os
import random
import warnings
import threading
import subprocess
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, List, Optional, Tuple
import platform

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
# Audio loading (FFMPEG - FASTEST for .opus)
# --------------------

def load_opus_ffmpeg(path: str, target_sr: int = 16000, timeout_seconds: int = 10) -> Tuple[torch.Tensor, int]:
    cmd = [
        'ffmpeg',
        '-i', path,
        '-f', 's16le',
        '-ar', str(target_sr),
        '-ac', '1',
        '-'
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8  # Large buffer to prevent pipe stalls
        )

        # Use communicate() with timeout — this is safer than run()
        stdout, stderr = proc.communicate(timeout=timeout_seconds)

        if proc.returncode != 0:
            raise RuntimeError(f"FFMPEG failed with code {proc.returncode} for {path}: {stderr.decode()}")

        audio_np = np.frombuffer(stdout, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(audio_np).unsqueeze(0)
        return waveform, target_sr

    except subprocess.TimeoutExpired:
        proc.kill()  # Explicitly kill if timeout
        proc.wait()  # Wait for cleanup
        raise RuntimeError(f"FFMPEG timed out after {timeout_seconds}s for {path}")

    except Exception as e:
        raise RuntimeError(f"Failed to load {path} with ffmpeg: {e}")

def load_raw_audio(path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Load the ENTIRE pre-decoded raw 16-bit PCM audio (no header).
    Must be preprocessed with ffmpeg: ffmpeg -i input.opus -f s16le -ar 16000 -ac 1 output.raw
    Returns: [1, T] tensor, sample_rate
    """
    try:
        with open(path, 'rb') as f:
            buf = f.read()
        original_size = len(buf)
        #if original_size % 2 != 0:
           # print(f"Warning: {path} has {original_size} bytes (not multiple of 2), trimming last byte")
        #    buf = buf[:-1]
        audio_np = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = torch.from_numpy(audio_np).unsqueeze(0)  # [1, T]
        return waveform, target_sr
    except Exception as e:
        raise RuntimeError(f"Failed to load raw audio {path}: {e}")


from chart.chart_processor import ChartProcessor


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
        difficulties: List[str] = ['Expert'],
        instruments: List[str] = ['Single'],
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
        self.use_predecoded_raw = use_predecoded_raw
        self.precomputed_windows = precomputed_windows
        self.decode_to_raw_on_init = decode_to_raw_on_init
        self.raw_dir = raw_dir

        self.chart_processor = ChartProcessor(difficulties, instruments)

        # Pre-cache chart data (CRITICAL optimization)
        self.chart_cache: Dict[Tuple[str, str], Tuple[List, List, int, float]] = {}
        print("[INFO] Pre-caching chart metadata...")
        for item in data:
            key = (item["chart_path"], item["difficulty"])
            if key not in self.chart_cache:
                try:
                    self.chart_processor.read_chart(chart_path=item["chart_path"], target_sections=item["difficulty"])
                    notes = self.chart_processor.notes[item["difficulty"]]
                    bpm_events = self.chart_processor.synctrack
                    resolution = int(self.chart_processor.song_metadata['Resolution'])
                    offset = float(self.chart_processor.song_metadata['Offset'])
                    self.chart_cache[key] = (notes, bpm_events, resolution, offset)
                except Exception as e:
                    print(f"[WARNING] Failed to cache chart {key}: {e}")
                    self.chart_cache[key] = None  # mark as bad

        # If auto-converting .opus to .raw
        if self.decode_to_raw_on_init and not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir, exist_ok=True)
            print(f"[INFO] Converting all .opus files to .raw in {self.raw_dir} (this may take time)...")
            self._convert_all_to_raw()

        # Precompute windows if enabled
        if self.precomputed_windows:
            print("[INFO] Precomputing fixed audio windows...")
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
                    except:
                        total_samples = 16000 * 120  # fallback: 2 min
                num_windows = max(1, total_samples // self.num_samples)
                for i in range(num_windows):
                    start = i * self.num_samples
                    end = min(start + self.num_samples, total_samples)
                    self.audio_windows.append((idx, start, end))
            print(f"[INFO] Precomputed {len(self.audio_windows)} windows.")

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
        self._audio_cache = {}
        self._current_chunk_id = None

        print(f"[INFO] Dataset initialized with {len(self.items)} items, chunk_size={chunk_size}, workers will reuse cache.")

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
            except Exception as e:
                print(f"[ERROR] Failed to convert {opus_path} to raw: {e}")

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count())) as executor:
            list(executor.map(convert_single, self.data))

    def _estimate_chunk_size(self, sample_n=3):
        sample_files = random.sample(self.data, min(sample_n, len(self.data)))
        sizes = []
        for item in sample_files:
            try:
                if self.use_predecoded_raw:
                    size_bytes = item.get("length_samples", 16000*60) * 2  # 16-bit = 2 bytes/sample
                else:
                    waveform, _ = load_opus_ffmpeg(item["audio_path"], self.sample_rate, timeout_seconds=30)
                    size_bytes = waveform.element_size() * waveform.numel()
                sizes.append(size_bytes)
            except:
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

    def _load_audio_file(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        if audio_path in self._audio_cache:
            return self._audio_cache[audio_path]

        try:
            if self.use_predecoded_raw:
                # Must have 'length_samples' in item metadata
                waveform, sr = load_raw_audio(audio_path, self.sample_rate)  # load longer buffer
            else:
                waveform, sr = load_opus_ffmpeg(audio_path, self.sample_rate, timeout_seconds=20)

            if len(self._audio_cache) < self.chunk_size * 2:
                self._audio_cache[audio_path] = (waveform, sr)
            return waveform, sr

        except Exception as e:
            print(f"Worker {self._get_worker_id()}: Failed to load {audio_path}: {e}")
            return torch.zeros(1, self.num_samples), self.sample_rate

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
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

    def _process_window(self, waveform: torch.Tensor, item: Dict, start_sample: int, end_sample: int) -> Dict:
        # Extract window (may be shorter than self.num_samples if file is short)
        chunk = waveform[:, start_sample:end_sample]  # Shape: [1, T], T <= self.num_samples

        actual_len = chunk.shape[-1]
        if actual_len < self.num_samples:
            # Pad with silence at the end
            pad_amount = self.num_samples - actual_len
            chunk = torch.nn.functional.pad(chunk, (0, pad_amount), mode='constant', value=0.0)
        elif actual_len > self.num_samples:
            # Shouldn't happen due to max_start logic, but just in case
            chunk = chunk[:, :self.num_samples]

        # Now chunk is guaranteed to be [1, self.num_samples]
        assert chunk.shape[-1] == self.num_samples, f"Chunk shape {chunk.shape} != {self.num_samples}"

        if self.augment:
            chunk = chunk.clone()  # only clone when augmenting
            chunk = self._augment(chunk)

        try:
            key = (item["chart_path"], item["difficulty"])
            notes, bpm_events, resolution, offset = self.chart_cache[key]
            if notes is None:
                raise ValueError("Chart was marked as bad")

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
            # Still return fixed-size audio tensor
            return {
                "audio": chunk,  # Already padded to [1, self.num_samples]
                "note_times": [],
                "note_values": [],
                "note_durations": [],
                "cond_diff": [-1],
            }


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> List[Dict]:
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                current_idx = idx if attempt == 0 else random.randint(0, len(self.items) - 1)
                item, chunk_id = self.items[current_idx]

                self._should_clear_cache(chunk_id)

                # Load audio
                waveform, sr = self._load_audio_file(item["raw_path"])

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Validate chart exists (already cached)
                key = (item["chart_path"], item["difficulty"])
                if self.chart_cache.get(key) is None:
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
                            item_windows = [w for w in self.audio_windows if w[0] == self.items[current_idx][0]]
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
                    except Exception as e:
                        print(f"Worker {self._get_worker_id()}: Window processing failed: {e}")
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
                    print(f"Worker {self._get_worker_id()}: __getitem__({current_idx}) failed: {e}")
                if attempt < max_attempts - 1:
                    continue

        print(f"Worker {self._get_worker_id()}: All {max_attempts} attempts failed for idx {idx}")
        empty_result = {
            "audio": torch.zeros(1, self.num_samples),
            "note_times": [],
            "note_values": [],
            "note_durations": [],
            "cond_diff": [-1],
        }
        return [empty_result] * self.num_pieces


# --------------------
# Optimized Collator (with optional torch.compile)
# --------------------

def _collate_batch_impl(batch: List[List[Dict]], bos_token: int, eos_token: int, pad_token: int, max_length: int, conditional: bool) -> Dict:
    flat_batch = [sample for sublist in batch for sample in sublist]
    if not flat_batch:
        return {}

    max_batch_len = min(
        max(len(sample["note_values"]) for sample in flat_batch) + 2,
        max_length
    )

    batch_audio, batch_note_values, batch_note_times, batch_note_durations = [], [], [], []
    attention_masks, batch_diff = [], []

    for sample in flat_batch:
        audio = sample['audio']
        note_times = [0.0] + sample["note_times"] + [1.0]
        note_durations = [0.0] + sample["note_durations"] + [0.0]
        note_values = [bos_token] + sample["note_values"] + [eos_token]

        note_times = note_times[:max_batch_len]
        note_durations = note_durations[:max_batch_len]
        note_values = note_values[:max_batch_len]

        seq_len = len(note_values)
        pad_len = max_batch_len - seq_len

        padded_values = note_values + [pad_token] * pad_len
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
        "cond_diff": torch.tensor(batch_diff, dtype=torch.long) if conditional else None,
    }

class AudioChartCollator:
    def __init__(self, bos_token: int, eos_token: int, pad_token: int = -100, max_length: int = 512, conditional: bool = False):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.conditional = conditional

        # Store args for passing to compiled function
        self._args = (bos_token, eos_token, pad_token, max_length, conditional)

        # Compile the top-level function
        #if torch.__version__ >= "2.1":
        #    self._collate_fn = torch.compile(_collate_batch_impl, mode="reduce-overhead")
        #else:
        self._collate_fn = _collate_batch_impl

    def __call__(self, batch: List[List[Dict]]) -> Dict:
        return self._collate_fn(batch, *self._args)
    
# --------------------
# Dataloader Factory
# --------------------

def create_chunked_audio_chart_dataloader(
    data: List[Dict],
    tokenizer,
    window_seconds: float = 10.0,
    difficulties: List[str] = ['Expert'],
    instruments: List[str] = ['Single'],
    batch_size: int = 32,
    max_length: int = 512,
    num_workers: int = 8,
    shuffle_chunks: bool = True,
    conditional: bool = False,
    num_pieces: int = 6,
    max_cache_gb: float = 100.0,
    chunk_size: Optional[int] = None,
    chunk_repeat: int = 1,
    use_predecoded_raw: bool = False,
    precomputed_windows: bool = False,
    decode_to_raw_on_init: bool = False,
    raw_dir: str = "raw_audio",
) -> Tuple[DataLoader, Dict]:
    """
    High-performance dataloader factory with all optimizations.
    Set use_predecoded_raw=True and predecode files with ffmpeg beforehand for maximum speed.
    """

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
        augment=False,  # Enable in training if needed
        use_predecoded_raw=use_predecoded_raw,
        precomputed_windows=precomputed_windows,
        decode_to_raw_on_init=decode_to_raw_on_init,
        raw_dir=raw_dir,
    )

    print(f"[INFO] Total items: {len(dataset)}, chunk_size={dataset.chunk_size}, workers={num_workers}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(num_workers, os.cpu_count() // 2),
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        timeout=60,
        multiprocessing_context='spawn' if platform.system() == 'Windows' else None,
    )
    return dataloader, vocab


# --------------------
# CLI Utility: Pre-decode all .opus to .raw (run separately)
# --------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pre-decode .opus files to .raw for faster loading.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing .opus files")
    parser.add_argument("--output-dir", type=str, default="raw_audio", help="Output directory for .raw files")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate (default: 16000)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel processes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    opus_files = [f for f in os.listdir(args.input_dir) if f.endswith('.opus')]
    print(f"Found {len(opus_files)} .opus files to convert...")

    def convert_one(f):
        src = os.path.join(args.input_dir, f)
        dst = os.path.join(args.output_dir, f.replace('.opus', '.raw'))
        if os.path.exists(dst):
            return
        cmd = [
            'ffmpeg',
            '-i', src,
            '-f', 's16le',
            '-ar', str(args.sr),
            '-ac', '1',
            dst
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"✓ {f}")
        except Exception as e:
            print(f"✗ {f}: {e}")

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(convert_one, opus_files))

    print(f"✅ Conversion complete. Output: {args.output_dir}")
