import os
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
##import librosa

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


#print(torchaudio.utils.ffmpeg_utils.get_audio_decoders())

from pydub import AudioSegment
import numpy as np
import torch

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
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr



class SimpleAudioTextDataset(Dataset):
    """
    Dataset for audio-text pairs.
    Each (audio, text) entry is treated as one sample.

    Args:
        data_file: JSON file with a list of {"audio_path": ..., "chart_path": ..., "target_section": ...}
        difficulties: possible diff to process
        instruments: possible instruments to process
        window_seconds: Duration of audio windows (in seconds)
        audio_processor: audio processor
        tokenizer: chart tokenizer
        max_retries: max retries if __getitem__ fails
    """

    def __init__(
        self,
        data,
        bos_token: int, 
        eos_token: int, 
        pad_token: int,
        max_length: int = 256,
        difficulties = ['Expert'],
        instruments = ['Single'],
        window_seconds: float = 10.0,
        audio_processor = None,
        tokenizer = None,
        conditional = False
    ):
       
        self.data = data       
        self.difficulties = difficulties
        self.instruments = instruments
        self.window_seconds = window_seconds
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.conditional = conditional

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.max_length = max_length

        self.chart_processor = ChartProcessor(difficulties, instruments)

    def __len__(self) -> int:
        return len(self.data)

    def _load_item(self, audio_path: str, chart_path: str, target_section: str) -> Dict[str, Union[torch.Tensor, str, float]]:
        
        if target_section:
            if not isinstance(target_sections, list):
                target_sections = [target_sections]
        assert len(target_section) == 1, 'This dataset requires 1 target section per item. Format input data files in triplets (audio_path,chart_path,target_section)'

        # --- load audio --- # TODO: audio window larger than chart notes time, multiple chunks per audio

        waveform, sr = torchaudio.load(audio_path)
        if sr != self.audio_processor.sampling_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.audio_processor.sampling_rate)
        
        window_samples = int(self.window_seconds * self.audio_processor.sampling_rate)
        max_start = waveform.shape[-1] - window_samples
        start_sample = random.randint(0, max_start) 
        end_sample = start_sample + window_samples

        waveform = waveform[:, start_sample:end_sample]
        waveform = self.audio_processor(
            raw_audio = waveform,
            sampling_rate = self.audio_processor.sampling_rate,
            return_tensors="pt" 
        )

        # --- load notes ---
        notes = self.chart_processor.read_chart(chart_path=chart_path, target_sections=target_section).notes
        bpm_events = self.chart_processor.synctrack
        resolution = int(self.chart_processor.song_metadata['Resolution'])
        offset = float(self.chart_processor.song_metadata['Offset'])

        start_seconds = start_sample / self.audio_processor.sampling_rate
        end_seconds = end_sample / self.audio_processor.sampling_rate

        # --- encode + convert to seconds ---
        tokenized_chart = self.tokenizer.encode(note_list=notes)
        tokenized_chart = self.tokenizer.format_seconds(
            tokenized_chart, bpm_events,
            resolution=resolution, offset=offset
        )

        # --- filter by window --- 
        filtered = [
            (t, v, d) for (t, v, d) in tokenized_chart
            if start_seconds <= t < end_seconds
        ]

        # Note times normalized in [0,1] - TODO: normalize duration with data stats
        if filtered:
            note_times, note_values, note_durations = map(list, zip(*filtered))
            note_times -= start_seconds
            note_times = note_times / self.window_seconds
        else:
            note_times, note_values, note_durations = [], [], []

        if self.conditional:
            diff = [
                mapped_diff for diff, mapped_diff in DIFF_MAPPING.items() if diff in target_section
            ]
        else:
            diff = [-1]
        
        # --- return sample, will pad in collator with max_batch_len ---
        return {
            "audio": waveform,  # already tensor
            "note_times": note_times,
            "note_values": note_values,
            "note_durations": note_durations,
            "cond_diff": diff,           
        }

    def __getitem__(self, idx: int):
        for attempt in range(self.max_retries):
            #try:
            item = self.data[idx] if attempt == 0 else random.choice(self.data)
            return self._load_item(item["audio_path"], item["chart_path"], item["target_section"])
            #except Exception as e:
            #    print(f"[Warning] Failed to load {item['audio_path']}: {e}")
                # try another random sample
            #    continue

        # if all retries fail, raise error
        raise RuntimeError(f"Failed to load a valid sample after {self.max_retries} attempts")
    

class WaveformDataset(Dataset):
    def __init__(
        self,
        data,
        bos_token: int, 
        eos_token: int, 
        pad_token: int,
        max_length: int = 256,
        difficulties = ['Expert'],
        instruments = ['Single'],
        window_seconds: float = 10.0,
        sample_rate: int = 16000,
        audio_processor = None,
        tokenizer = None,
        conditional = False,
        augment = False,
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

        self.chart_processor = ChartProcessor(difficulties, instruments)

    
    def __len__(self) -> int:
        return len(self.data)

    def _augment(self, waveform):
        # Random gain
        if random.random() < 0.5:
            gain_db = random.uniform(-6, 6)  # -6dB to +6dB
            waveform = waveform * (10 ** (gain_db / 20))

        # Additive Gaussian noise
        if random.random() < 0.5:
            noise_amp = 0.005 * waveform.abs().max() * random.random()
            waveform = waveform + noise_amp * torch.randn_like(waveform)

        # Random polarity flip
        if random.random() < 0.3:
            waveform = -waveform

        return waveform

    def _load_item(self, audio_path: str, chart_path: str, target_section: str) -> Dict[str, Union[torch.Tensor, str, float]]:
            
        if target_section:
            if not isinstance(target_section, list):
                target_section = [target_section]
        assert len(target_section) == 1, 'This dataset requires 1 target section per item. Format input data files in triplets (audio_path,chart_path,target_section)'

        # --- load audio --- # TODO: audio window larger than chart notes time, multiple chunks per audio

        waveform, sr = load_audio_pydub(audio_path, target_sr=16000)
        #waveform, sr = torchaudio.load(audio_path)
        #waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        #waveform = torch.from_numpy(waveform).unsqueeze(0)
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        # Pad/crop to fixed length, TODO allow to train on uncomplete chunks
        if waveform.size(1) < self.num_samples:
            pad_len = self.num_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        #if waveform.size(1) > self.num_samples:
        #    start = random.randint(0, waveform.size(1) - self.num_samples)
        #    waveform = waveform[:, start:start+self.num_samples]

        window_samples = int(self.window_seconds * self.sample_rate)
        max_start = waveform.shape[-1] - window_samples
        start_sample = random.randint(0, max_start) 
        end_sample = start_sample + window_samples

        waveform = waveform[:, start_sample:end_sample]
        if self.augment:
            waveform = self._augment(waveform)

        # --- load notes ---
        self.chart_processor.read_chart(chart_path=chart_path, target_sections=target_section)
        notes = self.chart_processor.notes
        notes = notes[target_section[0]]
        bpm_events = self.chart_processor.synctrack
        resolution = int(self.chart_processor.song_metadata['Resolution'])
        offset = float(self.chart_processor.song_metadata['Offset'])

        start_seconds = start_sample / self.sample_rate
        end_seconds = end_sample / self.sample_rate
        
        # --- encode + convert to seconds ---
        tokenized_chart = self.tokenizer.encode(note_list=notes)
        tokenized_chart = self.tokenizer.format_seconds(
            tokenized_chart, bpm_events,
            resolution=resolution, offset=offset
        )

        # --- filter by window --- 
        filtered = [
            (t, v, d) for (t, v, d, _) in tokenized_chart
            if start_seconds <= t < end_seconds
        ]

        # Note times normalized in [0,1] - TODO: normalize duration with data stats
        if filtered:
            note_times, note_values, note_durations = map(list, zip(*filtered))
            note_times = [(t - start_seconds) / self.window_seconds for t in note_times]
        else:
            note_times, note_values, note_durations = [], [], []

        if self.conditional:
            #print('target_section: ', target_section)
            diff = [
                mapped_diff for diff, mapped_diff in DIFF_MAPPING.items() if diff in target_section[0]
            ]
            #print('diff in load item: ', diff)

            #for diff, mapped_diff in DIFF_MAPPING.items():
                
            #    print('condition output: ', )
        else:
            diff = [-1]
        
        # --- return sample, will pad in collator with max_batch_len ---
        return {
            "audio": waveform,  # already tensor
            "note_times": note_times,
            "note_values": note_values,
            "note_durations": note_durations,
            "cond_diff": diff,           
        }
    
    def __getitem__(self, idx: int):
        for attempt in range(10):
            try:
                item = self.data[idx] if attempt == 0 else random.choice(self.data)
                return self._load_item(item["audio_path"], item["chart_path"], item["difficulty"])
            except Exception as e:
                print(f"[Warning] Failed to load {item['audio_path']}: {e}")
                # try another random sample
                continue

        # if all retries fail, raise error
        raise RuntimeError(f"Failed to load a valid sample after 10 attempts")
    


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
        vocab = None
    ):
    """
    Create a DataLoader for (audio,notes) pairs with proper batching and tokenization and audio processing
    
    Args:
        data_file: str for file with triplets (audio_path, chart_path, target_section)
        difficulties: Difficulties to include
        instruments: Instruments to include  
        batch_size: Batch size
        max_length: Maximum sequence length (will pad/truncate to this)
        num_workers: Number of worker processes
    """
    
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
        data = data,
        difficulties = difficulties,
        instruments = instruments,
        window_seconds = window_seconds,
        audio_processor = audio_processor,
        tokenizer = tokenizer,
        conditional = conditional,
        bos_token = vocab['<bos>'], 
        eos_token = vocab['<eos>'], 
        pad_token = vocab['<PAD>'],
        max_length = max_length
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

class AudioChartCollator:
    def __init__(self, bos_token, eos_token, pad_token=-100, max_length=512, conditional=False):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.max_length = max_length
        self.conditional = conditional
    
    def __call__(self, batch):
        return chart_collate_fn(
            batch,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            pad_token=self.pad_token,
            max_length=self.max_length,
            conditional=self.conditional
        )


def chart_collate_fn(batch, bos_token, eos_token, pad_token=-100, max_length=512, conditional=False):
    """Custom collate function with proper batching and padding.
       Sequences are padded to max_batch_len (capped by max_length).
    """
    if not batch:
        return {}
    
    # --- find max batch sequence length (with BOS/EOS), but cap at max_length
    max_batch_len = max(len(sample["note_values"]) for sample in batch) + 2  # +2 for BOS/EOS
    max_batch_len = min(max_batch_len, max_length)

    batch_audio = []
    batch_audio_mask = []
    batch_note_times = []
    batch_note_durations = []
    batch_note_values = []
    batch_diff = []
    attention_masks = []
    
    for sample in batch:
        audio = sample['audio']
        #audio_mask = sample['waveform']['padding_mask']
        note_times = sample["note_times"]
        note_durations = sample["note_durations"]
        note_values = sample["note_values"]
        diff = sample["cond_diff"]
        
        # add BOS/EOS + placeholders values 
        note_times = [0.0] + note_times + [1.0]
        note_durations = [0.0] + note_durations + [0.0]
        note_values = [bos_token] + note_values + [eos_token]

        # truncate if longer than max_batch_len
        note_times = note_times[:max_batch_len]
        note_durations = note_durations[:max_batch_len]
        note_values = note_values[:max_batch_len]


        # --- pad up to max_batch_len
        seq_len = len(note_values)
        pad_len = max_batch_len - seq_len

        padded_values = note_values + [pad_token] * pad_len
        padded_times = note_times + [0.0] * pad_len
        padded_durations = note_durations + [0.0] * pad_len

        attn_len = seq_len - 1
        attention_mask = [1] * attn_len + [0] * (max_batch_len - attn_len)

        # append
        batch_audio.append(audio)
        #batch_audio_mask.append(audio_mask)
        batch_note_values.append(padded_values)
        batch_note_times.append(padded_times)
        batch_note_durations.append(padded_durations)
        attention_masks.append(attention_mask)
        batch_diff.append(diff)

    # convert to tensors
    #batch_audio = torch.tensor(batch_audio, dtype=torch.float)
    batch_note_values = torch.tensor(batch_note_values, dtype=torch.long)
    batch_note_times = torch.tensor(batch_note_times, dtype=torch.float)
    batch_note_durations = torch.tensor(batch_note_durations, dtype=torch.float)
    attention_masks = torch.tensor(attention_masks, dtype=torch.long)
    if conditional:
        batch_diff = torch.tensor(batch_diff, dtype=torch.long)
    else:
        batch_diff = None  # No diff for unconditional case
    
    return {
        "audio": torch.stack(batch_audio, dim=0).float(), #.to(torch.float),
        #"audio_mask": batch_audio_mask,
        "note_values": batch_note_values,
        #"note_times": batch_note_times,
        #"note_durations": batch_note_durations,
        "attention_mask": attention_masks,
        "cond_diff": batch_diff,
    }

        
