import os
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio

import random
from typing import Dict, Union, List, Optional
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chart.chart_processor import ChartProcessor
from chart.tokenizer import SimpleTokenizerGuitar

DIFFICULTIES = ['Expert', 'Hard', 'Medium', 'Easy']
INSTRUMENTS = ['Single']
DIFF_MAPPING = {
    'Expert': 0,
    'Hard': 1,
    'Medium': 2,
    'Easy': 3
}

import json

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
        data_file: str,
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
        with open(data_file, "r") as f:
            self.data = json.load(f)

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
        
        # pad for easily truncate as note seq - for duration and note time compute loss only when
        # note values target is not pad or eos, need to compute loss only on real notes time/duration
        note_times = [0.0] + note_times + [1.0]
        note_durations = [0.0] + note_durations + [0.0]
        
        note_values = [self.bos_token] + note_values + [self.eos_tokens]
        # --- handle padding and attention mask for note sequence, LLM like --- 
        if len(note_values) >= self.max_length:
            note_values = note_values[:self.max_length]
            note_times = note_times[:self.max_length]
            note_durations = note_durations[:self.max_length]
            attention_mask = [1] * (self.max_length - 1)
            padded_tokens = note_values
        else:
            attention_mask = [1] * len(note_values) + [0] * (self.max_length -1 - len(note_values))
            padded_tokens = note_values + [self.pad_token] * (self.max_length - len(note_values))


        # --- return sample ---
        return {
            "audio": waveform,  # already tensor
            "note_times": torch.tensor(note_times, dtype=torch.float32),
            "note_values": torch.tensor(padded_tokens, dtype=torch.long),
            "note_durations": torch.tensor(note_durations, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "cond_diff": torch.tensor(diff, dtype=torch.long)
        }

    def __getitem__(self, idx: int):
        for attempt in range(self.max_retries):
            try:
                item = self.data[idx] if attempt == 0 else random.choice(self.data)
                return self._load_item(item["audio_path"], item["chart_path"], item["target_section"])
            except Exception as e:
                print(f"[Warning] Failed to load {item['audio_path']}: {e}")
                # try another random sample
                continue

        # if all retries fail, raise error
        raise RuntimeError(f"Failed to load a valid sample after {self.max_retries} attempts")
    

def create_audio_chart_dataloader(
        data_file: str,
        audio_processor,
        window_seconds,
        tokenizer, 
        difficulties: List[str] = ['Expert'], 
        instruments: List[str] = ['Single'],
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 4,
        shuffle: bool = True,
        conditional: bool = False,
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

    #collator = AudioChartCollator(
    #    bos_token=vocab['<bos>'], 
    #    eos_token=vocab['<eos>'], 
    #    pad_token=vocab['<PAD>'],
    #    max_length=max_length,
    #    conditional=conditional
    #)
    

    dataset = SimpleAudioTextDataset(
        data_file = data_file,
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
        #collate_fn=collator
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
       Input batch should be already tokenized.
    """
    if not batch:
        return {}
    
    # Tokenize all samples
    padded_batch = []
    attention_masks = []
    diff_batch = []
    
    for sample, diff in batch:

        sample = [bos_token] + sample + [eos_token]

        # Handle padding and attention mask
        if len(sample) >= max_length:
            sample = sample[:max_length]
            attention_mask = [1] * (max_length-1)
            padded_tokens = sample
        else:
            attention_mask = [1] * len(sample) + [0] * (max_length -1 - len(sample))
            padded_tokens = sample + [pad_token] * (max_length - len(sample))

        padded_batch.append(padded_tokens)
        attention_masks.append(attention_mask)
        diff_batch.append(diff)
        
        # Keep metadata for reference
        #metadata = {
        #    'file_path': sample.get('file_path', ''),
        #    'section_name': sample.get('section_name', ''),
        #    'song_metadata': sample.get('song_metadata', {}),
        #    'original_length': len(tokens)
        #}
        #metadata_batch.append(metadata)
    
    # Convert to tensors
    input_ids = torch.tensor(padded_batch, dtype=torch.long)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long)
    if conditional:
        input_diff = torch.tensor(diff_batch, dtype=torch.long)
    else:
        input_diff = None  # No diff for unconditional case

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'cond_diff': input_diff
    }