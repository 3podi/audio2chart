import os
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio

import random
from typing import Dict, Union, List
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
        difficulties = ['Expert'],
        instruments = ['Single'],
        window_seconds: float = 10.0,
        audio_processor = None,
        tokenizer = None,
    ):
        with open(data_file, "r") as f:
            self.data = json.load(f)

        self.difficulties = difficulties
        self.instruments = instruments
        self.window_seconds = window_seconds
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer

        self.chart_processor = ChartProcessor(difficulties, instruments)

    def __len__(self) -> int:
        return len(self.data)

    def _load_item(self, audio_path: str, chart_path: str, target_section: str) -> Dict[str, Union[torch.Tensor, str, float]]:
        
        if target_section:
            if not isinstance(target_sections, list):
                target_sections = [target_sections]
        assert len(target_section) == 1, 'This dataset requires 1 target section per item. Format input data files in triplets (audio_path,chart_path,target_section)'

        # --- load audio --- # TODO: audio window larger than chart notes time, multiple chunks per audio
        # TODO: process/load only the target label, change how read_chart works in the chart_processor

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


        # --- return sample ---
        return {
            "audio": waveform,
            "note_times": note_times,
            "note_values": note_values,
            "note_durations": note_durations,
            "cond_diff": diff
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
    

