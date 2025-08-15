import os
import random
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Dict, Any, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chart.chart_processor import ChartProcessor
from chart.tokenizer import SimpleTokenizerGuitar

from utils_dataloader import find_chart_files

DIFFICULTIES = ['Expert', 'Hard', 'Medium', 'Easy']
INSTRUMENTS = ['Single']
DIFF_MAPPING = {
    'Expert': 0,
    'Hard': 1,
    'Medium': 2,
    'Easy': 3
}

class ChartChunksDataset(Dataset):
    def __init__(self, chart_paths, difficulties, instruments, seq_len):
        self.seq_len = seq_len
        self.chunks = []
        self.chunks_diff = []
        
        proc = ChartProcessor(difficulties, instruments)
        self.tokenizer = SimpleTokenizerGuitar()

        for path in chart_paths:
            proc.read_chart(path)
            notes = proc.notes
            # Text like task we dont need to handle time
            #bpm_events = proc.synctrack
            #resolution = int(proc.song_metadata['Resolution'])
            #offset = float(proc.song_metadata['Offset'])
            #self.prepare_chunks(notes, bpm_events,resolution, offset)
            self.prepare_chunks(notes)

    def prepare_chunks(self, notes):
        for section_name, note_seq in notes.items():
            encoded_notes = self.tokenizer(note_list=note_seq)
            encoded_list = [n[1] for n in encoded_notes]
            chunks = [
                encoded_list[i:i+self.seq_len]
                for i in range(0, len(encoded_list) - self.seq_len +1, self.seq_len)
            ]
            self.chunks.extend(chunks)
            diff = [
                mapped_diff for diff, mapped_diff in DIFF_MAPPING.items() if diff in section_name
            ]
            self.chunks_diff.extend(diff * len(chunks))

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx], self.chunks_diff[idx]
    

def create_chart_dataloader(chart_root: List[str], 
                          difficulties: List[str] = ['Expert'], 
                          instruments: List[str] = ['Single'],
                          batch_size: int = 32,
                          max_length: int = 512,
                          num_workers: int = 4,
                          vocab: Optional[Dict] = None):
    """
    Create a DataLoader for chart files with proper batching and tokenization
    
    Args:
        chart_root: Root to find.chart file paths
        difficulties: Difficulties to include
        instruments: Instruments to include  
        batch_size: Batch size
        max_length: Maximum sequence length (will pad/truncate to this)
        num_workers: Number of worker processes
        vocab: Custom vocabulary dict, if None will create default
    """
    
    # Get chart paths from root folder
    chart_paths = find_chart_files(chart_root)
    
    # Create collate function with proper parameters
    def collate_fn(batch):
        return chart_collate_fn(batch, max_length=max_length, pad_token=-100)
    

    dataset = ChartChunksDataset(chart_paths, difficulties, instruments, max_length)
    shuffle = True
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return dataloader


def chart_collate_fn(batch, max_length=512, pad_token=-100):
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
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(sample) + [0] * (max_length - len(sample))
        
        # Pad sequence to max_length
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
    input_diff = torch.tensor(diff_batch, dtype=torch.long)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'input_diff': input_diff
    }