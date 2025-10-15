import os
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Dict, Any, Optional

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

class ChartChunksDataset(Dataset):
    def __init__(self, chart_paths, difficulties, instruments, seq_len, conditional=False):
        
        self.seq_len = seq_len - 2 #Add bos and eos in collate 
        self.chunks = []
        self.chunks_diff = []
        self.conditional = conditional

        proc = ChartProcessor(difficulties, instruments)
        self.tokenizer = SimpleTokenizerGuitar()

        n_failed_paths = 0

        for path in chart_paths:
            try:
                proc.read_chart(path)
                notes = proc.notes
                # Text like task we dont need to handle time
                self.prepare_chunks(notes)
            except Exception as e:
                print(f"Error processing chart {path}: {e}")
                n_failed_paths += 1
                continue
        
        print(f"Processed {len(chart_paths) - n_failed_paths} charts, failed on {n_failed_paths} paths.")

    def prepare_chunks(self, notes):
        for section_name, note_seq in notes.items():
            encoded_notes = self.tokenizer.encode(note_list=note_seq)
            encoded_list = [n[1] for n in encoded_notes]
            chunks = [
                encoded_list[i:i+self.seq_len]
                for i in range(0, len(encoded_list) - self.seq_len +1, self.seq_len)
            ]
            self.chunks.extend(chunks)
            if self.conditional:
                diff = [
                    mapped_diff for diff, mapped_diff in DIFF_MAPPING.items() if diff in section_name
                ]
                self.chunks_diff.extend(diff * len(chunks))
            else:
                self.chunks_diff.extend([-1] * len(chunks)) # placeholder for no diff

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx], self.chunks_diff[idx]    

def create_chart_dataloader(
        chart_paths: List[str], 
        difficulties: List[str] = ['Expert'], 
        instruments: List[str] = ['Single'],
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 4,
        shuffle: bool = True,
        conditional: bool = False,
        vocab: Optional[Dict] = None
    ):
    """
    Create a DataLoader for chart files with proper batching and tokenization
    
    Args:
        chart_paths: List of str with file paths
        difficulties: Difficulties to include
        instruments: Instruments to include  
        batch_size: Batch size
        max_length: Maximum sequence length (will pad/truncate to this)
        num_workers: Number of worker processes
        vocab: Custom vocabulary dict, if None will create default
    """
    
    if vocab is None:
        vocab = SimpleTokenizerGuitar().mapping_noteseqs2int
        bos_token_id = len(vocab.keys())
        eos_token_id = bos_token_id + 1 
        pad_token_id = eos_token_id + 1
        vocab['<bos>'] = bos_token_id
        vocab['<eos>'] = eos_token_id 
        vocab['<PAD>'] = pad_token_id

    collator = ChartCollator(
        bos_token=vocab['<bos>'], 
        eos_token=vocab['<eos>'], 
        pad_token=vocab['<PAD>'],
        max_length=max_length,
        conditional=conditional
    )
    

    dataset = ChartChunksDataset(chart_paths, difficulties, instruments, max_length, conditional)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator
    )
    
    return dataloader, vocab

class ChartCollator:
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