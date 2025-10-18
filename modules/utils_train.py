import random
import torch
import os
import numpy as np
import lightning as L
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chart.chart_processor import ChartProcessor
from chart.tokenizer import SimpleTokenizerGuitar


MAX_NOTES = 5000


def set_seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across:
      - Python's `random`
      - NumPy
      - PyTorch
      - PyTorch Lightning
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Lightning built-in helper
    L.seed_everything(seed, workers=True)

    print(f"[Seed] Global seed set to {seed}")


class LogGradientNorm(L.pytorch.callbacks.Callback):
    """
    Logs the gradient norm (L2 norm) before the optimizer step (pre-clipping/scaling).
    """
    def on_before_optimizer_step(self, trainer: L.Trainer, *args, **kwargs) -> None:
        total_norm = 0.0
        for param in trainer.lightning_module.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        trainer.lightning_module.log("train/grad_norm", total_norm)


def validate_dataset(data, difficulties, instruments, grid_ms):
    valid_items = []
    chart_processor = ChartProcessor(difficulties, instruments)
    tokenizer = SimpleTokenizerGuitar()
    for item in data:
        try:
            #chart_processor = ChartProcessor(difficulties, instruments)
            chart_processor.read_chart(chart_path=item["chart_path"], target_sections=item["difficulty"])
            notes = chart_processor.notes[item["difficulty"]]
            bpm_events = chart_processor.synctrack
            resolution = int(chart_processor.song_metadata['Resolution'])
            offset = float(chart_processor.song_metadata['Offset'])

            tokenized_chart = tokenizer.encode(note_list=notes)
            tokenized_chart = tokenizer.format_seconds(
                tokenized_chart, bpm_events, resolution=resolution, offset=offset
            )
            
            time_deltas = []
            for i in range(1, len(tokenized_chart)):
                delta = tokenized_chart[i][0] - tokenized_chart[i-1][0]
                time_deltas.append(delta)

            min_delta = min(time_deltas)

            if len(notes) > 0 and len(notes) < MAX_NOTES and min_delta > grid_ms/1000.0 :
                valid_items.append(item)
        except Exception as e:
            print(f"Skipping invalid chart: {item['chart_path']} - {e}")
    
    print(f"Filtered dataset: {len(valid_items)}/{len(data)} valid charts")
    return valid_items


def validate_dataset_notes(data, difficulties, instruments, grid_ms):
    valid_items = []
    chart_processor = ChartProcessor(difficulties, instruments)
    tokenizer = SimpleTokenizerGuitar()
    for item in data:
        try:
            chart_processor.read_chart(chart_path=item)
            bpm_events = chart_processor.synctrack
            resolution = int(chart_processor.song_metadata['Resolution'])
            offset = float(chart_processor.song_metadata['Offset'])

            for difficulty in chart_processor.notes.keys():
                notes = chart_processor.notes[difficulty]
                tokenized_chart = tokenizer.encode(note_list=notes)
                tokenized_chart = tokenizer.format_seconds(
                    tokenized_chart, bpm_events, resolution=resolution, offset=offset
                )
                
                time_deltas = []
                for i in range(1, len(tokenized_chart)):
                    delta = tokenized_chart[i][0] - tokenized_chart[i-1][0]
                    time_deltas.append(delta)

                min_delta = min(time_deltas)

                if len(notes) > 0 and len(notes) < MAX_NOTES and min_delta > grid_ms/1000.0 :
                    valid_items.append(item)

        except Exception as e:
            print(f"Skipping invalid chart: {item} - {e}")
    
    print(f"Filtered dataset: {len(valid_items)}/{len(data)} valid charts")
    return valid_items