import json
import math
from dataclasses import dataclass, asdict
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EncodecModel, AutoProcessor
from huggingface_hub import hf_hub_download
import torchaudio

from inference.model_inference import TransformerDecoderAudioConditioned
from tqdm import tqdm

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
@dataclass
class TransformerConfig:
    vocab_size: int
    pad_token_id: int
    eos_token_id: int
    bos_token_id: int
    d_model: int = 512
    n_heads: int = 8
    num_kv_heads: int = 2
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    audio_drop: float = 0.0
    compression: Optional[int] = None
    rope_base: float = 10000.0
    conditional: bool = False
    use_flash: bool = False
    codebook_size: int = 128,
    grid_ms: int = 20


# ------------------------------------------------------------------
# Inference Model
# ------------------------------------------------------------------
class Charter(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.encoder = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        self.transformer = TransformerDecoderAudioConditioned(**asdict(config))

        self.encoder.eval()
        self.transformer.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.transformer.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(cls, repo_id: str):
        cfg_path = hf_hub_download(repo_id, "config.json")
        with open(cfg_path) as f:
            cfg = TransformerConfig(**json.load(f))
        model = cls(cfg)
        bin_path = hf_hub_download(repo_id, "pytorch_model.bin")
        state = torch.load(bin_path, map_location="cpu")
        model.transformer.load_state_dict(state)
        return model

    def _preprocess_audio_old(self, audio_path: str, device: torch.device):
            wav, sr = torchaudio.load(audio_path)
            if wav.size(0) > 1:
                wav = wav.mean(0, keepdim=True)  # to mono

            inputs = self.processor(
                raw_audio=wav.squeeze(0).numpy(),  # [T]
                sampling_rate=sr,
                return_tensors="pt"
            ).to(device)

            return inputs["input_values"], inputs["padding_mask"]


    def _read_audio(self, audio_path: str, device: torch.device):
        import librosa
        wav, sr = librosa.load(audio_path, sr=24000, mono=True)
        inputs = self.processor(
            raw_audio=wav,
            sampling_rate=24000,
            return_tensors="pt"
        ).to(device)
        return inputs["input_values"], inputs["padding_mask"]


    def generate(
        self,
        audio_path: str,
        temperature: float = -1.0,
        top_k: int = 0,
        class_id: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ) -> List[torch.Tensor]:
        """
        Fast batched generation with KV-cache + pre-allocation.
        """
        self.to(device)
        self.eval()

        input_values, padding_mask = self._read_audio(audio_path, device)  # [1,1,T]
        total_samples = input_values.size(-1)
        target_sr = 24000
        chunk_sec = 30
        chunk_samples = chunk_sec * target_sr
        ms_resolution = self.config.grid_ms

        if total_samples < chunk_samples:
            raise ValueError(f"Audio must be >= {chunk_sec}s, got {total_samples/target_sr:.2f}s")


        starts = list(range(0, total_samples, chunk_samples))
        if starts[-1] + chunk_samples > total_samples:
            starts[-1] = max(0, total_samples - chunk_samples)   # force full 30 s

        chunks = []
        masks  = []
        for s in starts:
            e = s + chunk_samples
            chunks.append(input_values[..., s:e])
            masks.append(padding_mask[..., s:e])

        audio_batch = torch.cat(chunks, dim=0).to(device)          # [B,1,720000]
        mask_batch  = torch.cat(masks , dim=0).to(device)          # [B,720000]


        with torch.no_grad():
            enc = self.encoder.encode(audio_batch, mask_batch, bandwidth=3.0)
            codes = enc.audio_codes.squeeze(0)                                  # [B,4,T]
            # sum the 4 codebook embeddings                                     
            audio_emb = sum(self.transformer.codes_embedding[i](codes[:, i]) for i in range(4))
            audio_emb = self.transformer.norm_audio(audio_emb)
            if self.transformer.compression:
                audio_emb = self.transformer.audio_compression(audio_emb)       # (B, T_audio', d_model)


        B = audio_batch.size(0)
        class_ids = (torch.full((B, 1), class_id, dtype=torch.long, device=device)
                    if self.config.conditional else None)


        full_seq_len = int(chunk_sec * 1000 / ms_resolution)   

        ids = torch.full((B, full_seq_len + 1), self.config.bos_token_id,
                        dtype=torch.long, device=device)
        ids[:, 0] = self.config.bos_token_id

        self_cache = [None for _ in range(self.transformer.n_layers)]
        cross_cache = [None for _ in range(self.transformer.n_layers)]
        sample_fn = self._make_sampler(temperature, top_k, device)

        for step in tqdm(range(full_seq_len), desc="Lets rock!"):
            cur_token = ids[:, step:step+1]                     # [B,1]

            logits, self_cache, cross_cache = self.transformer(
                cur_token, audio_emb, attention_mask=None, class_ids=class_ids,
                step=step, use_cache=True, self_cache=self_cache, cross_cache=cross_cache
            )                                                   # logits: [B,1,V]

            next_id = sample_fn(logits[:, -1])                  # [B,1]
            ids[:, step + 1] = next_id.squeeze(-1)

        # drop BOS + extract 'new' tokens for last chunk
        sequences = []
        new_tokens = int((total_samples - starts[-1]) * 1000 / (target_sr * ms_resolution))

        for b in range(B):
            seq = ids[b, 1:]                                    # remove BOS
            if b == B - 1:                                      # last (possibly overlapping) chunk
                sequences.append(seq[-new_tokens:])
            else:
                sequences.append(seq[:full_seq_len])

        return sequences


    def _make_sampler(self, temperature: float, top_k: int, device: torch.device):
        if temperature <= 0:                     # greedy
            def sample(logits):
                return logits.argmax(dim=-1, keepdim=True)
            return sample

        def sample(logits):
            if temperature != 1.0:
                logits = logits / temperature

            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < topk_vals[..., -1:], -float('inf'))

            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        return sample



model = Charter.from_pretrained("3podi/charter-v1.0-40-M-best-acc")

# This will RAISE ERROR if < 30s
seqs = model.generate("../black.mp3", temperature=1.0, top_k=0)

seqs = torch.cat(seqs).flatten()

print('N chords: ', torch.sum((seqs>4) & (seqs < 34)))

seqs = seqs.cpu().tolist()
print('Length song: ', len(seqs))

print(seqs)
print('First 10 notes: ', seqs[:10])



from chart.tokenizer import SimpleTokenizerGuitar

def time_to_tick(time_sec, bpm, resolution):
    return int(round(time_sec * bpm / 60.0 * resolution))

def convert_notes_to_ticks(tokens_list, time_list, fixed_bpm=200, resolution=480):

    attrs = {
        'is5': False,
        'is6': False,
        'isS': False,
    }

    tick_notes = []

    for t, note in zip(time_list, tokens_list):
        if note != 34:
            tick = time_to_tick(t, fixed_bpm, resolution)
            tick_notes.append((tick,note,0,attrs))

    return tick_notes

time_list = [i*40/1000 for i in range(len(seqs))]
ticked_notes = convert_notes_to_ticks(seqs, time_list)

print(ticked_notes)
print(len(ticked_notes))

tokenizer = SimpleTokenizerGuitar()

decoded_full = tokenizer.decode(ticked_notes)


print(len(decoded_full))

for i in range(10):
    print(decoded_full[i])


template = """
[Song]

{
  Name = "dummy"
  Artist = "dummy"
  Charter = "3podi"
  Album = "dummy"
  Year = ", 2022"
  Offset = 0
  Resolution = 192
  Player2 = bass
  Difficulty = 3
  PreviewStart = 0
  PreviewEnd = 0
  Genre = "J-pop/rock"
  MediaType = "cd"
  MusicStream = "song.ogg"
}

[SyncTrack]
{
  0 = TS 4
  0 = B 200000
}

[ExpertSingle]
{

} 
"""

def fill_expert_single(template_text: str, notes: list[tuple]) -> str:
    # Build the new ExpertSingle block content
    new_lines = [f'  {t} = {typ} {a} {b}' for (t, typ, a, b) in notes]
    new_block = "[ExpertSingle]\n{\n" + "\n".join(new_lines) + "\n}"

    # Replace the old ExpertSingle block with the new one
    import re
    filled = re.sub(r"\[ExpertSingle\]\s*\{[^}]*\}", new_block, template_text, flags=re.DOTALL)

    return filled

new_text = fill_expert_single(template, decoded_full)


def save_chart_file(filled_text: str, filename: str):
    """Save the filled chart text to a .chart file."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(filled_text)
    print(f"✅ Saved chart to {filename}")


save_chart_file(new_text, "notes.chart")
