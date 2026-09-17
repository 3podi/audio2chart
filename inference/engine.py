import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EncodecModel, AutoProcessor
from huggingface_hub import hf_hub_download

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
    # Seconds of audio per decoded chunk. Must match what the model was trained on:
    # the decoder emits window_seconds * 1000 / grid_ms tokens, and asking a model
    # trained on 1502 for 3002 produces nonsense rather than an error. It was hardcoded
    # to 30 here while the trainer read it from config, so changing one broke the other
    # silently.
    window_seconds: float = 30.0


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
        """Load from a Hugging Face repo id, or from a local directory holding
        config.json and pytorch_model.bin -- which is what export_checkpoint.py writes,
        and the only way to run a model trained here rather than a released one."""
        local = Path(repo_id)
        if (local / "config.json").is_file() and (local / "pytorch_model.bin").is_file():
            with (local / "config.json").open(encoding="utf-8") as handle:
                cfg = TransformerConfig(**json.load(handle))
            model = cls(cfg)
            model.transformer.load_state_dict(
                torch.load(local / "pytorch_model.bin", map_location="cpu")
            )
            return model

        cfg_path = hf_hub_download(repo_id, "config.json")
        with open(cfg_path) as f:
            cfg = TransformerConfig(**json.load(f))
        model = cls(cfg)
        bin_path = hf_hub_download(repo_id, "pytorch_model.bin")
        state = torch.load(bin_path, map_location="cpu")
        model.transformer.load_state_dict(state)
        return model


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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        allowed_ids: Optional[List[int]] = None,
        guidance: float = 0.0,
        pad_bias: float = 0.0,
        max_parallel_chunks: int = 4,
    ) -> List[torch.Tensor]:
        """
        Fast batched generation with KV-cache + pre-allocation.
        """
        self.to(device)
        self.eval()

        input_values, padding_mask = self._read_audio(audio_path, device)  # [1,1,T]
        total_samples = input_values.size(-1)
        target_sr = 24000
        chunk_sec = float(self.config.window_seconds)
        chunk_samples = int(chunk_sec * target_sr)
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

        sample_fn = self._make_sampler(temperature, top_k, device, allowed_ids, pad_bias)
        neg_emb = self._contrast_audio(audio_emb) if guidance > 0 else None

        # Guidance runs a second branch with its own KV caches, so it doubles cache
        # memory. Decoding every chunk at once then pushes a full song past the card
        # and the driver pages it to host RAM instead of failing: measured 20 it/s
        # without guidance against 12 s/it with it, a ~240x collapse on the same song.
        # Halving how many chunks decode at once restores the original footprint.
        group = B if neg_emb is None else max(1, (B + 1) // 2)
        # Halving only makes guidance no worse than no guidance; it does not bound
        # anything, because half of a long song's chunks is still a lot of chunks. A
        # four-minute song paged at 13.2 s/it where the same code on a two-minute clip
        # ran at 0.1 s/it. Cap it so peak memory depends on this number rather than on
        # how long the song happens to be.
        if max_parallel_chunks > 0:
            group = min(group, max_parallel_chunks)

        layers = self.transformer.n_layers
        total = full_seq_len * ((B + group - 1) // group)
        progress = tqdm(total=total, desc="Let's rock!")

        # LIMITATION: chunks are decoded independently -- each starts from BOS with its
        # own KV cache and no memory of the one before it. A generated chart therefore
        # cannot have song-level structure: no build, no chorus busier than its verse,
        # no resolution. Even with perfect audio conditioning the model cannot know it
        # is in the second chorus rather than the first, because nothing carries over.
        #
        # It also leaves each chunk starting cold. Measured within one chunk, the first
        # quarter held 15 notes against 47 in the rest, and a generated chart is 25%
        # sparse over the first five seconds of every 30s window against a human 7% --
        # suggestive at -1.28 sd on a single chart, not established.
        #
        # The fix is to prime each chunk with the tail of the previous one so note
        # history is continuous across the boundary. That changes every generated
        # chart, so it is a decision rather than a patch.
        for lo in range(0, B, group):
            hi = min(lo + group, B)
            emb_g = audio_emb[lo:hi]
            neg_g = None if neg_emb is None else neg_emb[lo:hi]
            ids_g = None if class_ids is None else class_ids[lo:hi]

            self_cache = [None for _ in range(layers)]
            cross_cache = [None for _ in range(layers)]
            neg_self = [None for _ in range(layers)] if neg_g is not None else None
            neg_cross = [None for _ in range(layers)] if neg_g is not None else None

            for step in range(full_seq_len):
                cur_token = ids[lo:hi, step:step+1]

                logits, self_cache, cross_cache = self.transformer(
                    cur_token, emb_g, attention_mask=None, class_ids=ids_g,
                    step=step, use_cache=True, self_cache=self_cache, cross_cache=cross_cache
                )

                if neg_g is not None:
                    # Same history, different music. What survives the subtraction is
                    # the part of the prediction this audio is responsible for; scaling
                    # it up is the point, because measurement says it is real but small.
                    neg_logits, neg_self, neg_cross = self.transformer(
                        cur_token, neg_g, attention_mask=None, class_ids=ids_g,
                        step=step, use_cache=True, self_cache=neg_self, cross_cache=neg_cross
                    )
                    logits = logits + guidance * (logits - neg_logits)

                ids[lo:hi, step + 1] = sample_fn(logits[:, -1]).squeeze(-1)
                progress.update(1)

            del self_cache, cross_cache, neg_self, neg_cross
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        progress.close()

        # drop BOS + extract 'new' tokens for last chunk
        sequences = []
        prev_end = starts[-2] + chunk_samples if len(starts) > 1 else 0
        new_tokens = int((total_samples - prev_end) * 1000 / (target_sr * ms_resolution))

        for b in range(B):
            seq = ids[b, 1:]                                    # remove BOS
            if b == B - 1:                                      # last (possibly overlapping) chunk
                sequences.append(seq[-new_tokens:])
            else:
                sequences.append(seq[:full_seq_len])

        return sequences


    @staticmethod
    def _contrast_audio(audio_emb: torch.Tensor) -> torch.Tensor:
        """The negative branch for guidance: real music, but the wrong music.

        A zeroed or mean embedding is the usual choice, but this model was never
        trained with audio dropout, so it has no notion of "no audio" -- feeding it
        one produces logits from far outside the training distribution, and guidance
        would amplify that garbage rather than the conditioning. Every option here
        stays on the manifold of embeddings the model actually saw.

        With several chunks, the negative for each is a neighbouring chunk. With a
        single chunk there is no neighbour, so the window is rolled halfway: still a
        real passage of this song, just not the one being charted now.
        """
        if audio_emb.size(0) > 1:
            return torch.roll(audio_emb, 1, dims=0)
        return torch.roll(audio_emb, audio_emb.size(1) // 2, dims=1)

    def _style_mask(self, allowed_ids, device: torch.device):
        """A [1, V] boolean of what the style forbids, or None.

        Applied before top-k so the two compose: top-k then picks among what the style
        already permits, rather than the style vetoing whatever top-k happened to choose.
        """
        if allowed_ids is None:
            return None
        blocked = torch.ones(1, self.config.vocab_size, dtype=torch.bool, device=device)
        index = torch.as_tensor(sorted(allowed_ids), dtype=torch.long, device=device)
        if index.numel() == 0:
            raise ValueError("Style permits no tokens; sampling would produce NaN")
        blocked[0, index] = False
        return blocked

    def _make_sampler(self, temperature: float, top_k: int, device: torch.device,
                      allowed_ids=None, pad_bias: float = 0.0):
        """`pad_bias` is added to the pad token's logit, which controls note density.

        Density and vocabulary richness pull against each other under a single
        temperature knob. Teacher-forced the model is well calibrated -- 0.343 of its
        mass on chords against a true 0.369, 0.133 on taps against 0.132 -- but free
        running it over-emits notes badly, and raising temperature to recover chords,
        taps and sustains multiplies that. Measured on one clip against a corpus 7.2
        notes per second: temperature 0.5 gives 12.2 nps and no sustains at all, while
        temperature 1.0 gives the right tap and sustain rates at ~44 nps.

        Every step is really two decisions -- emit or stay silent, and what to emit --
        and one temperature governs both, so biasing pad ought to separate them.

        MEASURED: it does not, and the way it fails is the useful part. At temperature
        1.0, top_k 128, against a corpus 7.2 notes per second:

            pad+0   43.7 nps   tap 0.424
            pad+1   30.0 nps   tap 0.353
            pad+2   14.8 nps   tap 0.166
            pad+3    0.66 nps  tap 0.063
            pad+4    0.04 nps

        Nothing lands near 7.2, and one unit between pad+2 and pad+3 collapses output
        22-fold. The bias is also not orthogonal: suppressing notes preferentially
        suppresses chords and taps, so density and vocabulary stay coupled.

        Both follow from the loop being self-reinforcing -- fewer notes emitted changes
        the history the model reads, which makes it expect fewer still, so any
        perturbation amplifies instead of settling. That gain is exposure bias, and it
        shows directly as drift: over one 30s chunk at temperature 1.0 the tap rate
        climbs 0.224 -> 0.454 -> 0.671 -> 0.723, while at temperature 0.5 it is a flat
        0.000 with no sustains. Sharpening suppresses the runaway and the vocabulary
        together.

        Kept as a diagnostic, defaulting to no-op, because it is the knob that made the
        feedback visible and the idea is an obvious one to try again. The fix for the
        vocabulary collapse is in training, not here.
        """
        blocked = self._style_mask(allowed_ids, device)
        pad_id = self.config.pad_token_id

        if temperature <= 0:                     # greedy
            def sample(logits):
                if blocked is not None:
                    logits = logits.masked_fill(blocked, -float('inf'))
                if pad_bias:
                    logits = logits.clone()
                    logits[..., pad_id] += pad_bias
                return logits.argmax(dim=-1, keepdim=True)
            return sample

        def sample(logits):
            if blocked is not None:
                logits = logits.masked_fill(blocked, -float('inf'))

            # Before temperature, so the bias means the same thing at any temperature.
            if pad_bias:
                logits = logits.clone()
                logits[..., pad_id] += pad_bias

            if temperature != 1.0:
                logits = logits / temperature

            if top_k > 0:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < topk_vals[..., -1:], -float('inf'))

            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        return sample



