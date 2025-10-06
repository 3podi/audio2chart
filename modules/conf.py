import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import torchaudio
import numpy as np
from typing import Optional, Tuple
from torch.backends.cuda import SDPBackend, sdpa_kernel

def apply_rotary_emb(x: torch.Tensor, dim: int, base: float = 10000.0) -> torch.Tensor:
    """
    Apply Rotary Position Embeddings (RoPE) to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, T, Dk).
        dim (int): The dimension to apply rotations (usually Dk).
        base (float): Base for frequency computation.

    Returns:
        torch.Tensor: Tensor with RoPE applied.
    """
    seq_len = x.size(2)
    device = x.device
    dtype = x.dtype

    theta = base ** (-torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
    positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    angles = positions * theta.unsqueeze(0)
    sin = torch.sin(angles)
    cos = torch.cos(angles)

    x1 = x[..., : dim // 2]
    x2 = x[..., dim // 2 : dim]
    rotated = torch.cat(
        [x1 * cos.unsqueeze(0).unsqueeze(1) - x2 * sin.unsqueeze(0).unsqueeze(1),
         x1 * sin.unsqueeze(0).unsqueeze(1) + x2 * cos.unsqueeze(0).unsqueeze(1)],
        dim=-1
    )
    return rotated

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with RoPE and GQA support using scaled_dot_product_attention.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_kv_heads: int = None,
        dropout: float = 0.1,
        is_causal: bool = False,
        use_rope: bool = True,
        rope_base: float = 10000.0
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else n_heads
        assert self.n_heads % self.num_kv_heads == 0, "n_heads must be divisible by num_kv_heads"
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.is_causal = is_causal
        self.use_rope = use_rope
        self.rope_base = rope_base

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.size()

        Q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            Q = apply_rotary_emb(Q, dim=self.d_k, base=self.rope_base)
            K = apply_rotary_emb(K, dim=self.d_k, base=self.rope_base)

        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :].expand(-1, self.n_heads, T, -1).bool()
        else:
            attn_mask = None

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out = F.scaled_dot_product_attention(
                query=Q,
                key=K,
                value=V,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=self.is_causal,
                enable_gqa=True
            )

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.linear_out(out)

class ConformerFeedForward(nn.Module):
    """
    Feed-forward module for Conformer with GeLU activation.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class ConformerConvolution(nn.Module):
    """
    Convolution module for Conformer with LayerNorm and GeLU.
    """
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, groups=d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)  # Transpose for LayerNorm
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # Back to (B, D, T)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return x

class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding for Conformer (simplified).
    """
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)
        self.max_len = max_len
        self.pe = None
        self.extend_pe(max_len)

    def extend_pe(self, max_len: int, device: torch.device = None, dtype: torch.dtype = torch.float32):
        self.pe = nn.Parameter(torch.randn(max_len * 2 + 1, 1, self.d_model, device=device, dtype=dtype))
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x * self.scale
        x = self.dropout(x)
        return x, self.pe

class ConformerLayer(nn.Module):
    """
    Single Conformer layer.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        num_kv_heads: int,
        conv_kernel_size: int,
        dropout: float,
        dropout_att: float
    ):
        super().__init__()
        self.fc_factor = 0.5
        self.norm_ff1 = nn.LayerNorm(d_model)
        self.ff1 = ConformerFeedForward(d_model, d_ff, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout_att,
            is_causal=False,
            use_rope=True
        )
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model, conv_kernel_size, dropout)
        self.norm_ff2 = nn.LayerNorm(d_model)
        self.ff2 = ConformerFeedForward(d_model, d_ff, dropout)
        self.norm_out = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = x
        x = self.norm_ff1(x)
        x = self.ff1(x)
        x = residual + self.dropout(x) * self.fc_factor

        residual = x
        x = self.norm_attn(x)
        x = self.self_attn(x, attention_mask=pad_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm_conv(x)
        x = self.conv(x, pad_mask=pad_mask)
        x = residual + self.dropout(x)

        residual = x
        x = self.norm_ff2(x)
        x = self.ff2(x)
        x = residual + self.dropout(x) * self.fc_factor

        x = self.norm_out(x)
        return x

class ConformerEncoder(nn.Module):
    """
    Conformer Encoder for ASR.
    """
    def __init__(
        self,
        feat_in: int,
        n_layers: int,
        d_model: int,
        subsampling_factor: int = 4,
        ff_expansion_factor: int = 4,
        n_heads: int = 4,
        num_kv_heads: int = None,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        dropout_att: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.feat_in = feat_in
        self.subsampling_factor = subsampling_factor

        self.pre_encode = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=(3, 3), stride=(2, subsampling_factor), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.GELU()
        )
        self.pre_encode_linear = nn.Linear(d_model * ((feat_in + 1) // subsampling_factor), d_model)

        self.pos_enc = RelPositionalEncoding(d_model, dropout, max_len=5000)

        self.layers = nn.ModuleList([
            ConformerLayer(
                d_model=d_model,
                d_ff=d_model * ff_expansion_factor,
                n_heads=n_heads,
                num_kv_heads=num_kv_heads if num_kv_heads is not None else n_heads // 4,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                dropout_att=dropout_att
            ) for _ in range(n_layers)
        ])

        self.max_audio_length = 5000

    def forward(
        self,
        audio_signal: torch.Tensor,
        length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if audio_signal.shape[1] != self.feat_in:
            raise ValueError(f"Expected audio_signal dimension {self.feat_in}, got {audio_signal.shape[1]}")

        self.update_max_seq_length(seq_length=audio_signal.size(2), device=audio_signal.device)

        audio_signal = audio_signal.unsqueeze(1)
        audio_signal = self.pre_encode(audio_signal)
        batch, channels, feat, time = audio_signal.shape
        audio_signal = audio_signal.permute(0, 3, 2, 1).reshape(batch, time, feat * channels)
        audio_signal = self.pre_encode_linear(audio_signal)
        length = (length // self.subsampling_factor).to(torch.int64)

        audio_signal, pos_emb = self.pos_enc(audio_signal)

        pad_mask = torch.arange(0, audio_signal.size(1), device=audio_signal.device).expand(
            length.size(0), -1
        ) < length.unsqueeze(-1)
        pad_mask = pad_mask.bool()

        for layer in self.layers:
            audio_signal = layer(
                x=audio_signal,
                att_mask=None,
                pos_emb=pos_emb,
                pad_mask=pad_mask
            )

        audio_signal = audio_signal.transpose(1, 2)
        return audio_signal, length

    def update_max_seq_length(self, seq_length: int, device: torch.device):
        if seq_length > self.max_audio_length:
            self.max_audio_length = seq_length
            self.pos_enc.extend_pe(self.max_audio_length, device, self.pos_enc.pe.dtype)

def load_and_preprocess_wav(wav_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, float]:
    """Load and resample .wav file."""
    waveform, sr = librosa.load(wav_path, sr=None, mono=True)
    if sr != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sample_rate)
    return waveform, sample_rate

def extract_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int = 80,
    hop_length: int = 160,
    win_length: int = 400,
    n_fft: int = 512
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract mel-spectrogram from a waveform chunk."""
    waveform = torch.from_numpy(waveform).float().unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=sample_rate // 2
    )
    mel_spec = mel_transform(waveform)
    mel_spec = torch.log(mel_spec + 1e-9)
    length = torch.tensor([mel_spec.size(2)], dtype=torch.int64)
    return mel_spec, length

def chunk_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    chunk_duration_seconds: float = 20.0,
    overlap_seconds: float = 1.0
) -> list:
    """Split waveform into overlapping chunks."""
    chunk_samples = int(chunk_duration_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)
    step_samples = chunk_samples - overlap_samples
    chunks = []
    start = 0
    while start < len(waveform):
        end = min(start + chunk_samples, len(waveform))
        chunks.append(waveform[start:end])
        if end == len(waveform):
            break
        start += step_samples
    return chunks

def process_wav_chunks(
    wav_path: str,
    model: ConformerEncoder,
    chunk_duration_seconds: float = 20.0,
    overlap_seconds: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process a .wav file by chunking, encoding, and concatenating outputs."""
    waveform, sample_rate = load_and_preprocess_wav(wav_path, sample_rate=16000)
    chunks = chunk_waveform(waveform, sample_rate, chunk_duration_seconds, overlap_seconds)

    all_outputs = []
    all_lengths = []
    overlap_frames = int(overlap_seconds * sample_rate / 160)
    subsampling_factor = model.subsampling_factor

    device = next(model.parameters()).device
    for chunk_wave in chunks:
        mel_spec, length = extract_mel_spectrogram(waveform=chunk_wave, sample_rate=sample_rate, n_mels=model.feat_in)
        mel_spec = mel_spec.to(device)
        length = length.to(device)

        with torch.no_grad():
            chunk_output, chunk_length = model(mel_spec, length)

        if overlap_frames > 0 and len(all_outputs) > 0:
            chunk_output = chunk_output[:, :, overlap_frames // subsampling_factor :]
            chunk_length -= overlap_frames // subsampling_factor

        all_outputs.append(chunk_output)
        all_lengths.append(chunk_length)

    full_output = torch.cat(all_outputs, dim=2)
    full_length = sum(all_lengths)
    return full_output, torch.tensor([full_length], dtype=torch.int64)

if __name__ == "__main__":
    # Initialize model
    model = ConformerEncoder(
        feat_in=80,
        n_layers=17,
        d_model=256,
        subsampling_factor=4,
        ff_expansion_factor=4,
        n_heads=8,
        num_kv_heads=2,
        conv_kernel_size=31,
        dropout=0.1,
        dropout_att=0.0
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Process a .wav file
    wav_path = "path/to/your/audio.wav"  # Replace with actual path
    outputs, encoded_lengths = process_wav_chunks(wav_path, model)
    print(f"Output shape: {outputs.shape}")
    print(f"Encoded lengths: {encoded_lengths}")
