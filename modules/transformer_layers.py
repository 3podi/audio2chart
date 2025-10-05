import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
import inspect


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_flash=False, is_causal=True):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.use_flash = use_flash
        self.is_causal = is_causal

    def forward(self, x, attention_mask=None):
        """
        x: (B, T, d_model)
        attention_mask: (B, T) with 1=keep, 0=pad
        """
        B, T, _ = x.size()

        # Project and split into heads
        Q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T, Dk)
        K = self.w_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Build attention mask
        if self.is_causal:
            # Causal mask: only allow attending to past and current tokens
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))  # (T, T)
        else:
            # No causal masking: full attention allowed
            causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool)  # (T, T)

        if attention_mask is not None:
            # Only need to mask keys (broadcast over queries & heads)
            pad_mask = attention_mask[:, None, None, :].bool()  # (B, 1, 1, T)
            attn_mask = causal_mask[None, None, :, :] & pad_mask
        else:
            attn_mask = causal_mask[None, None, :, :]  # (1, 1, T, T)

        # FlashAttention
        if self.use_flash:
            out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,  # boolean mask
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False       # we already handle causal
            )
        else:
            # Regular attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            scores = scores.masked_fill(~attn_mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, V)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.linear_out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear_out(self.dropout(F.relu(self.linear1(x))))
    

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_flash=False, is_causal=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, use_flash, is_causal)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        # Self-attention with layer norm and residual connection
        x = x + self.self_attn(self.norm1(x), attention_mask)

        # Feed-forward with residual connection and layer norm
        x = x + self.feed_forward(self.norm2(x))        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding implementation"""
    
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create the div_term for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x.size(1) is seq_len
        # self.pe[:, :seq_len, :] gives us (1, seq_len, d_model)
        # Broadcasting handles the batch dimension
        return x + self.pe[:, :x.size(1), :]

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_flash=False):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.use_flash = use_flash

    def forward(self, query, key_value, query_mask=None, kv_mask=None):
        """
        Cross attention between query and key-value sequences
        
        Args:
            query: (B, T_q, d_model) - decoder/target sequence
            key_value: (B, T_kv, d_model) - encoder/source sequence  
            query_mask: (B, T_q) with 1=keep, 0=pad for query sequence
            kv_mask: (B, T_kv) with 1=keep, 0=pad for key-value sequence
        
        Returns:
            output: (B, T_q, d_model)
        """
        B, T_q, _ = query.size()
        T_kv = key_value.size(1)

        # Project queries from decoder, keys/values from encoder
        Q = self.w_q(query).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_q, Dk)
        K = self.w_k(key_value).view(B, T_kv, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_kv, Dk)
        V = self.w_v(key_value).view(B, T_kv, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_kv, Dk)

        # Create attention mask
        attn_mask = None
        if kv_mask is not None:
            # Mask out padded positions in key-value sequence
            # Shape: (B, 1, 1, T_kv) -> broadcast to (B, H, T_q, T_kv)
            attn_mask = kv_mask[:, None, None, :].bool()  # (B, 1, 1, T_kv)
            
            # If query also has padding, combine masks
            if query_mask is not None:
                # This create a (B, 1, T_q, T_kv) mask
                query_expanded = query_mask[:, None, :, None].bool()  # (B, 1, T_q, 1)
                attn_mask = attn_mask & query_expanded

        # FlashAttention
        if self.use_flash:
            out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,  # boolean mask
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False       # Cross attention is not causal
            )
        else:
            # Regular attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T_q, T_kv)
            
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float('-inf'))
                
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            out = torch.matmul(attn_weights, V)  # (B, H, T_q, Dk)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)  # (B, T_q, d_model)
        return self.linear_out(out)


class DecoderBlockCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_flash=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, use_flash)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout, use_flash)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, decoder_input, encoder_output, decoder_mask=None, encoder_mask=None):
        # Self-attention on decoder sequence
        x = decoder_input
        x = x + self.self_attn(self.norm1(x), decoder_mask)
        
        # Cross-attention
        x = x + self.cross_attn(
            query=self.norm2(x),        # decoder sequence
            key_value=encoder_output,   # encoder sequence  
            query_mask=decoder_mask,
            kv_mask=encoder_mask
        )
        
        # Feed forward
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x
    

### Update Transformer Blocks ### 
### Inspired from gpt-oss     ###

class RMSNorm(torch.nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-05
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)
    

def _apply_rotary_emb_batched(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T, H, q_mult, D)
    cos, sin: (T, D//2)
    """
    B, T, H, M, D = x.shape
    cos = cos.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, T, 1, 1, D/2)
    sin = sin.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # query/key: (B, T, H, q_mult, D)
        B, T, H, M, D = query.shape
        _, inv_freq = self._compute_concentration_and_inv_freq() 
        t = torch.arange(T, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        
        query = _apply_rotary_emb_batched(query, cos, sin)
        key = _apply_rotary_emb_batched(key, cos, sin)
        return query, key

def sdpa(Q, K, V, S, sm_scale, sliding_window=0, attention_mask=None):
    """
    Batched scaled dot-product attention with optional sliding window and sink logits.

    Args:
        Q: (B, T, H, q_mult, D)
        K: (B, T, H, D)
        V: (B, T, H, D)
        S: (H,) - sink logits per head
        sm_scale: float
        sliding_window: int (0 = full attention)
        attention_mask: (B, T) or (B, 1, 1, T) with 1=keep, 0=pad
    Returns:
        attn_out: (B, T, H*q_mult*D)
    """
    B, T, H, q_mult, D = Q.shape
    assert K.shape == (B, T, H, D)
    assert V.shape == (B, T, H, D)

    # Expand K/V for multi-query heads
    K = K.unsqueeze(3).expand(-1, -1, -1, q_mult, -1)  # (B, T, H, q_mult, D)
    V = V.unsqueeze(3).expand(-1, -1, -1, q_mult, -1)  # (B, T, H, q_mult, D)

    # Sink logits per (head, q_mult, query_pos)
    S = S.view(1, H, 1, 1, 1).expand(B, -1, q_mult, T, -1)  # (B, H, q_mult, T, 1)

    # --- Build causal/sliding mask (T, T) ---
    causal_mask = torch.triu(Q.new_full((T, T), -float("inf")), diagonal=1)
    if sliding_window > 0:
        causal_mask += torch.tril(
            Q.new_full((T, T), -float("inf")), diagonal=-sliding_window
        )

    # Expand to (1, 1, 1, T, T)
    full_mask = causal_mask[None, None, None, :, :]

    # --- Apply attention mask (1=keep, 0=pad) ---
    if attention_mask is not None:
        if attention_mask.dim() == 2:  # (B, T)
            pad_mask = (attention_mask == 0)
            full_mask = full_mask.masked_fill(
                pad_mask[:, None, None, None, :], -float("inf")
            )
        elif attention_mask.dim() == 4:  # (B, 1, 1, T)
            full_mask = full_mask.masked_fill(
                (attention_mask.unsqueeze(1) == 0), -float("inf")
            )
        else:
            raise ValueError(f"Unexpected attention_mask shape: {attention_mask.shape}")

    # --- Compute attention logits ---
    # (B, H, q_mult, T_q, T_k)
    QK = torch.einsum("bqhm d, bkhm d -> bhm qk", Q, K) * sm_scale
    QK = QK + full_mask  # (B, H, q_mult, T, T)

    # Add sink logits (extra key slot)
    QK = torch.cat([QK, S], dim=-1)  # (B, H, q_mult, T, T+1)

    # Softmax over keys (+sink)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]  # remove sink column

    # Weighted sum
    attn = torch.einsum("bhm qk, bkhm d -> bqhm d", W, V)  # (B, T, H, q_mult, D)
    return attn.reshape(B, T, -1)


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        d_model,
        head_dim,
        n_heads,
        num_key_value_heads,
        sliding_window,
        apply_sliding,
        rope_theta,
        rope_scaling_factor,
        rope_ntk_alpha,
        rope_ntk_beta,
        initial_context_length,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = n_heads
        self.num_key_value_heads = num_key_value_heads
        # Only apply sliding window to every other layer
        self.sliding_window = sliding_window if apply_sliding and layer_idx % 2 == 0 else 0
        self.sinks = torch.nn.Parameter(
            torch.empty(n_heads, dtype=torch.bfloat16)
        )
        self.norm = RMSNorm(d_model)
        qkv_dim = head_dim * (
            n_heads + 2 * num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            d_model, qkv_dim, dtype=torch.bfloat16
        )
        self.out = torch.nn.Linear(
            head_dim * self.num_attention_heads,
            d_model,
            dtype=torch.bfloat16,
        )
        self.sm_scale = 1 / math.sqrt(head_dim)
        self.rope = RotaryEmbedding(
            head_dim,
            rope_theta,
            torch.float32,
            initial_context_length=initial_context_length,
            scaling_factor=rope_scaling_factor,
            ntk_alpha=rope_ntk_alpha,
            ntk_beta=rope_ntk_beta,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        t = self.norm(x)
        qkv = self.qkv(t)
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            self.num_attention_heads
            * self.head_dim : (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()

        q = q.view(B, T, self.num_key_value_heads,
                self.num_attention_heads // self.num_key_value_heads, self.head_dim)
        k = k.view(B, T, self.num_key_value_heads, self.head_dim)
        v = v.view(B, T, self.num_key_value_heads, self.head_dim)
        q, k = self.rope(q, k)
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window, attention_mask=attention_mask)
        t = self.out(t)
        t = x + t
        return t


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class FeedForward2(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear_out(self.dropout(swiglu(self.linear1(self.norm(x)))))
    

class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        config,
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx)
        self.mlp = FeedForward2(d_model=config.d_model, d_ff=config.d_ff, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = x + self.mlp(x)
        return x
    

class DecoderBlockCrossAttention2(nn.Module):
    def __init__(
            self,
            d_model, 
            n_heads, 
            d_ff, 
            dropout, 
            head_dim,
            num_key_value_heads,
            sliding_window,
            apply_sliding,
            rope_theta,
            rope_scaling_factor,
            rope_ntk_alpha,
            rope_ntk_beta,
            initial_context_length,
            use_flash=False, 
            layer_idx=None
        ):
        super().__init__()
        self.self_attn = AttentionBlock(
            d_model=d_model,
            head_dim=head_dim,
            n_heads=n_heads,
            num_key_value_heads=num_key_value_heads,
            sliding_window=sliding_window,
            apply_sliding=apply_sliding,
            rope_theta=rope_theta,
            rope_scaling_factor=rope_scaling_factor,
            rope_ntk_alpha=rope_ntk_alpha,
            rope_ntk_beta=rope_ntk_beta,
            initial_context_length=initial_context_length,
            layer_idx=layer_idx
        )
        self.cross_attn = CrossAttention(d_model, n_heads, dropout, use_flash)
        self.feed_forward = FeedForward2(d_model, d_ff, dropout)
                
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, decoder_input, encoder_output, decoder_mask=None, encoder_mask=None):
        # Self-attention on decoder sequence
        x = decoder_input
        x = x + self.self_attn(x, decoder_mask)
        
        # Cross-attention
        x = x + self.cross_attn(
            query=self.norm(x),         # decoder sequence
            key_value=encoder_output,   # encoder sequence  
            query_mask=decoder_mask,
            kv_mask=encoder_mask
        )
        
        # Feed forward
        x = x + self.dropout(self.feed_forward(x))
        return x
    

class Transformer(torch.nn.Module):
    def __init__(
        self,
        # --- Model dimensions ---
        d_model: int = 512,
        head_dim: int = 64,
        n_heads: int = 8,
        num_key_value_heads: int = 4,
        d_ff: int = 2048,
        dropout: float = 0.1,
        # --- Architecture ---
        num_hidden_layers: int = 8,
        vocab_size: int = 0,
        max_audio_len: int = 5000,
        apply_sliding: bool = False,
        sliding_window: int = 0,
        use_flash: bool = True,
        conditional: bool = False,
        # --- Rotary Embedding ---
        rope_theta: float = 10000.0,
        initial_context_length: int = 1024,
        rope_scaling_factor: float = 1.0,
        rope_ntk_alpha: float = 1.0,
        rope_ntk_beta: float = 32.0,
        # --- RMSNorm ---
        rms_eps: float = 1e-5,
        **kwargs, 
    ):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.num_key_value_heads = num_key_value_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.max_audio_len = max_audio_len
        self.apply_sliding = apply_sliding
        self.sliding_window = sliding_window
        self.use_flash = use_flash
        self.conditional = conditional
        self.rope_theta = rope_theta
        self.initial_context_length = initial_context_length
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_ntk_alpha = rope_ntk_alpha
        self.rope_ntk_beta = rope_ntk_beta
        self.rms_eps = rms_eps

        self.embedding = torch.nn.Embedding(
            vocab_size, d_model, dtype=torch.bfloat16
        )
        self.audio_positional_encoding = PositionalEncoding(d_model, max_audio_len)

        self.block = torch.nn.ModuleList(
            [
                DecoderBlockCrossAttention2(
                    d_model, 
                    n_heads, 
                    d_ff, 
                    dropout, 
                    head_dim,
                    num_key_value_heads,
                    sliding_window,
                    apply_sliding,
                    rope_theta,
                    rope_scaling_factor,
                    rope_ntk_alpha,
                    rope_ntk_beta,
                    initial_context_length,
                    use_flash=use_flash, 
                    layer_idx=layer_idx
                )
                for layer_idx in range(num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.unembedding = torch.nn.Linear(
            d_model,
            vocab_size,
            bias=False,
            dtype=torch.bfloat16,
        )

    def forward(self, input_ids, input_audio, attention_mask=None, class_ids=None):
        input_audio = self.audio_positional_encoding(input_audio)#.permute(0, 2, 1))
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)
        
        # Token embeddings and positional encoding
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        if self.conditional:
            assert class_ids is not None, "class_idx must be provided for conditional transformer"
            # Embed the class index and add to the input
            x = x + self.cond_embedding(class_ids)
        
        # Pass through decoder layers
        for layer in self.block:
            x = layer(decoder_input=x, encoder_output=input_audio, decoder_mask=attention_mask)
        
        # Output projection to vocabulary
        x = self.norm(x)
        logits = self.unembedding(x)
        
        return logits