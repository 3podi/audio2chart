import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import inspect

def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)

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

def apply_rotary_emb(x: torch.Tensor, dim: int, base: float = 10000.0) -> torch.Tensor:
    """
    Apply Rotary Position Embeddings (RoPE) to the input tensor.
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
        self.is_causal = is_causal  # Will be used to generate the causal mask
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

        # Create combined attention mask
        if attention_mask is not None:
            # attention_mask: [B, T] -> [B, 1, T] (broadcastable to [B, n_heads, T, T])
            attn_mask = attention_mask.unsqueeze(1)  # Shape: [B, 1, T]
            attn_mask = attn_mask.bool()  # 1 = True (keep), 0 = False (mask)

            # Create causal mask (lower triangular)
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()  # [T, T]
            causal_mask = causal_mask[None, None, :, :]  # [1, 1, T, T]
            causal_mask = causal_mask.expand(B, 1, -1, -1)  # [B, 1, T, T]

            # Combine causal and custom mask: True only where both are True
            attn_mask = attn_mask.unsqueeze(2).expand(-1, -1, T, -1)  # [B, 1, T, T]
            attn_mask = attn_mask & causal_mask  # Element-wise AND
        else:
            # Default to causal mask if no custom mask provided and is_causal is True
            attn_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()[None, None, :, :]  # [1, 1, T, T]
            attn_mask = attn_mask.expand(B, 1, -1, -1)  # [B, 1, T, T] if is_causal else None
            attn_mask = attn_mask if self.is_causal else None

        # Apply scaled dot-product attention
        out = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,  # Disabled since we handle causality manually, lets get it
            enable_gqa=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.linear_out(out)
    

    
class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_kv_heads: int = None,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        use_flash: bool = False
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else n_heads
        assert self.n_heads % self.num_kv_heads == 0, "n_heads must be divisible by num_kv_heads"
        self.d_k = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.use_flash = use_flash

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key_value, query_mask=None, kv_mask=None, return_attention_weights=True):
        """
        Cross attention between query and key-value sequences.
        
        Args:
            query: (B, T_q, d_model) - decoder/target sequence
            key_value: (B, T_kv, d_model) - encoder/source sequence
            query_mask: (B, T_q) with 1=keep, 0=pad for query sequence
            kv_mask: (B, T_kv) with 1=keep, 0=pad for key-value sequence
            return_attention_weights: If True, return attention weights (only during evaluation)
        
        Returns:
            output: (B, T_q, d_model)
            attention_weights: (B, n_heads, T_q, T_kv) (if return_attention_weights=True and not training)
        """
        B, T_q, _ = query.size()
        T_kv = key_value.size(1)

        # Project and reshape
        Q = self.w_q(query).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, T_q, Dk)
        K = self.w_k(key_value).view(B, T_kv, self.num_kv_heads, self.d_k).transpose(1, 2)  # (B, H_kv, T_kv, Dk)
        V = self.w_v(key_value).view(B, T_kv, self.num_kv_heads, self.d_k).transpose(1, 2)  # (B, H_kv, T_kv, Dk)

        # Apply RoPE if enabled
        if self.use_rope:
            Q = apply_rotary_emb(Q, dim=self.d_k, base=self.rope_base)
            K = apply_rotary_emb(K, dim=self.d_k, base=self.rope_base)

        # Create combined attention mask
        attn_mask = None
        if kv_mask is not None or query_mask is not None:
            if kv_mask is not None:
                attn_mask = kv_mask[:, None, None, :].bool()  # (B, 1, 1, T_kv)
            else:
                attn_mask = torch.ones(B, 1, 1, T_kv, device=query.device, dtype=torch.bool)
            if query_mask is not None:
                query_expanded = query_mask[:, None, :, None].bool()  # (B, 1, T_q, 1)
                attn_mask = attn_mask.expand(-1, 1, T_q, -1)  # (B, 1, T_q, T_kv)
                attn_mask = attn_mask & query_expanded.expand(-1, 1, -1, T_kv)  # Element-wise AND

        # During training, use scaled_dot_product_attention without computing weights
        out = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p,
            is_causal=False,
            enable_gqa=True
        )

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)  # (B, T_q, d_model)
        out = self.linear_out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear_out = nn.Linear(d_ff // 2, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear_out(self.dropout(swiglu(self.linear1(x))))
    


class DecoderBlockCrossAttention(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, num_kv_heads, rope_base, dropout=0.1, use_cross=True, use_flash=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads, 
            num_kv_heads=num_kv_heads, 
            dropout=dropout, 
            is_causal=True, 
            use_rope=True, 
            rope_base=rope_base
        )
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.use_cross = use_cross
        if use_cross:
            self.cross_attn = CrossAttention(        
                d_model=d_model,
                n_heads=n_heads*2,
                num_kv_heads=num_kv_heads,
                dropout=dropout,
                use_rope=True,
                rope_base=rope_base,
                use_flash=True
            )
            self.norm2 = RMSNorm(d_model)
        
        self.norm1 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_input, encoder_output, decoder_mask=None, encoder_mask=None):
        # Self-attention on decoder sequence
        x = decoder_input
        x = x + self.self_attn(self.norm1(x), decoder_mask)
        
        # Cross-attention
        if self.use_cross:
            x = x + self.cross_attn(
                query=self.norm2(x),        # decoder sequence
                key_value=encoder_output,   # encoder sequence  
                query_mask=decoder_mask,
                kv_mask=encoder_mask
            )

        # Feed forward
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x
    

class TemporalConvPool(nn.Module):
    def __init__(self, dim: int, compression: int = 2, kernel_size: int = 5):
        """
        Temporal Conv1D pooling layer.

        Args:
            dim (int): embedding dimension (in_channels = out_channels)
            compression (int): stride factor (must be >= 1)
                               e.g. 2 means sequence length is halved
            kernel_size (int): temporal kernel size (should be odd)
        """
        super().__init__()
        assert compression >= 1, "compression must be >= 1"
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=compression,
            padding=padding
        )
        self.stride = compression

    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)   # [B, D, T]
        x = self.conv(x)        # downsample in time
        x = x.transpose(1, 2)   # [B, T_down, D]
        return x



class TransformerDecoderAudioConditioned(nn.Module):
    def __init__(
            self,
            vocab_size, 
            pad_token_id, 
            eos_token_id, 
            d_model=512, 
            n_heads=8, 
            num_kv_heads=2,
            n_layers=6, 
            d_ff=2048, 
            dropout=0.1,
            audio_drop=0.0,
            compression=None,
            rope_base=10000.0, 
            conditional=False, 
            use_flash=False,
            codebook_size=1024
        ):
        super().__init__()
        
        self.d_model = d_model
        d_ff = 4 * d_model
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.codes_embedding = nn.ModuleList(nn.Embedding(codebook_size, d_model) for _ in range(4))
        self.conditional = conditional
        if self.conditional:
            self.cond_embedding = nn.Embedding(4, d_model)
        
        # Decoder layers
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            use_cross = True
            self.layers.append(
                DecoderBlockCrossAttention(
                    d_model=d_model,
                    d_ff=d_ff,
                    n_heads=n_heads,
                    num_kv_heads=num_kv_heads, 
                    rope_base=rope_base, 
                    dropout=dropout, 
                    use_cross=use_cross, 
                    use_flash=use_flash
                )
            )
    
        # Audio projection layer to adapt codebook_dim to d_model
        #self.audio_projection = nn.Linear(128, d_model, bias=False)
        self.norm_audio = nn.LayerNorm(d_model)
        self.audio_drop = nn.Dropout(audio_drop)
        
        self.compression = compression
        if compression:
            self.audio_compression = TemporalConvPool(dim=d_model,compression=compression)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        # Special init, gpt2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('linear_out.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_attention_mask(self, input_ids):
        """Create attention mask where 1 means attend, 0 means don't attend (pad token)"""
        return (input_ids != self.pad_token_id).long()
    
    def forward(self, input_ids, input_audio, attention_mask=None, class_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - Token indices
            attention_mask: (batch_size, seq_len) - Mask where 1 means attend, 0 means pad token
        
        Returns:
            logits: (batch_size, seq_len, vocab_size) - Output logits
        """
        
        # Token embeddings
        x = self.token_embedding(input_ids)

        if self.conditional:
            assert class_ids is not None, "class_idx must be provided for conditional transformer"
            # Embed the class index and add to the input
            x = x + self.cond_embedding(class_ids)

        # Adapt audio codes        
        input_audio = sum( self.codes_embedding[i](input_audio[:,i,:]) for i in range(4)) 
        input_audio = self.norm_audio(input_audio)
        if self.compression:
            input_audio = self.audio_compression(input_audio)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(decoder_input=x, encoder_output=input_audio, decoder_mask=attention_mask)
        
        # Output projection to vocabulary
        logits = self.output_projection(x)
        
        return logits
