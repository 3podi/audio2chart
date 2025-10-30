from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.transformer2 import apply_rotary_emb, swiglu, RMSNorm, TemporalConvPool, FeedForward

class MultiHeadAttention(nn.Module):
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

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        *,
        use_cache: bool = False,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: int = 0
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.size()

        Q = self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        if use_cache and cache is not None:
            K_past, V_past = cache
            K_new = self.w_k(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2).contiguous()
            V_new = self.w_v(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2).contiguous()
            K_past[:,:,step:step+1,:] = K_new
            V_past[:,:,step:step+1,:] = V_new
            K = K_past[:, :, :step+1, :]
            V = V_past[:, :, :step+1, :]
        else:
            K = self.w_k(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)
            V = self.w_v(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            Q = apply_rotary_emb(Q, dim=self.d_k, base=self.rope_base)
            K = apply_rotary_emb(K, dim=self.d_k, base=self.rope_base)

        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1)
            attn_mask = attn_mask.bool()

            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
            causal_mask = causal_mask[None, None, :, :]
            causal_mask = causal_mask.expand(B, 1, -1, -1)

            attn_mask = attn_mask.unsqueeze(2).expand(-1, -1, T, -1)
            attn_mask = attn_mask & causal_mask
        else:
            attn_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()[None, None, :, :]
            attn_mask = attn_mask.expand(B, 1, -1, -1) if self.is_causal else None

        out = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
            enable_gqa=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.linear_out(out)

        if use_cache:
            return out, (K_past, V_past)
        return out, None
    

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

    def forward(
        self,
        query,
        key_value,
        query_mask=None,
        kv_mask=None,
        *,
        use_cache: bool = False,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: int = 0
    ):
        B, T_q, _ = query.size()
        T_kv = key_value.size(1)

        Q = self.w_q(query).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)

        if use_cache and cache is not None:
            K, V = cache
        else:
            K = self.w_k(key_value).view(B, T_kv, self.num_kv_heads, self.d_k).transpose(1, 2).contiguous()
            V = self.w_v(key_value).view(B, T_kv, self.num_kv_heads, self.d_k).transpose(1, 2).contiguous()

        if self.use_rope:
            Q = apply_rotary_emb(Q, dim=self.d_k, base=self.rope_base)
            K = apply_rotary_emb(K, dim=self.d_k, base=self.rope_base)

        attn_mask = None
        if kv_mask is not None or query_mask is not None:
            if kv_mask is not None:
                attn_mask = kv_mask[:, None, None, :].bool()
            else:
                attn_mask = torch.ones(B, 1, 1, T_kv, device=query.device, dtype=torch.bool)
            if query_mask is not None:
                query_expanded = query_mask[:, None, :, None].bool()
                attn_mask = attn_mask.expand(-1, 1, T_q, -1)
                attn_mask = attn_mask & query_expanded.expand(-1, 1, -1, T_kv)

        out = F.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p,
            is_causal=False,
            enable_gqa=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.linear_out(out)

        if use_cache:
            return out, (K, V)
        return out, None
    

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

    def forward(
        self,
        decoder_input,
        encoder_output,
        decoder_mask=None,
        encoder_mask=None,
        *,
        use_cache: bool = False,
        self_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        step: int = 0
    ):
        x = decoder_input

        # Self-attention
        x_attn, self_kv = self.self_attn(
            self.norm1(x), decoder_mask,
            use_cache=use_cache, cache=self_kv,
            step=step
        )
        x = x + x_attn

        # Cross-attention
        if self.use_cross:
            x_cross, cross_kv = self.cross_attn(
                query=self.norm2(x),
                key_value=encoder_output,
                query_mask=decoder_mask,
                kv_mask=encoder_mask,
                use_cache=use_cache,
                cache=cross_kv,
                step=step
            )
            x = x + x_cross
        else:
            cross_kv = None

        # Feed forward
        x = x + self.dropout(self.feed_forward(self.norm3(x)))

        if use_cache:
            return x, (self_kv, cross_kv)
        return x, None
    

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
            codebook_size=1024,
            **kwargs,
        ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.codes_embedding = nn.ModuleList(
            nn.Embedding(codebook_size, d_model) for _ in range(4)
        )
        self.conditional = conditional
        if self.conditional:
            self.cond_embedding = nn.Embedding(4, d_model)
        
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                DecoderBlockCrossAttention(
                    d_model=d_model,
                    d_ff=4 * d_model,
                    n_heads=n_heads,
                    num_kv_heads=num_kv_heads, 
                    rope_base=rope_base, 
                    dropout=dropout,
                    use_cross=True,
                    use_flash=use_flash
                )
            )
        
        self.norm_audio = nn.LayerNorm(d_model)
        self.audio_drop = nn.Dropout(audio_drop)
        self.compression = compression
        if compression:
            self.audio_compression = TemporalConvPool(dim=d_model, compression=compression)

        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """(seq_len, seq_len) lower-triangular mask of 1s (attend)"""
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.long))

    def init_kv_cache(self, batch_size, max_len, device):
        """
        Init KV-cache both for self-attention and crossattention.
        
        Returns:
            self_cache: list of (self_kv, cross_kv) per layer
            cross_cache: list of None per layer, it gets init with the first forward pass
        """

        dtype = next(self.parameters()).dtype
        num_kv_heads = num_kv_heads if self.num_kv_heads is not None else self.n_heads
        d_k = self.d_model // self.n_heads
        self_cache = [
            (torch.zeros(batch_size,num_kv_heads, max_len, d_k, device=device, dtype=dtype),
             torch.zeros(batch_size,num_kv_heads, max_len, d_k, device=device, dtype=dtype))
            for _ in range(self.n_layers)        
            ]
        cross_cache = [None for _ in range(self.n_layers)]
        return self_cache, cross_cache


    def forward(
        self,
        input_ids: torch.Tensor,                       # [B, S] (S=1 during generation)
        audio_emb: torch.Tensor,                       # [B, T_audio', d_model] (precomputed encoder/audio embedding)
        attention_mask: Optional[torch.Tensor] = None, # [B, S] optional
        class_ids: Optional[torch.Tensor] = None,      # [B, 1] optional conditional class
        step: Optional[int] = None, 
        *,
        use_cache: bool = False,
        self_cache: Optional[
            List[Tuple[torch.Tensor, torch.Tensor]]
        ] = None,  # [(self_k: [B, n_kv_heads, max_len, d_k], self_v: same), ...] per layer
        cross_cache: Optional[
            List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,  # [(cross_k: [B, n_kv_heads, T_audio', d_k], cross_v: same), ...] per layer, may contain None before first forward
    ) -> Union[
        torch.Tensor,
        Tuple[
            torch.Tensor,
            Tuple[
                List[Tuple[torch.Tensor, torch.Tensor]],                # self_cache
                List[Optional[Tuple[torch.Tensor, torch.Tensor]]],      # cross_cache
            ],
        ],
    ]:
        """
        Forward pass of the decoder with optional self- and cross-attention caching.

        Args:
            input_ids: torch.Tensor
                Token indices for the current step. Shape: [B, S]
            audio_emb: torch.Tensor
                Precomputed encoder/audio embeddings used for cross-attention.
                Shape: [B, T_audio', d_model]
            attention_mask: Optional[torch.Tensor]
                Attention mask over input tokens. Shape: [B, S]. 1 = valid, 0 = pad.
            class_ids: Optional[torch.Tensor]
                Class-conditioning IDs, if used. Shape: [B, 1]
            step: Optional[int] = None
                Step in the autoregressive generation to slice the self kv cache
            use_cache: bool
                Whether to use and return KV cache for autoregressive generation.
            self_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
                One tuple per decoder layer: (self_k, self_v),
                each shaped [B, n_kv_heads, max_len, d_k].
                Updated in-place during generation.
            cross_cache: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]
                One tuple per decoder layer: (cross_k, cross_v),
                each shaped [B, n_kv_heads, T_audio', d_k].
                If None, computed from `audio_emb` on first forward.

        Returns:
            logits: torch.Tensor
                Next-token logits. Shape: [B, S, vocab_size]
            if use_cache:
                (logits, (self_cache, cross_cache))
        """
        device = input_ids.device
        B, S = input_ids.shape

        x = self.token_embedding(input_ids)                 # (B, S, d_model)

        if self.conditional:
            assert class_ids is not None
            x = x + self.cond_embedding(class_ids)          # broadcast over S

        # sum the 4 codebook embeddings -> (B, T_audio, d_model)
        #audio_emb = sum(self.codes_embedding[i](input_audio[:, i]) for i in range(4))
        #audio_emb = self.norm_audio(audio_emb)
        #if self.compression:
        #    audio_emb = self.audio_compression(audio_emb)   # (B, T_audio', d_model)


        for i, layer in enumerate(self.layers):
            self_kv = self_cache[i]
            cross_kv = cross_cache[i]

            x, self_kv, cross_kv = layer(
                decoder_input=x,
                encoder_output=audio_emb,
                decoder_mask=attention_mask,
                use_cache=use_cache,
                self_kv=self_kv,
                cross_kv=cross_kv,
                step=step
            )

            if use_cache:
                self_cache[i] = self_kv
                cross_cache[i] = cross_kv

        logits = self.output_projection(x)                  # (B, S, vocab)

        if use_cache:
            return logits, (self_cache, cross_cache)
        return logits