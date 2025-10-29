from typing import Optional, Tuple, Union
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
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_kv_heads = num_kv_heads or n_heads
        assert self.n_heads % self.num_kv_heads == 0
        self.d_k = d_model // n_heads
        self.dropout = dropout
        self.is_causal = is_causal
        self.use_rope = use_rope
        self.rope_base = rope_base

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)

    # reshape (B, T, d) -> (B, H, T, d_k)
    def _reshape_q(self, x):
        B, T, _ = x.shape
        return self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _reshape_kv(self, x):
        B, T, _ = x.shape
        return (
            self.w_k(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2),
            self.w_v(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        use_cache: bool = False,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns:
            out          – (B, T, d_model)
            new_cache    – (k_cache, v_cache)  each (B, num_kv_heads, T_cache, d_k)
        """
        B, T, _ = x.shape
        device = x.device

        Q = self._reshape_q(x)                               # (B, H, T, d_k)

        if use_cache and cache is not None:
            K_cache, V_cache = cache
            K = self._reshape_kv(x)[0]                       # new key (B, H_kv, T, d_k)
            V = self._reshape_kv(x)[1]                       # new value
            K = torch.cat([K_cache, K], dim=2)               # (B, H_kv, T_cache+T, d_k)
            V = torch.cat([V_cache, V], dim=2)
        else:
            K, V = self._reshape_kv(x)                       # (B, H_kv, T, d_k)

        if self.use_rope:
            Q = apply_rotary_emb(Q, dim=self.d_k, base=self.rope_base)
            K = apply_rotary_emb(K, dim=self.d_k, base=self.rope_base)

        if self.is_causal:
            causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
            causal = causal[None, None, :, :]               # (1,1,T,T)
        else:
            causal = None

        if attention_mask is not None:
            pad = attention_mask[:, None, :, None].bool()    # (B,1,T,1)
            if causal is not None:
                mask = pad.expand(-1, -1, -1, T) & causal.expand(B, -1, -1, -1)
            else:
                mask = pad.expand(-1, -1, -1, T)
        else:
            mask = causal

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            enable_gqa=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.linear_out(out)

        if use_cache:
            return out, (K, V)
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
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_kv_heads = num_kv_heads or n_heads
        assert self.n_heads % self.num_kv_heads == 0
        self.d_k = d_model // n_heads
        self.dropout = dropout
        self.use_rope = use_rope
        self.rope_base = rope_base

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.w_v = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)

    def _reshape_q(self, x):
        B, T, _ = x.shape
        return self.w_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

    def _reshape_kv(self, x):
        B, T, _ = x.shape
        return (
            self.w_k(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2),
            self.w_v(x).view(B, T, self.num_kv_heads, self.d_k).transpose(1, 2),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None,
        *,
        use_cache: bool = False,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T_q, _ = query.shape
        T_kv = key_value.shape[1]
        device = query.device

        Q = self._reshape_q(query)                                 # (B, H, T_q, d_k)

        if use_cache and cache is not None:
            K_cache, V_cache = cache
            K_new, V_new = self._reshape_kv(key_value)
            K = torch.cat([K_cache, K_new], dim=2)
            V = torch.cat([V_cache, V_new], dim=2)
        else:
            K, V = self._reshape_kv(key_value)                     # (B, H_kv, T_kv, d_k)

        if self.use_rope:
            Q = apply_rotary_emb(Q, dim=self.d_k, base=self.rope_base)
            K = apply_rotary_emb(K, dim=self.d_k, base=self.rope_base)

        if kv_mask is not None:
            kv_mask = kv_mask[:, None, None, :].bool()             # (B,1,1,T_kv)

        if query_mask is not None:
            q_mask = query_mask[:, None, :, None].bool()           # (B,1,T_q,1)
            if kv_mask is not None:
                attn_mask = q_mask.expand(-1, -1, -1, T_kv) & kv_mask.expand(-1, -1, T_q, -1)
            else:
                attn_mask = q_mask.expand(-1, -1, -1, T_kv)
        else:
            attn_mask = kv_mask

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            enable_gqa=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.linear_out(out)

        if use_cache:
            return out, (K, V)
        return out, None
    

class DecoderBlockCrossAttention(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        n_heads,
        num_kv_heads,
        rope_base,
        dropout=0.1,
        use_cross=True,
        use_flash=False,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            is_causal=True,
            use_rope=True,
            rope_base=rope_base,
        )
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.use_cross = use_cross
        if use_cross:
            self.cross_attn = CrossAttention(
                d_model=d_model,
                n_heads=n_heads * 2,            
                num_kv_heads=num_kv_heads,
                dropout=dropout,
                use_rope=True,
                rope_base=rope_base,
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
        cache: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor],
                             Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor,
               Optional[Tuple[Tuple[torch.Tensor, torch.Tensor],
                             Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        cache = (self_attn_cache, cross_attn_cache)
                each is (k, v) tuple
        """
        x = decoder_input

        # self-attention
        self_cache = cache[0] if use_cache and cache else None
        x, new_self = self.self_attn(
            self.norm1(x),
            attention_mask=decoder_mask,
            use_cache=use_cache,
            cache=self_cache,
        )
        x = x + decoder_input

        # cross-attention 
        cross_cache = cache[1] if use_cache and cache else None
        if self.use_cross:
            x, new_cross = self.cross_attn(
                query=self.norm2(x),
                key_value=encoder_output,
                query_mask=decoder_mask,
                kv_mask=encoder_mask,
                use_cache=use_cache,
                cache=cross_cache,
            )
            x = x + x 
        else:
            new_cross = None

        # feed-forward 
        x = x + self.dropout(self.feed_forward(self.norm3(x)))

        if use_cache:
            return x, (new_self, new_cross)
        return x, None
    


from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

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


    def forward(
        self,
        input_ids: torch.Tensor,                     # [B, S]  (S=1 during generation)
        input_audio: torch.Tensor,                   # [B, 4, T_audio]
        attention_mask: Optional[torch.Tensor] = None,
        class_ids: Optional[torch.Tensor] = None,
        *,
        use_cache: bool = False,
        cache: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        """
        Args:
            input_ids:      (B, S)
            input_audio:    (B, 4, T_audio)  – raw codebook indices
            attention_mask: (B, S)  – 1 = real token, 0 = pad (optional)
            class_ids:      (B, 1)  – conditioning class (if conditional)
            use_cache:      bool – whether to return/update KV cache
            cache:          tuple of (self_kv, cross_kv) per layer, or None

        Returns:
            logits: (B, S, vocab_size)
            if use_cache=True → (logits, new_cache)
        """
        device = input_ids.device
        B, S = input_ids.shape

        x = self.token_embedding(input_ids)                 # (B, S, d_model)

        if self.conditional:
            assert class_ids is not None
            x = x + self.cond_embedding(class_ids)          # broadcast over S

        # sum the 4 codebook embeddings → (B, T_audio, d_model)
        audio_emb = sum(self.codes_embedding[i](input_audio[:, i]) for i in range(4))
        audio_emb = self.norm_audio(audio_emb)
        if self.compression:
            audio_emb = self.audio_compression(audio_emb)   # (B, T_audio', d_model)

        causal_mask = self._causal_mask(S, device)       # (S, S)

        # pad mask (if provided) – broadcast to (B, S, S)
        if attention_mask is not None:
            # attention_mask: (B, S) -> (B, 1, S)
            pad_mask = attention_mask.unsqueeze(1)
            # combine: pad_mask * causal_mask
            attn_mask = pad_mask * causal_mask.unsqueeze(0)
        else:
            attn_mask = causal_mask.unsqueeze(0)        # (1, S, S)

        # ------------------- KV-cache initialisation -------------------
        new_cache = [] if use_cache else None
        layer_cache_idx = 0

        for layer in self.layers:
            if use_cache:
                self_kv, cross_kv = (cache[layer_cache_idx] if cache else (None, None))
            else:
                self_kv, cross_kv = None, None

            x, self_kv_new, cross_kv_new = layer(
                decoder_input=x,
                encoder_output=audio_emb,
                decoder_mask=attn_mask,
                use_cache=use_cache,
                cache=(self_kv, cross_kv),
            )

            if use_cache:
                new_cache.append((self_kv_new, cross_kv_new))
                layer_cache_idx += 1

        logits = self.output_projection(x)                  # (B, S, vocab)

        if use_cache:
            return logits, tuple(new_cache)
        return logits