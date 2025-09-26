import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
import inspect


class MultiHeadAttention(nn.Module):
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

        # causal mask (T, T)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))  # (T, T)

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
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, use_flash=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, use_flash)
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
    

