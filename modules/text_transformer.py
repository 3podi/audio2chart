import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import math
import lightning as L
from torchmetrics import Accuracy
import inspect

from scheduler import LinearWarmupCosineAnnealingLR

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
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

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations and split into heads
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)   # (B, nh, T, hs)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        
        # Create causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask shape: (batch_size, seq_len) with 0 == padding_tokens
            # Convert to (batch_size, 1, 1, seq_len) for broadcasting
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.linear_out(out)
        
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear_out(self.dropout(F.relu(self.linear1(x))))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        # Self-attention with layer norm and residual connection
        x = self.self_attn(self.norm1(x), attention_mask)

        # Feed-forward with residual connection and layer norm
        x = x + self.feed_forward(self.norm2(x))        
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :].transpose(0, 1)


class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, pad_token_id, eos_token_id, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_seq_len=5000, dropout=0.1, weigth_tying=False):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        if weigth_tying:
            self.token_embedding.weight = self.output_projection.weight
        
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
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - Token indices
            attention_mask: (batch_size, seq_len) - Mask where 1 means attend, 0 means pad token
        
        Returns:
            logits: (batch_size, seq_len, vocab_size) - Output logits
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)
        
        # Token embeddings and positional encoding
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Output projection to vocabulary
        logits = self.output_projection(x)
        
        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, attention_mask=None):
        """Simple greedy generation with temperature and top-k sampling"""
        self.eval()
        with torch.no_grad():
            generated = input_ids.clone()
            
            for _ in range(max_length - input_ids.size(1)):
                # Create or update attention mask
                if attention_mask is not None:
                    current_mask = torch.cat([
                        attention_mask, 
                        torch.ones(attention_mask.size(0), 1, device=attention_mask.device, dtype=torch.long)
                    ], dim=1)
                else:
                    current_mask = None
                
                # Forward pass
                logits = self.forward(generated, current_mask)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Update attention mask
                if attention_mask is not None:
                    attention_mask = current_mask
                
                # Stop if we generate pad token
                if (next_token == self.pad_token_id or next_token == self.eos_token_id).all():
                    print('Found exit token, stopping generation')
                    break
                    
        return generated
    

class NotesTransformer(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        
        self.transformer = TransformerDecoderOnly(cfg)
        self.vocab_size = cfg 
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size)
        
        # For perplexity calculation
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        
        # Input: x[:,:-1], Target: x[:,1:]
        input_tokens = x[:, :-1] 
        target_tokens = x[:, 1:] 
        
        logits = self.transformer(input_tokens)  # Shape: (batch_size, seq_len-1, vocab_size)
        
        logits_flat = logits.view(-1, self.vocab_size)  # (batch_size * seq_len, vocab_size)
        targets_flat = target_tokens.view(-1)  # (batch_size * seq_len)
        
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=vocab_size-1)  # padding token is last vocab idx
        
        # Calculate accuracy
        preds = torch.argmax(logits_flat, dim=-1)
        acc = self.train_accuracy(preds, targets_flat)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_perplexity", perplexity, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        
        input_tokens = x[:, :-1]
        target_tokens = x[:, 1:]
        
        logits = self.transformer(input_tokens)
        
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = target_tokens.view(-1)
        
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.vocab_size-1)
        
        # Calculate accuracy
        preds = torch.argmax(logits_flat, dim=-1)
        acc = self.val_accuracy(preds, targets_flat)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Log validation metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_perplexity", perplexity, on_epoch=True, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        # Similar to validation_step
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = self.transformer.configure_optimizers(
            weight_decay = self.cfg.optimizer.weight_decay,
            learning_rate = self.cfg.optimizer.lr
        )

        ### Define number of steps based on dataloader
        
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer = optimizer,
            warmup_steps = self.cfg.optimizer.warmup_steps,
            max_steps = self.cfg.optimizer.max_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            },
        }

    def on_train_epoch_end(self):
        # Reset metrics at the end of each epoch
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        self.val_accuracy.reset()


# Example usage and testing
if __name__ == "__main__":
    # Model parameters
    vocab_size = 16
    d_model = 512
    n_heads = 8
    n_layers = 6
    max_seq_len = 512
    pad_token_id = vocab_size - 1
    eos_token_id = pad_token_id - 1
    
    # Create model
    model = TransformerDecoderOnly(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        weigth_tying=False
    )
    
    # Example input with padding
    batch_size = 1
    seq_len = 10
    
    # Create sample input with some padding tokens
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    input_ids[:, -3:] = pad_token_id  # Add padding to last 3 positions
    
    print("Input shape:", input_ids.shape)
    print("Input tokens:", input_ids)
    
    # Create attention mask (1 = attend, 0 = ignore padding)
    attention_mask = (input_ids != pad_token_id).long()
    print("Attention mask:", attention_mask)
    
    # Forward pass
    logits = model(input_ids, attention_mask)
    print("Output logits shape:", logits.shape)
    
    # Test generation
    prompt = torch.randint(1, vocab_size, (1, 5))  # Start with a short prompt
    print('Prompt for generation: ', prompt)
    generated = model.generate(prompt, max_length=20, temperature=0.8, top_k=10)
    print("Generated sequence shape:", generated.shape)
    print("Generated tokens:", generated)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")