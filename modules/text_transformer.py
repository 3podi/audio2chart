import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import math
import lightning as L
from torchmetrics import Accuracy
import inspect

from modules.scheduler import LinearWarmupCosineAnnealingLR

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


class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, pad_token_id, eos_token_id, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_seq_len=5000, dropout=0.1, conditional=False, use_flash=False, weigth_tying=False):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.conditional = conditional
        if self.conditional:
            self.cond_embedding = nn.Embedding(4, d_model)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout, use_flash)
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
    
    def forward(self, input_ids, attention_mask=None, class_ids=None):
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

        if self.conditional:
            assert class_ids is not None, "class_idx must be provided for conditional transformer"
            # Embed the class index and add to the input
            x = x + self.cond_embedding(class_ids).unsqueeze(1)
        
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
    def __init__(self, pad_token_id, eos_token_id, vocab_size, cfg_model, cfg_optimizer=None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.transformer = TransformerDecoderOnly(
            vocab_size = self.vocab_size,
            pad_token_id = pad_token_id,
            eos_token_id = eos_token_id,
            d_model = cfg_model.d_model,
            n_heads = cfg_model.n_heads,
            n_layers = cfg_model.n_layers,
            d_ff = cfg_model.d_ff,
            max_seq_len = cfg_model.max_seq_len,
            dropout = cfg_model.dropout,
            conditional = cfg_model.conditional,
            use_flash = cfg_model.use_flash
        )

        self.cfg_optimizer = cfg_optimizer
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size-1, ignore_index=self.vocab_size-1)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size-1, ignore_index=self.vocab_size-1)
        
        # For perplexity calculation
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        
        x = batch.get('input_ids', None)
        mask = batch.get('attention_mask', None)
        class_ids = batch.get('cond_diff', None)
        
        input_tokens = x[:, :-1].contiguous()
        target_tokens = x[:, 1:].contiguous()
        
        # Forward pass
        logits = self.transformer(input_tokens, attention_mask=mask, class_ids=class_ids)
        
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_tokens.reshape(-1)
        
        # Compute loss
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.vocab_size-1)
        
        preds = torch.argmax(logits_flat, dim=-1)
        acc = self.train_accuracy(preds, targets_flat)
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/perplexity", perplexity, on_step=True, on_epoch=True)

        # Compute per-class metrics
        if class_ids is not None and batch_idx % 100 == 0:
            with torch.no_grad():
                # Expand class_ids to match flattened logits dimensions
                # class_ids: [batch_size] -> [batch_size * seq_len]
                seq_len = target_tokens.shape[1]
                class_ids_expanded = class_ids.unsqueeze(1).expand(-1, seq_len).reshape(-1)
                
                # First filter out ignored tokens globally
                valid_mask = targets_flat != (self.vocab_size - 1)
                
                if valid_mask.sum() > 0:  # Only proceed if there are valid tokens
                    valid_logits = logits_flat[valid_mask]
                    valid_targets = targets_flat[valid_mask]
                    valid_preds = preds[valid_mask]
                    valid_class_ids = class_ids_expanded[valid_mask]
                    
                    # Get unique classes among valid tokens
                    unique_classes = torch.unique(valid_class_ids)
                    
                    for class_id in unique_classes:
                        # Create mask for current class among valid tokens
                        class_mask = (valid_class_ids == class_id)

                        if class_mask.sum() > 0:  # Only compute if class has valid tokens
                            # Filter predictions and targets for this class
                            class_logits = valid_logits[class_mask]
                            class_targets = valid_targets[class_mask]
                            class_preds = valid_preds[class_mask]
                            
                            # Compute class-specific metrics
                            class_loss = F.cross_entropy(class_logits, class_targets)
                            class_acc = (class_preds == class_targets).float().mean()
                            class_perplexity = torch.exp(class_loss)
                            
                            # Log class-specific metrics
                            class_name = f"class_{int(class_id.item())}"
                            self.log(f"train/loss_{class_name}", class_loss, on_step=True, on_epoch=True)
                            self.log(f"train/acc_{class_name}", class_acc, on_step=True, on_epoch=True)
                            self.log(f"train/perplexity_{class_name}", class_perplexity, on_step=True, on_epoch=True)
        
    
        return loss
        

    def validation_step(self, batch, batch_idx):
        
        x = batch.get('input_ids', None)
        mask = batch.get('attention_mask', None)
        class_ids = batch.get('cond_diff', None)
        
        input_tokens = x[:, :-1].contiguous()
        target_tokens = x[:, 1:].contiguous()
        
        # Forward pass
        logits = self.transformer(input_tokens, attention_mask=mask, class_ids=class_ids)
        
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_tokens.reshape(-1)
        
        # Compute loss
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.vocab_size-1)
        
        preds = torch.argmax(logits_flat, dim=-1)
        acc = self.train_accuracy(preds, targets_flat)
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/perplexity", perplexity, on_step=True, on_epoch=True)

        # Compute per-class metrics
        if class_ids is not None and batch_idx % 100 == 0:
            # Expand class_ids to match flattened logits dimensions
            # class_ids: [batch_size] -> [batch_size * seq_len]
            seq_len = target_tokens.shape[1]
            class_ids_expanded = class_ids.unsqueeze(1).expand(-1, seq_len).reshape(-1)
            
            # First filter out ignored tokens globally
            valid_mask = targets_flat != (self.vocab_size - 1)
            
            if valid_mask.sum() > 0:  # Only proceed if there are valid tokens
                valid_logits = logits_flat[valid_mask]
                valid_targets = targets_flat[valid_mask]
                valid_preds = preds[valid_mask]
                valid_class_ids = class_ids_expanded[valid_mask]
                
                # Get unique classes among valid tokens
                unique_classes = torch.unique(valid_class_ids)
                
                for class_id in unique_classes:
                    # Create mask for current class among valid tokens
                    class_mask = (valid_class_ids == class_id)
                    
                    if class_mask.sum() > 0:  # Only compute if class has valid tokens
                        # Filter predictions and targets for this class
                        class_logits = valid_logits[class_mask]
                        class_targets = valid_targets[class_mask]
                        class_preds = valid_preds[class_mask]
                        
                        # Compute class-specific metrics
                        class_loss = F.cross_entropy(class_logits, class_targets)
                        class_acc = (class_preds == class_targets).float().mean()
                        class_perplexity = torch.exp(class_loss)
                        
                        # Log class-specific metrics
                        class_name = f"class_{int(class_id.item())}"
                        self.log(f"val/loss_{class_name}", class_loss, on_step=True, on_epoch=True)
                        self.log(f"val/acc_{class_name}", class_acc, on_step=True, on_epoch=True)
                        self.log(f"val/perplexity_{class_name}", class_perplexity, on_step=True, on_epoch=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        # Similar to validation_step
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = self.transformer.configure_optimizers(
            weight_decay = self.cfg_optimizer.weight_decay,
            learning_rate = self.cfg_optimizer.lr,
            betas = (0.9, 0.95),
            device_type=self.device
        )

        ### Define number of steps based on dataloader
        
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer = optimizer,
            warmup_steps = self.cfg_optimizer.warmup_steps,
            max_steps = self.cfg_optimizer.max_steps
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