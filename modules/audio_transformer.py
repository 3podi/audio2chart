from modules.text_transformer import MultiHeadAttention, FeedForward, PositionalEncoding
from modules.audio_compression import SEANetEncoder2d
#from transformers import EncodecModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import math
import inspect
from modules.scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import Accuracy

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
    

class TransformerDecoderAudioConditioned(nn.Module):
    def __init__(self, vocab_size, pad_token_id, eos_token_id, d_model=512, n_heads=8, n_layers=6, 
                 d_ff=2048, max_seq_len=512, max_audio_len=10000, dropout=0.1, conditional=False, use_flash=False, weigth_tying=False):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.audio_positional_encoding = PositionalEncoding(d_model, max_audio_len)
        self.conditional = conditional
        if self.conditional:
            self.cond_embedding = nn.Embedding(4, d_model)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderBlockCrossAttention(d_model, n_heads, d_ff, dropout, use_flash)
            for _ in range(n_layers)
        ])

        # Audio adapter
        self.adapter = nn.Conv1d(
            in_channels=128,
            out_channels=d_model,
            kernel_size=4,
            stride=2,
            padding=0,
            bias=False
        )
        
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
    
    def forward(self, input_ids, input_audio, attention_mask=None, class_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - Token indices
            attention_mask: (batch_size, seq_len) - Mask where 1 means attend, 0 means pad token
        
        Returns:
            logits: (batch_size, seq_len, vocab_size) - Output logits
        """

        # Process audio
        #input_audio = input_audio.permute(0, 2, 1)
        #input_audio = self.adapter(input_audio) 
        print('input audio shape iniziale: ', input_audio.shape)
        input_audio = self.audio_positional_encoding(input_audio.permute(0, 2, 1))
        print('input_audio shape after positional: ', input_audio.shape)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_ids)
        
        # Token embeddings and positional encoding
        print('x shape: ', input_ids.shape)
        x = self.token_embedding(input_ids)
        print('x shape after token embed: ', x.shape)
        x = self.positional_encoding(x)
        print('x shape after pos enc: ', x.shape)

        if self.conditional:
            assert class_ids is not None, "class_idx must be provided for conditional transformer"
            # Embed the class index and add to the input
            #print(self.cond_embedding(class_ids).unsqueeze(1).shape)
            x = x + self.cond_embedding(class_ids)
        
        print('shape input ids: ', x.shape)
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(decoder_input=x, encoder_output=input_audio, decoder_mask=attention_mask)
        
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
    
    # TODO: generate/inference
    #def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, attention_mask=None):


class AudioTransformer(L.LightningModule):
    def __init__(self, pad_token_id, eos_token_id, vocab_size, cfg_model, cfg_optimizer=None):
        super().__init__()
        
        self.vocab_size = vocab_size

        # Transformer
        self.transformer = TransformerDecoderAudioConditioned(
            vocab_size = self.vocab_size,
            pad_token_id = pad_token_id,
            eos_token_id = eos_token_id,
            d_model = cfg_model.d_model,
            n_heads = cfg_model.n_heads,
            n_layers = cfg_model.n_layers,
            d_ff = cfg_model.d_ff,
            max_seq_len = cfg_model.max_seq_len,
            max_audio_len = cfg_model.max_audio_len,
            dropout = cfg_model.dropout,
            conditional = cfg_model.conditional,
            use_flash = cfg_model.use_flash
        )

        # Audio encoder
        self.audio_encoder = EncodecModel.from_pretrained("facebook/encodec_48khz").eval()

        # Optimizer
        self.cfg_optimizer = cfg_optimizer
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size-1, ignore_index=self.vocab_size-1)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size-1, ignore_index=self.vocab_size-1)
        
        # For perplexity calculation
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        
        audio = batch.get('audio', None)
        audio_mask = batch.get('audio_mask', None)
        x = batch.get('note_values', None)
        #x_t = batch.get('note_times', None)
        #x_dt = batch.get('note_durations', None)
        mask = batch.get('attention_mask', None)
        class_ids = batch.get('cond_diff', None)
        
        input_tokens = x[:, :-1].contiguous()
        target_tokens = x[:, 1:].contiguous()
        x_t = x_t[:,1:].contiguous()
        x_dt = x_dt[:,1:].contiguous()
        
        # Forward pass
        with torch.no_grad():
            audio_encoded = self.audio_encoder.encode(audio, audio_mask)
            decoded_frames = []
            for frame in audio_encoded.audio_codes:
                frame = frame.transpose(0, 1)
                embeddings = self.audio_encoder.quantizer.decode(frame)
                decoded_frames.append(embeddings)

            decoded_frames = torch.stack(decoded_frames)  # [n_frames, batch, embed_dim, time_per_frame]
            final_sequence = decoded_frames.permute(1, 0, 3, 2).reshape(decoded_frames.shape[1], -1, decoded_frames.shape[2])
        logits = self.transformer(input_tokens, final_sequence, attention_mask=mask, class_ids=class_ids)
        
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
        
        audio = batch.get('audio', None)
        audio_mask = batch.get('audio_mask', None)
        x = batch.get('note_values', None)
        #x_t = batch.get('note_times', None)
        #x_dt = batch.get('note_durations', None)
        mask = batch.get('attention_mask', None)
        class_ids = batch.get('cond_diff', None)
        
        input_tokens = x[:, :-1].contiguous()
        target_tokens = x[:, 1:].contiguous()
        x_t = x_t[:,1:].contiguous()
        x_dt = x_dt[:,1:].contiguous()
        
        # Forward pass
        with torch.no_grad():
            audio_encoded = self.audio_encoder.encode(audio, audio_mask)
            decoded_frames = []
            for frame in audio_encoded.audio_codes:
                frame = frame.transpose(0, 1)
                embeddings = self.audio_encoder.quantizer.decode(frame)
                decoded_frames.append(embeddings)

            decoded_frames = torch.stack(decoded_frames)  # [n_frames, batch, embed_dim, time_per_frame]
            final_sequence = decoded_frames.permute(1, 0, 3, 2).reshape(decoded_frames.shape[1], -1, decoded_frames.shape[2])
        logits = self.transformer(input_tokens, final_sequence, attention_mask=mask, class_ids=class_ids)
       
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



class WaveformTransformer(L.LightningModule):
    def __init__(self, pad_token_id, eos_token_id, vocab_size, cfg_model, cfg_optimizer=None):
        super().__init__()
        
        self.vocab_size = vocab_size

        # Transformer
        self.transformer = TransformerDecoderAudioConditioned(
            vocab_size = self.vocab_size,
            pad_token_id = pad_token_id,
            eos_token_id = eos_token_id,
            d_model = cfg_model.d_model,
            n_heads = cfg_model.n_heads,
            n_layers = cfg_model.n_layers,
            d_ff = cfg_model.d_ff,
            max_seq_len = cfg_model.max_seq_len,
            max_audio_len = cfg_model.max_audio_len,
            dropout = cfg_model.dropout,
            conditional = cfg_model.conditional,
            use_flash = cfg_model.use_flash
        )

        # Audio encoder
        self.audio_encoder = SEANetEncoder2d()

        # Optimizer
        self.cfg_optimizer = cfg_optimizer
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size-1, ignore_index=self.vocab_size-1)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size-1, ignore_index=self.vocab_size-1)
        
        #self = self.to(torch.bfloat16)  # Convert model to bfloat16
        # For perplexity calculation
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        
        audio = batch.get('audio', None)
        #audio = torch.randn(2, 1, 2048, device=self.device, dtype=torch.float32)
        #audio_mask = batch.get('audio_mask', None)
        x = batch.get('note_values', None)
        #x_t = batch.get('note_times', None)
        #x_dt = batch.get('note_durations', None)
        mask = batch.get('attention_mask', None)
        class_ids = batch.get('cond_diff', None)
        
        assert audio.device == next(self.parameters()).device, "Device mismatch!"
        #print("Audio dtype:", audio.dtype)  # Should be torch.float32
        #print("Model dtype:", next(self.parameters()).dtype)  # Should be torch.float32
        
        #print(dict(self.named_parameters()).keys())
       
        #print('\n printing next params')
        #print(next(self.audio_encoder.parameters()).device)


        input_tokens = x[:, :-1].contiguous()
        target_tokens = x[:, 1:].contiguous()
        mask = mask[:, :-1].contiguous()
        #x_t = x_t[:,1:].contiguous()
        #x_dt = x_dt[:,1:].contiguous()
       

        assert not torch.isnan(audio).any(), "NaN in audio"
        assert not torch.isinf(audio).any(), "Inf in audio"
        assert not torch.isnan(x).any(), "NaN in"


        print('audio shape: ', audio.shape)
        print('notes values shape: ', x.shape)
        print('conditional diff: ', class_ids.shape)

        # Forward pass
        print('about to encode')
        self.audio_encoder = self.audio_encoder
        audio_encoded = self.audio_encoder(audio.contiguous())
        #print('audio encoded: ', audio_encoded)
        #print('encoded with shape: ', audio_encoded.shape)
        #print('decoder attention mask shape before transformer forward: ', mask.shape)
        #print('batch 0 decodet attn mask: ', mask[0])
        #print('batch 1 decoder attn mask: ', mask[1])
        logits = self.transformer(input_tokens, audio_encoded, attention_mask=mask, class_ids=class_ids)
        
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_tokens.reshape(-1)
        
        # Compute loss
        print('About to compute loss.')
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
                class_ids_expanded = class_ids.expand(-1, seq_len).reshape(-1)
                
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
        
        audio = batch.get('audio', None)
        #audio_mask = batch.get('audio_mask', None)
        x = batch.get('note_values', None)
        #x_t = batch.get('note_times', None)
        #x_dt = batch.get('note_durations', None)
        mask = batch.get('attention_mask', None)
        class_ids = batch.get('cond_diff', None)
        
        input_tokens = x[:, :-1].contiguous()
        target_tokens = x[:, 1:].contiguous()
        #x_t = x_t[:,1:].contiguous()
        #x_dt = x_dt[:,1:].contiguous()
        
        # Forward pass
        audio_encoded = self.audio_encoder(audio).unsqueeze(1)
        logits = self.transformer(input_tokens, audio_encoded, attention_mask=mask, class_ids=class_ids)
       
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

    def configure_optimizers_old(self):
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


    def configure_optimizers(self):
        # Create single optimizer with parameter groups
        optimizer = torch.optim.AdamW([
            {
                'params': self.transformer.parameters(),
                'lr': self.cfg_optimizer.lr,
                'weight_decay': self.cfg_optimizer.weight_decay
            },
            {
                'params': self.audio_encoder.parameters(), 
                'lr': self.cfg_optimizer.lr,  # Could use different LR: lr * 0.1
                'weight_decay': self.cfg_optimizer.weight_decay
            }
        ], betas=(0.9, 0.95))
    
        # Single scheduler for the combined optimizer
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_steps=self.cfg_optimizer.warmup_steps,
            max_steps=self.cfg_optimizer.max_steps
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


