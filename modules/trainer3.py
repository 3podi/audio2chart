import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score
from modules.scheduler import LinearWarmupCosineAnnealingLR
from modules.models import TransformerDecoderOnly
from hydra.utils import instantiate
import inspect

import torch.nn as nn
from modules.transformer_layers import (
    PositionalEncoding,
    DecoderBlock,
)
import math

class TransformerEncoderCTC(nn.Module):
    def __init__(self, input_vocab, vocab_size, pad_token_id, d_model=512, n_heads=8, n_layers=6, 
                 max_seq_len=5000, dropout=0.1, use_flash=False, is_causal=False):
        super().__init__()
        
        self.d_model = d_model
        self.input_vocab = input_vocab
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        
        # Token embedding and positional encoding
        self.token_embedding = nn.ModuleList(nn.Embedding(input_vocab, d_model) for _ in range(4))
        self.token_norm = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
 
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, 4 * d_model, dropout, use_flash, is_causal) #is_causal=False -> EncoderBlock
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
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
            input_ids: (batch_size, seq_len) - Token indices or Token embeddings
            attention_mask: (batch_size, seq_len) - Mask where 1 means attend, 0 means pad token
        
        Returns:
            x: (batch_size, seq_len, vocab_size) - Output logits or Output embedding
        """

        x = self.token_norm(self.token_embedding(input_ids))
        x = self.positional_encoding(x)
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        # Output projection to vocabulary
        x = self.output_projection(x) 
        return x


class WaveformTransformerDiscrete(L.LightningModule):
    def __init__(self, vocab_size, pad_token_id, eos_token_id, cfg_model, cfg_optimizer=None):
        super().__init__()

        self.vocab_size = vocab_size - 2 # remove eos bos
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        
        self.audio_encoder = instantiate(
            cfg_model.encoder,
            vocab_size=None,
            pad_token_id=pad_token_id, # is passed to the transformer for the seanet
        )

        # Instantiate submodels from config
        self.transformer = instantiate(
            cfg_model.transformer,
            vocab_size=self.vocab_size,
            pad_token_id=-1, #self.pad_token_id, use only if there are tokens to not attend to but at the moment i have discrete full seqs
            input_vocab = self.audio_encoder.model.config.codebook_size
        )
        #self.transformer = torch.compile(self.transformer)

        self.freeze_encoder=cfg_model.freeze_encoder
        if self.freeze_encoder:
            self.audio_encoder.eval()
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        
        # Optimizer
        self.cfg_optimizer = cfg_optimizer

        # Metrics
        #self.train_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size)
        #self.val_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size)

        #self.class_weights = torch.ones(self.vocab_size)
        #self.class_weights[self.pad_token_id] = 0.1

        self.save_hyperparameters()
    
    def _step(self, batch, batch_idx, split):

        audio = batch.get('input_values', None)
        padding_mask = batch.get('padding_mask', None)
        x = batch.get('note_values', None)
        class_ids = batch.get('cond_diff', None)

        target_tokens = x[:, 1:-1]
        target_lengths = (target_tokens != self.pad_token_id).sum(dim=1)
        targets_flat = target_tokens[target_tokens != self.pad_token_id]  
        #target_tokens = x[:, 1:]

        assert not torch.isnan(audio).any(), "NaN in audio"
        assert not torch.isinf(audio).any(), "Inf in audio"
        assert not torch.isnan(x).any(), "NaN in audio"

        # Forward pass
        if self.freeze_encoder:
            with torch.no_grad():
                audio_codes, audio_scales, last_frame_pad_length = self.audio_encoder(audio, padding_mask, bandwidth=3.0, return_embeddings=False)
        else:
            audio_codes, audio_scales, last_frame_pad_length, audio_encoded = self.audio_encoder(audio, padding_mask, bandwidth=3.0, return_embeddings=False)
            #audio_encoded = self.audio_encoder(audio)
        

        audio_codes = audio_codes.squeeze()

        B = audio_codes.size(0)
        logits = self.transformer(audio_codes, attention_mask=None)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T_audio, B, vocab)
        input_lengths = torch.full((B,), 2250, dtype=torch.long)


        # Compute loss
        loss = F.ctc_loss(
            log_probs,
            targets_flat,
            input_lengths,
            target_lengths,
            blank=self.pad_token_id,
            zero_infinity=True
        )


        if batch_idx % 50 == 0:
            list_of_target_sequences = [
                t.tolist() for t in torch.split(targets_flat, target_lengths.tolist())
                ]
            preds = ctc_greedy_decode(log_probs, self.pad_token_id)
            cer = token_error_rate(preds, list_of_target_sequences)
            self.log(f"{split}/CER", cer, prog_bar=True)

        return loss
    

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, split='train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, split='val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, split='test')

    def configure_optimizers(self):
        weight_decay = self.cfg_optimizer.weight_decay
        lr_t = self.cfg_optimizer.lr          # LR for transformer
        lr_e = self.cfg_optimizer.lr_audio    # LR for encoder
        betas = (0.9, 0.95)
        device_type = "cuda" 

        # ---- Build base param dict ----
        param_dict = {n: p for n,p in self.named_parameters() if p.requires_grad}

        # Separate params by 2D vs <2D
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2 and ("audio_encoder" not in n)]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2 and ("audio_encoder" not in n)]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': lr_t},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': lr_t},
        ]

        # ---- Add encoder with its own LR ----
        if not self.freeze_encoder:
            enc_params = list(self.audio_encoder.parameters())
            optim_groups.append({
                'params': enc_params,
                'weight_decay': weight_decay,
                'lr': lr_e,
            })

        # ---- Create optimizer ----
        #fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        #use_fused = fused_available and device_type == 'cuda'
        #extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(optim_groups, betas=betas)#, **extra_args)

        # ---- Scheduler ----
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_steps=self.cfg_optimizer.warmup_steps,
            max_steps=self.cfg_optimizer.max_steps,
            eta_min=1e-4,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


    def configure_optimizers_old(self):
        # Create single optimizer with parameter groups
        param_groups = [
            {
                'params': self.transformer.parameters(),
                'lr': self.cfg_optimizer['lr'],
                'weight_decay': self.cfg_optimizer['weight_decay']
            }
        ]
        if not self.freeze_encoder:
            param_groups.append({
                'params': self.audio_encoder.parameters(),
                'lr': self.cfg_optimizer['lr_audio'],
                'weight_decay': self.cfg_optimizer['weight_decay']
            })
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

        # Single scheduler for the combined optimizer
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_steps=self.cfg_optimizer.warmup_steps,
            max_steps=self.cfg_optimizer.max_steps,
            eta_min= 0.0001
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
            },
        }


    def on_after_backward(self):
        if self.freeze_encoder:
            return

        # run only every 100 steps
        if self.global_step % 100 != 0:
            return

        total = 0
        nonzero = 0
        grad_norm_sum = 0.0
        max_abs = 0.0

        for p in self.audio_encoder.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            total += 1
            if g.abs().sum() != 0:
                nonzero += 1
                grad_norm_sum += g.norm().item()
                max_abs = max(max_abs, g.abs().max().item())

        avg_grad_norm = grad_norm_sum / max(nonzero, 1)

        self.log("grad/audioencoder_nonzero_frac", nonzero / max(total,1),
                on_step=True, prog_bar=True)
        self.log("grad/audioencoder_maxabs", max_abs,
                on_step=True, prog_bar=True)
        self.log("grad/audioencoder_avg_norm", avg_grad_norm,
                on_step=True, prog_bar=True)





import numpy as np

def ctc_greedy_decode(log_probs, blank_id=0):
    # log_probs: (T, B, V)
    best_path = log_probs.argmax(dim=-1)  # (T, B)
    sequences = []
    for b in range(best_path.shape[1]):
        seq = []
        prev = blank_id
        for t in range(best_path.shape[0]):
            tok = best_path[t, b].item()
            if tok != blank_id and tok != prev:
                seq.append(tok)
            prev = tok
        sequences.append(seq)
    return sequences


def levenshtein_distance(ref, hyp):
    m, n = len(ref), len(hyp)
    dp = np.zeros((m+1, n+1), dtype=int)
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,       # deletion
                dp[i][j-1] + 1,       # insertion
                dp[i-1][j-1] + cost   # substitution
            )
    return dp[m][n]

def token_error_rate(pred_sequences, target_sequences):
    total_err, total_len = 0, 0
    for pred, target in zip(pred_sequences, target_sequences):
        total_err += levenshtein_distance(target, pred)
        total_len += len(target)
    return total_err / max(total_len, 1)