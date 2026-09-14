import math
import os
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy
from modules.scheduler import LinearWarmupCosineAnnealingLR
from modules.models import TransformerDecoderOnly
from hydra.utils import instantiate
from omegaconf import OmegaConf


###############################
#                             #
#         Trainer             #
#       Notes only            #
#                             #
#                             #
###############################


@contextmanager
def _stage_timer(name):
    """Split the forward pass into encoder versus decoder. Off unless A2C_TIME_STEP is set.

    Lightning's `trainer.profiler=simple` is the right tool for finding which *phase* of
    a step owns the time -- it is what showed the dataloader at 0.1% -- but it cannot see
    inside training_step. This can: it reports the frozen Encodec encoder at 0.065 s
    against 0.089 s for the whole 245 M-parameter decoder, at a 15 s window.

    The synchronize calls are why this is opt-in. They serialise the pipeline, so leaving
    it on would slow down the thing it is measuring.
    """
    if not os.environ.get("A2C_TIME_STEP"):
        yield
        return
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        print(f"A2C_TIME {name}={time.perf_counter() - started:.4f}", flush=True)


class NotesTransformer(L.LightningModule):
    def __init__(self, pad_token_id, eos_token_id, vocab_size, cfg_model, cfg_optimizer=None, is_discrete=False):
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
        self.is_discrete = is_discrete
        self.pad_token_id = pad_token_id
        
        # Metrics
        if is_discrete:
            self.class_weights = torch.ones(self.vocab_size)
            self.class_weights[pad_token_id] = 0.1
            self.train_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size)
            self.val_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size)
        else:
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
        if self.is_discrete:
            weights = self.class_weights.to(logits.device)
            loss = F.cross_entropy(logits_flat, targets_flat, weight=weights)
        else:
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.vocab_size-1)
        
        preds = torch.argmax(logits_flat, dim=-1)
        acc = self.train_accuracy(preds, targets_flat)
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/perplexity", perplexity, on_step=True, on_epoch=True)

        # Non-pad (non-silence) mask
        if self.is_discrete:
            nonpad_mask = targets_flat != self.pad_token_id

            if nonpad_mask.any():
                nonpad_logits = logits_flat[nonpad_mask]
                nonpad_targets = targets_flat[nonpad_mask]
                nonpad_preds = preds[nonpad_mask]

                nonpad_loss = F.cross_entropy(nonpad_logits, nonpad_targets)
                nonpad_acc = (nonpad_preds == nonpad_targets).float().mean()
                nonpad_perplexity = torch.exp(nonpad_loss)

                # Log only for monitoring (not backprop)
                self.log("train/loss_nonpad", nonpad_loss, on_step=True, on_epoch=True)
                self.log("train/acc_nonpad", nonpad_acc, on_step=True, on_epoch=True)
                self.log("train/perplexity_nonpad", nonpad_perplexity, on_step=True, on_epoch=True)
            
            # === PAD-related metrics ===
            pad_preds = preds == self.pad_token_id
            pad_targets = targets_flat == self.pad_token_id

            total_tokens = targets_flat.numel()
            total_nonpad_targets = (~pad_targets).sum()
            total_pad_targets = pad_targets.sum()
            total_pad_preds = pad_preds.sum()

            # False positive rate for pad:
            pad_fp = (pad_preds & (~pad_targets)).sum()
            pad_fp_rate = pad_fp.float() / (total_nonpad_targets.float() + 1e-8)

            # Pad prediction rate:
            pad_pred_rate = total_pad_preds.float() / (total_tokens + 1e-8)

            # True positive rate for pad (optional context):
            pad_tp = (pad_preds & pad_targets).sum()
            pad_tp_rate = pad_tp.float() / (total_pad_targets.float() + 1e-8)

            # Non-pad recall (1 - FP rate):
            nonpad_recall = 1.0 - pad_fp_rate

            # Log everything
            self.log("train/pad_fp_rate", pad_fp_rate, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train/pad_pred_rate", pad_pred_rate, on_step=True, on_epoch=True)
            self.log("train/pad_tp_rate", pad_tp_rate, on_step=True, on_epoch=True)
            self.log("train/nonpad_recall", nonpad_recall, on_step=True, on_epoch=True)

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
        if self.is_discrete:
            weights = self.class_weights.to(logits.device)
            loss = F.cross_entropy(logits_flat, targets_flat, weight=weights)
        else:
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.vocab_size-1)
                
        preds = torch.argmax(logits_flat, dim=-1)
        acc = self.train_accuracy(preds, targets_flat)
        perplexity = torch.exp(loss)
        
        # Log metrics
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/perplexity", perplexity, on_step=True, on_epoch=True)

        # Non-pad (non-silence) mask
        if self.is_discrete:
            nonpad_mask = targets_flat != self.pad_token_id

            if nonpad_mask.any():
                nonpad_logits = logits_flat[nonpad_mask]
                nonpad_targets = targets_flat[nonpad_mask]
                nonpad_preds = preds[nonpad_mask]

                nonpad_loss = F.cross_entropy(nonpad_logits, nonpad_targets)
                nonpad_acc = (nonpad_preds == nonpad_targets).float().mean()
                nonpad_perplexity = torch.exp(nonpad_loss)

                # Log only for monitoring (not backprop)
                self.log("val/loss_nonpad", nonpad_loss, on_step=True, on_epoch=True)
                self.log("val/acc_nonpad", nonpad_acc, on_step=True, on_epoch=True)
                self.log("val/perplexity_nonpad", nonpad_perplexity, on_step=True, on_epoch=True)

            # === PAD-related metrics ===
            pad_preds = preds == self.pad_token_id
            pad_targets = targets_flat == self.pad_token_id

            total_tokens = targets_flat.numel()
            total_nonpad_targets = (~pad_targets).sum()
            total_pad_targets = pad_targets.sum()
            total_pad_preds = pad_preds.sum()

            # False positive rate for pad:
            pad_fp = (pad_preds & (~pad_targets)).sum()
            pad_fp_rate = pad_fp.float() / (total_nonpad_targets.float() + 1e-8)

            # Pad prediction rate:
            pad_pred_rate = total_pad_preds.float() / (total_tokens + 1e-8)

            # True positive rate for pad (optional context):
            pad_tp = (pad_preds & pad_targets).sum()
            pad_tp_rate = pad_tp.float() / (total_pad_targets.float() + 1e-8)

            # Non-pad recall (1 - FP rate):
            nonpad_recall = 1.0 - pad_fp_rate

            # Log everything
            self.log("val/pad_fp_rate", pad_fp_rate, on_step=True, on_epoch=True, prog_bar=True)
            self.log("val/pad_pred_rate", pad_pred_rate, on_step=True, on_epoch=True)
            self.log("val/pad_tp_rate", pad_tp_rate, on_step=True, on_epoch=True)
            self.log("val/nonpad_recall", nonpad_recall, on_step=True, on_epoch=True)

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


###############################
#                             #
#         Trainer             #
#       Audio Conditioned     #
#                             #
#                             #
###############################


class WaveformTransformerDiscrete(L.LightningModule):
    def __init__(self, vocab_size, pad_token_id, eos_token_id, cfg_model, cfg_optimizer=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        # Fraction of training input tokens replaced at random, to blunt exposure bias.
        # Off by default: it changes what the model is trained on, so it is a decision
        # for a run rather than something to inherit silently.
        self.input_noise = float(OmegaConf.select(cfg_model, "input_noise", default=0.0))
        # Mean length of a corrupted run. 1.0 is the original behaviour, where every
        # token is corrupted independently; above 1.0 the same total corruption is
        # delivered in contiguous bursts instead. See _corrupt_inputs for why that
        # matters more than the rate does.
        self.input_noise_burst = float(
            OmegaConf.select(cfg_model, "input_noise_burst", default=1.0))
        # Note tokens occupy [0, n_note_tokens); bos/eos/pad sit above them and must
        # never be drawn as noise -- an injected eos would teach the model to stop.
        self.n_note_tokens = min(pad_token_id, eos_token_id)

        self.audio_encoder = instantiate(
            cfg_model.encoder,
            vocab_size=None,
            pad_token_id=pad_token_id, # is passed to the transformer for the seanet
        )

        self.audio_output_kind = getattr(self.audio_encoder, "output_kind", "codes")
        codebook_size = None
        audio_feature_dim = None
        num_audio_codebooks = 4
        if self.audio_output_kind == "codes":
            codebook_size = self.audio_encoder.model.config.codebook_size
            num_audio_codebooks = self.audio_encoder.num_quantizers
        elif self.audio_output_kind == "features":
            audio_feature_dim = self.audio_encoder.output_dim
        else:
            raise ValueError(f"Unsupported audio encoder output: {self.audio_output_kind}")

        # Instantiate submodels from config
        self.transformer = instantiate(
            cfg_model.transformer,
            vocab_size=self.vocab_size,
            pad_token_id=-1,
            eos_token_id=self.eos_token_id,
            codebook_size=codebook_size,
            audio_feature_dim=audio_feature_dim,
            num_audio_codebooks=num_audio_codebooks,
        )

        self.freeze_encoder=cfg_model.freeze_encoder
        if self.freeze_encoder:
            self.audio_encoder.eval()
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        
        # Optimizer
        self.cfg_optimizer = cfg_optimizer

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size)

        # Pad is ~86% of every target sequence, so predicting it everywhere already
        # scores 86% accuracy -- hence down-weighting it, to stop the model taking that
        # shortcut. But 0.1 overshoots badly into the opposite failure.
        #
        # At 0.1 the loss is 0.86 * 0.1 = 0.086 from pad positions against 0.14 * 1.0 =
        # 0.14 from real notes, so notes dominate the objective roughly 62/38 while
        # being 14% of the data. The model is then trained for *which note* and hardly
        # at all for *note or silence*, and it shows: measured on both runs of
        # 2026-09-11, two consecutive notes in the context collapse P(pad) from 0.90 to
        # 0.09 and it never recovers, so generation saturates and never rests. Human
        # tapping charts rest 15 times a minute; those models rest 0.35-0.85 times.
        #
        # Both runs showed it, including one that never saw the other's checkpoint, so
        # it is produced here rather than inherited.
        self.pad_class_weight = float(
            OmegaConf.select(cfg_model, "pad_class_weight", default=0.1))
        self.class_weights = torch.ones(self.vocab_size)
        self.class_weights[self.pad_token_id] = self.pad_class_weight

        self.save_hyperparameters()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            # Lightning recursively toggles child modules; pretrained Encodec
            # must remain deterministic while frozen.
            self.audio_encoder.eval()
        return self

    @staticmethod
    def _extract_batch(batch):
        required = ("input_values", "padding_mask", "note_values")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Batch is missing required keys: {missing}")
        notes = batch["note_values"]
        return (
            batch["input_values"], batch["padding_mask"],
            notes[:, :-1].contiguous(), notes[:, 1:].contiguous(),
            batch.get("cond_diff"),
        )

    def _corrupt_inputs(self, input_tokens):
        """Randomly replace training input tokens, so the model learns to recover.

        Teacher forcing only ever shows the model a correct history, and free running
        it then drifts: measured over one 30s chunk, the generated tap rate climbs
        0.224 -> 0.454 -> 0.671 -> 0.723 as it conditions on more of its own output,
        and at a lower sampling temperature it collapses instead to a flat zero taps
        and zero sustains. The model is not miscalibrated -- teacher-forced it puts
        0.181 of its mass on taps against a true 0.197 -- it has simply never seen a
        wrong history and has no idea how to continue from one.

        Scheduled sampling is the usual answer and costs a second forward pass per
        step. Corrupting a fraction of inputs approximates it for free.

        WHAT THE RATE ALONE DOES NOT FIX. Corrupting tokens independently teaches
        recovery from a typo, and the failure that survives it is a rut. A generated
        LESBIAN VAMPIRE KILLERS opened with 46 consecutive five-fret tap chords, one
        every 30 ms, which Robert reported as impossible to play. Probed directly with
        chord_response.py, the model recovers from a short run of them and cannot
        recover from a long one:

            history          P(same again)   5-note mass   (corpus: 0.223%)
            clean                   0.0000        0.000%
            2 GRYBO taps            0.0000        0.001%
            8 GRYBO taps            0.0164        9.770%
            16 GRYBO taps           0.1292       41.760%

        That boundary is exactly where the training signal stops. At input_noise 0.08,
        i.i.d., the chance of eight consecutive corrupted positions is 0.08**8, about
        1.7e-9; across the 72 million token positions of a 6000-step run the expected
        number of runs of sixteen is 2e-10. The model was never once shown a history
        it had been wrong in for long enough to matter, so "get out of a rut" is not
        something it could have learned, at any rate.

        Tracing a real generation confirms entry is not a sampling accident either:
        the first two-note chord arrived at rank 19 with probability 0.0065 -- a tail
        draw -- and twelve steps later a five-fret chord was the model's rank-1 token
        at probability 0.184. One unlikely token puts the history off-distribution and
        the predictions there are junk. No temperature or top_k setting fixes a rank-1
        prediction; this is the same shape as the pad collapse, and the same place to
        fix it.

        So corruption is delivered in bursts of a single repeated token, which is the
        pathology verbatim, while `input_noise` still sets the overall fraction of
        corrupted positions. `input_noise_burst` is the mean burst length; 1.0 restores
        the original independent behaviour exactly.

        Targets are untouched, so the model is still asked for the right answer -- the
        point is to make it right despite a damaged history, not to teach it the
        damage. Padding is left alone as well; corrupting it would teach the model to
        emit notes where a chart has ended.
        """
        if not self.input_noise or not self.training:
            return input_tokens

        keep = input_tokens != self.pad_token_id
        burst = max(1.0, self.input_noise_burst)

        if burst <= 1.0:
            corrupt = keep & (torch.rand_like(input_tokens, dtype=torch.float)
                              < self.input_noise)
            if not corrupt.any():
                return input_tokens
            noise = torch.randint_like(input_tokens, 0, self.n_note_tokens)
            return torch.where(corrupt, noise, input_tokens)

        # A burst starts rarely enough that the expected fraction of corrupted
        # positions is still `input_noise`: each start covers `burst` positions on
        # average, so starts happen at input_noise / burst.
        start_rate = self.input_noise / burst
        starts = keep & (torch.rand_like(input_tokens, dtype=torch.float) < start_rate)
        if not starts.any():
            return input_tokens

        # Geometric lengths, so most bursts are short and a few are long -- the long
        # ones are the whole point, and a fixed length would never produce the range
        # the probe walks. Mean length is `burst` for success probability 1 / burst.
        rows, width = input_tokens.shape
        device = input_tokens.device
        uniform = torch.rand(rows, width, device=device).clamp_min(torch.finfo(
            torch.float32).tiny)
        lengths = 1 + torch.floor(
            torch.log(uniform) / math.log(1.0 - 1.0 / burst)).long()

        # How far each burst reaches, then a running maximum: a position is corrupt if
        # some earlier start still covers it. cummax also reports *which* start that
        # was, which is how a burst repeats one token instead of drawing a new one
        # every step -- the 46-chord run is one token held, not 46 random mistakes.
        #
        # Done this way rather than by walking the sequence: the obvious loop is one
        # tensor op per position, 1500 of them per training step.
        index = torch.arange(width, device=device).expand(rows, width)
        reach = torch.where(starts, index + lengths, torch.full_like(index, -1))
        furthest, origin = torch.cummax(reach, dim=1)
        corrupt = (furthest > index) & keep
        if not corrupt.any():
            return input_tokens

        noise = torch.randint(0, self.n_note_tokens, (rows, width),
                              device=device, dtype=input_tokens.dtype)
        repeated = torch.gather(noise, 1, origin.clamp_min(0))
        return torch.where(corrupt, repeated, input_tokens)

    @staticmethod
    def _flatten_encodec_frames(audio_codes):
        """Convert [frames, batch, quantizers, time] to [batch, quantizers, time]."""
        if audio_codes.dim() != 4:
            raise ValueError(f"Expected four-dimensional Encodec codes, got {audio_codes.shape}")
        frames, batch, quantizers, frame_length = audio_codes.shape
        return audio_codes.permute(1, 2, 0, 3).reshape(
            batch, quantizers, frames * frame_length
        )

    def _encode_audio(self, audio, padding_mask):
        context = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        # Lightning wraps the whole step in bf16 autocast, which is a 3.4x win on the
        # transformer and a 5.5x loss for Encodec: cuDNN has no fused LSTM kernel for
        # bf16, so SEANet's recurrence over 2250 timesteps falls back to an unfused path.
        # Measured 0.130s per call in fp32 against 0.712s under autocast, for codes
        # that are quantiser indices either way -- there is no precision to gain.
        #
        # That reasoning is about an LSTM, so the encoder declares it rather than the
        # trainer assuming it. MERT is a plain transformer with no recurrence, and
        # forcing 95M parameters over 1124 frames into fp32 costs both speed and the
        # VRAM headroom this card does not have -- the Encodec path already peaked at
        # 7.7 GB of 8.
        wants_fp32 = getattr(self.audio_encoder, "prefers_fp32", True)
        with context, torch.autocast(self.device.type, enabled=not wants_fp32):
            if self.audio_output_kind == "codes":
                output = self.audio_encoder(
                    audio.float(), padding_mask, return_embeddings=False
                )
                return self._flatten_encodec_frames(output[0])
            return self.audio_encoder(audio.float())
    
    def _step(self, batch, batch_idx, split):

        audio, padding_mask, input_tokens, target_tokens, class_ids = self._extract_batch(batch)

        if split == "train":
            input_tokens = self._corrupt_inputs(input_tokens)

        if not torch.isfinite(audio).all():
            raise ValueError("Audio batch contains non-finite values")

        # Forward pass
        with _stage_timer("encode"):
            audio_codes = self._encode_audio(audio, padding_mask)
        with _stage_timer("transformer"):
            logits = self.transformer(input_tokens, audio_codes, class_ids=class_ids)
        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_tokens.reshape(-1)

        # Compute loss
        weights = self.class_weights.to(logits.device)
        loss = F.cross_entropy(logits_flat, targets_flat, weight=weights)

        preds = torch.argmax(logits_flat, dim=-1)
        if split == 'train':
            acc = self.train_accuracy(preds, targets_flat)
        else:
            acc = self.val_accuracy(preds, targets_flat)


        perplexity = torch.exp(loss)

        # Log metrics
        self.log(f"{split}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{split}/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{split}/perplexity", perplexity, on_step=True, on_epoch=True)

        # Non-pad (non-silence) mask
        nonpad_mask = targets_flat != self.pad_token_id

        if nonpad_mask.any():
            nonpad_logits = logits_flat[nonpad_mask]
            nonpad_targets = targets_flat[nonpad_mask]
            nonpad_preds = preds[nonpad_mask]

            nonpad_loss = F.cross_entropy(nonpad_logits, nonpad_targets)
            nonpad_acc = (nonpad_preds == nonpad_targets).float().mean()
            nonpad_perplexity = torch.exp(nonpad_loss)

            # Log only for monitoring (not backprop)
            self.log(f"{split}/loss_nonpad", nonpad_loss, on_step=True, on_epoch=True)
            self.log(f"{split}/acc_nonpad", nonpad_acc, on_step=True, on_epoch=True)
            self.log(f"{split}/perplexity_nonpad", nonpad_perplexity, on_step=True, on_epoch=True)

        # === PAD-related metrics ===
        pad_preds = preds == self.pad_token_id
        pad_targets = targets_flat == self.pad_token_id

        total_tokens = targets_flat.numel()
        total_nonpad_targets = (~pad_targets).sum()
        total_pad_targets = pad_targets.sum()
        total_pad_preds = pad_preds.sum()

        # False positive rate for pad:
        pad_fp = (pad_preds & (~pad_targets)).sum()
        pad_fp_rate = pad_fp.float() / (total_nonpad_targets.float() + 1e-8)

        # Pad prediction rate:
        pad_pred_rate = total_pad_preds.float() / (total_tokens + 1e-8)

        # True positive rate for pad (optional context):
        pad_tp = (pad_preds & pad_targets).sum()
        pad_tp_rate = pad_tp.float() / (total_pad_targets.float() + 1e-8)

        # Non-pad recall (1 - FP rate):
        nonpad_recall = 1.0 - pad_fp_rate

        # Log everything
        self.log(f"{split}/pad_fp_rate", pad_fp_rate, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{split}/pad_pred_rate", pad_pred_rate, on_step=True, on_epoch=True)
        self.log(f"{split}/pad_tp_rate", pad_tp_rate, on_step=True, on_epoch=True)
        self.log(f"{split}/nonpad_recall", nonpad_recall, on_step=True, on_epoch=True)


        # Compute per-class metrics
        if class_ids is not None and batch_idx % 100 == 0:
            with torch.no_grad():
                # Expand class_ids to match flattened logits dimensions
                # class_ids: [batch_size] -> [batch_size * seq_len]
                seq_len = target_tokens.shape[1]
                class_ids_expanded = class_ids.reshape(-1, 1).expand(-1, seq_len).reshape(-1)

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
                            self.log(f"{split}/loss_{class_name}", class_loss, on_step=True, on_epoch=True)
                            self.log(f"{split}/acc_{class_name}", class_acc, on_step=True, on_epoch=True)
                            self.log(f"{split}/perplexity_{class_name}", class_perplexity, on_step=True, on_epoch=True)


        if  batch_idx % 100 == 0:
            with torch.no_grad():
                original_loss = F.cross_entropy(logits_flat, targets_flat, weight=weights)
                
                if audio_codes.size(0) > 1:
                    perm = torch.randperm(audio_codes.size(0), device=audio_codes.device)
                    shuffled_audio = audio_codes[perm]
                    shuffled_logits = self.transformer(input_tokens, shuffled_audio, class_ids=class_ids)
                    shuffled_logits_flat = shuffled_logits.reshape(-1, self.vocab_size)
                    shuffled_loss = F.cross_entropy(shuffled_logits_flat, targets_flat, weight=weights)
                    shuffled_delta = (shuffled_loss - original_loss).abs() / (original_loss + 1e-8)
                    self.log(f"ablation/{split}_cond_shuffled_delta_loss", shuffled_delta, on_step=True)



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
            enc_params = [p for p in self.audio_encoder.parameters() if p.requires_grad]
            optim_groups.append({
                'params': enc_params,
                'weight_decay': weight_decay,
                'lr': lr_e,
            })

        # ---- Create optimizer ----
        optimizer = torch.optim.AdamW(optim_groups, betas=betas)

        # ---- Scheduler ----
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_steps=self.cfg_optimizer.warmup_steps,
            max_steps=self.cfg_optimizer.max_steps,
            eta_min=lr_t*0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


    def on_train_epoch_end(self):
        # Reset metrics at the end of each epoch
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self.val_accuracy.reset()
        for i, block in enumerate(self.transformer.layers):
            if hasattr(block, "cross_gamma"):
                self.log(f"gamma/layer_{i}", block.cross_gamma.item(), on_epoch=True)


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
