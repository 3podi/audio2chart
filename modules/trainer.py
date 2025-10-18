import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score
from modules.scheduler import LinearWarmupCosineAnnealingLR
from modules.models import TransformerDecoderOnly
from hydra.utils import instantiate



###############################
#                             #
#         Trainer             #
#       Notes only            #
#                             #
#                             #
###############################


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


class WaveformTransformer(L.LightningModule):
    def __init__(self, vocab_size, pad_token_id, eos_token_id, cfg_model, cfg_optimizer=None, ablation_run=False):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.ablation_run = ablation_run

        # Instantiate submodels from config
        self.transformer = instantiate(
            cfg_model.transformer,
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        self.audio_encoder = instantiate(
            cfg_model.encoder,
            vocab_size=None,
            pad_token_id=pad_token_id,
        )

        # Optimizer
        self.cfg_optimizer = cfg_optimizer
        
        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size-1, ignore_index=self.vocab_size-1)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size-1, ignore_index=self.vocab_size-1)
        
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        
        audio = batch.get('audio', None)
        #audio_mask = batch.get('audio_mask', None)
        x = batch.get('note_values', None)
        #x_t = batch.get('note_times', None)
        #x_dt = batch.get('note_durations', None)
        mask = batch.get('attention_mask', None)
        class_ids = batch.get('cond_diff', None)
        
        input_tokens = x[:, :-1].contiguous()
        target_tokens = x[:, 1:].contiguous()
        mask = mask[:, :-1].contiguous()
        #x_t = x_t[:,1:].contiguous()
        #x_dt = x_dt[:,1:].contiguous()
       

        assert not torch.isnan(audio).any(), "NaN in audio"
        assert not torch.isinf(audio).any(), "Inf in audio"
        assert not torch.isnan(x).any(), "NaN in audio"

        # Forward pass
        audio_encoded = self.audio_encoder(audio.contiguous())
        logits = self.transformer(input_tokens, audio_encoded, attention_mask=mask, class_ids=class_ids)
        
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


        # Conditioning check: Run ablations
        if self.ablation_run and batch_idx % 100 == 0:
            with torch.no_grad():
                # Original loss
                original_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.vocab_size-1)
                
                # Ablation 1: Zero-out audio encoding
                zero_audio = torch.zeros_like(audio_encoded)
                zero_logits = self.transformer(input_tokens, zero_audio, attention_mask=mask, class_ids=class_ids)
                zero_logits_flat = zero_logits.reshape(-1, self.vocab_size)
                zero_loss = F.cross_entropy(zero_logits_flat, targets_flat, ignore_index=self.vocab_size-1)
                zero_delta = (zero_loss - original_loss).abs() / (original_loss + 1e-8)
                self.log("train/cond_zero_delta_loss", zero_delta, on_step=True)
                
                # Ablation 2: Noisy audio encoding
                if audio_encoded.numel() > 0:
                    noise_scale = audio_encoded.std().clamp(min=1e-6)
                    noise = torch.randn_like(audio_encoded) * noise_scale
                    noisy_audio = audio_encoded + noise
                    noisy_logits = self.transformer(input_tokens, noisy_audio, attention_mask=mask, class_ids=class_ids)
                    noisy_logits_flat = noisy_logits.reshape(-1, self.vocab_size)
                    noisy_loss = F.cross_entropy(noisy_logits_flat, targets_flat, ignore_index=self.vocab_size-1)
                    noisy_delta = (noisy_loss - original_loss).abs() / (original_loss + 1e-8)
                    self.log("train/cond_noisy_delta_loss", noisy_delta, on_step=True)
                
                # Ablation 3: Random shuffle of audio features across batch 
                if audio_encoded.size(0) > 1:
                    perm = torch.randperm(audio_encoded.size(0))
                    shuffled_audio = audio_encoded[perm]
                    shuffled_logits = self.transformer(input_tokens, shuffled_audio, attention_mask=mask, class_ids=class_ids)
                    shuffled_logits_flat = shuffled_logits.reshape(-1, self.vocab_size)
                    shuffled_loss = F.cross_entropy(shuffled_logits_flat, targets_flat, ignore_index=self.vocab_size-1)
                    shuffled_delta = (shuffled_loss - original_loss).abs() / (original_loss + 1e-8)
                    self.log("train/cond_shuffled_delta_loss", shuffled_delta, on_step=True)
                
                # If deltas are low (e.g., <0.1), model may be ignoring conditioning
        
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
        mask = mask[:, :-1].contiguous()
        #x_t = x_t[:,1:].contiguous()
        #x_dt = x_dt[:,1:].contiguous()
        
        # Forward pass
        audio_encoded = self.audio_encoder(audio.contiguous())
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
                'lr': self.cfg_optimizer.lr_audio,
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



class WaveformTransformerDiscrete(L.LightningModule):
    def __init__(self, vocab_size, pad_token_id, eos_token_id, cfg_model, cfg_optimizer=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        #cfg_model.transformer['vocab_size'] = self.vocab_size

        # Instantiate submodels from config
        self.transformer = instantiate(
            cfg_model.transformer,
            vocab_size=self.vocab_size,
            pad_token_id=-1, #self.pad_token_id, use only if there are tokens to not attend to but at the moment i have discrete full seqs
            eos_token_id=self.eos_token_id, # useless, TODO:delete
        )
        
        self.audio_encoder = instantiate(
            cfg_model.encoder,
            vocab_size=None,
            pad_token_id=pad_token_id, # is passed to the transformer for the seanet
        )

        self.freeze_encoder=cfg_model.freeze_encoder
        if self.freeze_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        
        # Optimizer
        self.cfg_optimizer = cfg_optimizer

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size)

        self.class_weights = torch.ones(self.vocab_size)
        self.class_weights[self.pad_token_id] = 0.1

        self.save_hyperparameters()
    
    def _step(self, batch, batch_idx, split):

        audio = batch.get('input_values', None)
        padding_mask = batch.get('padding_mask', None)
        x = batch.get('note_values', None)
        class_ids = batch.get('cond_diff', None)

        input_tokens = x[:, :-1].contiguous()
        target_tokens = x[:, 1:].contiguous()

        assert not torch.isnan(audio).any(), "NaN in audio"
        assert not torch.isinf(audio).any(), "Inf in audio"
        assert not torch.isnan(x).any(), "NaN in audio"

        # Forward pass
        if self.freeze_encoder:
            with torch.no_grad():
                audio_codes, audio_scales, last_frame_pad_length, audio_encoded = self.audio_encoder(audio, padding_mask, bandwidth=3.0, return_embeddings=True)
        else:
            audio_encoded = self.audio_encoder(audio)
                
        logits = self.transformer(input_tokens, audio_encoded, class_ids=class_ids)
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
                            self.log(f"{split}/loss_{class_name}", class_loss, on_step=True, on_epoch=True)
                            self.log(f"{split}/acc_{class_name}", class_acc, on_step=True, on_epoch=True)
                            self.log(f"{split}/perplexity_{class_name}", class_perplexity, on_step=True, on_epoch=True)


        return loss
    

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, split='train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, split='val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, split='test')

    def configure_optimizers(self):
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