import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy
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


###############################
#                             #
#         Trainer             #
#       Audio Conditioned     #
#                             #
#                             #
###############################


class WaveformTransformer(L.LightningModule):
    def __init__(self, vocab_size, pad_token_id, eos_token_id, cfg_model, cfg_optimizer=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

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
        assert not torch.isnan(x).any(), "NaN in"

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
            pad_token_id=pad_token_id, # is passed to the transformer
        )

        # Optimizer
        self.cfg_optimizer = cfg_optimizer

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size)#, ignore_index=self.vocab_size-1)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.vocab_size)#, ignore_index=self.vocab_size-1)

        self.class_weights = torch.ones(self.vocab_size)
        self.class_weights[self.pad_token_id] = 0.1

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):

        audio = batch.get('audio', None)
        x = batch.get('note_values', None)
        class_ids = batch.get('cond_diff', None)

        #print('Tokens shape: ', x.shape)
        input_tokens = x[:, :-1].contiguous()
        target_tokens = x[:, 1:].contiguous()

        assert not torch.isnan(audio).any(), "NaN in audio"
        assert not torch.isinf(audio).any(), "Inf in audio"
        assert not torch.isnan(x).any(), "NaN in"

        # Forward pass
        #print('Audio shape: ', audio.squeeze().shape)
        audio_encoded = self.audio_encoder(audio.contiguous())
        print('Input tokens shape: ', input_tokens.shape)
        print('Encoded shape: ', audio_encoded.shape)
        logits = self.transformer(input_tokens, audio_encoded, class_ids=class_ids)

        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_tokens.reshape(-1)

        # Compute loss
        weights = self.class_weights.to(logits.device)
        loss = F.cross_entropy(logits_flat, targets_flat, weight=weights)#, ignore_index=self.vocab_size-1)

        preds = torch.argmax(logits_flat, dim=-1)
        acc = self.train_accuracy(preds, targets_flat)
        perplexity = torch.exp(loss)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/perplexity", perplexity, on_step=True, on_epoch=True)


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
            self.log("train/loss_nonpad", nonpad_loss, on_step=True, on_epoch=True)
            self.log("train/acc_nonpad", nonpad_acc, on_step=True, on_epoch=True)
            self.log("train/perplexity_nonpad", nonpad_perplexity, on_step=True, on_epoch=True)



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
        x = batch.get('note_values', None)
        class_ids = batch.get('cond_diff', None)

        input_tokens = x[:, :-1].contiguous()
        target_tokens = x[:, 1:].contiguous()

        # Forward pass
        audio_encoded = self.audio_encoder(audio.contiguous())
        logits = self.transformer(input_tokens, audio_encoded, class_ids=class_ids)

        logits_flat = logits.reshape(-1, self.vocab_size)
        targets_flat = target_tokens.reshape(-1)

        # Compute loss
        weights = self.class_weights.to(logits.device)
        loss = F.cross_entropy(logits_flat, targets_flat, weight=weights)#, ignore_index=self.vocab_size-1)

        preds = torch.argmax(logits_flat, dim=-1)
        acc = self.train_accuracy(preds, targets_flat)
        perplexity = torch.exp(loss)

        # Log metrics
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/perplexity", perplexity, on_step=True, on_epoch=True)


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
            self.log("val/loss_nonpad", nonpad_loss, on_step=True, on_epoch=True)
            self.log("val/acc_nonpad", nonpad_acc, on_step=True, on_epoch=True)
            self.log("val/perplexity_nonpad", nonpad_perplexity, on_step=True, on_epoch=True)

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



