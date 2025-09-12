import wandb
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import random

from dataloader.audio_loader import create_audio_chart_dataloader
from dataloader.utils_dataloader import find_audio_files, split_json_entries_by_audio
from modules.audio_transformer import WaveformTransformer

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import hydra

#from transformers import AutoProcessor, EncodecModel
from chart.tokenizer import SimpleTokenizerGuitar

import json

class LogGradientNorm(L.pytorch.callbacks.Callback):
    """
    Logs the gradient norm (L2 norm) before the optimizer step (pre-clipping/scaling).
    """
    def on_before_optimizer_step(self, trainer: L.Trainer, *args, **kwargs) -> None:
        total_norm = 0.0
        for param in trainer.lightning_module.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        trainer.lightning_module.log("train/grad_norm", total_norm)

@hydra.main(version_base=None, config_path="configs",config_name="audio")
def main(config: DictConfig):

    # Wandb
    wandb_config = OmegaConf.to_container(
        config,
        resolve=True,
        throw_on_missing=True,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{config.model.name}_"
        #f"{config.root_folder}_"
        f"seq{config.max_length}_"
        f"d{config.model.d_model}_"
        f"n{config.model.n_layers}_"
        f"lr{config.optimizer.lr}_"
        f"bs{config.batch_size}_"
        f"{timestamp}"
    )

    wandb.init(
        project="audio2chart",
        config=wandb_config,
        name=run_name,
        tags=config.tags,
        reinit=True
    )

    wandb_logger = WandbLogger(log_model="all")

    # Data
    random.seed(42)
    if config.data_split_folder:
        with open(f"{config.data_split_folder}/train.json", "r", encoding="utf-8") as f:
            train_files = json.load(f) 
        with open(f"{config.data_split_folder}/val.json", "r", encoding="utf-8") as f:
            val_files = json.load(f) 
    else:
        find_audio_files(
            root=config.root_folder,
            difficulties=list(config.diff_list),
            instruments=list(config.inst_list),
            output_json=f"{config.root_folder}/audio_dataset.json",
            skipped_json=f"{config.root_folder}/audio_skipped.json",
        )

        split_json_entries_by_audio(
            input_json=f"{config.root_folder}/audio_dataset.json",
            train_json=f"{config.root_folder}/train.json",
            val_json=f"{config.root_folder}/val.json",
            val_ratio=config.validation_split,
        )

        with open(f"{config.root_folder}/train.json", "r", encoding="utf-8") as f:
            train_files = json.load(f) 
        with open(f"{config.root_folder}/val.json", "r", encoding="utf-8") as f:
            val_files = json.load(f) 

    #audio_processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")
    audio_processor = None
    tokenizer = SimpleTokenizerGuitar()

    train_dataloader, vocab = create_audio_chart_dataloader(
        train_files,
        audio_processor=audio_processor,
        window_seconds=config.window_seconds,
        tokenizer=tokenizer,
        difficulties=list(config.diff_list),
        instruments=list(config.inst_list),
        batch_size=config.batch_size,
        max_length=config.max_length,
        conditional=config.model.conditional,
    )

    val_dataloader, _ = create_audio_chart_dataloader(
        val_files,
        audio_processor=audio_processor,
        window_seconds=config.window_seconds,
        tokenizer=tokenizer,
        difficulties=list(config.diff_list),
        instruments=list(config.inst_list),
        batch_size=config.batch_size,
        max_length=config.max_length,
        conditional=config.model.conditional,
        shuffle=False
    )

    # Model
    model = WaveformTransformer(
        pad_token_id=vocab['<PAD>'],
        eos_token_id=['<eos>'],
        vocab_size=len(vocab),
        cfg_model=config.model,
        cfg_optimizer=config.optimizer
    )

    print(model.audio_encoder.compute_receptive_field())

    # Callbacks
    #checkpoint_cb = L.pytorch.callbacks.ModelCheckpoint(
    #    monitor="val_loss",
    #    save_top_k=1,
    #    mode="min",
    #    filename="best-checkpoint"
    #)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(monitor="val/acc_epoch", min_delta=0.001, patience=10, verbose=False, mode="max")
    track_grad_norm = LogGradientNorm()

    # Trainer
    trainer = L.Trainer(
        max_epochs = config.max_epochs,
        accelerator = "gpu" if config.gpus > 0 else "cpu",
        devices = config.gpus if config.gpus > 0 else 1,
        enable_checkpointing = False,
        callbacks = [lr_monitor, early_stop_callback, track_grad_norm],
        log_every_n_steps = 10,
        logger = wandb_logger,
        precision = config.precision,
        #default_root_dir=checkpoint_path,
        #check_val_every_n_epoch=cfg["val_frequency"],
        num_sanity_val_steps=0,
        #accumulate_grad_batches=cfg.accumulate_grad_batches,
        #gradient_clip_val=1.0,
    )

    trainer.fit(
        model,
        train_dataloaders = train_dataloader,
        val_dataloaders = val_dataloader
    )

    wandb.finish() 

if __name__ == "__main__":
    main()
