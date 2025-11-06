import wandb
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from dataloader.audio_loader import create_chunked_audio_chart_dataloader as create_audio_chart_dataloader
from dataloader.utils_dataloader import find_audio_files, split_json_entries_by_audio_raw
from modules.trainer import WaveformTransformerDiscrete
from modules.utils_train import set_seed_everything, LogGradientNorm, validate_dataset

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import hydra

from chart.tokenizer import SimpleTokenizerGuitar
import json



@hydra.main(version_base=None, config_path="configs",config_name="audio")
def main(config: DictConfig):

    set_seed_everything(config.seed)

    # Wandb
    wandb_config = OmegaConf.to_container(
        config,
        resolve=True,
        throw_on_missing=True,
    )


    encoder_cfg = config.model.encoder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{config.model.name}_"
        f"ws{config.window_seconds}_"
        f"grid{config.grid_ms}_"
        f"seq{config.max_length}_"
        f"d{config.model.transformer.d_model}_"
        f"n{config.model.transformer.n_layers}_"
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

    wandb_logger = WandbLogger(log_model=False)

    # Data
    if config.data_split_folder:
        with open(f"{config.data_split_folder}/train.json", "r", encoding="utf-8") as f:
            train_files = json.load(f) 
        with open(f"{config.data_split_folder}/val.json", "r", encoding="utf-8") as f:
            val_files = json.load(f) 
    else:
        split_json_entries_by_audio_raw(
            input_json=f"{config.root_folder}/audio_dataset_with_raw.json",
            train_json=f"{config.root_folder}/train.json",
            val_json=f"{config.root_folder}/val.json",
            val_ratio=config.validation_split,
        )

        with open(f"{config.root_folder}/train.json", "r", encoding="utf-8") as f:
            train_files = json.load(f) 
        with open(f"{config.root_folder}/val.json", "r", encoding="utf-8") as f:
            val_files = json.load(f) 


    tokenizer = SimpleTokenizerGuitar()
    
    train_files = validate_dataset(train_files, list(config.diff_list), list(config.inst_list),config.grid_ms)
    val_files = validate_dataset(val_files, list(config.diff_list), list(config.inst_list), config.grid_ms)

    train_dataloader, vocab = create_audio_chart_dataloader(
        train_files,
        window_seconds=config.window_seconds,
        sample_rate=config.model.sample_rate,
        tokenizer=tokenizer,
        difficulties=list(config.diff_list),
        instruments=list(config.inst_list),
        batch_size=config.batch_size,
        num_pieces = 6,
        max_length=config.max_length,
        conditional=config.model.transformer.conditional,
        use_predecoded_raw=True,
        is_discrete=config.is_discrete,
        augment=config.augment,
        grid_ms=config.grid_ms,
        use_processor=config.model.use_processor
    )

    val_dataloader, _ = create_audio_chart_dataloader(
        val_files,
        window_seconds=config.window_seconds,
        sample_rate=config.model.sample_rate,
        tokenizer=tokenizer,
        difficulties=list(config.diff_list),
        instruments=list(config.inst_list),
        batch_size=64,
        num_pieces=1,
        max_length=config.max_length,
        conditional=config.model.transformer.conditional,
        use_predecoded_raw=True,
        is_discrete=config.is_discrete,
        augment=False,
        grid_ms=config.grid_ms,
        use_processor=config.model.use_processor
    )
    
    print('Length train dataloader: ', len(train_dataloader))
    print('Length val dataloader: ', len(val_dataloader))

    # Model
    model = WaveformTransformerDiscrete(
        pad_token_id=vocab['<PAD>'],
        eos_token_id=vocab['<eos>'],
        vocab_size=len(vocab),
        cfg_model=config.model,
        cfg_optimizer=config.optimizer
    )
    
    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(monitor="val/acc_nonpad_epoch", min_delta=0.0001, patience=5, verbose=False, mode="max")
    track_grad_norm = LogGradientNorm()
    
    callback_list = [lr_monitor, early_stop_callback, track_grad_norm]

    if config.save_run:
        checkpoint_cb = L.pytorch.callbacks.ModelCheckpoint(
            monitor="val/acc_nonpad_epoch",
            save_top_k=1,
            mode="max",
            filename="best-checkpoint"
        )
        callback_list.append(checkpoint_cb)    

    # Trainer
    trainer = L.Trainer(
        max_epochs = config.max_epochs,
        accelerator = "gpu" if config.gpus > 0 else "cpu",
        devices = config.gpus if config.gpus > 0 else 1,
        enable_checkpointing = False,
        callbacks = callback_list,
        log_every_n_steps = 10,
        logger = wandb_logger,
        precision = config.precision,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
    )

    trainer.fit(
        model,
        train_dataloaders = train_dataloader,
        val_dataloaders = val_dataloader
    )

    wandb.finish() 

if __name__ == "__main__":
    main()
