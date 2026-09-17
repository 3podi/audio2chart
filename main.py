"""Train the audio-conditioned discrete chart model."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from chart.tokenizer import SimpleTokenizerGuitar
from dataloader.audio_loader import create_chunked_audio_chart_dataloader
from dataloader.utils_dataloader import split_json_entries_by_audio_raw
import torch

from modules.pretrained import load_pretrained_transformer
from modules.run_utils import build_trainer, configure_logging, experiment_logger
from modules.trainer import WaveformTransformerDiscrete
from modules.utils_train import set_seed_everything, validate_dataset


MONITORED_METRIC = "val/acc_nonpad_epoch"


def build_run_name(config: DictConfig) -> str:
    transformer = config.model.transformer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"{config.model.name}_ws{config.data.window_seconds}_"
        f"grid{config.data.grid_ms}_seq{config.data.max_length}_"
        f"d{transformer.d_model}_n{transformer.n_layers}_"
        f"lr{config.optimizer.lr}_bs{config.loader.batch_size}_{timestamp}"
    )


def _read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, list):
        raise ValueError(f"Manifest must contain a JSON list: {path}")
    return value


def load_data_splits(config: DictConfig) -> tuple[list[dict], list[dict]]:
    folder = Path(config.data.split_folder or config.data.root_folder)
    if config.data.split_folder is None:
        split_json_entries_by_audio_raw(
            input_json=str(folder / "audio_dataset_with_raw.json"),
            train_json=str(folder / "train.json"),
            val_json=str(folder / "val.json"),
            val_ratio=config.data.validation_split,
            random_seed=config.seed,
        )
    return _read_manifest(folder / "train.json"), _read_manifest(folder / "val.json")


def build_dataloaders(
    config: DictConfig,
    tokenizer: SimpleTokenizerGuitar,
    train_files: list[dict],
    val_files: list[dict],
):
    difficulties = list(config.data.difficulties)
    instruments = list(config.data.instruments)
    validation = dict(
        difficulties=difficulties,
        instruments=instruments,
        grid_ms=config.data.grid_ms,
        error_policy=config.data.error_policy,
        # Judge a song by whether any window is usable, the way the loader does, rather
        # than by its single tightest transition anywhere in the chart.
        window_seconds=config.data.window_seconds,
        # A cold validation pass over the 13,720-entry corpus is ~9.4 minutes idle, and
        # much worse under disk contention. Verdicts are reused while the chart and these
        # settings are unchanged.
        cache_path=Path(config.data.root_folder) / "validation_cache.json",
    )
    train_files = validate_dataset(train_files, **validation)
    val_files = validate_dataset(val_files, **validation)

    common = dict(
        tokenizer=tokenizer,
        window_seconds=config.data.window_seconds,
        sample_rate=config.model.sample_rate,
        difficulties=difficulties,
        instruments=instruments,
        max_length=config.data.max_length,
        num_workers=config.loader.num_workers,
        conditional=config.model.transformer.conditional,
        use_predecoded_raw=True,
        is_discrete=True,
        grid_ms=config.data.grid_ms,
        use_processor=config.model.use_processor,
        processor_checkpoint=OmegaConf.select(
            config, "model.encoder.checkpoint", default="facebook/encodec_24khz"
        ),
        error_policy=config.data.error_policy,
        max_sample_retries=config.loader.max_sample_retries,
        # Per dataloader, split across its workers. persistent_workers keeps the training
        # workers alive through validation, so the host holds two of these at once.
        max_cache_gb=config.loader.max_cache_gb,
        # Bounded and lazy. Caching every chart up front cost ~18 GB across the main
        # process and its spawned workers, because Windows pickles the dataset into
        # each one. Each song is visited about once per epoch, so it bought little.
        chart_cache_size=config.loader.chart_cache_size,
    )
    train_loader, vocab = create_chunked_audio_chart_dataloader(
        train_files,
        batch_size=config.loader.batch_size,
        num_pieces=config.loader.train_num_pieces,
        augment=config.loader.augment,
        **common,
    )
    val_loader, _ = create_chunked_audio_chart_dataloader(
        val_files,
        batch_size=config.loader.val_batch_size,
        num_pieces=config.loader.val_num_pieces,
        augment=False,
        shuffle_chunks=False,
        **common,
    )
    return train_loader, val_loader, vocab


def freeze_lower_layers(model, count: int) -> None:
    """Freeze the bottom `count` decoder blocks, lowest first.

    VRAM here is dominated by trainable parameters, not activations: each one costs 4
    bytes of gradient plus 8 of AdamW moments on top of its own 4, so 230 M trainable
    parameters is ~2.8 GB before a single activation. Trading batch shape away moved peak
    usage by 14 MiB; freezing does the arithmetic that matters. The lower blocks carry the
    general audio-to-chart mapping the released checkpoint already learned, so they are
    the ones worth holding fixed while the upper blocks adapt to the new vocabulary.

    `configure_optimizers` already filters on requires_grad, so frozen parameters get no
    optimizer state.
    """
    layers = model.transformer.layers
    if not 0 < count <= len(layers):
        raise ValueError(f"freeze_layers must be in 1..{len(layers)}, got {count}")
    frozen = 0
    for layer in layers[:count]:
        for parameter in layer.parameters():
            if parameter.requires_grad:
                parameter.requires_grad = False
                frozen += parameter.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.getLogger(__name__).info(
        "Froze %d of %d decoder layers (%.1f M parameters); %.1f M still trainable, "
        "saving ~%.2f GB of gradients and optimizer state",
        count, len(layers), frozen / 1e6, trainable / 1e6, frozen * 12 / 1e9,
    )


def run(config: DictConfig) -> None:
    configure_logging(config.logging.level)
    set_seed_everything(config.seed)
    tokenizer = SimpleTokenizerGuitar()
    train_files, val_files = load_data_splits(config)
    train_loader, val_loader, vocab = build_dataloaders(config, tokenizer, train_files, val_files)
    model = WaveformTransformerDiscrete(
        pad_token_id=vocab["<PAD>"],
        eos_token_id=vocab["<eos>"],
        vocab_size=len(vocab),
        cfg_model=config.model,
        cfg_optimizer=config.optimizer,
    )
    freeze_layers = OmegaConf.select(config, "model.freeze_layers", default=0)
    if freeze_layers:
        freeze_lower_layers(model, freeze_layers)

    pretrained = OmegaConf.select(config, "model.pretrained", default=None)
    if pretrained:
        load_pretrained_transformer(
            model,
            pretrained,
            tokenizer,
            jitter=OmegaConf.select(config, "model.pretrained_jitter", default=0.01),
            seed=config.seed,
            allow_partial=OmegaConf.select(
                config, "model.pretrained_allow_partial", default=False
            ),
        )
    # Start from a checkpoint we produced, with a fresh optimizer. `resume_path` cannot
    # do this: it restores the optimizer, the epoch and the step count, so a checkpoint
    # taken past max_steps would stop immediately. What is wanted for a comparison run
    # is the weights alone.
    init_from = OmegaConf.select(config, "model.init_from", default=None)
    if init_from:
        path = Path(init_from)
        if not path.is_file():
            raise FileNotFoundError(f"model.init_from does not exist: {path}")
        # weights_only=False executes pickled code; this is only ever pointed at a
        # checkpoint produced locally, never a downloaded one.
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        logging.getLogger(__name__).info(
            "Initialised from %s (%d missing, %d unexpected)",
            path, len(missing), len(unexpected))

    # An epoch here is over a day and the card runs at ~97% VRAM, so a run that dies part
    # way through is the expected case, not the exceptional one. `save_last` keeps a
    # last.ckpt beside the monitored best; point trainer.resume_path at it to continue.
    resume_path = OmegaConf.select(config, "trainer.resume_path", default=None)
    if resume_path and not Path(resume_path).is_file():
        raise FileNotFoundError(f"trainer.resume_path does not exist: {resume_path}")
    if resume_path:
        logging.getLogger(__name__).info("Resuming from %s", resume_path)

    with experiment_logger(config, build_run_name(config)) as logger:
        build_trainer(config, logger, MONITORED_METRIC).fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_path,
        )


@hydra.main(version_base=None, config_path="configs", config_name="audio")
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    main()
