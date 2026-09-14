"""Shared training entry-point utilities."""

from __future__ import annotations

import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.plugins.io import TorchCheckpointIO
from omegaconf import DictConfig, OmegaConf

from modules.utils_train import LogGradientNorm, WatchPadCollapse


logger = logging.getLogger(__name__)


class DirectCheckpointIO(TorchCheckpointIO):
    """Serialize checkpoints straight to disk instead of through a memory buffer.

    Lightning's `_atomic_save` builds the entire checkpoint in a BytesIO before writing
    it, so saving costs twice the checkpoint's size in RAM. This one is ~2.9 GB (245 M
    parameters plus AdamW's two moments), and the four persistent dataloader workers each
    hold up to 2 GB of cached audio at the same time. Two runs died at exactly this point,
    once with a truncated zip ("unexpected pos ... vs ...") and once with a segfault --
    different symptoms, same cause.

    Writing to a sibling temporary file and renaming keeps the atomicity the buffer was
    there to provide, and the file is checked for a plausible size before the rename so a
    short write surfaces here rather than at resume time.
    """

    MINIMUM_BYTES = 1 << 20

    # A checkpoint here is ~2.42 GB. Refuse to start a write without room for three, so
    # the run stops with a clear message instead of dying mid-write.
    #
    # This is not hypothetical. trainer.log_dir was pointed at C: on 2026-09-12 for the
    # write speed -- 704 MB/s against 108 MB/s on the corpus drive -- and five runs of
    # checkpoints filled a 464 GB system drive to 7 MB free. The run then died inside
    # torch.save, and the cleanup `unlink` raised PermissionError on the locked .partial,
    # which *replaced* the original exception: the traceback said the file was in use by
    # another process and said nothing at all about the disk being full.
    REQUIRED_HEADROOM_BYTES = 3 * 2_500_000_000

    def save_checkpoint(self, checkpoint: dict[str, Any], path, storage_options=None) -> None:
        if storage_options is not None:
            raise TypeError(
                f"{type(self).__name__} does not accept storage_options={storage_options!r}"
            )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        free = shutil.disk_usage(destination.parent).free
        if free < self.REQUIRED_HEADROOM_BYTES:
            raise RuntimeError(
                f"Refusing to write {destination}: only {free / 1e9:.1f} GB free on that "
                f"drive and a checkpoint is ~2.4 GB. Free space or point trainer.log_dir "
                f"somewhere with room."
            )

        temporary = destination.with_name(destination.name + ".partial")
        try:
            torch.save(checkpoint, temporary)
            written = temporary.stat().st_size
            if written < self.MINIMUM_BYTES:
                raise RuntimeError(f"Checkpoint write produced only {written} bytes")
            os.replace(temporary, destination)
        except BaseException:
            # Never let cleanup mask the real failure. On Windows the temporary can still
            # be locked by the very write that just failed, and an exception raised here
            # replaces the one actually worth reading.
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                logger.warning("Could not remove %s: %s", temporary, cleanup_error)
            raise
        logger.info("Wrote checkpoint %s (%.2f GB)", destination, written / 1e9)


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown logging level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _log_dir(config: DictConfig) -> str:
    """Where checkpoints and CSV metrics are written.

    Needs room before speed. A checkpoint is 2.42 GB and a run writes about eight of
    them.

    This docstring used to say to keep it off the corpus drive, on the grounds that G:
    runs at 108 MB/s against C:'s 704 MB/s and the dataloader reads audio from G:. That
    was mostly wrong: the profiler puts the entire dataloader at 0.1% of a step, so the
    contention hardly exists, and the real throughput fix was writing checkpoints less
    often. Following the old advice filled a 464 GB system drive to 7 MB free and killed
    a run mid-write, which cost far more than the 2.6 minutes per run the slower drive
    costs.
    """
    return str(OmegaConf.select(config, "trainer.log_dir", default="lightning_logs"))


@contextmanager
def experiment_logger(config: DictConfig, run_name: str) -> Iterator[object]:
    """Create an optional W&B logger and always finalize the active run."""
    if not config.tracking.enabled:
        yield CSVLogger(save_dir=_log_dir(config), name=run_name)
        return
    wandb.init(
        project=config.tracking.project,
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        name=run_name,
        tags=list(config.tracking.tags),
        reinit=True,
    )
    try:
        yield WandbLogger(log_model=False)
    finally:
        wandb.finish()


def build_callbacks(config: DictConfig, monitor: str, mode: str = "max") -> list[object]:
    callbacks: list[object] = [
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor=monitor,
            min_delta=0.0001,
            patience=config.trainer.early_stopping_patience,
            mode=mode,
        ),
        LogGradientNorm(),
        # Teacher-forced metrics cannot see the failure this watches for.
        WatchPadCollapse(),
    ]
    if config.trainer.save_run:
        callbacks.append(L.pytorch.callbacks.ModelCheckpoint(
            monitor=monitor,
            save_top_k=1,
            mode=mode,
            filename="best-checkpoint",
            # save_last belongs on the step-based callback below, not here. With it on
            # both, the two collide on last.ckpt and Lightning versions one to
            # last-v1.ckpt -- a redundant 1.98 GB write every validation, and two files
            # named "last" with different contents, which is worse than useless when
            # choosing what to resume from.
            save_last=False,
        ))
        # Validation is what makes the monitored checkpoint expensive (200 batches, ~13
        # minutes), not the 1.98 GB write (~45 s). Tying resume points to it puts them
        # hours apart, so a pause or a crash costs hours. This one owns last.ckpt and
        # writes it on a plain optimizer-step count with no validation -- note Lightning
        # counts optimizer steps here, so the value is batches / accumulate_grad_batches.
        interval = OmegaConf.select(config, "trainer.checkpoint_every_n_steps", default=0)
        if interval:
            callbacks.append(L.pytorch.callbacks.ModelCheckpoint(
                save_top_k=0,
                save_last=True,
                every_n_train_steps=interval,
            ))
    return callbacks


VALIDATION_BUDGET = 0.25
"""Fraction of a run's time that may be validation before it is worth saying so."""

VALIDATION_BATCH_COST = 11.0
"""How much longer a validation batch takes than a training one, measured.

Counting raw batches understates validation badly. A validation item is `val_num_pieces`
windows where a training item is `train_num_pieces` -- 4 against 1 in the configs used
here -- so a validation batch is several times the work despite having no backward pass.
Measured on the burst-corruption run: 2.2 it/s training against 0.19 it/s validating,
which is 11.6x. Rounded down, so the warning errs toward staying quiet.
"""


def validation_budget(config: DictConfig, val_check_interval, limit_val_batches):
    """Report what validation will actually cost, in the units the run was asked in.

    configs/default.yaml has carried a comment about this for a while -- Lightning's
    val_check_interval counts BATCHES while ModelCheckpoint's every_n_train_steps counts
    OPTIMIZER STEPS -- and the comment did not stop it happening again. A run meant to
    validate twice was set to val_check_interval=500 with accumulate_grad_batches=8 and
    max_steps=1000, which is 8000 batches, so it validates sixteen times: about four
    hours of validation around one hour of training.

    A comment is not a guard. This does the arithmetic and prints it, so the mistake is
    visible in the first seconds of a run rather than in its wall-clock time.

    Returns the lines to print, so it can be tested without capturing stdout.
    """
    if not val_check_interval or not isinstance(val_check_interval, int):
        return []

    accumulate = int(OmegaConf.select(config, "trainer.accumulate_grad_batches",
                                      default=1) or 1)
    max_steps = int(OmegaConf.select(config, "trainer.max_steps", default=-1) or -1)

    steps_between = val_check_interval / accumulate
    lines = [f"Validating every {val_check_interval} batches "
             f"= every {steps_between:.4g} optimizer steps "
             f"(accumulate_grad_batches={accumulate})."]
    if max_steps <= 0:
        return lines

    total_batches = max_steps * accumulate
    runs = total_batches / val_check_interval
    lines[0] += f" {runs:.4g} validation run(s) over {max_steps} steps."

    if limit_val_batches and isinstance(limit_val_batches, int):
        val_batches = runs * limit_val_batches
        val_cost = val_batches * VALIDATION_BATCH_COST
        share = val_cost / (total_batches + val_cost)
        if share > VALIDATION_BUDGET:
            lines.append(
                f"  WARNING: {val_batches:.0f} validation batches against "
                f"{total_batches} training batches, and a validation batch costs about "
                f"{VALIDATION_BATCH_COST:.0f}x a training one -- roughly {share:.0%} of "
                f"this run will be validation. val_check_interval counts BATCHES; to "
                f"validate every N optimizer steps set it to N * {accumulate}.")
    return lines


def build_trainer(config: DictConfig, logger: object, monitor: str) -> L.Trainer:
    use_gpu = config.trainer.gpus > 0
    # val_check_interval also sets how often a checkpoint can be written, because the
    # monitored ModelCheckpoint only fires on validation. Left unset, Lightning validates
    # once per epoch, which on this corpus is over a day between save points.
    limits = {
        name: OmegaConf.select(config, f"trainer.{name}")
        for name in ("limit_train_batches", "limit_val_batches", "val_check_interval")
    }
    for line in validation_budget(config, limits["val_check_interval"],
                                  limits["limit_val_batches"]):
        print(line)
    return L.Trainer(
        max_epochs=config.trainer.max_epochs,
        max_steps=OmegaConf.select(config, "trainer.max_steps", default=-1),
        **{name: value for name, value in limits.items() if value is not None},
        accelerator="gpu" if use_gpu else "cpu",
        devices=config.trainer.gpus if use_gpu else 1,
        default_root_dir=_log_dir(config),
        plugins=[DirectCheckpointIO()],
        enable_checkpointing=bool(config.trainer.save_run),
        callbacks=build_callbacks(config, monitor),
        log_every_n_steps=config.trainer.log_every_n_steps,
        logger=logger,
        precision=config.trainer.precision,
        num_sanity_val_steps=config.trainer.num_sanity_val_steps,
        gradient_clip_val=config.trainer.gradient_clip_val,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        # `trainer.profiler=simple` splits a step into waiting for data versus running
        # the model. Guessing which one owns the time has been wrong repeatedly here:
        # a hand-rolled profile ran fp32 with a warm 60-file cache and reported the
        # encoder at 0.246s and the loader at 0.292s, when the encoder actually costs
        # 0.712s under the bf16 autocast training really uses.
        profiler=OmegaConf.select(config, "trainer.profiler", default=None),
    )
