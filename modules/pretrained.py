"""Load released Charter weights into the Lightning training model.

`export_checkpoint.py` goes one way: a Lightning checkpoint's ``state_dict`` has its
``transformer.`` prefix stripped and is saved as ``pytorch_model.bin`` for inference.
This module inverts that, and additionally widens the token embedding / output
projection when the training vocabulary is the expressive one (1283 tokens) while the
released checkpoint was trained on the legacy 35-token vocabulary.

The expressive layout was chosen so legacy token ``c`` sits at ``c * N_FLAG * N_SUSTAIN``
(see :class:`chart.tokenizer.SimpleTokenizerGuitar`), so a legacy row can simply be
broadcast onto the whole block of expressive tokens that share its pressed lanes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from chart.tokenizer import SimpleTokenizerGuitar


logger = logging.getLogger(__name__)

VOCAB_PARAMETERS = ("transformer.token_embedding.weight", "transformer.output_projection.weight")


def _download(repo_id: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, "pytorch_model.bin"))


def resolve_source(source: str | Path) -> Path:
    """Accept a local .bin/.ckpt path, a directory holding one, or a HF repo id."""
    candidate = Path(source)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        weights = candidate / "pytorch_model.bin"
        if weights.is_file():
            return weights
        raise FileNotFoundError(f"No pytorch_model.bin in {candidate}")
    return _download(str(source))


def read_transformer_state(path: Path) -> dict[str, torch.Tensor]:
    """Return weights keyed as the LightningModule expects (``transformer.*``)."""
    blob: Any = torch.load(path, map_location="cpu", weights_only=True)
    state = blob.get("state_dict") if isinstance(blob, dict) and "state_dict" in blob else blob
    if not isinstance(state, dict):
        raise ValueError(f"Unrecognised checkpoint payload in {path}")

    prefixed = {name: value for name, value in state.items() if name.startswith("transformer.")}
    if prefixed:
        return prefixed
    # An exported inference checkpoint: keys are relative to the transformer itself.
    return {f"transformer.{name}": value for name, value in state.items()}


def expand_vocab_rows(
    rows: torch.Tensor,
    tokenizer: SimpleTokenizerGuitar,
    jitter: float = 0.01,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Broadcast legacy row ``c`` onto every expressive token sharing its lanes.

    Legacy ``c`` covers chords 0..n_chords-1 plus bos/eos/pad; the expressive vocabulary
    keeps the specials at the top in the same order, so they map one to one.
    `jitter` adds a small seeded perturbation to the copies so the duplicated output
    rows do not start out producing bit-identical logits.
    """
    legacy = SimpleTokenizerGuitar(
        exclude_open_chords=tokenizer.exclude_open_chords, expressive=False
    )
    if rows.shape[0] != legacy.vocab_size:
        raise ValueError(
            f"Expected a {legacy.vocab_size}-row legacy parameter, got {rows.shape[0]}"
        )

    block = tokenizer.N_FLAG * tokenizer.N_SUSTAIN
    expanded = rows.new_zeros((tokenizer.vocab_size, rows.shape[1]))
    for chord in range(legacy.n_chords):
        base = tokenizer.compose(chord, 0, 0)
        expanded[base : base + block] = rows[chord]
    for legacy_id, new_id in (
        (legacy.bos_id, tokenizer.bos_id),
        (legacy.eos_id, tokenizer.eos_id),
        (legacy.pad_id, tokenizer.pad_id),
    ):
        expanded[new_id] = rows[legacy_id]

    if jitter:
        noise = torch.empty_like(expanded).normal_(generator=generator)
        expanded += noise * (jitter * rows.std())
        # Keep the exact legacy row wherever the legacy token had a meaning of its own.
        for chord in range(legacy.n_chords):
            expanded[tokenizer.compose(chord, 0, 0)] = rows[chord]
        expanded[tokenizer.bos_id] = rows[legacy.bos_id]
        expanded[tokenizer.eos_id] = rows[legacy.eos_id]
        expanded[tokenizer.pad_id] = rows[legacy.pad_id]
    return expanded


def load_pretrained_transformer(
    model: torch.nn.Module,
    source: str | Path,
    tokenizer: SimpleTokenizerGuitar,
    jitter: float = 0.01,
    seed: int = 0,
    allow_partial: bool = False,
) -> dict[str, list[str]]:
    """Load released weights into `model` in place; return a report of what happened.

    Raises unless every transformer parameter is accounted for. `strict=False` is needed
    only because the frozen Encodec encoder is rebuilt from facebook/encodec_24khz and is
    absent from a transformer-only checkpoint; anything else left uninitialised means the
    configured architecture does not match the checkpoint, and a run started that way
    trains a partly random model while looking perfectly healthy.
    """
    path = resolve_source(source)
    state = read_transformer_state(path)
    target = model.state_dict()

    generator = torch.Generator().manual_seed(seed)
    for name in VOCAB_PARAMETERS:
        if name not in state or name not in target:
            continue
        if state[name].shape == target[name].shape:
            continue
        if state[name].shape[0] < target[name].shape[0]:
            state[name] = expand_vocab_rows(state[name], tokenizer, jitter, generator)
            logger.info("Expanded %s to the expressive vocabulary (%d rows)", name, len(state[name]))

    mismatched = [
        name
        for name, value in state.items()
        if name in target and value.shape != target[name].shape
    ]
    if mismatched:
        detail = ", ".join(
            f"{name} {tuple(state[name].shape)} != {tuple(target[name].shape)}"
            for name in mismatched[:5]
        )
        raise ValueError(
            f"Checkpoint architecture does not match the configured model: {detail}"
            + (f" (and {len(mismatched) - 5} more)" if len(mismatched) > 5 else "")
        )

    result = model.load_state_dict(state, strict=False)
    loaded = [name for name in state if name in target]
    if not loaded:
        raise ValueError(f"No parameters from {path} matched the model")

    unexpected = list(result.unexpected_keys)
    # The audio encoder is rebuilt from facebook/encodec_24khz and is frozen, so it is
    # expected to be missing from a transformer-only checkpoint.
    missing = [name for name in result.missing_keys if not name.startswith("audio_encoder.")]
    logger.info(
        "Loaded %d/%d transformer parameters from %s (%d missing, %d unexpected)",
        len(loaded), len(target), path, len(missing), len(unexpected),
    )
    if (missing or unexpected) and not allow_partial:
        raise ValueError(
            f"{path} does not fully match the configured model: "
            f"{len(missing)} parameters left uninitialised "
            f"(e.g. {missing[:3]}), {len(unexpected)} unused in the checkpoint "
            f"(e.g. {unexpected[:3]}). Check configs/model against the checkpoint's "
            f"config.json, or pass model.pretrained_allow_partial=true to accept this."
        )
    for name in missing:
        logger.warning("Not initialised from checkpoint: %s", name)
    for name in unexpected:
        logger.warning("Present in checkpoint but not in the model: %s", name)
    return {"loaded": loaded, "missing": missing, "unexpected": unexpected}
