"""Export an Encodec-conditioned Lightning checkpoint for the inference engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from inference.engine import TransformerConfig
from inference.model_inference import TransformerDecoderAudioConditioned


def extract_transformer_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract the decoder state dict and remove Lightning's module prefix."""
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a Lightning 'state_dict'")

    prefix = "transformer."
    transformer_state = {
        name[len(prefix):]: value
        for name, value in state_dict.items()
        if name.startswith(prefix)
    }
    if not transformer_state:
        raise ValueError("Checkpoint contains no 'transformer.*' parameters")
    return transformer_state


def export_checkpoint(checkpoint_path: Path, config_path: Path, output_dir: Path,
                      trust: bool = False) -> None:
    """Validate and export weights in the format consumed by Charter."""
    with config_path.open(encoding="utf-8") as stream:
        config_data = json.load(stream)

    # grid_ms and window_seconds decide how many tokens the decoder emits, so a model
    # exported with the wrong pair generates nonsense rather than failing. They used to
    # be implicit -- grid_ms defaulted to 20 and the window was hardcoded to 30 in the
    # engine -- which was survivable only while nothing changed them.
    missing = [key for key in ("grid_ms", "window_seconds") if key not in config_data]
    if missing:
        raise ValueError(
            f"{config_path} is missing {', '.join(missing)}. The exported model has to "
            "carry the grid and window it was trained on; defaulting them silently "
            "produces a model that emits the wrong number of tokens."
        )
    config = TransformerConfig(**config_data)

    # Lightning's save_hyperparameters() stores the whole Hydra config in the checkpoint, so
    # a weights_only load refuses it -- and allowlisting the classes one at a time turns into
    # DictConfig, then dict, then whatever is nested next. weights_only=False deserializes
    # arbitrary Python, so it stays opt-in: pass trust=True only for a checkpoint you
    # produced yourself, never for one that was downloaded.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=not trust)
    transformer_state = extract_transformer_state_dict(checkpoint)

    # The inference transformer takes architecture arguments only; the timing fields
    # travel in config.json for the engine to read.
    architecture = {k: v for k, v in config_data.items()
                    if k not in ("grid_ms", "window_seconds")}
    inference_model = TransformerDecoderAudioConditioned(**architecture)
    try:
        inference_model.load_state_dict(transformer_state, strict=True)
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint is not compatible with the current Encodec inference architecture"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(transformer_state, output_dir / "pytorch_model.bin")
    with (output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config.__dict__, stream, indent=2)
        stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Lightning checkpoint path")
    parser.add_argument("config", type=Path, help="Inference config.json")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--trust", action="store_true",
                        help="Deserialize arbitrary Python from the checkpoint. Only for "
                             "checkpoints you produced yourself.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    export_checkpoint(arguments.checkpoint, arguments.config, arguments.output, arguments.trust)
