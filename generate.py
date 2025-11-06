import argparse
import os
import torch

from inference.engine import Charter
from chart.time_conversion import convert_notes_to_ticks
from chart.tokenizer import SimpleTokenizerGuitar
from chart.chart_writer import fill_expert_single


def main():
    parser = argparse.ArgumentParser(
        description="🎵 Convert an audio file into a Guitar Hero-style chart using Charter."
    )

    # Required argument
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the input audio file (must be >= 30 seconds)."
    )

    # Optional model + sampling args
    parser.add_argument(
        "--model_name",
        type=str,
        default="3podi/charter-v1.0-40-M-best-acc",
        help="Model identifier or path. (default: 3podi/charter-v1.0-40-M-best-acc)"
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=32, help="Top-k sampling parameter.")

    # Optional metadata
    parser.add_argument("--name", type=str, default=None, help="Song title.")
    parser.add_argument("--artist", type=str, default=None, help="Artist name.")
    parser.add_argument("--album", type=str, default=None, help="Album name.")
    parser.add_argument("--genre", type=str, default=None, help="Genre.")
    parser.add_argument("--charter", type=str, default=None, help="Charter name.")
    parser.add_argument("--bpm", type=int, default=200, help="Chart bpm.")
    parser.add_argument("--resolution", type=int, default=480, help="Chart resolution.")

    # Output path (optional)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination folder for output. Defaults to ./<song_name>/notes.chart"
    )

    args = parser.parse_args()

    # Load model + tokenizer
    print(f"Loading model: {args.model_name}")
    model = Charter.from_pretrained(args.model_name)
    tokenizer = SimpleTokenizerGuitar()
    ms_resolution = model.config.grid_ms

    # Generate tokens
    print(f"Generating chart for: {args.audio_path}")
    seqs = model.generate(
        args.audio_path,
        temperature=args.temperature,
        top_k=args.top_k
    )
    seqs = torch.cat(seqs).flatten().cpu().tolist()

    # Convert to ticked notes
    time_list = [i * ms_resolution / 1000 for i in range(len(seqs))]
    ticked_notes = convert_notes_to_ticks(seqs, time_list, fixed_bpm=args.bpm, resolution=args.resolution)
    decoded_full = tokenizer.decode(ticked_notes)

    # Prepare metadata
    model_tag = args.model_name.split("/")[-1]
    default_charter = args.charter or f"audio2chart/{model_tag}-{args.temperature}-{args.top_k}"

    song_name = args.name or os.path.splitext(os.path.basename(args.audio_path))[0]
    metadata = {
        "name": song_name,
        "artist": args.artist or "audio2chart",
        "album": args.album or "audio2chart",
        "genre": args.genre or "audio2chart",
        "charter": default_charter,
        "bpm": args.bpm,
        "resolution": args.resolution
    }

    # Fill and save chart
    filled_text = fill_expert_single(decoded_full, metadata=metadata)

    # Determine output folder and file path
    if args.output:
        output_folder = args.output
    else:
        output_folder = os.path.join(os.getcwd(), song_name)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "notes.chart")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(filled_text)

    print(f"✅ Chart saved to: {output_path}")


if __name__ == "__main__":
    main()
