"""How accurate is tempo detection, measured against charts that already know?

Every human chart carries its tempo in [SyncTrack] and its first note's tick. That is
ground truth for both halves of what detect_tempo produces -- the tempo itself, and the
phase that decides where beat one sits.

The distinction matters because the two fail differently. A wrong tempo drifts: the chart
pulls further from the audio as the song goes on. A wrong phase is a constant offset, and
on its own would be almost harmless -- except that `--snap auto` quantises every note onto
the beat grid, so an offset grid shoves each note by a different amount and the result
reads as jitter rather than as lateness. Robert, playing a generated chart: "the patterns
seem to drift out of sync but the song stays mostly in sync".

On LESBIAN VAMPIRE KILLERS the tempo came out exactly right (152.000, matching the human
chart) while the first beat landed 35 ms late. This measures whether that is typical.

Only single-tempo charts are used. A chart with tempo changes has no one right answer to
compare against, and detect_tempo does not claim to find them.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chart.tempo import detect_tempo, first_onset, parse_sync_track


def chart_offset(chart_path: str) -> float:
    """The [Song] Offset in seconds, which is where the chart's own beat grid starts."""
    import re

    try:
        with open(chart_path, encoding="utf-8-sig", errors="replace") as stream:
            head = stream.read(4000)
    except OSError:
        return 0.0
    match = re.search(r'Offset\s*=\s*"?([-\d.]+)"?', head)
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path,
                        default=Path(r"G:\a2c_data\splits_tapping\val.json"))
    parser.add_argument("--songs", type=int, default=40)
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="How much audio to analyse. Tempo detection scores a dense "
                             "grid over the onset envelope, so this dominates runtime.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import librosa

    entries = json.loads(args.split.read_text(encoding="utf-8"))
    random.Random(0).shuffle(entries)

    tempo_errors, phase_errors, octave = [], [], 0
    onset_phase_errors: list[float] = []
    checked = 0

    for entry in entries:
        if checked >= args.songs:
            break
        try:
            bpm_events, _, resolution = parse_sync_track(entry["chart_path"])
        except (OSError, ValueError):
            continue
        if len(bpm_events) != 1:
            continue                      # tempo changes: no single right answer
        true_bpm = bpm_events[0][1] / 1000.0

        try:
            waveform, rate = librosa.load(entry["audio_path"], sr=24000, mono=True,
                                          duration=args.seconds)
        except Exception:
            continue
        if len(waveform) < rate * 20:
            continue

        try:
            found_bpm, phase, _ = detect_tempo(waveform, rate)
            onset = first_onset(waveform, rate)
        except Exception:
            continue
        checked += 1

        # An octave error is a different failure from being slightly off, and averaging
        # the two together would hide both.
        ratios = [found_bpm / true_bpm * factor for factor in (1.0, 2.0, 0.5)]
        closest = min(ratios, key=lambda r: abs(r - 1.0))
        if abs(closest - 1.0) < 0.02 and abs(found_bpm / true_bpm - 1.0) >= 0.02:
            octave += 1
            continue
        tempo_errors.append(abs(found_bpm - true_bpm))

        # Compare our beat grid against the chart's own, not against the first onset.
        #
        # Measuring to the first onset conflates two different things: a grid that is
        # genuinely offset, and a song that simply opens on a pickup note. A pickup is off
        # the beat on purpose, so counting its distance as error overstates the problem.
        # That version reported an 18.7 ms median and a 158.7 ms p90, and some of that was
        # the music rather than the analysis.
        #
        # A single-tempo chart puts a beat line every `resolution` ticks from tick 0, so in
        # seconds its grid sits at Offset + k * period. The comparison has to be circular:
        # being one whole beat out is being exactly right.
        period = 60.0 / found_bpm
        target = chart_offset(entry["chart_path"])

        def circular_error(candidate):
            difference = (candidate - target) % period
            return min(difference, period - difference) * 1000.0

        phase_errors.append(circular_error(phase))
        # Robert's own rule says the first note lands on a beat line, which makes the
        # first onset a phase estimate in its own right -- and a far more direct one than
        # maximising mean onset strength over every beat in the song. Measured alongside,
        # because the search is what is currently 45-51 ms out and something has to
        # replace it.
        onset_phase_errors.append(circular_error(onset))

    print(f"{checked} single-tempo songs, first {args.seconds:.0f}s of each\n")
    if tempo_errors:
        exact = sum(1 for e in tempo_errors if e < 0.05)
        close = sum(1 for e in tempo_errors if e < 1.0)
        print(f"  TEMPO")
        print(f"    median error   {statistics.median(tempo_errors):.3f} BPM")
        print(f"    within 0.05    {exact}/{len(tempo_errors)}")
        print(f"    within 1.0     {close}/{len(tempo_errors)}")
        print(f"    octave errors  {octave} (counted separately)")
    if phase_errors:
        print(f"\n  PHASE -- our beat grid against the chart's own, circular")
        print(f"    median         {statistics.median(phase_errors):.1f} ms")
        print(f"    p90            {sorted(phase_errors)[int(0.9 * len(phase_errors))]:.1f} ms")
        print(f"    worst          {max(phase_errors):.1f} ms")
    if onset_phase_errors:
        print(f"\n  PHASE taken from the first onset instead of the search")
        print(f"    median         {statistics.median(onset_phase_errors):.1f} ms")
        print(f"    p90            "
              f"{sorted(onset_phase_errors)[int(0.9 * len(onset_phase_errors))]:.1f} ms")
        better = sum(1 for a, b in zip(onset_phase_errors, phase_errors) if a < b)
        print(f"    beats the search on {better}/{len(phase_errors)} songs")
        print(f"\n  At a 10 ms grid, anything above ~5 ms moves notes onto the wrong slot")
        print(f"  once --snap auto quantises against this grid.")


if __name__ == "__main__":
    main()
