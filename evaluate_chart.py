"""Score a generated chart against what human charters actually do.

Four independent questions, because "is this a good chart?" is not one measurement:

*What is it made of?* Chord, tap, forced and sustain rates against the per-chart
corpus median. This is where the current model fails hardest -- at the shipped
sampling defaults it emits zero chords, zero taps and almost no sustains despite
predicting all three at near-human rates when teacher-forced.

*Are the shapes real?* Catalogue coverage above the chart's own chance floor. Raw
coverage is not comparable between charts; the floor ranges 13% to 35% depending on
how a chart uses the neck.

*Does it phrase?* Rests per minute against the human Expert baseline. A human chart
breathes constantly -- 34 short rests a minute, a third of the song -- where the
current model plays continuously and then stops dead for seconds at a time.

*Is it legal?* Universal rule 1, no overlapping sustains. Zero violations in 836,327
human notes, so any violation is a defect -- though a chart with no sustains passes
vacuously, which is worth knowing when reading the result.

ONE CHART DOES NOT CHARACTERISE A MODEL. Measured across eight charts from eight
songs, this model's output ranges from 0.000 to 0.655 sustains, -0.172 to +0.383
pattern lift, and 0 to 80 rests a minute. A whole day of this project's conclusions
came from a single chart and several of them were wrong in both directions -- taps
called absent when the median chart matches humans, pattern vocabulary called
human-level when the median chart is at chance, density called too low when it is 3.2x
too high. Score several charts before saying anything about the model.

NOR DOES A DELTA FROM THE MEDIAN MEAN MUCH ON ITS OWN. Twelve *human* charts from the
tapping validation split, on these same measurements:

    metric        median     human range
    pct_chord      0.051     0.000 - 0.237
    pct_tap        0.940     0.810 - 1.000
    pct_sustain    0.010     0.000 - 0.084
    nps           20.97      3.97  - 33.58
    lift           0.518     0.308 - 0.614
    rests/min     16.95      0.53  - 55.30

Density spans 8x between human charts and phrasing spans 100x. A generated chart at
10 NPS is half the median and squarely inside what humans do; one with 3 rests a minute
is unusual but not outside the range. Read the ! flags as "look at this", not as
"wrong", and weigh a metric by how tightly humans agree on it -- pct_tap is narrow and
therefore informative, rests/min is not.

*Does it follow the music?* Only with --audio. Note density against the onset
envelope, which is meaningful in aggregate and noisy for one chart (per-song sd 0.284).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chart.chart_processor import ChartProcessor
from chart.discover import (HUMAN_LIFT_MEAN, HUMAN_LIFT_SD, TAPPING_LIFT_MEAN,
                           TAPPING_LIFT_SD, catalogue_lift)
from chart.metrics import (BASELINES, TAPPING_REST_SECONDS,
                          TAPPING_RESTS_PER_MINUTE, profile_from_timed)
from chart.patterns import single_note_runs
from chart.tokenizer import SimpleTokenizerGuitar

NOTE_LINE = re.compile(r"^\s*(\d+)\s*=\s*N\s+(\d+)\s+(\d+)")
MAX_FRET = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chart", type=Path)
    parser.add_argument("--section", default="ExpertSingle")
    parser.add_argument("--audio", type=Path, default=None,
                        help="Also score how well note density tracks the audio.")
    parser.add_argument("--baseline", choices=sorted(BASELINES), default="all",
                        help="Which human population to compare against. 'all' is 250 "
                             "Expert charts of every style; 'tapping' is the 1454-song "
                             "tapping split. Use 'tapping' for a model trained on the "
                             "tapping subset -- against 'all' a correct tapping chart "
                             "fails every metric, because taps run 0.96 against 0.19 "
                             "and density 21 NPS against 7.8.")
    return parser.parse_args()


REST_SECONDS = 0.25
HUMAN_RESTS_PER_MINUTE = 34.0
HUMAN_REST_SECONDS = 0.38
"""Measured over 120 human Expert charts; only 1 of them has no rests at all.

Tapping charts phrase differently -- 15.1 rests a minute, median 0.40 s -- so --baseline
tapping swaps these too. See chart.metrics.TAPPING_PER_CHART_MEDIAN."""


def phrasing(times: list[float]) -> tuple[float, float]:
    """(rests per minute, median rest) for gaps longer than REST_SECONDS."""
    import statistics

    ordered = sorted(set(times))
    if len(ordered) < 2:
        return 0.0, 0.0
    span = ordered[-1] - ordered[0]
    if span <= 0:
        return 0.0, 0.0
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b - a > REST_SECONDS]
    return len(gaps) / (span / 60.0), (statistics.median(gaps) if gaps else 0.0)


def overlapping_sustains(chart_path: Path) -> tuple[int, int]:
    """Sustains still ringing when the same fret is struck again."""
    per_fret: dict[int, list[tuple[int, int]]] = defaultdict(list)
    inside = False
    with chart_path.open(encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("["):
                inside = stripped == "[ExpertSingle]"
                continue
            if not inside:
                continue
            match = NOTE_LINE.match(line)
            if match:
                tick, fret, length = (int(g) for g in match.groups())
                if fret <= MAX_FRET:
                    per_fret[fret].append((tick, length))
    bad = total = 0
    for items in per_fret.values():
        items.sort()
        for index in range(len(items) - 1):
            tick, length = items[index]
            total += 1
            if length > 0 and tick + length > items[index + 1][0]:
                bad += 1
    return bad, total


def line(name: str, actual: float, target: float, fmt: str = "{:.3f}") -> str:
    delta = actual - target
    flag = "  " if abs(delta) <= max(0.25 * abs(target), 0.02) else " !"
    return (f"  {name:<16}{fmt.format(actual):>10}{fmt.format(target):>10}"
            f"{delta:>+10.3f}{flag}")


def main():
    args = parse_args()
    tokenizer = SimpleTokenizerGuitar(expressive=True)
    processor = ChartProcessor(["Expert"], ["Single"])
    processor.read_chart(str(args.chart), target_sections=args.section)
    resolution = int(processor.song_metadata["Resolution"])
    offset = float(processor.song_metadata["Offset"])
    encoded = tokenizer.encode(processor.notes[args.section], resolution=resolution)
    timed = tokenizer.format_seconds(encoded, processor.synctrack, resolution, offset)
    profile = profile_from_timed(timed, tokenizer)

    print(f"{args.chart}")
    print("  NOTE: one chart does not characterise a model -- this model's output")
    print("        varies enormously by song. Score several before concluding.")
    print(f"  {profile.positions} positions over {profile.duration_seconds:.1f}s\n")
    print(f"  {'metric':<16}{'chart':>10}{'human':>10}{'delta':>10}"
          f"   (human = {args.baseline})")
    # Report the chord rate, not the single-note rate. A chart with no chords at all
    # reads as chord_1 1.000 against a human 0.872, and a relative tolerance on a
    # number that close to 1 waves through the most conspicuous defect there is.
    baseline = BASELINES[args.baseline]
    for key, value in (("pct_chord", 1.0 - profile.chord_hist.get(1, 0.0)),
                       ("pct_tap", profile.pct_tap),
                       ("pct_forced", profile.pct_forced),
                       ("pct_sustain", profile.pct_sustain),
                       ("nps", profile.nps)):
        target = (1.0 - baseline["chord_1"] if key == "pct_chord" else baseline[key])
        print(line(key, value, target, "{:.2f}" if key == "nps" else "{:.3f}"))

    runs = [[fret for _, fret in run] for run in single_note_runs(timed, tokenizer)]
    coverage, floor = catalogue_lift(runs)
    lift = coverage - floor
    lift_mean = TAPPING_LIFT_MEAN if args.baseline == "tapping" else HUMAN_LIFT_MEAN
    lift_sd = TAPPING_LIFT_SD if args.baseline == "tapping" else HUMAN_LIFT_SD
    sigmas = (lift - lift_mean) / lift_sd if lift_sd else 0.0
    print(f"\n  pattern lift    {lift:>+10.3f}{lift_mean:>+10.3f}"
          f"{sigmas:>+9.1f}sd")
    print(f"    coverage {coverage:.3f} against a chance floor of {floor:.3f}")

    per_minute, typical = phrasing([item[0] for item in timed])
    rest_target = (TAPPING_RESTS_PER_MINUTE if args.baseline == "tapping"
                   else HUMAN_RESTS_PER_MINUTE)
    rest_length = (TAPPING_REST_SECONDS if args.baseline == "tapping"
                   else HUMAN_REST_SECONDS)
    print("")
    print(f"  rests/min       {per_minute:>10.1f}{rest_target:>10.1f}"
          f"{per_minute - rest_target:>+10.1f}"
          + ("  " if abs(per_minute - rest_target) < 10 else " !"))
    print(f"  typical rest    {typical:>10.2f}{rest_length:>10.2f}"
          f"{typical - rest_length:>+10.2f}"
          + ("  " if abs(typical - rest_length) < 0.3 else " !"))

    bad, total = overlapping_sustains(args.chart)
    print(f"\n  overlapping sustains {bad}/{total}"
          + ("  (vacuous: this chart has almost no sustains)"
             if profile.pct_sustain < 0.01 else ""))

    if args.audio:
        import librosa
        from chart.responsiveness import note_density, onset_envelope, responsiveness
        waveform, rate = librosa.load(str(args.audio), sr=24000, mono=True)
        duration = len(waveform) / rate
        score = responsiveness(
            note_density([item[0] for item in timed], duration),
            onset_envelope(waveform, rate))
        print(f"\n  responsiveness  {score:>+10.3f}  "
              f"(human median +0.261; per-song sd 0.284, so one chart says little)")


if __name__ == "__main__":
    main()
