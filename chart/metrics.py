"""Distributional metrics for judging whether a generated chart looks like a chart.

Loss and token accuracy cannot answer this. Measured over this corpus, 85.6% of 20 ms grid
slots are padding, so a model that emits nothing at all scores 85.6% token accuracy; the
interesting range is the last 14 points. Cross-entropy is also position-exact, so a chart
shifted by one 20 ms slot scores zero on every note despite being musically identical, and
it rewards under-charting because padding is always the safe guess.

That last failure mode is not hypothetical for this project. The earlier attempt in
`Clone Hero Autocharter` produced 99.2% single notes against 75.3% for humans, and 18.2-note
average runs against 203.4 -- degenerate output that was obvious to a charter and invisible
to any loss curve. A chord-size histogram catches it immediately.

These metrics compare *distributions* between a chart and a reference, on axes a charter
would notice. They say nothing about whether the chart matches the audio -- for that see
onset alignment, which needs the waveform and is deliberately not in this module.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

OPEN_LANE = 7

# Measured over 2,995 charts / 8,175,963 note positions (G:\a2c_data\chart_stats.json).
#
# These are POOLED over positions, so a long dense chart contributes more than a short one.
# ChartProfile measures a single chart, and averaging profiles across charts gives an
# unweighted mean, which is a different estimator: on a 40-chart sample it read pct_sustain
# 0.080 against this 0.045, and nps 10.8 against 7.2. Both are correct for what they
# measure. Compare like with like -- pooled against pooled, or per-chart against per-chart.
# Measured over 250 Expert charts, unweighted per chart -- the estimator that matches a
# ChartProfile, which is also one chart. Use this to judge a generated chart;
# CORPUS_REFERENCE above is pooled over positions and answers a different question.
#
# Mean and median diverge sharply on pct_tap (0.317 vs 0.194) and pct_sustain (0.152 vs
# 0.049) because chart style is skewed: a minority of charts are almost entirely taps
# and drag the mean. The median is the better target for "a typical chart", and the gap
# between the two is a reminder that a single generated chart cannot be scored against
# a mean without accounting for that spread.
PER_CHART_REFERENCE: dict[str, float] = {
    "chord_1": 0.8073,
    "chord_2": 0.1364,
    "chord_3": 0.0482,
    "pct_tap": 0.3172,
    "pct_forced": 0.0869,
    "pct_sustain": 0.1526,
    "lane_0": 0.1743,
    "lane_2": 0.2499,
    "lane_open": 0.0271,
    "nps": 10.1317,
}

PER_CHART_MEDIAN: dict[str, float] = {
    "chord_1": 0.8717,
    "chord_2": 0.0883,
    "chord_3": 0.0099,
    "pct_tap": 0.1942,
    "pct_forced": 0.0437,
    "pct_sustain": 0.0490,
    "lane_0": 0.1677,
    "lane_2": 0.2508,
    "lane_open": 0.0021,
    "nps": 7.8127,
}

# Same estimator, different population: the 1454-song tapping training split, unweighted
# per chart.
#
# Use this whenever the model being judged was trained on the tapping subset, because
# PER_CHART_MEDIAN above is measured across every charting style and a tapping overchart
# is nothing like a typical chart:
#
#              tapping   all styles
#   pct_tap     0.960       0.194
#   nps        21.01        7.81
#   pct_sustain 0.007       0.049
#   pct_forced  0.0009      0.044
#   rests/min  15.1        34.0
#
# Scored against the all-styles median, a correct tapping chart fails every one of those
# -- taps five times too high, density near three times too high, sustains and forced
# notes almost absent, half the rests. A human chart pulled straight out of the tapping
# validation split reads that way, which is how this was noticed. Judging a
# tapping-trained model against the wrong population would have condemned it for doing
# exactly what it was trained to do.
TAPPING_PER_CHART_MEDIAN: dict[str, float] = {
    "chord_1": 0.9536,
    "chord_2": 0.0280,
    "chord_3": 0.0051,
    "pct_tap": 0.9601,
    "pct_forced": 0.0009,
    "pct_sustain": 0.0067,
    "lane_0": 0.1914,
    "lane_2": 0.2503,
    "lane_open": 0.0004,
    "nps": 21.0105,
}

# Phrasing, same split and estimator. The all-styles figures are 34.0 and 0.38.
TAPPING_RESTS_PER_MINUTE = 15.1
TAPPING_REST_SECONDS = 0.40

BASELINES = {
    "all": PER_CHART_MEDIAN,
    "tapping": TAPPING_PER_CHART_MEDIAN,
}


# What separates the four difficulties, measured over 120 songs charted at all four
# levels -- same song, same charter, so the differences are difficulty itself.
#
# Difficulty is not density. Notes per second rises only 1.8x across the whole ladder,
# while mean run length is flat through Hard and then quadruples at Expert, and the tap
# rate jumps 72% at that same step. Chords peak at HARD and fall at Expert; sustains
# decrease monotonically as difficulty rises. So the Expert signature is long unbroken
# runs and tap technique, not more notes -- a difficulty control that scales density
# produces a dense Hard chart, not an Expert one.
DIFFICULTY_LADDER: dict[str, dict[str, float]] = {
    "Easy":   {"nps": 5.68, "pct_chord": 0.067, "pct_tap": 0.127,
               "pct_sustain": 0.112, "pattern_lift": 0.135, "mean_run": 55.7},
    "Medium": {"nps": 6.20, "pct_chord": 0.131, "pct_tap": 0.142,
               "pct_sustain": 0.091, "pattern_lift": 0.144, "mean_run": 56.5},
    "Hard":   {"nps": 6.93, "pct_chord": 0.154, "pct_tap": 0.152,
               "pct_sustain": 0.078, "pattern_lift": 0.203, "mean_run": 58.1},
    "Expert": {"nps": 10.03, "pct_chord": 0.139, "pct_tap": 0.262,
               "pct_sustain": 0.062, "pattern_lift": 0.220, "mean_run": 232.0},
}


# What the current fine-tuned model actually produces, over 8 charts from 8 songs at
# the shipped sampling defaults. Kept so a future run can be compared against where
# this one stood, and because the spread is the point: sustain ran 0.000 to 0.655,
# pattern lift -0.172 to +0.383, rests 0.0 to 80.3 a minute. No single chart
# characterises this model.
GENERATED_BASELINE: dict[str, float] = {
    "pct_chord": 0.043,      # human median 0.128
    "pct_tap": 0.224,        # human median 0.194 -- normal
    "pct_sustain": 0.000,    # human median 0.049 -- collapsed, bimodal
    "nps": 26.297,           # human median 7.813 -- 3.2x too dense
    "pattern_lift": 0.007,   # human mean 0.209 -- at chance
    "rests_per_minute": 1.192,   # human 34.0
}
"""Medians. Compare a future model against these to see whether it improved."""


CORPUS_REFERENCE: dict[str, float] = {
    "chord_1": 0.8726,
    "chord_2": 0.0891,
    "chord_3": 0.0266,
    "chord_4": 0.0062,
    "chord_5": 0.0055,
    "pct_tap": 0.4709,
    "pct_forced": 0.0591,
    "pct_sustain": 0.0454,
    "lane_0": 0.1770,
    "lane_1": 0.2258,
    "lane_2": 0.2480,
    "lane_3": 0.1999,
    "lane_4": 0.1232,
    "lane_open": 0.0256,
    "nps": 7.2,
}


@dataclass
class ChartProfile:
    """What a chart looks like, independent of which song it is."""

    positions: int = 0
    duration_seconds: float = 0.0
    nps: float = 0.0
    nps_by_bucket: list[float] = field(default_factory=list)
    chord_hist: dict[int, float] = field(default_factory=dict)
    lane_hist: dict[int, float] = field(default_factory=dict)
    pct_tap: float = 0.0
    pct_forced: float = 0.0
    pct_sustain: float = 0.0
    ioi_hist: dict[str, float] = field(default_factory=dict)
    mean_run_length: float = 0.0
    longest_run: int = 0

    @property
    def nps_variation(self) -> float:
        """Std/mean of density over time. A chart with no dynamics reads as flat."""
        if len(self.nps_by_bucket) < 2 or self.nps <= 0:
            return 0.0
        mean = sum(self.nps_by_bucket) / len(self.nps_by_bucket)
        if mean <= 0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in self.nps_by_bucket) / len(self.nps_by_bucket)
        return math.sqrt(variance) / mean

    def as_flat(self) -> dict[str, float]:
        """The subset comparable against CORPUS_REFERENCE."""
        flat = {f"chord_{size}": self.chord_hist.get(size, 0.0) for size in range(1, 6)}
        for lane in range(5):
            flat[f"lane_{lane}"] = self.lane_hist.get(lane, 0.0)
        flat["lane_open"] = self.lane_hist.get(OPEN_LANE, 0.0)
        flat["pct_tap"] = self.pct_tap
        flat["pct_forced"] = self.pct_forced
        flat["pct_sustain"] = self.pct_sustain
        flat["nps"] = self.nps
        return flat


# Inter-onset interval buckets, in seconds. Named rather than numeric because the useful
# question is "is it emitting plausible rhythms", and the boundaries are perceptual.
IOI_BUCKETS: tuple[tuple[str, float], ...] = (
    ("<=30ms", 0.030),      # chords/rolls at the edge of the grid
    ("<=60ms", 0.060),      # very fast runs
    ("<=125ms", 0.125),     # 16ths at ~120bpm
    ("<=250ms", 0.250),     # 8ths
    ("<=500ms", 0.500),     # quarters
    ("<=1s", 1.000),
    (">1s", float("inf")),
)
RUN_GAP_SECONDS = 0.25
"""Notes closer together than this count as part of the same run."""


def _bucket_ioi(delta: float) -> str:
    for name, upper in IOI_BUCKETS:
        if delta <= upper:
            return name
    return IOI_BUCKETS[-1][0]


def profile_from_timed(timed, tokenizer, bucket_seconds: float = 10.0) -> ChartProfile:
    """Build a profile from `tokenizer.format_seconds(...)` output.

    `timed` is a list of (absolute_time, token, duration, attrs), which is what both a
    reference chart and a decoded generation produce, so the same code judges both.
    """
    if not timed:
        return ChartProfile()

    times = [entry[0] for entry in timed]
    tokens = [entry[1] for entry in timed]
    positions = len(times)
    span = max(times) - min(times)

    chords: Counter[int] = Counter()
    lanes: Counter[int] = Counter()
    taps = forced = sustains = 0
    for token in tokens:
        chord_index, flag, sustain = tokenizer.split(token)
        note_lanes = tokenizer.reverse_chord[chord_index]
        fret_count = len([lane for lane in note_lanes if lane != OPEN_LANE])
        chords[max(1, fret_count)] += 1
        for lane in note_lanes:
            lanes[lane] += 1
        taps += bool(flag & 2)
        forced += bool(flag & 1)
        sustains += bool(sustain)

    ordered = sorted(times)
    deltas = [b - a for a, b in zip(ordered, ordered[1:])]
    ioi: Counter[str] = Counter(_bucket_ioi(d) for d in deltas)

    runs: list[int] = []
    current = 1
    for delta in deltas:
        if delta <= RUN_GAP_SECONDS:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)

    buckets: Counter[int] = Counter()
    origin = min(times)
    for time_value in times:
        buckets[int((time_value - origin) // bucket_seconds)] += 1
    n_buckets = int(span // bucket_seconds) + 1
    nps_by_bucket = [buckets.get(i, 0) / bucket_seconds for i in range(n_buckets)]

    total_lane_hits = sum(lanes.values()) or 1
    return ChartProfile(
        positions=positions,
        duration_seconds=span,
        nps=positions / span if span > 0 else 0.0,
        nps_by_bucket=nps_by_bucket,
        chord_hist={size: count / positions for size, count in sorted(chords.items())},
        lane_hist={lane: count / total_lane_hits for lane, count in sorted(lanes.items())},
        pct_tap=taps / positions,
        pct_forced=forced / positions,
        pct_sustain=sustains / positions,
        ioi_hist={name: ioi.get(name, 0) / max(1, len(deltas)) for name, _ in IOI_BUCKETS},
        mean_run_length=sum(runs) / len(runs),
        longest_run=max(runs),
    )


def profile_chart_file(chart_path: str, section: str, processor, tokenizer) -> ChartProfile:
    """Profile a .chart on disk. `processor` is a configured ChartProcessor."""
    processor.read_chart(chart_path, target_sections=section)
    resolution = int(processor.song_metadata["Resolution"])
    offset = float(processor.song_metadata["Offset"])
    encoded = tokenizer.encode(processor.notes[section], resolution=resolution)
    timed = tokenizer.format_seconds(encoded, processor.synctrack, resolution, offset)
    return profile_from_timed(timed, tokenizer)


def compare(profile: ChartProfile, reference: dict[str, float] | None = None) -> dict[str, dict]:
    """Per-metric deviation from a reference. Defaults to the measured corpus."""
    reference = CORPUS_REFERENCE if reference is None else reference
    flat = profile.as_flat()
    report: dict[str, dict] = {}
    for name, expected in reference.items():
        actual = flat.get(name, 0.0)
        report[name] = {
            "actual": actual,
            "reference": expected,
            "delta": actual - expected,
            "ratio": (actual / expected) if expected else float("inf"),
        }
    return report


def total_variation(profile: ChartProfile, reference: dict[str, float] | None = None) -> float:
    """Half the L1 distance between the chord-size distributions, in [0, 1].

    Do NOT read a single chart's score as a verdict. Measured over 40 real charts this is
    mean 0.120 with a max of 0.596 -- individual charts deviate from the corpus average all
    the time, because a chording chart legitimately has more chords than average. A high
    score means "unusual", not "bad".

    It earns its keep in aggregate: compare the *distribution* of scores over many
    generated charts against the same distribution over real ones. If generated charts
    cluster tighter than real ones, the model is averaging rather than charting; if they
    cluster somewhere real charts never go, it is producing something else entirely.
    For judging one chart, use `paired_deltas` against that song's own reference.
    """
    reference = CORPUS_REFERENCE if reference is None else reference
    total = 0.0
    for size in range(1, 6):
        total += abs(profile.chord_hist.get(size, 0.0) - reference.get(f"chord_{size}", 0.0))
    return total / 2


def paired_deltas(generated: ChartProfile, reference: ChartProfile) -> dict[str, float]:
    """Generated minus the reference chart *of the same song*.

    This is the comparison that means something for a single chart: the reference controls
    for the song, so a difference is attributable to the model rather than to the music
    being unusually chord-heavy or unusually fast.
    """
    left, right = generated.as_flat(), reference.as_flat()
    deltas = {name: left.get(name, 0.0) - right.get(name, 0.0) for name in right}
    deltas["nps_variation"] = generated.nps_variation - reference.nps_variation
    deltas["mean_run_length"] = generated.mean_run_length - reference.mean_run_length
    return deltas


def scorecard(profile: ChartProfile, reference: dict[str, float] | None = None) -> str:
    """Human-readable comparison, for eyeballing one chart."""
    report = compare(profile, reference)
    lines = [
        f"positions {profile.positions}  span {profile.duration_seconds:.1f}s  "
        f"nps {profile.nps:.2f}  nps variation {profile.nps_variation:.2f}",
        f"runs: mean {profile.mean_run_length:.1f}, longest {profile.longest_run}",
        f"chord-size total variation from corpus: {total_variation(profile, reference):.4f}",
        "",
        f"{'metric':<14}{'actual':>10}{'corpus':>10}{'delta':>10}",
    ]
    for name, values in report.items():
        lines.append(
            f"{name:<14}{values['actual']:>10.4f}{values['reference']:>10.4f}"
            f"{values['delta']:>+10.4f}"
        )
    lines.append("")
    lines.append("inter-onset intervals: " + "  ".join(
        f"{name} {profile.ioi_hist.get(name, 0):.3f}" for name, _ in IOI_BUCKETS
    ))
    return "\n".join(lines)
