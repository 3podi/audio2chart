"""Bootstrap new patterns from the corpus using the ones already validated.

Robert's proposal: scan songs, find the patterns we have already confirmed, box those
regions off, and treat whatever is left between them as candidate new patterns -- then he
confirms or rejects the frequent ones, they join the vocabulary, and the loop repeats.

That is lexicon induction, and it works here because the known vocabulary already explains
a large share of the corpus: measured over 400 charts, the 124-entry catalogue claims 68.2%
of all fret motion. The remaining third is where new patterns live.

Two design choices worth stating:

*Longest-first matching.* A short shape would otherwise consume part of a longer one --
`(1, 1)` eats the front of every ascending run -- leaving shredded residue that looks like
noise. Claiming the longest known shapes first keeps the leftovers meaningful.

*A length floor on candidates.* Most 2-delta leftovers are transition notes, not patterns:
Robert's account is that a transition is whatever joins one pattern to the next, so it
belongs to the join and will always look like unexplained residue. Filtering to longer
fragments surfaces patterns; the short residue is itself the transition vocabulary, and is
worth mining separately.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from chart.catalogue import CATALOGUE, body_signature
from chart.patterns import single_note_runs

FRETS = "GRYBO"


@dataclass
class Discovery:
    """What the corpus contains that the known vocabulary does not explain."""

    deltas_seen: int = 0
    deltas_claimed: int = 0
    candidates: Counter = field(default_factory=Counter)
    charts_per_candidate: Counter = field(default_factory=Counter)

    @property
    def coverage(self) -> float:
        return self.deltas_claimed / self.deltas_seen if self.deltas_seen else 0.0


def known_signatures(catalogue: dict[str, str] | None = None) -> dict[tuple[int, ...], str]:
    """Catalogue bodies, keyed by signature. Bodies, so transitions do not block a match."""
    source = CATALOGUE if catalogue is None else catalogue
    known: dict[tuple[int, ...], str] = {}
    for name, text in source.items():
        sig = body_signature(text)
        if len(sig) >= 2:
            known.setdefault(sig, name)
    return known


def render(signature: tuple[int, ...]) -> str:
    """A signature as playable notes, placed wherever it fits on the neck."""
    for start in range(5):
        frets = [start]
        for delta in signature:
            frets.append(frets[-1] + delta)
        if all(0 <= f <= 4 for f in frets):
            return " ".join(FRETS[f] for f in frets)
    return "(does not fit on five frets)"


def claim_run(deltas: list[int], order: list[tuple[int, ...]]) -> list[bool]:
    """Mark which deltas a known pattern accounts for, longest patterns first."""
    claimed = [False] * len(deltas)
    for sig in order:
        width = len(sig)
        for start in range(len(deltas) - width + 1):
            if not any(claimed[start:start + width]) and tuple(deltas[start:start + width]) == sig:
                for index in range(start, start + width):
                    claimed[index] = True
    return claimed


def unexplained(claimed: list[bool], deltas: list[int], min_length: int) -> list[tuple[int, ...]]:
    """Contiguous stretches no known pattern claimed."""
    fragments, current = [], []
    for index, is_claimed in enumerate(claimed + [True]):
        if is_claimed:
            if len(current) >= min_length:
                fragments.append(tuple(current))
            current = []
        else:
            current.append(deltas[index])
    return fragments


def scan(entries, processor, tokenizer, min_length: int = 4,
         catalogue: dict[str, str] | None = None) -> Discovery:
    """Segment each chart with the known vocabulary and collect what is left over."""
    known = known_signatures(catalogue)
    order = sorted(known, key=len, reverse=True)
    result = Discovery()

    for entry in entries:
        try:
            processor.read_chart(entry["chart_path"], target_sections=entry["difficulty"])
            resolution = int(processor.song_metadata["Resolution"])
            offset = float(processor.song_metadata["Offset"])
            encoded = tokenizer.encode(processor.notes[entry["difficulty"]], resolution=resolution)
            timed = tokenizer.format_seconds(encoded, processor.synctrack, resolution, offset)
        except (KeyError, OSError, TypeError, ValueError):
            continue

        seen_here: set[tuple[int, ...]] = set()
        for run in single_note_runs(timed, tokenizer):
            frets = [fret for _, fret in run]
            deltas = [b - a for a, b in zip(frets, frets[1:])]
            if not deltas:
                continue
            result.deltas_seen += len(deltas)
            claimed = claim_run(deltas, order)
            result.deltas_claimed += sum(claimed)
            for fragment in unexplained(claimed, deltas, min_length):
                result.candidates[fragment] += 1
                seen_here.add(fragment)
        for fragment in seen_here:
            result.charts_per_candidate[fragment] += 1
    return result


def report(discovery: Discovery, top: int = 20) -> str:
    lines = [
        f"coverage {discovery.coverage:.1%} "
        f"({discovery.deltas_claimed}/{discovery.deltas_seen} fret motions claimed)",
        f"{sum(discovery.candidates.values())} candidate fragments, "
        f"{len(discovery.candidates)} distinct shapes",
        "",
        f"{'count':>7}{'charts':>8}  {'notes':<26}signature",
    ]
    for signature, count in discovery.candidates.most_common(top):
        charts = discovery.charts_per_candidate[signature]
        lines.append(f"{count:>7}{charts:>8}  {render(signature):<26}{signature}")
    return "\n".join(lines)


def bar_lift(observed_on_bar: int, instances: int, expected_rate: float) -> float:
    """How much more often a shape starts on a bar line than its own charts predict.

    This is the signal that separates an authored pattern from leftover residue, and it
    has to be measured against the charts the shape actually occurs in. Pooling across the
    corpus buries it: dense tapping charts put almost nothing on a bar line, so a global
    comparison reads 7.4% against a 7.3% baseline and looks like noise. Controlled per
    chart, known catalogue patterns run at a median 1.47x and up to 3.46x.

    Measured on discovery candidates:

        G G G G G G G G   2.97x   tremolo, real
        O Y B R Y G       2.63x   the catalogue's split ladder, written two notes too long
        G B R G O         0.09x
        G B R G Y         0.08x

    A shape that starts a bar twelve times *less* often than chance is not a pattern that
    happens to be unlisted -- it is the middle of a longer figure whose front the catalogue
    already claimed. Near-zero lift is the fragment signature.
    """
    if instances <= 0 or expected_rate <= 0:
        return 0.0
    return (observed_on_bar / instances) / expected_rate


FRAGMENT_LIFT = 0.5
"""Below this, a candidate is more likely residue than a pattern."""

PATTERN_LIFT = 1.5
"""Above this, a candidate behaves like the known catalogue entries."""


def classify(lift: float) -> str:
    if lift < FRAGMENT_LIFT:
        return "fragment"
    if lift >= PATTERN_LIFT:
        return "pattern-like"
    return "unclear"


def catalogue_lift(runs: list[list[int]], shuffles: int = 3,
                   catalogue: dict[str, str] | None = None) -> tuple[float, float]:
    """Catalogue coverage, and the coverage the same notes reach by chance.

    Raw coverage cannot be compared between charts. The catalogue holds two-delta
    shapes -- anchor is (1, -1), trip ascending is (1, 1) -- that random fret motion
    produces constantly, and how often depends on how a chart uses the neck: measured
    floors ranged from 13% to 35% across three charts. A chart that wanders the whole
    neck scores a high floor for free, one that camps on two frets scores a low one.

    Shuffling the frets within each run destroys every real pattern while preserving
    note count, run lengths and which frets appear, so the difference isolates authored
    structure. Measured over 60 human Expert charts the lift is mean +0.209, median
    +0.227, sd 0.167, quartiles +0.078 to +0.336.

    That spread is wide enough that a single chart says little on its own, which is the
    same caution the responsiveness metric needs. It was still decisive once: a
    generated chart read 52.8% against a human 69.1% pooled, which looks like half the
    vocabulary, but its floor was 29.8% and its lift +0.230 -- the human median, with
    31 of 60 real charts scoring lower.

    `runs` is a list of fret sequences, as `single_note_runs` produces them.

    Returns (coverage, chance_floor).
    """
    import random as _random
    import statistics

    order = sorted(known_signatures(catalogue), key=len, reverse=True)

    def cover(sequences: list[list[int]]) -> float | None:
        seen = claimed = 0
        for frets in sequences:
            deltas = [b - a for a, b in zip(frets, frets[1:])]
            if not deltas:
                continue
            seen += len(deltas)
            claimed += sum(claim_run(deltas, order))
        return claimed / seen if seen else None

    real = cover(runs)
    if real is None:
        return 0.0, 0.0

    floors = []
    for seed in range(shuffles):
        rng = _random.Random(seed)
        shuffled = []
        for frets in runs:
            copy = list(frets)
            rng.shuffle(copy)
            shuffled.append(copy)
        value = cover(shuffled)
        if value is not None:
            floors.append(value)
    return real, statistics.mean(floors) if floors else 0.0


HUMAN_LIFT_MEAN = 0.268
HUMAN_LIFT_SD = 0.170
"""Catalogue lift over 180 human Expert charts sampled from the whole corpus.

Was 0.209 / 0.167 over 60 charts. Two things moved it, and they were measured apart:

  stored, 60 charts, catalogue of 123 entries      +0.209  sd 0.167
  whole corpus, 180 charts, catalogue of 100       +0.268  sd 0.170
  tapping split, 180 charts, catalogue of 100      +0.399  sd 0.127

The first step (+0.059) is the catalogue rework alone -- castles cut to green, twenty
anchors demoted to UNCONFIRMED_SHAPES, cakes corrected, the skip family added. A lift
baseline is only meaningful against the catalogue it was measured with, so changing the
catalogue silently invalidated this number and nothing flagged it.

The second step (+0.131) is population. Note the sample here is the whole corpus, which
contains overcharts, so it is not the same definition as the original 60 "general"
charts; it is the right reference for a model trained on everything."""

TAPPING_LIFT_MEAN = 0.416
TAPPING_LIFT_SD = 0.127
"""Catalogue lift over 330 human charts from the tapping split, train and val pooled.

Tapping charts carry far more catalogue vocabulary than the corpus at large --
coverage around 0.68 against 0.489 -- and vary less doing it. Judging a
tapping-trained model against HUMAN_LIFT_MEAN would call a chart at +0.30 lift a
quarter of a standard deviation above human when it is nearly a full one below. That
is a reversed verdict, so use these whenever the model was trained on the tapping
subset.

Pooled deliberately. Measured separately, the two halves of the split disagree:

    train, 180 charts   +0.399  sd 0.127
    val,   150 charts   +0.436  sd 0.124

a gap of 0.037, about 2.7 sigma on those sample sizes. The split groups by song
rather than by pack, so both halves draw from one pool and this reads as sampling
rather than a real population difference -- but it is large enough that taking either
half alone would move a generated chart score by 0.3 sd. score_run draws its clips
from the val half, which is precisely the half that would have biased the comparison
had the train figure been used on its own."""
