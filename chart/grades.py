"""How hard each catalogue pattern actually is, measured rather than assigned.

Robert's section-conditioning plan annotates patterns with difficulty grades. Hand
grading is slow and subjective, and it is the process that put castles in the
catalogue that do not exist. The corpus grades them instead: the same songs are
charted at four levels by the same charter, so where a pattern lands on that ladder is
its grade, from real charts rather than judgement.

A grade is the occurrence-weighted mean level, Easy=1 to Expert=4, over 165,792
occurrences in 150 songs charted at all four levels.

**Read grades against NULL_GRADE, not against 2.5.** Expert sections hold far more
notes than Easy ones -- the level shares here are {100*pooled[l]/total:.0f}/{100*pooled[l]/total:.0f}/{100*pooled[l]/total:.0f}/{100*pooled[l]/total:.0f}
-- so a pattern with no difficulty preference whatsoever still scores 2.76. Only
the distance from that null means anything, which is the same correction
`catalogue_lift` needs and for the same reason.

What it shows: most of the catalogue is universal vocabulary. Anchors, sweeps and
trills appear at every level and mark nothing. Thirteen patterns are genuine Expert
markers, led by the split zigs.

One result worth a charter's eye: `trip zig split G-Y-O` grades +1.02 above null while
`trip zig split G-B-O` grades -0.29 below it. Same named family, opposite ends of the
scale -- G-Y-O splits evenly (deltas 2, 2) and G-B-O does not (3, 1). Either split
regularity matters more than split width, or the 47 occurrences behind G-B-O are too
few to trust. Worth confirming before any of this drives generation.
"""

NULL_GRADE = 2.76
"""A pattern with no difficulty preference. Above is harder, below is easier."""

MIN_OCCURRENCES = 40
"""Below this a grade is noise; 67 patterns clear it."""

# name -> (grade, occurrences)
PATTERN_GRADES: dict[str, tuple[float, int]] = {
    'quad zig split G-R-B-O': (3.94, 104),
    'trip zig split G-Y-O': (3.77, 171),
    'roll quad gapped G-Y-B-O': (3.37, 79),
    'quad desc gapped O-B-R-G': (3.36, 714),
    'roll quint desc': (3.23, 78),
    'quad chimney G-R-Y-B, O peak': (3.19, 181),
    'ladder wide desc': (3.16, 118),
    'quad zig split G-Y-B-O': (3.16, 163),
    'raked trill 2-finger': (3.16, 157),
    'sweep 12-note': (3.11, 74),
    'trip chimney G-R-Y, B peak': (3.11, 555),
    'quad asc gapped G-Y-B-O': (3.08, 1137),
    'trip zig G-R-Y': (3.07, 1586),
    'reverse chimney quad': (3.03, 228),
    'quad desc gapped O-Y-R-G': (3.03, 1894),
    'quad chimney R-Y-B-O, O peak': (3.02, 489),
    'quad desc gapped O-B-Y-G': (3.01, 1876),
    'quad asc gapped G-R-Y-O': (3.00, 1611),
    'quad zig split G-R-Y-O': (3.00, 143),
    'trill full span O-G': (2.99, 511),
    'quad zig G-R-Y-B': (2.89, 923),
    'trill 2-split B-G': (2.89, 2564),
    'trip chimney G-R-B, Y peak': (2.86, 241),
    'trill adjacent R-G': (2.85, 13431),
    'trill 1-split Y-G': (2.85, 6133),
    # Was 'triangle slide 8-note'. That name held the identical sequence to
    # 'sweep 8-note' and has been removed from the catalogue as misfiled, so these
    # 595 occurrences belong to the surviving name -- they are occurrences of the
    # shape, and only ever got this label because it matched first.
    'sweep 8-note': (2.84, 595),
    'ladder triplet step desc': (2.83, 441),
    'quad asc gapped G-R-B-O': (2.81, 477),
    'triangle slide 6-note': (2.80, 3004),
    'trip zig split G-R-B': (2.80, 311),
    'shape O-G-O': (2.79, 724),
    'trip desc gapped O-Y-G': (2.78, 1658),
    'shape G-O-G': (2.77, 3092),
    'trip asc gapped G-B-O': (2.77, 1186),
    'quad desc contiguous O-B-Y-R': (2.76, 11091),
    'ladder triplet step asc': (2.76, 540),
    'trip desc gapped O-B-R': (2.75, 4883),
    'trip desc contiguous O-B-Y': (2.74, 17106),
    'quint zig G-R-Y-B-O': (2.73, 323),
    'trip chimney G-Y-B, O peak': (2.72, 64),
    'quint asc sweep': (2.72, 1215),
    'shape G-R-G': (2.72, 11285),
    'trip zig split G-Y-B': (2.71, 286),
    'quad asc contiguous G-R-Y-B': (2.71, 5428),
    'reverse chimney trip a': (2.71, 893),
    'trip asc gapped G-Y-B': (2.71, 4050),
    'roll quad desc B-Y-R-G': (2.70, 1256),
    'quad adjacent overlapping': (2.70, 745),
    'trip asc gapped G-R-B': (2.70, 4905),
    'trip desc gapped O-R-G': (2.70, 1179),
    'trip asc contiguous G-R-Y': (2.70, 22980),
    'quint chimney full neck': (2.69, 245),
    'quint desc sweep': (2.68, 2554),
    'shape R-G-R': (2.66, 4888),
    'shape G-Y-G': (2.66, 10682),
    'shape G-B-G': (2.66, 7114),
    'trip zig split G-R-O': (2.66, 64),
    'castle green anchor (TTFAF)': (2.64, 44),
    'reverse chimney quint to base': (2.61, 41),
    'shape Y-G-Y': (2.60, 1839),
    'roll quad asc G-R-Y-B': (2.59, 647),
    'sweep 6-note': (2.58, 495),
    'quad chimney G-R-B-O, Y peak': (2.57, 256),
    'ladder wide asc': (2.56, 143),
    'raked trill 3-finger': (2.53, 431),
    'trip zig split G-B-O': (2.47, 47),
    'shape B-G-B': (2.46, 1424),
}


def relative_grade(name: str) -> float | None:
    """Grade relative to NULL_GRADE. Positive means the pattern marks difficulty.

    Names here track the catalogue, so entries removed from it (the unattested castles,
    the raked chimneys that duplicate cakes) are gone, and the twenty "anchor X-Y-X"
    figures are "shape X-Y-X" now that they live in UNCONFIRMED_SHAPES.
    """
    entry = PATTERN_GRADES.get(name)
    return None if entry is None else entry[0] - NULL_GRADE


HUMAN_MARKER_SHARE = 0.027
HUMAN_MARKER_SHARE_SD = 0.038
"""How much of a human Expert chart's pattern vocabulary is Expert markers.

Measured over 80 Expert charts: mean 2.7%, median 1.4%. Real Expert charts are NOT
built from the hardest patterns -- their mean pattern grade is 2.753, sitting on the
null. They use the same universal vocabulary as every other level and arrange it
differently, which is what the difficulty ladder says too: run length quadruples and
the tap rate jumps while the shapes stay ordinary.

A generated chart measured 14.9%, 5.5x the human rate and +3.23 sd out. The model
reaches for split zigs constantly where a charter spends one as a moment of emphasis.
That is a distinct defect from the vocabulary collapse and the weak conditioning,
because it survives at the pattern level where those two do not.

Two claims that looked like the same defect did NOT survive their baselines, and are
recorded so they are not repeated: the generated chart's density is *more* variable
than 93% of human charts (nps variation 0.606 against a human mean 0.409), and its
235-note mean run is normal for Expert, whose own mean is 232. What does survive
alongside the marker overuse is phrasing. A human Expert chart breathes constantly:
34 rests a minute, typically 0.38s, a third of the song spent at rest, and only 1 of
120 charts has none. The generated chart rests 1.1 times a minute for a median 5.82s
-- 31x fewer, 15x longer, roughly the same total silence arranged completely
differently. It plays continuously and then stops dead. The long runs follow from
that rather than from long phrases.

Stated first as "no gap exceeds 250 ms anywhere, not rare, none", which was wrong:
there are about three, and they showed as 0.00 in a histogram printed to two decimals.
"""


def markers(threshold: float = 0.3) -> list[str]:
    """Patterns that genuinely indicate a harder chart, hardest first."""
    scored = [(grade - NULL_GRADE, name) for name, (grade, _) in PATTERN_GRADES.items()]
    return [name for delta, name in sorted(scored, reverse=True) if delta >= threshold]
