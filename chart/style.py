"""Chart styles expressed as constraints on the token vocabulary.

Several charting styles are not vague aesthetics but hard properties of individual notes,
and the expressive vocabulary puts those properties in known bit positions:

    token = (chord_index * N_FLAG + flag) * N_SUSTAIN + sustain_bucket
    flag bit0 = forced (HOPO), bit1 = tap

A Wii chart is defined by "every note is a tap, no strums". That is not something a model
has to be taught -- it is a subset of the vocabulary. Masking the rest to -inf at sampling
time makes the constraint impossible to violate, on any checkpoint, with no retraining.

This module is deliberately torch-free: it answers "which token ids satisfy this style",
and the caller turns that into a mask.

Styles that are distributional rather than absolute -- technical, variety-skillset -- are
NOT expressible here and need the pattern control track instead. See
docs/section-conditioning.md.
"""

from __future__ import annotations

from dataclasses import dataclass

OPEN_LANE = 7


@dataclass(frozen=True)
class StyleConstraint:
    """Which notes a style permits. Every field narrows; defaults permit everything."""

    taps_only: bool = False
    """Wii charts: every note carries the tap flag."""

    forbid_taps: bool = False
    """The inverse -- strum/HOPO only."""

    min_lanes: int = 1
    """Chording charts: require at least this many frets pressed at once."""

    max_lanes: int = 5
    """1 restricts to single notes."""

    max_fret_span: int | None = None
    """One-hand charts: the pressed frets must lie within this many frets of each other."""

    allow_open: bool = True
    """Whether open notes (lane 7) may be used."""

    forbid_sustains: bool = False
    """Tapping styles often want everything staccato."""

    def __post_init__(self) -> None:
        if self.taps_only and self.forbid_taps:
            raise ValueError("taps_only and forbid_taps are mutually exclusive")
        if self.min_lanes > self.max_lanes:
            raise ValueError(f"min_lanes {self.min_lanes} exceeds max_lanes {self.max_lanes}")
        if self.max_fret_span is not None and self.max_fret_span < 0:
            raise ValueError("max_fret_span must be non-negative")


# Named styles. Only the ones expressible as per-note constraints belong here.
PRESETS: dict[str, StyleConstraint] = {
    # "all the notes are tap notes, absolutely no strum notes"
    "wii": StyleConstraint(taps_only=True),
    "chording": StyleConstraint(min_lanes=2),
    "single_notes": StyleConstraint(max_lanes=1),
    # A hand covering three adjacent frets without moving.
    "one_hand": StyleConstraint(max_fret_span=2, max_lanes=3),
    "no_sustains": StyleConstraint(forbid_sustains=True),
    "no_opens": StyleConstraint(allow_open=False),
}


def _lane_ok(lanes: tuple[int, ...], constraint: StyleConstraint) -> bool:
    if not constraint.allow_open and OPEN_LANE in lanes:
        return False
    frets = [lane for lane in lanes if lane != OPEN_LANE]
    # An open note on its own presses no frets; judge it only by allow_open.
    if not frets:
        return constraint.min_lanes <= 1
    if not constraint.min_lanes <= len(frets) <= constraint.max_lanes:
        return False
    if constraint.max_fret_span is not None:
        if max(frets) - min(frets) > constraint.max_fret_span:
            return False
    return True


def allowed_note_tokens(tokenizer, constraint: StyleConstraint) -> list[int]:
    """Note token ids this style permits, ascending. Excludes bos/eos/pad."""
    if not tokenizer.expressive:
        raise ValueError("Style constraints need the expressive vocabulary")
    allowed: list[int] = []
    for chord_index, lanes in tokenizer.reverse_chord.items():
        if not _lane_ok(lanes, constraint):
            continue
        for flag in range(tokenizer.N_FLAG):
            is_tap = bool(flag & 2)
            if constraint.taps_only and not is_tap:
                continue
            if constraint.forbid_taps and is_tap:
                continue
            for sustain in range(tokenizer.N_SUSTAIN):
                if constraint.forbid_sustains and sustain:
                    continue
                allowed.append(tokenizer.compose(chord_index, flag, sustain))
    return sorted(allowed)


def allowed_tokens(tokenizer, constraint: StyleConstraint) -> list[int]:
    """Everything the sampler may emit, including the specials it needs.

    `pad` is always permitted and that is not a detail: the grid is ~90% padding, and a
    mask that forbids it would force a note into every 20 ms slot -- roughly 3,000 notes a
    minute, which is not a style, it is a wall.
    """
    tokens = allowed_note_tokens(tokenizer, constraint)
    if not tokens:
        raise ValueError("Style constraint permits no notes at all")
    return sorted(tokens + [tokenizer.pad_id, tokenizer.eos_id])


def resolve(name_or_constraint: str | StyleConstraint) -> StyleConstraint:
    if isinstance(name_or_constraint, StyleConstraint):
        return name_or_constraint
    try:
        return PRESETS[name_or_constraint]
    except KeyError:
        raise ValueError(
            f"Unknown style {name_or_constraint!r}; known: {sorted(PRESETS)}"
        ) from None


def describe(tokenizer, constraint: StyleConstraint) -> str:
    """One line for logs, so a run records what it was constrained to."""
    permitted = len(allowed_note_tokens(tokenizer, constraint))
    total = tokenizer.n_notes
    return f"{permitted}/{total} note tokens permitted ({100 * permitted / total:.1f}%)"
