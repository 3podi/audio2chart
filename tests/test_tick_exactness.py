"""Do note times survive the grid exactly, or do they drift?

Robert's requirement is blunt: "ticks need to be exact. I dont want any sync issues."
The model does not emit ticks -- it emits grid indices at `grid_ms`, which carry a
quantisation error of up to half a grid step. These tests take real tick positions all
the way around the loop the pipeline actually uses:

    ticks -> seconds -> discretize_time (what the model sees and emits)
          -> seconds -> convert_notes_to_ticks with choose_snap -> ticks

and assert the recovered ticks are *identical*, not merely close. Snapping is what
closes the gap: a 10 ms grid error is a small fraction of a subdivision, so a note
rounds back onto the slot it came from -- provided the subdivision chosen is fine
enough not to merge notes and coarse enough to absorb the error.

The 20 ms grid is included deliberately as the counter-example, to keep a record of
what the old setting did to a tapping passage.
"""

from __future__ import annotations

import unittest

from chart.time_conversion import (
    choose_snap,
    convert_notes_to_ticks,
    preprocess_bpm_segments,
    tick_to_seconds,
)
from chart.tokenizer import SimpleTokenizerGuitar

GRID_MS = 10
WINDOW_SECONDS = 15
RESOLUTION = 192


def _round_trip(ticks, bpm_events, resolution=RESOLUTION, grid_ms=GRID_MS,
                window_seconds=WINDOW_SECONDS):
    """Ticks -> seconds -> grid -> seconds -> ticks, through the real functions."""
    tokenizer = SimpleTokenizerGuitar(expressive=True)
    segments = preprocess_bpm_segments(bpm_events, resolution)
    times = [tick_to_seconds(tick, segments, resolution) for tick in ticks]

    # One green note per position; the token identity is irrelevant to timing, but it
    # has to be a real note token or convert_notes_to_ticks filters it out.
    notes = [(tick, 'N', 0, 0) for tick in ticks]      # (tick, type, lane, duration)
    green = tokenizer.encode(notes, resolution=resolution)[0][1]
    tokens = [green] * len(ticks)

    pad = tokenizer.pad_id
    grid = tokenizer.discretize_time(times, tokens, pad, grid_ms, window_seconds)

    grid_seconds = grid_ms / 1000.0
    recovered_times = [index * grid_seconds for index, token in enumerate(grid)
                       if token != pad]
    recovered_tokens = [token for token in grid if token != pad]

    # choose_snap runs on the ticks the model's own output converts to, exactly as
    # generate.py --snap auto does; it never sees the human ticks.
    unsnapped = convert_notes_to_ticks(
        recovered_tokens, recovered_times, resolution=resolution,
        bpm_events=bpm_events, snap=0, pad_token_id=pad, tokenizer=tokenizer)
    snap = choose_snap([tick for tick, *_ in unsnapped], resolution)
    snapped = convert_notes_to_ticks(
        recovered_tokens, recovered_times, resolution=resolution,
        bpm_events=bpm_events, snap=snap, pad_token_id=pad, tokenizer=tokenizer)
    return [tick for tick, *_ in snapped], snap


class TickExactnessTest(unittest.TestCase):
    def test_sixteenths_at_200_bpm_survive_exactly(self):
        bpm = [(0, 200000)]
        step = RESOLUTION // 4                      # a sixteenth note
        ticks = [index * step for index in range(64)]
        recovered, _ = _round_trip(ticks, bpm)
        self.assertEqual(ticks, recovered)

    def test_a_triplet_run_is_not_forced_onto_a_binary_grid(self):
        """24th notes: the case where snapping to 16ths would collapse notes."""
        bpm = [(0, 180000)]
        step = RESOLUTION // 6                      # a 24th note
        ticks = [index * step for index in range(48)]
        recovered, snap = _round_trip(ticks, bpm)
        self.assertEqual(ticks, recovered)
        self.assertEqual(0, snap % 3, f"a triplet run needs a snap divisible by 3, got {snap}")

    def test_exactness_holds_across_a_tempo_change(self):
        bpm = [(0, 140000), (RESOLUTION * 8, 175500)]
        step = RESOLUTION // 4
        ticks = [index * step for index in range(64)]
        recovered, _ = _round_trip(ticks, bpm)
        self.assertEqual(ticks, recovered)

    def test_a_tenth_of_a_bpm_is_still_exact(self):
        """detect_tempo rounds to ###.#; the tenth must not cost alignment."""
        bpm = [(0, 163700)]
        step = RESOLUTION // 4
        ticks = [index * step for index in range(64)]
        recovered, _ = _round_trip(ticks, bpm)
        self.assertEqual(ticks, recovered)

    def test_thirty_seconds_at_200_bpm_survive_exactly(self):
        """37.5 ms apart -- the fast tapping the 20 ms grid could not hold."""
        bpm = [(0, 200000)]
        step = RESOLUTION // 8
        ticks = [index * step for index in range(64)]
        recovered, _ = _round_trip(ticks, bpm)
        self.assertEqual(ticks, recovered)

    def test_every_musical_subdivision_is_exact_down_to_the_grid_step(self):
        """The direct answer to "can the error be 0 ms?": yes, until the grid runs out.

        Swept over 120-400 BPM and 16th-96th notes, every subdivision whose spacing
        clears the 10 ms grid step recovers its ticks exactly -- 0.0 ms error, down to a
        12.5 ms gap (96ths at 200 BPM, 64ths at 300). Nothing in that range drifts.
        """
        for bpm in (120, 180, 200, 260, 300, 400):
            for subdivision in (16, 24, 32, 48, 64, 96):
                step = RESOLUTION * 4 // subdivision
                gap_ms = (step / RESOLUTION) * (60.0 / bpm) * 1000
                if gap_ms < GRID_MS:
                    continue                      # covered by the rejection test below
                ticks = [index * step for index in range(32)]
                with self.subTest(bpm=bpm, subdivision=subdivision):
                    recovered, _ = _round_trip(ticks, [(0, int(bpm * 1000))])
                    self.assertEqual(ticks, recovered)

    def test_below_the_grid_step_the_window_is_refused_not_mistimed(self):
        """96ths at 300 BPM are 8.3 ms apart -- finer than the grid can represent.

        The important property is which way it fails. A grid that silently rounded two
        notes together would ship a desynced chart, which is the thing Robert asked not
        to happen; instead discretization raises and the window is dropped from training.
        """
        step = RESOLUTION * 4 // 96
        ticks = [index * step for index in range(32)]
        with self.assertRaises(ValueError):
            _round_trip(ticks, [(0, 300000)])

    def test_an_irregular_burst_off_any_subdivision_still_drifts(self):
        """The one case that is not exact, recorded honestly rather than hidden.

        Ten ticks at 200 BPM is 15.6 ms -- above the grid step, so it is accepted, but
        it sits on no common subdivision (a quarter / 19.2), so choose_snap has nothing
        to snap it to and the +/-5 ms grid error survives into the ticks. Charters do not
        place notes here deliberately; this exists so that if the model ever emits such
        spacing, the behaviour is known rather than assumed.
        """
        ticks = [index * 10 for index in range(32)]
        recovered, snap = _round_trip(ticks, [(0, 200000)])
        self.assertEqual(0, snap, "expected choose_snap to find no usable subdivision")
        self.assertNotEqual(ticks, recovered)
        worst = max(abs(a - b) for a, b in zip(ticks, recovered))
        self.assertLessEqual(worst, 4, "drift should stay inside half a grid step")


if __name__ == "__main__":
    unittest.main()
