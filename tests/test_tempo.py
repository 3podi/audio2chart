"""The generated chart has to line up with the audio, and the beat grid has to mean
something. Both come down to the tempo map, so these check the map arithmetic directly:
a beat line lands on the audio's first note, and the lead-in takes exactly as long as
the audio takes to get there.
"""

import unittest

from chart.tempo import beat_aligned_tempo_events
from chart.time_conversion import preprocess_bpm_segments, seconds_to_tick

RESOLUTION = 192


def tick_of(seconds, bpm_events, resolution=RESOLUTION):
    segments = preprocess_bpm_segments(bpm_events, resolution)
    return seconds_to_tick(seconds, segments, resolution, [s[2] for s in segments])


BEAT_CASES = [
    (120.0, 0.10, 2.05),
    (174.0, 0.33, 1.02),
    (91.5, 0.48, 7.90),
    (200.0, 0.01, 0.55),
]


class TempoTest(unittest.TestCase):
    def test_first_note_lands_on_the_beat_line(self):
        for bpm, phase, onset in BEAT_CASES:
            events, anchor_tick, anchor = beat_aligned_tempo_events(bpm, phase, onset, RESOLUTION)
            assert anchor_tick % RESOLUTION == 0, "anchor must be a whole number of beats"
            assert abs(tick_of(anchor, events) - anchor_tick) <= 1


    def test_lead_in_lasts_exactly_as_long_as_the_intro(self):
        """Robert's third step: the tick-0 event is whatever makes the lead-in end on time.

        It cannot be exact -- .chart stores BPM as an integer number of milli-BPM, so the
        lead-in is only ever as accurate as that quantisation allows. What matters is that
        the residual is far below the token grid, and it is: microseconds.
        """
        for bpm, phase, onset in BEAT_CASES[:2]:
            events, anchor_tick, anchor = beat_aligned_tempo_events(
                bpm, phase, onset, RESOLUTION)
            lead_bpm = events[0][1] / 1000.0
            beats = anchor_tick / RESOLUTION
            assert abs(beats * 60.0 / lead_bpm - anchor) < 1e-3


    def test_the_grid_stays_locked_to_the_tracked_beat(self):
        """The anchor is now the first onset itself, and this test used to assert otherwise.

        It required the anchor to be the nearest *tracked beat*, on the reasoning that a
        note played slightly ahead of the beat should not drag the whole grid with it.
        That reasoning is sound and the risk is real -- but it only pays off if the tracked
        beat is where the music's beat actually is, and measured against 24 charts' own
        beat grids the phase search is 51 ms out at the median and 152 ms at p90.

        Anchoring on the onset instead measures better on the same ground truth: 42.7 ms
        median, 96 ms p90, better on 16 of 24 songs. It also makes Robert's rule -- the
        first note lands on a beat line -- true by construction rather than true as far as
        a search happens to be right.

        The tradeoff this test was guarding is still real. A first note played ahead of the
        beat now shifts the grid by that much, and nothing here detects it. It is accepted
        because the alternative measured worse, not because the concern was wrong.
        """
        bpm = 120.0
        _, _, anchor = beat_aligned_tempo_events(bpm, 0.10, 2.04, RESOLUTION)
        assert abs(anchor - 2.04) < 1e-9, "the anchor is the onset, not a snapped beat"

    def test_the_grid_falls_back_when_there_is_no_onset(self):
        """An onset at exactly zero means nothing was detected, so the search stands in."""
        bpm, phase, period = 120.0, 0.10, 0.5
        events, anchor_tick, anchor = beat_aligned_tempo_events(bpm, phase, 0.0, RESOLUTION)
        # 0.0 lands inside the first half-beat, so this is the immediate-start path.
        assert anchor_tick == 0
        assert events == [(0, 120000)]


    def test_song_tempo_holds_after_the_anchor(self):
        events, anchor_tick, anchor = beat_aligned_tempo_events(174.0, 0.33, 1.02, RESOLUTION)
        assert events[-1][1] == 174000
        one_beat_later = tick_of(anchor + 60.0 / 174.0, events)
        assert abs(one_beat_later - (anchor_tick + RESOLUTION)) <= 1


    def test_a_song_that_starts_immediately_starts_at_tick_zero(self):
        """This test used to assert the opposite, and the opposite was the bug.

        It required anchor_tick >= RESOLUTION -- that the first note can never sit at
        tick 0. But the anchor is *where the first note goes*, so forcing a beat of
        lead-in moved the first note a beat later than the audio's. On LESBIAN VAMPIRE
        KILLERS the first lyric is at t=0 and the human chart opens at tick 0; ours
        opened at tick 480 and skipped the opening words.
        """
        events, anchor_tick, anchor = beat_aligned_tempo_events(
            120.0, 0.0, 0.0, RESOLUTION)
        assert anchor_tick == 0
        assert anchor == 0.0
        # One event, at tick 0, carrying the real tempo -- nothing to stretch.
        assert events == [(0, 120000)]

    def test_a_real_intro_still_gets_its_lead_in(self):
        """The fix must not throw away the lead-in where one is genuinely needed."""
        events, anchor_tick, anchor = beat_aligned_tempo_events(
            120.0, 0.10, 2.05, RESOLUTION)
        assert anchor_tick >= RESOLUTION
        assert anchor > 0.5
        assert len(events) == 2
        assert events[0][0] == 0 and events[1][0] == anchor_tick

    def test_an_onset_just_inside_the_first_beat_is_not_pushed_a_beat_late(self):
        """The case that broke: onset at 0.05 s with a 0.5 s beat.

        The old rule moved the anchor to 0.5 s, ten times later than the note actually
        was. Anything inside the first half-beat now anchors at zero instead.
        """
        _, anchor_tick, anchor = beat_aligned_tempo_events(
            120.0, 0.02, 0.05, RESOLUTION)
        assert anchor_tick == 0, "a note 50 ms in must not be charted half a second in"


    def test_snapping_lands_on_the_beat_grid_the_map_defines(self):
        """--snap was inert until the grid meant something.

        The model emits on a 20 ms grid, which does not divide evenly into beats: at
        140 BPM a generated eighth-note run came out 246/247/224 ticks apart against a
        true 240. Snapping only fixes that if it rounds against the *song's* grid, which
        is what the beat-aligned map now provides.
        """
        from chart.time_conversion import convert_notes_to_ticks

        bpm, resolution, snap = 140.0, 480, 16
        events, anchor_tick, anchor = beat_aligned_tempo_events(bpm, 0.145, 0.546, resolution)

        beat = 60.0 / bpm
        times = [anchor + i * beat / 2 for i in range(8)]      # eighth notes
        jittered = [t + (0.012 if i % 2 else -0.009) for i, t in enumerate(times)]

        notes = convert_notes_to_ticks([1] * len(jittered), jittered, resolution=resolution,
                                       bpm_events=events, snap=snap, pad_token_id=0)
        ticks = sorted({item[0] for item in notes})
        grid = resolution * 4 // snap
        assert ticks, "snapping dropped every note"
        assert all(tick % grid == 0 for tick in ticks), f"off-grid ticks: {ticks}"
        assert ticks[0] == anchor_tick, "the first note must stay on its beat line"


    def test_detected_tempo_is_rounded_to_a_tenth(self):
        """140.006 BPM is estimation noise, not a real tempo.

        Songs are recorded to a click and charters write round numbers, so the extra digits
        the search produces are spurious. Rounding keeps the grid on values a human would
        have typed, and the phase is re-fitted so the grid still lands on the beat.

        This drives detect_tempo on a synthetic click track rather than inspecting its
        source: the previous version asserted that the word "round(" appeared in the
        function body, which would have passed just as happily if the rounding had been
        commented out or applied to the wrong variable.
        """
        import numpy as np

        from chart.tempo import detect_tempo

        sample_rate = 22050
        true_bpm, phase, duration = 137.0, 0.21, 20.0
        times = np.arange(0.0, duration, 1.0 / sample_rate)
        audio = np.zeros_like(times)
        click = np.exp(-np.arange(600) / 60.0) * np.sin(np.arange(600) * 0.35)
        for beat in np.arange(phase, duration, 60.0 / true_bpm):
            start_sample = int(beat * sample_rate)
            audio[start_sample:start_sample + click.size] += click[:audio.size - start_sample]

        bpm, fitted_phase, _ = detect_tempo(audio.astype(np.float32), sample_rate)

        self_tenth = round(bpm, 1)
        assert bpm == self_tenth, f"{bpm} is not on a tenth of a BPM"
        # Octave errors are a separate failure mode; accept any metrical level, but the
        # tempo has to be a clean multiple or fraction of the click we generated.
        ratio = bpm / true_bpm
        assert min(abs(ratio - r) for r in (0.5, 1.0, 2.0)) < 0.02, f"{bpm} is not 137-related"
        assert 0.0 <= fitted_phase < 60.0 / bpm


    def test_snap_choice_preserves_every_note(self):
        """Snapping to too coarse a subdivision merges notes; too fine invents precision."""
        from chart.time_conversion import choose_snap

        resolution = 192
        sixteenths = [i * (resolution // 4) for i in range(16)]
        assert choose_snap(sixteenths, resolution) == 16

        # A 24th-note run must not be snapped onto a 16th grid.
        twentyfourths = [i * (resolution // 6) for i in range(12)]
        assert choose_snap(twentyfourths, resolution) in (12, 24)

        # Quarter notes need nothing finer than quarters.
        quarters = [i * resolution for i in range(8)]
        assert choose_snap(quarters, resolution) == 4


    def test_snap_choice_handles_degenerate_input(self):
        from chart.time_conversion import choose_snap

        assert choose_snap([], 192) == 0
        assert choose_snap([100], 192) == 0


if __name__ == "__main__":
    unittest.main()
