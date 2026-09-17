"""A song that changes tempo needs a map, and the map has to keep ticks exact.

detect_tempo returns one number. Measured against charts carrying a single [SyncTrack]
event it is very accurate, but those are 23.2% of the corpus; the other 76.8% carry a
map, and for 190 of 307 of them a flat tempo has drifted past the 10 ms token grid by the
last event (measure_tempo_maps.py). Robert played one of them: "Leave it all to me is out
of sync a small bit and the beat lines are out of place".

Robert's standing requirement is that ticks are exact and there are no sync issues, so
these check the arithmetic rather than the estimator: every tempo change lands on a whole
beat, every event is on an integer tick, and a constant-tempo song is unchanged.
"""

import unittest

from chart.tempo import beat_aligned_tempo_events, beat_aligned_tempo_map
from chart.time_conversion import preprocess_bpm_segments, seconds_to_tick

RESOLUTION = 192


def tick_of(seconds, events, resolution=RESOLUTION):
    segments = preprocess_bpm_segments(events, resolution)
    return seconds_to_tick(seconds, segments, resolution, [s[2] for s in segments])


class TempoMapTests(unittest.TestCase):

    def test_one_tempo_is_exactly_what_the_old_path_produced(self):
        """A constant-tempo song must not change at all, or every chart moves."""
        for bpm, onset in ((120.0, 2.05), (174.0, 1.02), (91.5, 7.90), (200.0, 0.01)):
            old = beat_aligned_tempo_events(bpm, 0.0, onset, RESOLUTION)
            new = beat_aligned_tempo_map([(0.0, bpm)], onset, RESOLUTION)
            self.assertEqual(old, new, f"{bpm} BPM at {onset}s")

    def test_every_event_lands_on_a_beat_line(self):
        """The whole point of placing changes on beats is that ticks stay integral."""
        segments = [(0.0, 120.0), (20.0, 140.0), (40.0, 90.0), (61.5, 120.0)]
        events, anchor_tick, _ = beat_aligned_tempo_map(segments, 2.05, RESOLUTION)
        for tick, _mbpm in events:
            self.assertEqual(tick % RESOLUTION, 0, f"tick {tick} is not on a beat")
            self.assertIsInstance(tick, int)

    def test_a_tempo_change_is_produced_for_each_distinct_segment(self):
        segments = [(0.0, 120.0), (20.0, 140.0), (40.0, 90.0)]
        events, _, _ = beat_aligned_tempo_map(segments, 2.05, RESOLUTION)
        # A lead-in event at tick 0, the anchor, then one per later segment.
        self.assertEqual([mbpm for _tick, mbpm in events][-3:],
                         [120000, 140000, 90000])

    def test_ticks_are_strictly_increasing(self):
        """Two events on one tick is a malformed chart, not a slow one."""
        segments = [(0.0, 120.0), (20.0, 140.0), (40.0, 90.0), (61.5, 120.0)]
        events, _, _ = beat_aligned_tempo_map(segments, 2.05, RESOLUTION)
        ticks = [tick for tick, _ in events]
        self.assertEqual(ticks, sorted(set(ticks)))

    def test_a_change_closer_than_half_a_beat_replaces_rather_than_duplicates(self):
        """Rounding to the nearest beat can land two segments on the same tick."""
        segments = [(0.0, 120.0), (2.05, 140.0)]
        events, anchor_tick, _ = beat_aligned_tempo_map(segments, 2.05, RESOLUTION)
        ticks = [tick for tick, _ in events]
        self.assertEqual(ticks, sorted(set(ticks)))
        self.assertEqual(events[-1], (anchor_tick, 140000))

    def test_the_map_tracks_a_changing_tempo_where_a_flat_one_drifts(self):
        """The failure being fixed, measured rather than asserted.

        A song at 120 BPM for 32 beats and then 90 BPM: by the end of the slow part, a
        flat 120 map puts that moment seconds away from where it is in the audio.
        """
        fast_beats, slow_beats = 32, 32
        switch = fast_beats * 60.0 / 120.0
        end = switch + slow_beats * 60.0 / 90.0

        mapped, _, _ = beat_aligned_tempo_map(
            [(0.0, 120.0), (switch, 90.0)], 0.0, RESOLUTION)
        flat, _, _ = beat_aligned_tempo_map([(0.0, 120.0)], 0.0, RESOLUTION)

        expected = (fast_beats + slow_beats) * RESOLUTION
        self.assertEqual(tick_of(end, mapped), expected)
        self.assertGreater(abs(tick_of(end, flat) - expected), RESOLUTION * 8)

    def test_an_empty_map_is_an_error_rather_than_a_silent_default(self):
        with self.assertRaises(ValueError):
            beat_aligned_tempo_map([], 0.0, RESOLUTION)


if __name__ == "__main__":
    unittest.main()
