"""A chart that follows the music has to score above one that does not -- otherwise
the metric cannot support the claim it exists to make."""

import numpy as np

from chart.responsiveness import note_density, responsiveness


def test_a_chart_tracking_the_music_beats_one_that_ignores_it():
    rng = np.random.default_rng(0)
    envelope = np.abs(rng.normal(size=200)) + np.sin(np.arange(200) / 10) + 2
    following = envelope * 3 + rng.normal(scale=0.2, size=200)
    ignoring = np.abs(rng.normal(size=200)) + 2
    assert responsiveness(following, envelope) > 0.9
    assert abs(responsiveness(ignoring, envelope)) < 0.3


def test_constant_density_is_zero_not_an_error():
    """The mechanical failure mode: notes at a fixed rate regardless of the music."""
    envelope = np.sin(np.arange(100) / 5) + 2
    assert responsiveness(np.full(100, 7.0), envelope) == 0.0


def test_density_counts_notes_into_bins():
    density = note_density([0.1, 0.9, 1.5, 3.2], duration=4.0, bin_seconds=1.0)
    assert list(density) == [2.0, 1.0, 0.0, 1.0]


def test_notes_past_the_end_are_dropped():
    assert list(note_density([0.5, 99.0], duration=2.0, bin_seconds=1.0)) == [1.0, 0.0]


def test_a_handful_of_songs_cannot_resolve_a_small_effect():
    """The trap this guards: three songs looked like enough, and is not."""
    from chart.responsiveness import minimum_songs
    assert minimum_songs(0.10) == 33
    assert minimum_songs(0.05) > 100
    assert minimum_songs(0.40) < 5


def test_separating_generated_from_human_needs_a_large_sample():
    """Measured human-versus-generated gap is +0.081, less than half the
    human-versus-stranger effect the metric was validated on."""
    from chart.responsiveness import minimum_songs
    assert minimum_songs(0.081) > 45
