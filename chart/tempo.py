"""Tempo-map helpers: reuse the SyncTrack of an existing chart, or estimate one from audio.

The chart writer only ever emitted a single fixed BPM, so generated charts stayed
aligned to the audio but their beat grid meant nothing. Both entry points here
produce `bpm_events` in the same (tick, bpm * 1000) form that
`preprocess_bpm_segments` already consumes.
"""

import re

DEFAULT_BPM_RANGE = (100.0, 200.0)


def parse_sync_track(chart_path):
    """Lift Resolution and the [SyncTrack] events out of an existing .chart file.

    Returns (bpm_events, ts_events, resolution).
    """
    with open(chart_path, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()

    match = re.search(r"^\s*Resolution\s*=\s*(\d+)", text, re.M)
    if not match:
        raise ValueError(f"no Resolution in {chart_path}")
    resolution = int(match.group(1))

    block = re.search(r"\[SyncTrack\]\s*\{(.*?)\}", text, re.S)
    if not block:
        raise ValueError(f"no [SyncTrack] block in {chart_path}")

    bpm_events, ts_events = [], []
    for tick, kind, rest in re.findall(
        r"^\s*(\d+)\s*=\s*(B|TS)\s+([\d ]+)", block.group(1), re.M
    ):
        values = [int(v) for v in rest.split()]
        if kind == "B":
            bpm_events.append((int(tick), values[0]))
        else:
            ts_events.append((int(tick), tuple(values)))

    if not bpm_events:
        raise ValueError(f"[SyncTrack] in {chart_path} has no B events")

    return sorted(bpm_events), sorted(ts_events), resolution


def detect_tempo(y, sr, bpm_range=DEFAULT_BPM_RANGE, hop_length=256):
    """Estimate (bpm, phase, score) by a phase-locked search over the onset envelope.

    librosa.beat.beat_track reports the tempo of the strongest metrical level, which
    is routinely a half- or double-time octave error and is too imprecise to hold
    sync over a full song. Scoring a dense (bpm, phase) grid directly against onset
    strength is slower but lands within a few thousandths of a BPM.
    """
    import numpy as np
    import librosa

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.times_like(env, sr=sr, hop_length=hop_length)
    spread = env.std()
    env = (env - env.mean()) / (spread if spread else 1.0)
    end = times[-1]

    def score(bpm, phase):
        """Mean onset strength on the beat lines this (bpm, phase) implies.

        Sampled by interpolation rather than by nearest frame. The previous version used
        `np.searchsorted(times, beats)`, which returns the first index where
        `times[idx] >= beat` -- always the frame at or *after* the beat, never before. At
        hop_length 256 and 24 kHz a frame is 10.67 ms, so every beat was attributed to its
        frame's right edge and the estimated phase came out systematically late, about
        half a frame on average.

        It showed up in generated charts as a lead-in that was too long, which is the same
        thing seen from the other end: Robert found the first BPM event at 117.019 where
        125.554 lined up. Measured against the charts' own beat grids, the phase error was
        45 ms median.

        Interpolating removes both the bias and the 10.67 ms quantisation floor, and costs
        nothing -- np.interp is vectorised over the whole beat array.
        """
        beats = np.arange(phase, end, 60.0 / bpm)
        if beats.size == 0:
            return -np.inf
        return float(np.interp(beats, times, env).mean())

    def best_phase(bpm, step):
        phases = np.arange(0.0, 60.0 / bpm, step)
        scores = [score(bpm, p) for p in phases]
        i = int(np.argmax(scores))
        return float(phases[i]), scores[i]

    low, high = bpm_range
    best = (low, 0.0, -float("inf"))
    for bpm in np.arange(low, high, 0.25):
        phase, value = best_phase(bpm, 0.01)
        if value > best[2]:
            best = (float(bpm), phase, value)

    for bpm in np.arange(best[0] - 0.5, best[0] + 0.5, 0.002):
        phase, value = best_phase(bpm, 0.005)
        if value > best[2]:
            best = (float(bpm), phase, value)

    # Round to a tenth. The search reports things like 140.006, and that precision is
    # estimation noise rather than signal -- songs are recorded to a click, so the true
    # tempo is almost always a round number and a charter writes one. Rounding is
    # therefore more accurate, not less, and it keeps the beat grid on values a human
    # would have typed. The phase is re-fitted for the rounded value so the grid still
    # lands on the beat.
    rounded = round(best[0], 1)
    phase, value = best_phase(rounded, 0.002)
    return rounded, phase, value


FIRST_ONSET_FLOOR = 0.25
"""An onset weaker than this fraction of the strongest one is not the song starting."""


def first_onset(y, sr, hop_length=256, floor=FIRST_ONSET_FLOOR):
    """Time in seconds of the first real note in the audio.

    Anchoring a chart on the earliest detected onset is fragile: a fade-in, a room
    tone or a click before the count-in all register, and anchoring on one drags the
    entire chart out of sync. Requiring a fraction of the strongest onset in the
    track skips those. Backtracking then moves the answer from the envelope peak to
    the attack that produced it, which is where a charter would put the note.
    """
    import librosa

    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    peaks = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop_length, backtrack=False
    )
    if peaks.size == 0:
        return 0.0
    strong = peaks[env[peaks] >= floor * env[peaks].max()]
    if strong.size == 0:
        strong = peaks
    start = librosa.onset.onset_backtrack(strong[:1], env)
    return float(librosa.frames_to_time(start, sr=sr, hop_length=hop_length)[0])


def beat_aligned_tempo_events(bpm, phase, onset, resolution):
    """Tempo map putting a beat line exactly on the audio's first note.

    Robert's rule for syncing a generated chart, in his order: the first note lands
    on a beat line; that line carries a BPM event at the song's real tempo; and the
    tick-0 event -- the one Clone Hero starts every chart with -- is then whatever
    value makes the lead-in last exactly as long as the audio takes to reach that
    note.

    The anchor is the *tracked beat* nearest the first onset rather than the onset
    itself, so the grid stays locked to the music even when the first note is played
    slightly ahead of or behind the beat. Beat tracking never identifies downbeats,
    so a beat line is the strongest claim the analysis actually supports -- which is
    also what Robert specified.

    A song whose first note falls inside the opening beat gets no lead-in at all: one
    event at tick 0 at the real tempo, and the note sits at tick 0 where the human
    charter put it.

    Returns (bpm_events, anchor_tick, anchor_seconds).
    """
    period = 60.0 / bpm

    # Anchor on the first onset itself, not on the nearest beat the phase search found.
    #
    # Robert's rule is that the first note lands on a beat line. Taking the phase from the
    # search and then snapping the onset to it makes that true only as far as the search is
    # right, and measured against 24 charts' own beat grids the search is 51 ms out at the
    # median and 152 ms at p90. Using the onset makes the rule true by construction.
    #
    # It is also the better estimator on its own terms: 42.7 ms median and 96 ms p90
    # against the same ground truth, better on 16 of 24 songs. Both are still far worse
    # than the 10 ms grid, so the beat grid remains the weakest part of a generated chart
    # -- but the first onset is where a charter would put the note, and it is what the rule
    # asks for.
    anchor = onset if onset > 0 else phase + round((onset - phase) / period) * period

    # A song that starts immediately starts at tick 0.
    #
    # This used to read `while anchor < period: anchor += period`, on the grounds that a
    # chart needs a beat of lead-in. But `anchor` is *where the first note goes*, so
    # pushing it forward moved the first note a whole beat later than the audio's. On
    # LESBIAN VAMPIRE KILLERS the first lyric is at t=0 and the human chart opens with a
    # note at tick 0; ours opened at tick 480, a beat late, having skipped the opening
    # words. Robert, playing it: "the first note should've been at position 0 because
    # that is when the first vocal lyric was".
    #
    # When the first onset lands inside the opening beat, the honest tempo map is a
    # single event at tick 0 carrying the real tempo. There is nothing to stretch.
    if anchor < period / 2:
        return [(0, int(round(bpm * 1000)))], 0, 0.0

    beats = max(1, int(round(anchor / period)))
    lead_bpm = beats * 60.0 / anchor

    events = [
        (0, int(round(lead_bpm * 1000))),
        (beats * resolution, int(round(bpm * 1000))),
    ]
    return events, beats * resolution, anchor


SEGMENT_SECONDS = 20.0
"""How much audio each local tempo estimate sees.

Long enough for the phase-locked search to be stable -- at 100 BPM that is 33 beats --
and short enough to sit inside one section of a song. Shorter segments track a moving
tempo more closely and estimate each one worse, and below about 10 s the search starts
returning octave errors on sparse material.
"""

MERGE_BPM = 0.5
"""Two neighbouring segments within this many BPM are the same tempo.

detect_tempo rounds to a tenth, so anything under 0.5 is the estimator disagreeing with
itself rather than the song changing. Human charts carry a median of four tempo events;
emitting one per segment would produce dozens and claim precision the audio does not
support.
"""


def detect_tempo_map(y, sr, segment_seconds=SEGMENT_SECONDS, bpm_range=DEFAULT_BPM_RANGE,
                     hop_length=256):
    """Local tempo over the whole song, as [(start_seconds, bpm), ...].

    detect_tempo returns one number for the whole song, and measured against charts that
    carry a single [SyncTrack] event it is very accurate -- median error 0.000 BPM, 34 of
    36 within 1.0. But those charts are 23.2% of the corpus. The other 76.8% carry a
    tempo map, median four events, and for 190 of 307 of them a flat map has drifted past
    the 10 ms token grid by the last event. See measure_tempo_maps.py.

    Robert reported the symptom from the other end, playing the generated iCarly chart:
    "Leave it all to me is out of sync a small bit and the beat lines are out of place".
    We detected 104.900 BPM where the chart's first event says 128.226 -- not an octave
    error, just a song that does not have one tempo to find.

    This runs the same phase-locked search over consecutive segments and merges
    neighbours that agree, so a constant-tempo song still yields exactly one entry and
    costs only the extra searches.
    """
    import numpy as np

    duration = len(y) / float(sr)
    if duration <= segment_seconds * 1.5:
        bpm, _phase, _score = detect_tempo(y, sr, bpm_range, hop_length)
        return [(0.0, bpm)]

    segments = []
    start = 0.0
    while start < duration - segment_seconds / 2:
        stop = min(start + segment_seconds, duration)
        piece = y[int(start * sr):int(stop * sr)]
        if piece.size < sr:                       # under a second is not measurable
            break
        bpm, _phase, _score = detect_tempo(piece, sr, bpm_range, hop_length)
        segments.append((start, bpm))
        start = stop

    if not segments:
        bpm, _phase, _score = detect_tempo(y, sr, bpm_range, hop_length)
        return [(0.0, bpm)]

    merged = [segments[0]]
    for start, bpm in segments[1:]:
        if abs(bpm - merged[-1][1]) <= MERGE_BPM:
            continue
        merged.append((start, bpm))
    return merged


def beat_aligned_tempo_map(segments, onset, resolution):
    """Tick-exact tempo map from local tempi, keeping every event on a beat line.

    Extends beat_aligned_tempo_events, and reduces to it exactly when the song has one
    tempo -- the first segment is anchored by the same rule, so Robert's ordering still
    holds: the first note lands on a beat line, that line carries the real tempo, and the
    tick-0 event is whatever makes the lead-in last as long as the audio takes to get
    there.

    Every later tempo change is placed at a whole number of beats from the previous one.
    That is what keeps ticks exact -- a change at an arbitrary time would put the
    following notes on fractional ticks, and Robert's requirement is that they are not.
    It also matches how charts are written: a tempo change happens on a beat.

    The cost is that an event lands up to half a beat from where the audio changed. At
    150 BPM that is 200 ms of the change arriving early or late, against the seconds of
    accumulated drift a flat map produces; and a tempo estimated over a 20 s segment
    cannot locate its own onset better than that anyway.

    Returns (bpm_events, anchor_tick, anchor_seconds).
    """
    if not segments:
        raise ValueError("no tempo segments")

    first_bpm = segments[0][1]
    events, anchor_tick, anchor = beat_aligned_tempo_events(
        first_bpm, 0.0, onset, resolution)
    if len(segments) == 1:
        return events, anchor_tick, anchor

    events = list(events)
    tick = anchor_tick
    when = anchor
    current = first_bpm

    for start, bpm in segments[1:]:
        period = 60.0 / current
        beats = int(round((start - when) / period))
        if beats < 1:
            # The segment boundary is less than half a beat past the previous event.
            # Moving the tempo there would put two events on one tick; the later tempo
            # wins, because it was measured over more of the music around that point.
            if events and events[-1][0] == tick:
                events[-1] = (tick, int(round(bpm * 1000)))
                current = bpm
            continue
        tick += beats * resolution
        when += beats * period
        events.append((tick, int(round(bpm * 1000))))
        current = bpm

    return events, anchor_tick, anchor
