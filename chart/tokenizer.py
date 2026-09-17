import itertools
import math
from chart.time_conversion import convert_notes_to_seconds


class SimpleTokenizerGuitar():
    """Chart tokenizer.

    Legacy behaviour (expressive=False) emits one token per distinct pressed-lane
    combination: 32 tokens that say *which* frets are hit and nothing else.

    Expressive mode (the default) folds two things the legacy vocabulary threw away
    into the same token: the forced/tap flags and a bucketed sustain length. Measured
    over a 3000-chart sample of the target corpus, 47.1% of note positions carry a tap
    flag, 5.9% a forced flag and 4.5% a sustain, so the legacy vocabulary was discarding
    roughly half of every chart.

        token = (chord_index * N_FLAG + flag) * N_SUSTAIN + sustain_bucket

    chord_index  0..31   pressed-lane combination (unchanged from legacy)
    flag         0..3    bit0 = forced (lane 5), bit1 = tap (lane 6)
    sustain      0..9    0 = none, 1..9 = SUSTAIN_BEATS[i-1] beats

    Laid out this way, legacy token c corresponds to new token c * N_FLAG * N_SUSTAIN,
    so pretrained embedding rows can be broadcast onto the expanded vocabulary.
    """

    # Sustain buckets in beats, chosen from the corpus histogram: 2 beats (16.3% of all
    # sustains), 3/4 (13.5%), 3/8 (7.3%), 1/2 (4.6%) and 1 (2.9%) dominate, with a long
    # tail of rounding artefacts from mixed chart resolutions that these absorb.
    SUSTAIN_BEATS = (0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
    N_SUSTAIN = len(SUSTAIN_BEATS) + 1
    N_FLAG = 4
    MIN_SUSTAIN_BEATS = 0.0625

    def __init__(self, exclude_open_chords=True, expressive=True):

        self.exclude_open_chords = exclude_open_chords
        self.expressive = expressive
        # Define a mapping between pressed lanes (5 fret + open) and note index
        # If exclude_open_chords is True, exclude every chord with 7 except 7 alone

        indices = [i for i in range(5)]
        if not exclude_open_chords:
            indices = indices + [7]
        all_combinations = []
        for r in range(1, len(indices) + 1):
            combos = list(itertools.combinations(indices, r))
            all_combinations.extend(combos)

        self.chord_map = {v: idx for idx, v in enumerate(all_combinations)}
        if self.exclude_open_chords:
            self.chord_map[(7,)] = len(all_combinations)
        self.reverse_chord = {v: k for k, v in self.chord_map.items()}
        self.n_chords = len(self.chord_map)

        if self.expressive:
            self.n_notes = self.n_chords * self.N_FLAG * self.N_SUSTAIN
            self.mapping_noteseqs2int = {
                (lanes, flag, sus): self.compose(chord, flag, sus)
                for lanes, chord in self.chord_map.items()
                for flag in range(self.N_FLAG)
                for sus in range(self.N_SUSTAIN)
            }
        else:
            self.n_notes = self.n_chords
            self.mapping_noteseqs2int = dict(self.chord_map)

        self.reverse_map = {v: k for k, v in self.mapping_noteseqs2int.items()}

        self._log_buckets = [math.log(b) for b in self.SUSTAIN_BEATS]

    # ------------------------------------------------------------------ vocabulary

    @property
    def bos_id(self):
        return self.n_notes

    @property
    def eos_id(self):
        return self.n_notes + 1

    @property
    def pad_id(self):
        return self.n_notes + 2

    @property
    def vocab_size(self):
        return self.n_notes + 3

    def compose(self, chord, flag=0, sustain=0):
        if not self.expressive:
            return chord
        return (chord * self.N_FLAG + flag) * self.N_SUSTAIN + sustain

    def split(self, token):
        """token -> (chord_index, flag, sustain_bucket)."""
        if not self.expressive:
            return token, 0, 0
        chord, rest = divmod(token, self.N_FLAG * self.N_SUSTAIN)
        flag, sustain = divmod(rest, self.N_SUSTAIN)
        return chord, flag, sustain

    def is_note_token(self, token):
        return 0 <= token < self.n_notes

    def lanes_of(self, token):
        return self.reverse_chord[self.split(token)[0]]

    def attrs_of(self, token):
        flag = self.split(token)[1]
        return {'is5': bool(flag & 1), 'is6': bool(flag & 2), 'isS': False}

    def sustain_beats(self, token):
        bucket = self.split(token)[2]
        return self.SUSTAIN_BEATS[bucket - 1] if bucket else 0.0

    def sustain_ticks(self, token, resolution):
        return int(round(self.sustain_beats(token) * resolution))

    def legacy_to_expressive(self, legacy_token):
        """Map a legacy 32-token id onto its expressive equivalent (no flags, no sustain)."""
        return self.compose(legacy_token, 0, 0)

    def _sustain_bucket(self, duration, resolution):
        if not self.expressive or not resolution or not duration or duration <= 0:
            return 0
        beats = duration / resolution
        if beats < self.MIN_SUSTAIN_BEATS:
            return 0
        target = math.log(beats)
        best = min(range(len(self._log_buckets)),
                   key=lambda i: abs(self._log_buckets[i] - target))
        return best + 1

    # ------------------------------------------------------------------ encoding

    def _normalize_lanes(self, lanes):
        lanes = sorted(set(lanes))
        if self.exclude_open_chords and len(lanes) > 1 and lanes[-1] == 7:
            lanes.pop()
        return lanes

    def _append_encoded_group(self, output, tick, lanes, duration, is5, is6, is_power,
                              resolution=None):
        lanes = self._normalize_lanes(lanes)
        if not lanes:
            return
        chord = self.chord_map.get(tuple(lanes))
        if chord is None:
            raise ValueError(f"Unknown note sequence {lanes} at tick {tick}")
        flag = (1 if is5 else 0) | (2 if is6 else 0)
        mapped = self.compose(chord, flag, self._sustain_bucket(duration, resolution))
        output.append((tick, mapped, duration, {
            'is5': is5, 'is6': is6, 'isS': is_power,
        }))

    @staticmethod
    def _power_intervals(note_list):
        return sorted(
            (tick, tick + duration)
            for tick, note_type, _, duration in note_list
            if note_type == 'S'
        )

    def encode(self, note_list, resolution=None):
        encoded_notes = []
        last_tick = None
        seq_notes = []
        last_duration = None

        has_is5 = False
        has_is6 = False

        # Extract star power intervals from 'S' notes
        power_intervals = self._power_intervals(note_list)
        power_index = 0

        def is_in_power(tick):
            nonlocal power_index
            while power_index < len(power_intervals) and power_intervals[power_index][1] <= tick:
                power_index += 1
            return (power_index < len(power_intervals)
                    and power_intervals[power_index][0] <= tick < power_intervals[power_index][1])

        for tick, note_type, lane, duration in note_list:
            if note_type == 'S':
                continue  # star powers already handled

            # If we changed tick, output previous group first
            if last_tick is not None and tick != last_tick:
                # Clone Hero charts may contain duplicate lanes at one tick.
                self._append_encoded_group(encoded_notes, last_tick, seq_notes,
                                           last_duration, has_is5, has_is6,
                                           is_in_power(last_tick), resolution)

                # reset for new tick
                seq_notes = []
                has_is5 = False
                has_is6 = False
                last_duration = None

            # Process lane 5 or 6 notes: mark flag but do not add lane to seq_notes
            if lane == 5:
                has_is5 = True
            elif lane == 6:
                has_is6 = True
            else:
                seq_notes.append(lane)
                # Update duration to max of all lanes for this tick
                if last_duration is None:
                    last_duration = duration
                else:
                    last_duration = max(last_duration, duration)

            last_tick = tick

        # Flush last group
        if last_tick is not None and seq_notes:
            # Clone Hero charts may contain duplicate lanes at one tick.
            self._append_encoded_group(encoded_notes, last_tick, seq_notes,
                                       last_duration, has_is5, has_is6,
                                       is_in_power(last_tick), resolution)

        return encoded_notes

    # ------------------------------------------------------------------ decoding

    def decode(self, encoded_notes, resolution=None):
        note_list = []

        # Step 1: Precompute star power (isS) intervals
        power_start_ticks = {}  # tick -> duration

        in_power = False
        sustain_start = None

        for tick, _, _, attrs in encoded_notes:
            if attrs.get('isS', False):
                if not in_power:
                    in_power = True
                    sustain_start = tick
            else:
                if in_power:
                    power_start_ticks[sustain_start] = tick - sustain_start
                    in_power = False

        if in_power:
            last_tick = encoded_notes[-1][0]
            last_duration = encoded_notes[-1][2]
            power_start_ticks[sustain_start] = last_tick + last_duration - sustain_start

        # Step 2: Decode notes inline, insert S immediately after first isS
        for tick, mapped, duration, attrs in encoded_notes:
            chord, flag, bucket = self.split(mapped)
            lanes = self.reverse_chord[chord]

            # Sustain now travels inside the token. Fall back to the tuple's duration
            # for callers that still carry it separately (or in legacy mode).
            if self.expressive and resolution:
                note_duration = int(round(self.SUSTAIN_BEATS[bucket - 1] * resolution)) if bucket else 0
            else:
                note_duration = duration

            for lane in lanes:
                note_list.append((tick, 'N', lane, note_duration))

            is5 = bool(flag & 1) if self.expressive else attrs.get('is5', False)
            is6 = bool(flag & 2) if self.expressive else attrs.get('is6', False)
            if is5:
                note_list.append((tick, 'N', 5, 0))
            if is6:
                note_list.append((tick, 'N', 6, 0))

            # Add S only when this tick is the start of a sustain
            if tick in power_start_ticks:
                sustain_duration = power_start_ticks[tick]
                note_list.append((tick, 'S', 2, sustain_duration))  # Use lane 2 or another if desired

        # Already in order, no need to sort
        return note_list

    def format_seconds(self, notes, bpm_events, resolution=192, offset=0):
        return convert_notes_to_seconds(notes, bpm_events, resolution, offset)


    def discretize_time(self, time_list, tokens_list, pad_token_id, grid_ms, window_seconds, start_time=0.0):
        """
        Map tokens to a time-grid, discretized relative to a given start time.

        Parameters:
        - time_list: List of float timestamps (in seconds) for each token.
        - tokens_list: List of corresponding tokens.
        - pad_token_id: Integer ID used for padding empty grid slots.
        - grid_ms: Grid resolution in milliseconds (e.g., 10 for 10ms bins).
        - window_seconds: Total window duration in seconds to cover with the grid.
        - start_time: Float start timestamp (in seconds); times are relative to this.

        The function computes relative times as (t - start_time), rounds them to the nearest
        grid step, and places tokens on the grid. Times before start_time or beyond the window
        are ignored/clipped as needed.
        """

        if not isinstance(pad_token_id, int):
            raise TypeError("pad_token_id must be an integer")
        if grid_ms <= 0 or window_seconds <= 0:
            raise ValueError("grid_ms and window_seconds must be positive")
        if len(tokens_list) != len(time_list):
            raise ValueError("tokens and times_sec must have the same length")

        grid_s = grid_ms / 1000.0
        steps = window_seconds / grid_s
        if not math.isclose(steps, round(steps), abs_tol=1e-9):
            raise ValueError("window_seconds must be divisible by the time grid")

        # Compute relative times and check min delta (among all pairs, for collision safety)
        # It could be a time window with no tokens
        rel_times = [t - start_time for t in time_list]
        if len(rel_times) > 1:
            min_dt = min_delta(time_list)
            if min_dt < grid_s:
                raise ValueError("Min dt too short will cause collision in discretization")

        n_steps = int(window_seconds / grid_s)  # Total bins in the window
        grid = [pad_token_id] * n_steps

        # Round each relative time to nearest grid step and place token if in bounds
        for token, rel_t in zip(tokens_list, rel_times):
            if rel_t < 0:
                continue
            idx = int(round(rel_t / grid_s))
            if 0 <= idx < len(grid): #with < a note exactly at rel_time=window_seconds is excluded
                grid[idx] = token

        return grid


def min_delta(times_sec):
    times_sorted = sorted(times_sec)
    return min(
        times_sorted[i] - times_sorted[i - 1]
        for i in range(1, len(times_sorted))
    )
