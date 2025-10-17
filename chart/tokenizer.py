import itertools
from chart.time_conversion import convert_notes_to_seconds
from chart.chart_processor import ChartProcessor
import timeit


class SimpleTokenizerGuitar():
    def __init__(self, exclude_open_chords=True):

        self.exclude_open_chords = exclude_open_chords
        # Define a mapping between pressed lanes (5 fret + open) and note index
        # If exclude_open_chords is True, exclude every chord with 7 except 7 alone

        indices = [i for i in range(5)]
        if not exclude_open_chords:
            indices = indices [7]
        all_combinations = []
        for r in range(1, len(indices) + 1):
            combos = list(itertools.combinations(indices, r))
            all_combinations.extend(combos)

        self.mapping_noteseqs2int = {v:idx for idx, v in enumerate(all_combinations)}
        if self.exclude_open_chords:
            self.mapping_noteseqs2int[(7,)] = len(all_combinations)
        self.reverse_map = {v: k for k, v in self.mapping_noteseqs2int.items()}


    def encode(self, note_list):
        encoded_notes = []
        last_tick = None
        seq_notes = []
        last_duration = None

        has_is5 = False
        has_is6 = False

        # Extract star power intervals from 'S' notes
        power_intervals = []
        for tick, note_type, lane, duration in note_list:
            if note_type == 'S':
                power_intervals.append((tick, tick + duration))

        def is_in_power(tick):
            for start, end in power_intervals:
                if start <= tick < end:
                    return True
            return False

        for tick, note_type, lane, duration in note_list:
            if note_type == 'S':
                continue  # star powers already handled

            # If we changed tick, output previous group first
            if last_tick is not None and tick != last_tick:
                # Handle mistake case, multiple identical notes at same tick (it happens with [3,3])
                #if len(seq_notes) > 1 and len(set(seq_notes)) <= 1: #No perche potrebbe esserci [1,1,3], fix: sort(set()) -> in un tick puo solo esserci un tipo di nota
                #    seq_notes = [seq_notes[0]]
                seq_notes = sorted(set(seq_notes))
                if self.exclude_open_chords:
                    if len(seq_notes) > 1 and seq_notes[-1] == 7:
                        seq_notes = seq_notes[:-1]
                mapped = self.mapping_noteseqs2int.get(tuple(seq_notes), None)
                if mapped is None:
                    raise ValueError(f"Unknown note sequence {seq_notes} at tick {last_tick}")

                attrs = {
                    'is5': has_is5,
                    'is6': has_is6,
                    'isS': is_in_power(last_tick),
                }
                encoded_notes.append((last_tick, mapped, last_duration, attrs))

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
            # Handle mistake case, multiple identical notes at same tick (it happens with [3,3])
            seq_notes = sorted(set(seq_notes))
            if self.exclude_open_chords:
                if len(seq_notes) > 1 and seq_notes[-1] == 7:
                    seq_notes = seq_notes[:-1]
            mapped = self.mapping_noteseqs2int.get(tuple(seq_notes), None)
            if mapped is None:
                raise ValueError(f"Unknown note sequence {seq_notes} at tick {last_tick}")

            attrs = {
                'is5': has_is5,
                'is6': has_is6,
                'isS': is_in_power(last_tick),
            }
            encoded_notes.append((last_tick, mapped, last_duration, attrs))

        return encoded_notes
    
    def decode(self, encoded_notes):
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
            lanes = self.reverse_map[mapped]

            for lane in lanes:
                note_list.append((tick, 'N', lane, duration))

            #sustain is not assigned to those notes
            if attrs.get('is5', False):
                note_list.append((tick, 'N', 5, 0))
            if attrs.get('is6', False):
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

        assert isinstance(pad_token_id, int), 'Pad token id must be an int'

        if len(tokens_list) != len(time_list):
            raise ValueError("tokens and times_sec must have the same length")

        # Compute relative times and check min delta (among all pairs, for collision safety)
        rel_times = [t - start_time for t in time_list]
        min_dt = min_delta(time_list)
        grid_s = grid_ms / 1000.0
        if min_dt < grid_s: 
            raise ValueError("Min dt too short will cause collision in discretization")

        assert window_seconds%grid_s==0 , 'Time window must be multiple of time resolution'
        n_steps = int(window_seconds / grid_s)  # Total bins in the window
        grid = [pad_token_id] * n_steps

        # Round each relative time to nearest grid step and place token if in bounds
        for token, rel_t in zip(tokens_list, rel_times):
            if rel_t < 0:
                print('Warning: Got value less than zero while discretizing time')
                continue
            idx = int(round(rel_t / grid_s))
            if 0 <= idx < len(grid):
                grid[idx] = token
            if idx >= len(grid):
                print('Warning: Got a token that falls outside the discretized time window')

        return grid



def min_delta(times_sec):
    times_sorted = sorted(times_sec)
    return min(
        times_sorted[i] - times_sorted[i - 1]
        for i in range(1, len(times_sorted))
    )


 #Per l errore quando nelle vere notes ho uno star power che inizia in un tick semza altre note, 
 # esso non viene salvato perche non salvo gli inizi degli star power.
 # quando pero decodo la prima nota in quello star power allora il decoder mettera S,2 con quella nota 
 # e quindi invertendo l ordine di apparizione delle note.

 #Nelle decoded notes avro sempre lo star power messo dopo la prima nota dello star power
 #con lo stesso tick start

if __name__ == "__main__":
    tok = SimpleTokenizerGuitar()

    processor = ChartProcessor(['Expert', 'Medium', 'Easy'], ['Single', 'Drums'])

    t1 = timeit.default_timer()
    processor.read_chart('notes_full.chart')
    t2 = timeit.default_timer()
    print('Time processing 2: ', t2-t1 )

    notes = processor.notes["ExpertSingle"]
    bpm_events = processor.synctrack
    resolution = int(processor.song_metadata['Resolution'])
    offset = float(processor.song_metadata['Offset'])
    print(resolution, offset)

    tok = SimpleTokenizerGuitar()

    t1 = timeit.default_timer()
    encoded_notes = tok.encode(notes)
    t2 = timeit.default_timer()
    print('Time encoding: ', t2-t1 )

    t1 = timeit.default_timer()
    note_times = tok.format_seconds(encoded_notes, bpm_events,resolution=480, offset=0.05)
    t2 = timeit.default_timer()

    print('Coversion time: ', t2-t1)
