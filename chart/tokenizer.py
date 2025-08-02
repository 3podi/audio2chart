import itertools


class SimpleTokenizerGuitar():
    def __init__(self):

        # Define a mapping between pressed lanes and note index

        indices = [i for i in range(5)] # 5 fret
        all_combinations = []
        for r in range(1, len(indices) + 1):
            combos = list(itertools.combinations(indices, r))
            all_combinations.extend(combos)

        self.mapping_noteseqs2int = {v:idx for idx, v in enumerate(all_combinations)}
        self.mapping_noteseqs2int[(6,)] = len(all_combinations) # Open note: key 31
    
    def encode(self, note_list):
        encoded_notes = []
        last_tick = None
        seq_notes = []
        last_duration = None

        has_is5 = False
        has_is6 = False

        # Extract sustain intervals from 'S' notes
        sustain_intervals = []
        for tick, note_type, lane, duration in note_list:
            if note_type == 'S':
                sustain_intervals.append((tick, tick + duration))

        def is_in_sustain(tick):
            for start, end in sustain_intervals:
                if start <= tick < end:
                    return True
            return False

        for tick, note_type, lane, duration in note_list:
            if note_type == 'S':
                continue  # sustains already handled

            # If we changed tick, output previous group first
            if last_tick is not None and tick != last_tick:
                mapped = self.mapping_noteseqs2int.get(tuple(sorted(seq_notes)), None)
                if mapped is None:
                    raise ValueError(f"Unknown note sequence {seq_notes} at tick {last_tick}")

                attrs = {
                    'is5': has_is5,
                    'is6': has_is6,
                    'isS': is_in_sustain(last_tick),
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
            mapped = self.mapping_noteseqs2int.get(tuple(sorted(seq_notes)), None)
            if mapped is None:
                raise ValueError(f"Unknown note sequence {seq_notes} at tick {last_tick}")

            attrs = {
                'is5': has_is5,
                'is6': has_is6,
                'isS': is_in_sustain(last_tick),
            }
            encoded_notes.append((last_tick, mapped, last_duration, attrs))

        return encoded_notes
    



        
