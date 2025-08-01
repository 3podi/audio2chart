from typing import List
import re
import timeit

DIFFICULTIES = ['Expert', 'Hard', 'Medium', 'Easy']
INSTRUMENTS = ['Single', 'Drums']



class ChartProcessor():
    def __init__(self, difficulties, instruments):
        
        if not isinstance(difficulties, List):
            difficulties = [difficulties]

        assert (all(element in DIFFICULTIES for element in difficulties))
        assert (all(element in INSTRUMENTS for element in instruments))

        self.difficulties = difficulties
        self.instruments = instruments

        # DELETE this init if end up using open_chart function
        self.synctrack = []     #Store SyncTrack events: (tick,BPM)
        self.notes = {}
        self.sections = []
        self.song_metadata = None

        for inst in self.instruments:
            for diff in self.difficulties:
                self.sections.append(diff+inst)
        self.sections.extend(['Song', 'SyncTrack'])

        # Build regexes to match chart sections
        self.regexes = {
            name: re.compile(rf'\[{name}\]\s*\{{(.*?)\}}', re.DOTALL)
            for name in self.sections
        }


    def open_chart(self, chart_path):
        with open(chart_path, 'r', encoding='utf-8-sig') as f:
            self.chart_text = f.read()

        self.synctrack = []
        self.notes = {}
        #self.sections = []
        #self.sections.extend(['Song', 'SyncTrack'])
        self.song_metadata = None

        #for inst in self.instruments:
        #    for diff in self.difficulties:
        #        self.sections.append(diff+inst)
        
    def extract_sections(self):
        # Extracts raw content from each [SectionName] { ... } using defined regexes

        section_content = {}
        for name, pattern in self.regexes.items():
            match = pattern.search(self.chart_text)
            if match:
                section_content[name] = match.group(1).strip()
        return section_content
    
    def parse_chart_sections(self, chart_path):
        
        self.open_chart(chart_path)

        sections = self.extract_sections()

        # === Parse SyncTrack BPMs ===
        if "SyncTrack" in sections:
            for line in sections["SyncTrack"].splitlines():
                line = line.strip()
                match = re.match(r"(\d+)\s*=\s*B\s*(\d+)", line)
                if match:
                    tick = int(match.group(1))
                    bpm = int(match.group(2))
                    self.synctrack.append((tick, bpm))
        
        # === Parse [Song] metadata ===
        if "Song" in sections:
            self.song_metadata = sections["Song"].splitlines()

        # === Parse notes in other sections ===
        #note_events = {}  # section_name -> list of notes
        note_pattern = re.compile(r"(\d+)\s*=\s*(N|S)\s*(\d+)\s*(\d+)")

        for name, content in sections.items():
            if name == "SyncTrack" or name == "Song":
                continue
            self.notes[name] = []
            for line in content.splitlines():
                line = line.strip()
                match = note_pattern.match(line)
                if match:
                    tick = int(match.group(1))
                    note_type = match.group(2)  # "N" or "S"
                    lane = int(match.group(3))
                    length = int(match.group(4))
                    self.notes[name].append((tick, note_type, lane, length))
    
    def read_chart(self, chart_path):

        with open(chart_path, 'r', encoding='utf-8-sig') as f:
            chart = f.readlines()

        sections = []
        
        for inst in self.instruments:
            self.notes[inst] = {}
            for diff in self.difficulties:
                self.notes[inst][diff] = []

                sections.append(diff+'-'+inst)

        for line in chart:
            line = line.strip()

            if line.startswith("}"):
                in_synctrack = False
                in_notes = False
                continue

            elif line.startswith("[SyncTrack]"):
                in_synctrack = True
                in_notes = False
                continue

            section_matched = False
            for idx, sec in enumerate(sections):
                section_name = f'[{"".join(sec.split('-'))}]'
                if line.startswith(section_name):
                    in_synctrack = False
                    in_notes = True
                    section_matched = True
                    diff, inst = sec.split('-')
                    break

            if section_matched:
                continue
            elif line.startswith("["):
                in_synctrack = False
                in_notes = False
                continue

            if in_synctrack:
                match = re.match(r"(\d+)\s*=\s*B\s*(\d+)", line)
                if match:
                    tick = int(match.group(1))
                    bpm = int(match.group(2))
                    self.synctrack.append((tick, bpm))
            elif in_notes:
                match = re.match(r"(\d+)\s*=\s*N\s*(\d+)\s*(\d+)", line)
                if match:
                    tick = int(match.group(1))
                    lane = int(match.group(2))
                    length = int(match.group(3))

                    self.notes[inst][diff].append((tick, lane, length))




processor = ChartProcessor(['Expert', 'Medium', 'Easy'], ['Single', 'Drums'])

t1 = timeit.default_timer()
processor.read_chart('notes_full.chart')
t2 = timeit.default_timer()

print('Time processing 1: ', t2-t1 )

#print(processor.notes['Single']['Expert'])


t1 = timeit.default_timer()
processor.parse_chart_sections('notes_full.chart')
t2 = timeit.default_timer()

print('Time processing 2: ', t2-t1 )


#print(processor.notes['ExpertSingle'])




        