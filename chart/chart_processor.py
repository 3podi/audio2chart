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

        self.sections = []
        for inst in self.instruments:
            for diff in self.difficulties:
                self.sections.append(diff+inst)
        self.sections.extend(['Song', 'SyncTrack'])

        # Build regexes to match chart sections
        self.regexes = {
            name: re.compile(rf'\[{name}\]\s*\{{(.*?)\}}', re.DOTALL)
            for name in self.sections
        }

        self.regex_metadata = r'(Resolution|Offset|Genre)\s*=\s*"?([^"\n]+)"?'

    def open_chart(self, chart_path):
        with open(chart_path, 'r', encoding='utf-8-sig') as f:
            self.chart_text = f.read()

        self.synctrack = []             #Store SyncTrack events: (tick,BPM)
        self.notes = {}                 #Store note events for section: {"section": List[(tick,N,lane,length)]}
        self.song_metadata = None       #Store Song section
        
    def extract_sections(self):
        # Extracts raw content from each [SectionName] { ... } using defined regexes

        section_content = {}
        for name, pattern in self.regexes.items():
            match = pattern.search(self.chart_text)
            print("Tryng to match: ", name)
            print("Matched: ", match)
            if match:
                section_content[name] = match.group(1).strip()
        return section_content
    
    def read_chart(self, chart_path):
        
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
            #print('SOOONG: ', sections["Song"])
            #self.song_metadata = sections["Song"].splitlines()
            self.song_metadata = {match[0]: match[1] for match in re.findall(self.regex_metadata, sections["Song"])}


        # === Parse notes in other sections ===
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
    

processor = ChartProcessor(['Expert', 'Medium', 'Easy'], ['Single', 'Drums'])

t1 = timeit.default_timer()
processor.read_chart('notes_full.chart')
t2 = timeit.default_timer()

print('Time processing 2: ', t2-t1 )


for k, v in processor.notes.items():
    print(k)
    print(len(v))

print(processor.synctrack)
print(processor.song_metadata)
        
for meta in processor.song_metadata:
    if 'Resolution' in meta:
        print(meta.split('=')[-1].strip())
    if 'Offset' in meta:
        print(meta.split('=')[-1].strip())
print(processor.notes["MediumSingle"][:10])