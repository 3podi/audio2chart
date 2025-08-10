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

        #self.regexes = {
        #    name: re.compile(rf'\[{name}\]\s*\{{(.*)\}}', re.DOTALL)
        #    for name in self.sections
        #}

        self.regex_metadata = r'(Resolution|Offset|Genre)\s*=\s*"?([^"\n]+)"?'
        #self.regex_metadata = r'(Resolution|Offset|Genre)\s*=\s*"?([^"\n]+?)"?'

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
            if match:
                section_content[name] = match.group(1).strip()
        return section_content
    
    def extract_sections2(self):
        section_content = {}
        
        for section_name in self.sections:
            # Find the section header
            header_pattern = rf'\[{re.escape(section_name)}\]\s*\{{'
            header_match = re.search(header_pattern, self.chart_text)
            
            if not header_match:
                continue
            
            # Start after the opening brace
            start_pos = header_match.end() - 1  # Position of opening brace
            brace_count = 0
            pos = start_pos
            
            # Count braces to find the matching closing brace
            while pos < len(self.chart_text):
                char = self.chart_text[pos]
                
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    
                    if brace_count == 0:  # Found matching closing brace
                        content = self.chart_text[start_pos + 1:pos]
                        section_content[section_name] = content.strip()
                        break
                
                pos += 1
        
        return section_content
    
    def read_chart(self, chart_path):
        
        self.open_chart(chart_path)

        sections = self.extract_sections2()

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

        #print("DEBUG: sections keys:", list(sections.keys()))
        #print("DEBUG: sections['Song'] exists:", "Song" in sections)

        #if "Song" in sections:
        #    print("DEBUG: sections['Song'] content:")
        #    print(repr(sections["Song"]))  # repr() will show hidden chars
        #    print("DEBUG: sections['Song'] length:", len(sections["Song"]))
        #    print("DEBUG: First 200 chars:", sections["Song"][:200])
        #else:
        #    print("DEBUG: 'Song' section not found in sections!")
        #    print("DEBUG: Available sections:", list(sections.keys()))


if __name__ == "__main__":
    
    processor = ChartProcessor(['Expert', 'Medium', 'Easy'], ['Single', 'Drums'])

    t1 = timeit.default_timer()
    processor.read_chart('notes_full.chart')
    t2 = timeit.default_timer()

    print('Time processing chart: ', t2-t1)