"""Render a Clone Hero .chart file from decoded notes."""

import datetime
import re


TEMPLATE = """
[Song]

{
Name = "###name###"
Artist = "###artist###"
Charter = "###charter###"
Album = "###album###"
Year = ", ###year###"
Offset = ###offset###
Resolution = ###resolution###
Player2 = ###player2###
Difficulty = ###difficulty###
PreviewStart = ###previewstart###
PreviewEnd = ###previewend###
Genre = "###genre###"
MediaType = "cd"
MusicStream = "###musicstream###"
}

[SyncTrack]
{
0 = TS 4
0 = B ###bpm###000
}

[ExpertSingle]
{

} 
"""

# Fields the template needs that a caller usually has no opinion about. Anything a caller
# does pass in `metadata` wins; these are only here so the placeholders never survive into
# the written file. `musicstream` must name the audio file sitting next to notes.chart or
# Clone Hero will not play the song.
DEFAULTS = {
    "offset": 0,
    "player2": "bass",
    "difficulty": 3,
    "previewstart": 0,
    "previewend": 0,
    "musicstream": "song.ogg",
}

# Fields that fall back to a marker string rather than to a number when left empty.
TEXT_FIELDS = ("name", "artist", "album", "genre", "charter")


def render_sync_track(bpm_events, ts_events=None) -> str:
    """Render a [SyncTrack] block from a tempo map.

    bpm_events is a list of (tick, bpm * 1000); ts_events a list of
    (tick, (numerator, ...)). Defaults to 4/4 when no time signatures are given.
    """
    events = [(tick, 'TS ' + ' '.join(str(v) for v in values))
              for tick, values in (ts_events or [(0, (4,))])]
    events += [(tick, f'B {raw}') for tick, raw in bpm_events]
    # a TS and a B on the same tick conventionally list TS first
    events.sort(key=lambda e: (e[0], not e[1].startswith('TS')))

    body = "\n".join(f'  {tick} = {event}' for tick, event in events)
    return "[SyncTrack]\n{\n" + body + "\n}"


def fill_expert_single(notes: list[tuple], metadata: dict,
                       bpm_events=None, ts_events=None) -> str:
    """Render the chart. `metadata` overrides DEFAULTS; empty text fields get a marker."""
    values = dict(DEFAULTS)
    values.setdefault("year", datetime.date.today().year)
    values.update({key: value for key, value in metadata.items() if value not in (None, "")})
    for field in TEXT_FIELDS:
        if not values.get(field):
            values[field] = "audio2chart"

    template_text = TEMPLATE
    for key, value in values.items():
        template_text = template_text.replace(f'###{key}###', str(value))

    # Build the new ExpertSingle block content
    new_lines = [f'  {t} = {typ} {a} {b}' for (t, typ, a, b) in notes]
    new_block = "[ExpertSingle]\n{\n" + "\n".join(new_lines) + "\n}"

    # Replace the old ExpertSingle block with the new one
    filled_chart = re.sub(r"\[ExpertSingle\]\s*\{[^}]*\}", new_block, template_text, flags=re.DOTALL)

    if bpm_events:
        filled_chart = re.sub(r"\[SyncTrack\]\s*\{[^}]*\}",
                              render_sync_track(bpm_events, ts_events),
                              filled_chart, flags=re.DOTALL)

    return filled_chart