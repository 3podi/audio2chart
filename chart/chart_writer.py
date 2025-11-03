

TEMPLATE = """
[Song]

{
Name = "###name###"
Artist = "###artist###"
Charter = "###charter###"
Album = "###album###"
Year = ", 2022"
Offset = 0
Resolution = 192
Player2 = bass
Difficulty = 3
PreviewStart = 0
PreviewEnd = 0
Genre = "###genre###"
MediaType = "cd"
MusicStream = "song.ogg"
}

[SyncTrack]
{
0 = TS 4
0 = B 200000
}

[ExpertSingle]
{

} 
"""

def fill_expert_single(notes: list[tuple], metadata: dict) -> str:
    
    template_text = TEMPLATE
    # Fill the Song block
    for key, value in metadata:
        if value:
            template_text = template_text.replace(f'###{str(key)}###', value)
        else:
            template_text = template_text.replace(f'###{str(key)}###', 'audio2chart')

    # Build the new ExpertSingle block content
    new_lines = [f'  {t} = {typ} {a} {b}' for (t, typ, a, b) in notes]
    new_block = "[ExpertSingle]\n{\n" + "\n".join(new_lines) + "\n}"

    # Replace the old ExpertSingle block with the new one
    import re
    filled_chart = re.sub(r"\[ExpertSingle\]\s*\{[^}]*\}", new_block, template_text, flags=re.DOTALL)

    return filled_chart