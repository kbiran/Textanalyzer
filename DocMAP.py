import re
from typing import List, Dict

def DocMAP(text: str) -> Dict:
    """
    DocMAP: Detects headings, builds an outline, creates mini-summaries,
    and returns navigation anchors for UI click-to-jump behavior.

    INPUT:
        text (str): Full document text.

    OUTPUT (dict):
        {
            "sections": [
                {
                    "heading": "Introduction",
                    "content": "Full text of the section...",
                    "mini_summary": "Short summary...",
                    "start_index": 120   # character index in original text
                },
                ...
            ],
            "outline": ["Introduction", "Background", "Conclusion"]
        }
    """

    # ------------------------------------------------------------
    # 1. Detect Headings (US4.1)
    # ------------------------------------------------------------
    # Simple heuristic:
    # A heading is a line that is:
    #   - short (1–8 words)
    #   - starts with a capital letter
    #   - not a full sentence (no period at the end)
    #
    # This is intentionally simple for students.
    # ------------------------------------------------------------

    lines = text.split("\n")
    headings = []
    for i, line in enumerate(lines):
        clean = line.strip()

        if (1 <= len(clean.split()) <= 8
            and clean[:1].isupper()
            and not clean.endswith(".")):
            headings.append((i, clean))  # (line_number, heading_text)

    # If no headings found, treat whole doc as one section
    if not headings:
        section_text = text.strip()
        sentences = re.split(r'(?<=[.!?])\s+', section_text)
        mini_summary = " ".join(sentences[:2]).strip()
        sections = [{
            "heading": "Document",
            "content": section_text,
            "mini_summary": mini_summary,
            "start_index": 0
        }]
        outline = ["Document"]
        return {
            "sections": sections,
            "outline": outline
        }

    sections = []

    # If there is content before the first heading, create a section for it
    if headings[0][0] > 0:
        pre_text = "\n".join(lines[:headings[0][0]]).strip()
        if pre_text:
            sentences = re.split(r'(?<=[.!?])\s+', pre_text)
            mini_summary = " ".join(sentences[:2]).strip()
            sections.append({
                "heading": "Preface",
                "content": pre_text,
                "mini_summary": mini_summary,
                "start_index": 0
            })

    # ------------------------------------------------------------
    # 2. Build Sections + Extract Content (US4.2)
    # ------------------------------------------------------------
    for idx, (line_num, heading) in enumerate(headings):
        start = line_num

        # Determine where this section ends
        if idx < len(headings) - 1:
            end = headings[idx + 1][0]
        else:
            end = len(lines)

        # Join the section's text
        section_text = "\n".join(lines[start+1:end]).strip()

        # Compute character index for click-to-navigate (US4.4)
        char_index = len("\n".join(lines[:start]))

        # ------------------------------------------------------------
        # 3. Mini Summary (US4.3)
        # ------------------------------------------------------------
        # Simple rule: take the first 1–2 sentences.
        # ------------------------------------------------------------
        sentences = re.split(r'(?<=[.!?]) +', section_text)
        mini_summary = " ".join(sentences[:2]).strip()

        sections.append({
            "heading": heading,
            "content": section_text,
            "mini_summary": mini_summary,
            "start_index": char_index
        })

    # ------------------------------------------------------------
    # 4. Build Outline (US4.2)
    # ------------------------------------------------------------
    outline = [sec["heading"] for sec in sections]

    return {
        "sections": sections,
        "outline": outline
    }
