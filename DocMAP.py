import re
from typing import List, Dict, Tuple

# Constants for maintainability
MIN_HEADING_WORDS = 1
MAX_HEADING_WORDS = 8
MAX_SUMMARY_SENTENCES = 2
PREFACE_HEADING = "Preface"
DEFAULT_HEADING = "Document"

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
    if not text.strip():
        return {"sections": [], "outline": []}

    lines = text.split("\n")
    headings = _detect_headings(lines)

    if not headings:
        return _handle_no_headings(text)

    sections = _build_sections(lines, headings)
    outline = [sec["heading"] for sec in sections]

    return {
        "sections": sections,
        "outline": outline
    }

def _detect_headings(lines: List[str]) -> List[Tuple[int, str]]:
    """
    Detects potential headings based on simple heuristics.

    A heading is a line that is:
    - Short (1–8 words)
    - Starts with a capital letter
    - Not a full sentence (no period at the end)
    """
    headings = []
    for i, line in enumerate(lines):
        clean = line.strip()
        word_count = len(clean.split())
        if (MIN_HEADING_WORDS <= word_count <= MAX_HEADING_WORDS
            and clean and clean[0].isupper()
            and not clean.endswith(".")):
            headings.append((i, clean))
    return headings

def _handle_no_headings(text: str) -> Dict:
    """Handles the case when no headings are detected."""
    section_text = text.strip()
    mini_summary = _create_mini_summary(section_text)
    sections = [{
        "heading": DEFAULT_HEADING,
        "content": section_text,
        "mini_summary": mini_summary,
        "start_index": 0
    }]
    outline = [DEFAULT_HEADING]
    return {
        "sections": sections,
        "outline": outline
    }

def _build_sections(lines: List[str], headings: List[Tuple[int, str]]) -> List[Dict]:
    """Builds sections from detected headings."""
    sections = []
    cumulative_length = 0  # Track cumulative length for start_index

    # Handle content before first heading
    if headings and headings[0][0] > 0:
        pre_lines = lines[:headings[0][0]]
        pre_text = "\n".join(pre_lines).strip()
        if pre_text:
            mini_summary = _create_mini_summary(pre_text)
            sections.append({
                "heading": PREFACE_HEADING,
                "content": pre_text,
                "mini_summary": mini_summary,
                "start_index": 0
            })
        cumulative_length += len("\n".join(pre_lines)) + 1  # +1 for the newline

    # Build sections for each heading
    for idx, (line_num, heading) in enumerate(headings):
        start = line_num
        end = headings[idx + 1][0] if idx < len(headings) - 1 else len(lines)
        section_lines = lines[start + 1:end]
        section_text = "\n".join(section_lines).strip()

        sections.append({
            "heading": heading,
            "content": section_text,
            "mini_summary": _create_mini_summary(section_text),
            "start_index": cumulative_length
        })

        # Update cumulative length
        cumulative_length += len(lines[start]) + 1 + len("\n".join(section_lines)) + 1

    return sections

def _create_mini_summary(section_text: str) -> str:
    """Creates a mini-summary by taking the first few sentences."""
    if not section_text:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', section_text)
    return " ".join(sentences[:MAX_SUMMARY_SENTENCES]).strip()
