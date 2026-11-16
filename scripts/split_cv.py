"""Day 2, task 2: split the CV text into structured chunks (section / item).

Run (inside the api container):
    docker compose exec api python scripts/split_cv.py

Rules:
1. SECTION: a line that is entirely UPPERCASE (RESUMO, PROJETOS, ...). Text before the
   first section (name + contact line) is dropped: it is personal data, not content.
2. PARAGRAPH: the PDF hard-wraps lines at a fixed width. A line that reaches ~the full
   width continues on the next line (soft wrap). A shorter line ends the paragraph.
3. ITEM:
   - EXPERIÊNCIA: a new job starts at a heading with a date range ("set/2023 – out/2026").
   - PROJETOS: each paragraph is one project.
   - any other section: the whole section is one chunk.
4. Each chunk keeps its section and title, and its text is prefixed with them, so a chunk
   like "Mentor de 10 estagiários..." still says it came from the BSA Tech job.
"""

import re

from app.schemas import Chunk
from extract_cv import CV_PATH, extract_text

FULL_LINE_RATIO = 0.9  # a line this close to the max width is considered "wrapped"
MONTHS = "jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez"
JOB_HEADING = re.compile(rf"\b({MONTHS})/\d{{4}}\s*[–-]\s*(({MONTHS})/\d{{4}}|atual|presente)", re.I)


def split_sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Group lines under their uppercase header. Lines before the first header are dropped."""
    sections: list[tuple[str, list[str]]] = []
    for line in lines:
        if line.isupper():
            sections.append((line, []))
        elif sections:
            sections[-1][1].append(line)
    return sections


def join_wrapped(lines: list[str], width: int) -> list[str]:
    """Rebuild paragraphs: a full-width line continues on the next one."""
    paragraphs: list[str] = []
    previous_was_full = False
    for line in lines:
        if paragraphs and previous_was_full:
            paragraphs[-1] += " " + line
        else:
            paragraphs.append(line)
        previous_was_full = len(line) >= FULL_LINE_RATIO * width
    return paragraphs


def build_chunks(text: str) -> list[Chunk]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    width = max(len(ln) for ln in lines)
    chunks: list[Chunk] = []

    for section, section_lines in split_sections(lines):
        paragraphs = join_wrapped(section_lines, width)

        if section.startswith("EXPERIÊNCIA"):
            job: list[str] = []
            for para in paragraphs + [""]:  # "" flushes the last job
                if (JOB_HEADING.search(para) or not para) and job:
                    chunks.append(Chunk(section, job[0], "\n".join(job)))
                    job = []
                if para:
                    job.append(para)
        elif section == "PROJETOS":
            for para in paragraphs:
                title = para.split(" (")[0]
                chunks.append(Chunk(section, title, para))
        else:
            chunks.append(Chunk(section, section, "\n".join(paragraphs)))

    return chunks


if __name__ == "__main__":
    chunks = build_chunks(extract_text(CV_PATH))
    for i, c in enumerate(chunks):
        print(f"{i} | {c.section} | {c.title[:55]!r} | {len(c.text)} chars")

    assert len(chunks) == 9, f"expected 9 chunks, got {len(chunks)}"
    print("check ok: 9 chunks")
