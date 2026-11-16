"""Split the CV text into structured chunks (section / item).

Rules:
1. SECTION: a line that is entirely UPPERCASE (RESUMO, PROJETOS, ...). Text before the
   first section (name + contact line) is dropped: it is personal data, not content.
2. PARAGRAPH: the PDF hard-wraps lines at a fixed width. A line that reaches ~the full
   width continues on the next line (soft wrap). A shorter line ends the paragraph.
3. ITEM:
   - EXPERIÊNCIA: a new job starts at a heading that ends with a date range
     ("set/2023 – out/2026"). The next line starts with the company ("BSA Tech | Recife").
   - PROJETOS: each paragraph is one project.
   - any other section: the whole section is one chunk.
4. The title carries the identity of the item (role | company | period, or project name)
   and is NOT repeated in the text, so nothing is embedded twice.
"""

import re

from app.schemas import Chunk

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


def _job_heading(paragraph: str) -> re.Match | None:
    """Match only when the date range closes the line, so a date inside a sentence is ignored."""
    match = JOB_HEADING.search(paragraph)
    return match if match and match.end() == len(paragraph) else None


def _job_chunk(section: str, job: list[str]) -> Chunk:
    """job[0] is the heading, job[1] the company line, the rest is the description."""
    heading, *rest = job
    match = _job_heading(heading)
    role, period = heading[: match.start()].strip(), match.group(0)

    company, details = "", rest
    if rest:
        company, _, location = rest[0].partition(" | ")
        details = ([location] if location else []) + rest[1:]

    title = " | ".join(part for part in (role, company, period) if part)
    return Chunk(section, title, "\n".join(details))


def build_chunks(text: str) -> list[Chunk]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    width = max(len(ln) for ln in lines)
    chunks: list[Chunk] = []

    for section, section_lines in split_sections(lines):
        paragraphs = join_wrapped(section_lines, width)

        if section.startswith("EXPERIÊNCIA"):
            job: list[str] = []
            for para in [*paragraphs, ""]:  # "" flushes the last job
                if (_job_heading(para) or not para) and job:
                    chunks.append(_job_chunk(section, job))
                    job = []
                if para:
                    job.append(para)
        elif section == "PROJETOS":
            for para in paragraphs:
                title = para.split(" (")[0]
                chunks.append(Chunk(section, title, para[len(title) :].strip()))
        else:
            chunks.append(Chunk(section, section, "\n".join(paragraphs)))

    return chunks
