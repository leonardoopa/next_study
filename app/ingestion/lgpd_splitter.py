"""Split the LGPD (Lei 13.709/2018, Planalto's compiled text) into one chunk per article.

Rules:
1. CONTEXT: "CAPÍTULO I" (+ the next line, its title) and "Seção I" set the section of every
   article below them, so a chunk says where in the law it lives.
2. ARTICLE: a line starting with "Art. N" opens a chunk. Parágrafos, incisos and alíneas
   that follow stay with their article. The chunk title is "Art. N", which is also the
   stable key the golden set uses to say which article answers a question.
3. NOISE removed: struck-through (revoked) text, amendment notes such as
   "(Redação dada pela Lei ...)", everything before "CAPÍTULO I" and the signatures after
   "Brasília, ...".
4. COMPILED-TEXT QUIRKS: the page lists superseded versions of some articles one after the
   other (Art. 55-A appears 4 times, and 55-H..55-K show up in two different blocks). The
   page is chronological, so the LAST occurrence of an article is the current text and
   replaces the older ones. The order of article numbers is NOT assumed to be increasing.
   Vetoed or revoked articles have no content and are dropped. A typo such as "Art. 5 7."
   is repaired to "Art. 57." before parsing.
5. LONG articles (Art. 55-J has ~7,000 characters) are split at line boundaries. Each part
   after the first repeats the start of the article (the caput) for context.
"""

import re
from dataclasses import dataclass

from app.ingestion.html_text import html_to_lines
from app.schemas import Chunk

MAX_CHARS = 2000  # about 500 tokens: far below the embedding model limit
CAPUT_CHARS = 300  # how much of the opening line is repeated in later parts

# "(Redação dada pela Lei nº 13.853, de 2019)", "(Incluído pela ...)", "(Vide ...)", ...
_NOTE = re.compile(
    r"\s*\((?:Reda[çc][ãa]o dada|Inclu[ií]d[oa]|Revogad[oa]|Vide|Vig[êe]ncia"
    r"|Regulamento|Produ[çc][ãa]o de efeito)[^)]*\)",
    re.I,
)
_CHAPTER = re.compile(r"^CAP[ÍI]TULO\s+([IVXLC]+)\s*$")
_SECTION = re.compile(r"^(Subse[çc][ãa]o|Se[çc][ãa]o)\s+([IVXLC]+)\s*$")
_ARTICLE = re.compile(r"^Art\.\s*(\d+)\s*[º°]?(?:-([A-Z]))?\s*\.?\s*(.*)$")
_SIGNATURE = re.compile(r"^Bras[ií]lia\b")
_NO_CONTENT = re.compile(r"^\(?\s*(VETADO|Revogado)\s*\)?\.?$", re.I)


@dataclass
class _Article:
    key: tuple[int, str]
    label: str  # "18", "55-A"
    section: str
    lines: list[str]  # lines[0] is the caput, without the "Art. N" prefix


def _clean(line: str) -> str:
    line = _NOTE.sub("", line).strip()
    # "Art. 5 7." is a typo for "Art. 57." in the source page.
    return re.sub(r"^(Art\.\s*\d+)\s+(\d)(?=\s*[.º°])", r"\1\2", line)


def _title_after(cleaned: list[str], i: int) -> tuple[str, int]:
    """The heading title is the next non-empty line (a note line may be cleaned to empty)."""
    while i < len(cleaned) and not cleaned[i]:
        i += 1
    # rstrip: the source leaves a stray closing quote on some titles (… PRIVACIDADE’).
    return (cleaned[i].rstrip("’'\"”"), i + 1) if i < len(cleaned) else ("", i)


def parse_articles(lines: list[str]) -> list[_Article]:
    chapter = section = ""
    started = False
    current: _Article | None = None
    latest: dict[tuple[int, str], _Article] = {}  # newest version of each article

    cleaned = [_clean(line) for line in lines]
    i = 0
    while i < len(cleaned):
        line = cleaned[i]
        i += 1
        if not line:
            continue
        if started and _SIGNATURE.match(line):
            break

        if m := _CHAPTER.match(line):
            title, i = _title_after(cleaned, i)
            chapter, section, started, current = f"CAPÍTULO {m.group(1)} – {title}", "", True, None
            continue
        if m := _SECTION.match(line):
            title, i = _title_after(cleaned, i)
            section, current = f"{m.group(1)} {m.group(2)} – {title}", None
            continue
        if not started:
            continue

        if m := _ARTICLE.match(line):
            caput = m.group(3)
            if not caput or _NO_CONTENT.match(caput):  # vetoed or revoked: nothing to index
                current = None
                continue
            key = (int(m.group(1)), m.group(2) or "")
            label = m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
            where = f"{chapter} > {section}" if section else chapter
            current = _Article(key, label, where, [caput])
            latest[key] = current  # the newest version replaces the older ones
            continue

        if current is not None:
            current.lines.append(line)

    return [latest[key] for key in sorted(latest)]


def _has_content(article: _Article) -> bool:
    caput = article.lines[0].strip()
    only_caput = len(article.lines) == 1
    return bool(caput) and not (only_caput and _NO_CONTENT.match(caput))


def _split(lines: list[str], max_chars: int) -> list[list[str]]:
    """Group lines into parts of at most max_chars (a single long line stays whole)."""
    parts: list[list[str]] = [[]]
    size = 0
    for line in lines:
        if parts[-1] and size + len(line) + 1 > max_chars:
            parts.append([])
            size = 0
        parts[-1].append(line)
        size += len(line) + 1
    return parts


def build_lgpd_chunks(raw: bytes) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in parse_articles(html_to_lines(raw)):
        if not _has_content(article):
            continue
        parts = _split(article.lines, MAX_CHARS)
        caput = article.lines[0][:CAPUT_CHARS]
        for n, part in enumerate(parts, start=1):
            title = f"Art. {article.label}"
            if len(parts) > 1:
                title += f" (parte {n}/{len(parts)})"
            body = part if n == 1 else [caput, "[...]", *part]
            chunks.append(Chunk(article.section, title, "\n".join(body)))
    return chunks
