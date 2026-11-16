import unicodedata
from pathlib import Path

from pypdf import PdfReader


def normalize(text: str) -> str:
    """NFKC turns compatibility characters into plain ones.

    PDF fonts store ligatures as single characters: 'ﬁ' (U+FB01) -> 'fi',
    'ﬂ' (U+FB02) -> 'fl', 'ﬀ' (U+FB00) -> 'ff'. Without this, a search for
    "fluxo" does not match "ﬂuxo".
    """
    return unicodedata.normalize("NFKC", text)


def extract_text(path: Path) -> str:
    """Return the normalized text of every page, joined by a blank line."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return normalize("\n\n".join(pages))
