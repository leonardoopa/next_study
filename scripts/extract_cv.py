import unicodedata
from pathlib import Path

from pypdf import PdfReader

CV_PATH = Path("data/cv.pdf")


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


if __name__ == "__main__":
    text = extract_text(CV_PATH)

    # Check: no ligature characters left (they live in the U+FB00..U+FB06 block).
    leftovers = [c for c in text if "ﬀ" <= c <= "ﬆ"]
    assert not leftovers, f"ligatures still present: {set(leftovers)}"
    print("check ok: no ligatures")

    print(f"{len(text)} chars")
    print(text[:400])
