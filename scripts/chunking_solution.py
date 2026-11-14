"""Reference solution for scripts/chunking.py (peek only after you tried).

Run:  python3 scripts/chunking_solution.py
"""

TEXT = (
    "Art. 482 - Constituem justa causa para rescisão do contrato de trabalho "
    "pelo empregador: a) ato de improbidade; b) incontinência de conduta."
)


def chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Split `text` into windows of `size` characters.

    Each window starts `size - overlap` characters after the previous one.
    Raise ValueError if `overlap >= size` (the window would never advance).
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")

    step = size - overlap
    return [text[start : start + size] for start in range(0, len(text), step)]


if __name__ == "__main__":
    # Check 1: no overlap must give exactly these 3 chunks.
    chunks = chunk_fixed(TEXT, size=56, overlap=0)
    assert chunks == [
        "Art. 482 - Constituem justa causa para rescisão do contr",
        "ato de trabalho pelo empregador: a) ato de improbidade; ",
        "b) incontinência de conduta.",
    ], chunks
    print("check 1 ok")

    # Check 2: invalid overlap must be rejected.
    try:
        chunk_fixed(TEXT, size=10, overlap=10)
    except ValueError:
        print("check 2 ok")
    else:
        raise AssertionError("expected ValueError when overlap >= size")

    # Check 3 (no assert): look at the output and think about the LAST chunk.
    # This version still has the issue. Fixing it is the follow-up exercise.
    for i, c in enumerate(chunk_fixed(TEXT, size=56, overlap=10)):
        print(i, repr(c))
