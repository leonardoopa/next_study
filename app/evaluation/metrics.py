"""Retrieval metrics. Pure functions: no database, no model, easy to test.

Ranking unit: the ARTICLE, not the chunk. A long article is stored as several chunks
("Art. 18 (parte 1/2)"), so the ranked list keeps only the first occurrence of each
article; otherwise two parts of the same article would push the next article down.
"""

from collections.abc import Sequence
from math import comb


def article_label(title: str) -> str:
    """"Art. 18 (parte 1/2)" -> "Art. 18"."""
    return title.split(" (")[0]


def ranked_articles(titles: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ranked: list[str] = []
    for title in titles:
        label = article_label(title)
        if label not in seen:
            seen.add(label)
            ranked.append(label)
    return ranked


def first_rank(ranked: Sequence[str], expected: Sequence[str]) -> int | None:
    """1-based position of the first expected article, or None if it is not in the list."""
    wanted = set(expected)
    for position, label in enumerate(ranked, start=1):
        if label in wanted:
            return position
    return None


def hit_at_k(rank: int | None, k: int) -> bool:
    """Did the answer show up in the first k? (1 hit = enough)."""
    return rank is not None and rank <= k


def recall_at_k(ranked: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Share of the expected articles found in the first k (matters when several answer)."""
    wanted = set(expected)
    return len(wanted & set(ranked[:k])) / len(wanted)


def reciprocal_rank(rank: int | None) -> float:
    """1 for 1st, 0.5 for 2nd, 0.33 for 3rd... and 0 when not found. The mean is the MRR."""
    return 0.0 if rank is None else 1.0 / rank


def compare_ranks(
    a: Sequence[int | None], b: Sequence[int | None]
) -> tuple[int, int, int]:
    """Question by question: how often A ranks the answer better than B (wins), worse
    (losses) or equal (ties). A missing answer is worse than any position."""
    wins = losses = ties = 0
    for rank_a, rank_b in zip(a, b, strict=True):
        key_a = float("inf") if rank_a is None else rank_a
        key_b = float("inf") if rank_b is None else rank_b
        if key_a < key_b:
            wins += 1
        elif key_a > key_b:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided exact sign test: how likely is a split this uneven if both methods were
    equally good? Only the questions where they differ count (ties carry no information)."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2**n)
