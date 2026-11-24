"""Run the retrieval modes over a golden set and record where each one ranked the answer.

Retrieval only (layer 1): no answer is generated, so a run costs one embedding per question
plus one rerank call for the rerank mode. Generation quality is a separate evaluation.
"""

import asyncio
import time
from dataclasses import dataclass, field

import psycopg

from app import repository
from app.embeddings import embed_query
from app.evaluation.golden import GoldenItem
from app.evaluation.metrics import first_rank, ranked_articles
from app.rerank import rerank
from app.schemas import RankedChunk

MODES = ("vector", "lexical", "hybrid", "hybrid_rerank")


@dataclass
class ModeOutcome:
    titles: list[str]  # chunk titles in the order the mode returned them
    ranked: list[str]  # the same, as distinct articles
    rank: int | None  # position of the first expected article
    ms: int  # time of this mode alone (the question embedding is shared)
    fallback: bool = False  # rerank only: it failed and the hybrid order was kept
    top_similarity: float | None = None  # vector only: cosine of the best chunk


@dataclass
class QuestionResult:
    item: GoldenItem
    outcomes: dict[str, ModeOutcome] = field(default_factory=dict)


def _outcome(item: GoldenItem, titles: list[str], started: float, **extra) -> ModeOutcome:
    ranked = ranked_articles(titles)
    return ModeOutcome(
        titles=titles,
        ranked=ranked,
        rank=first_rank(ranked, item.expected),
        ms=int((time.perf_counter() - started) * 1000),
        **extra,
    )


def evaluate_question(
    conn: psycopg.Connection,
    item: GoldenItem,
    document: str | None,
    depth: int,
    modes: tuple[str, ...] = MODES,
) -> QuestionResult:
    result = QuestionResult(item)
    embedding = embed_query(item.question)

    if "vector" in modes:
        t = time.perf_counter()
        rows = repository.search_chunks(conn, embedding, depth, document)
        top = rows[0].similarity if rows else None
        result.outcomes["vector"] = _outcome(
            item, [r.title for r in rows], t, top_similarity=top
        )

    if not item.answerable:
        # No expected article: ranks mean nothing. The vector run above is kept only to see
        # how similar the best chunk is when nothing in the document answers.
        return result

    if "lexical" in modes:
        t = time.perf_counter()
        rows = repository.search_lexical(conn, item.question, depth, document)
        result.outcomes["lexical"] = _outcome(item, [r.title for r in rows], t)

    hybrid: list[RankedChunk] = []
    if "hybrid" in modes or "hybrid_rerank" in modes:
        t = time.perf_counter()
        hybrid = repository.search_hybrid(
            conn, item.question, embedding, limit=depth, document=document
        )
        if "hybrid" in modes:
            result.outcomes["hybrid"] = _outcome(item, [r.title for r in hybrid], t)

    if "hybrid_rerank" in modes:
        t = time.perf_counter()
        reranked = asyncio.run(rerank(item.question, hybrid, top_k=depth))
        result.outcomes["hybrid_rerank"] = _outcome(
            item, [c.title for c in reranked.chunks], t, fallback=reranked.used_fallback
        )
    return result


def evaluate(
    conn: psycopg.Connection,
    items: list[GoldenItem],
    document: str | None,
    depth: int = 10,
    modes: tuple[str, ...] = MODES,
    progress=None,
) -> list[QuestionResult]:
    results = []
    for n, item in enumerate(items, start=1):
        results.append(evaluate_question(conn, item, document, depth, modes))
        if progress:
            progress(n, len(items), item)
    return results
