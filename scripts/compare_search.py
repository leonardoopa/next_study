"""Compare vector, lexical, hybrid and hybrid+rerank on a few questions (informal).

Run (inside the api container):
    docker compose exec api python scripts/compare_search.py

For each question it prints the position (1 = best) of the first chunk that holds the
answer, or "-" when that search did not return it, plus the time of the rerank step.
With 9 chunks this only illustrates the mechanism; a real comparison needs a bigger
corpus and a golden set (day 5).
"""

import asyncio
import time

import psycopg
from pgvector.psycopg import register_vector

from app import repository
from app.config import get_settings
from app.embeddings import embed_query
from app.rerank import rerank

# (question, titles that hold the answer: any of them counts)
CASES = [
    ("Quantos estagiários ele mentorou?", ("Engenheiro de Software Back-End",)),
    ("Ele usou RabbitMQ?", ("Engenheiro de Software Back-End", "COMPETÊNCIAS TÉCNICAS")),
    ("Ele trabalhou com filas de mensagens?", ("Engenheiro de Software Back-End",)),
    ("Quais bancos de dados ele conhece?", ("COMPETÊNCIAS TÉCNICAS",)),
    ("Onde ele estudou?", ("FORMAÇÃO",)),
    ("Qual projeto monitora preços?", ("Bot de ofertas",)),
]
LIMIT = 9  # the whole corpus, so positions are visible


def position(titles: list[str], wanted: tuple[str, ...]) -> str:
    for i, title in enumerate(titles, start=1):
        if title.startswith(wanted):
            return str(i)
    return "-"


if __name__ == "__main__":
    print(f"{'pergunta':40} {'vetor':>6} {'lexical':>8} {'hibrida':>8} {'+rerank':>8} {'rerank_ms':>10}")
    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        register_vector(conn)
        for question, wanted in CASES:
            embedding = embed_query(question)
            vector = [r.title for r in repository.search_chunks(conn, embedding, LIMIT)]
            lexical = [r.title for r in repository.search_lexical(conn, question, LIMIT)]
            hybrid = repository.search_hybrid(conn, question, embedding, limit=LIMIT)

            started = time.perf_counter()
            reranked = asyncio.run(rerank(question, hybrid, top_k=LIMIT))
            rerank_ms = int((time.perf_counter() - started) * 1000)
            flag = " (FALLBACK)" if reranked.used_fallback else ""

            print(
                f"{question:40} {position(vector, wanted):>6} "
                f"{position(lexical, wanted):>8} "
                f"{position([r.title for r in hybrid], wanted):>8} "
                f"{position([r.title for r in reranked.chunks], wanted):>8} "
                f"{rerank_ms:>10}{flag}"
            )
