"""SQL for documents and chunks. Only data access here: no splitting, no embedding."""

import psycopg
from pgvector import Vector

from app.schemas import EmbeddedChunk, RankedChunk, SearchResult


def upsert_document(conn: psycopg.Connection, name: str, source_path: str) -> int:
    """Create the document, or reuse it if the name already exists. Returns its id."""
    row = conn.execute(
        """
        INSERT INTO documents (name, source_path) VALUES (%s, %s)
        ON CONFLICT (name) DO UPDATE SET source_path = EXCLUDED.source_path
        RETURNING id
        """,
        (name, source_path),
    ).fetchone()
    return row[0]


def existing_hashes(conn: psycopg.Connection, document_id: int) -> set[str]:
    rows = conn.execute(
        "SELECT content_hash FROM chunks WHERE document_id = %s", (document_id,)
    ).fetchall()
    return {r[0] for r in rows}


def insert_chunks(
    conn: psycopg.Connection, document_id: int, items: list[EmbeddedChunk]
) -> None:
    """Insert new chunks. A (document_id, content_hash) that already exists is skipped."""
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks
                (document_id, position, section, title, content, content_hash, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id, content_hash) DO NOTHING
            """,
            [
                (
                    document_id,
                    it.position,
                    it.chunk.section,
                    it.chunk.title,
                    it.chunk.text,
                    it.content_hash,
                    Vector(it.embedding),
                )
                for it in items
            ],
        )


def update_positions(
    conn: psycopg.Connection, document_id: int, positions: dict[str, int]
) -> None:
    """Keep the order right for chunks that already existed (hash -> new position)."""
    with conn.cursor() as cur:
        cur.executemany(
            "UPDATE chunks SET position = %s WHERE document_id = %s AND content_hash = %s",
            [(pos, document_id, h) for h, pos in positions.items()],
        )


def delete_stale(
    conn: psycopg.Connection, document_id: int, keep_hashes: set[str]
) -> int:
    """Remove chunks whose text no longer exists in the document. Returns how many."""
    cur = conn.execute(
        "DELETE FROM chunks WHERE document_id = %s AND NOT (content_hash = ANY(%s))",
        (document_id, list(keep_hashes)),
    )
    return cur.rowcount


# Optional filter: NULL means "all documents". One SQL for both cases keeps the queries
# (and their plans) the same shape; the filter is just a condition.
_DOC_FILTER = "(%(doc)s::text IS NULL OR d.name = %(doc)s)"

_SEARCH_SQL = f"""
    SELECT c.section, c.title, c.content, 1 - (c.embedding <=> %(emb)s) AS similarity
    FROM chunks c JOIN documents d ON d.id = c.document_id
    WHERE {_DOC_FILTER}
    ORDER BY c.embedding <=> %(emb)s
    LIMIT %(limit)s
"""


def _to_result(row) -> SearchResult:
    return SearchResult(
        section=row[0], title=row[1], content=row[2], similarity=float(row[3])
    )


def search_chunks(
    conn: psycopg.Connection,
    query_embedding: list[float],
    k: int = 5,
    document: str | None = None,
) -> list[SearchResult]:
    """Top-k chunks closest to the query by cosine distance.

    `<=>` is cosine DISTANCE (0 = identical direction), so ORDER BY ascending puts the
    closest first, and similarity = 1 - distance. `document` limits the search to one
    document by name (None = all).
    """
    params = {"emb": Vector(query_embedding), "limit": k, "doc": document}
    return [_to_result(r) for r in conn.execute(_SEARCH_SQL, params).fetchall()]


async def asearch_chunks(
    conn: psycopg.AsyncConnection,
    query_embedding: list[float],
    k: int = 5,
    document: str | None = None,
) -> list[SearchResult]:
    """Async version of search_chunks, for the API (the connection comes from the pool)."""
    params = {"emb": Vector(query_embedding), "limit": k, "doc": document}
    cur = await conn.execute(_SEARCH_SQL, params)
    return [_to_result(r) for r in await cur.fetchall()]


# --- Lexical and hybrid search ---------------------------------------------------------
# The question becomes a tsquery with OR between terms. websearch_to_tsquery joins terms
# with AND, which is too strict for natural questions ("conhece" would have to appear in
# the chunk). With OR, any matching term qualifies and ts_rank orders by how well.
_TSQUERY_SQL = """
    SELECT CASE
        WHEN %(tsq)s::text IS NULL THEN replace(
            websearch_to_tsquery('portuguese', immutable_unaccent(%(text)s))::text, '&', '|'
        )::tsquery
        -- already-stemmed lexemes ("a" | "b"): the 'simple' config must not stem them again
        ELSE to_tsquery('simple', %(tsq)s::text)
    END AS tsq
"""

_LEXICAL_SQL = f"""
    WITH q AS ({_TSQUERY_SQL})
    SELECT c.section, c.title, c.content, ts_rank(c.tsv, q.tsq, 1) AS score
    FROM chunks c JOIN documents d ON d.id = c.document_id, q
    WHERE c.tsv @@ q.tsq AND {_DOC_FILTER}
    ORDER BY score DESC, c.id
    LIMIT %(limit)s
"""

# RRF: each list contributes 1 / (k + rank). Only positions are used, never raw scores,
# because cosine and ts_rank live on different scales.
_HYBRID_SQL = f"""
    WITH q AS ({_TSQUERY_SQL}),
    vec AS (
        SELECT id, row_number() OVER (ORDER BY dist, id) AS rnk
        FROM (
            SELECT c.id, c.embedding <=> %(emb)s AS dist
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE {_DOC_FILTER}
            ORDER BY dist
            LIMIT %(n)s
        ) s
    ),
    lex AS (
        SELECT id, row_number() OVER (ORDER BY r DESC, id) AS rnk
        FROM (
            SELECT c.id, ts_rank(c.tsv, q.tsq, 1) AS r
            FROM chunks c JOIN documents d ON d.id = c.document_id, q
            WHERE c.tsv @@ q.tsq AND {_DOC_FILTER}
            ORDER BY r DESC
            LIMIT %(n_lex)s
        ) s
    ),
    fused AS (
        SELECT id, sum(w * 1.0 / (%(k)s + rnk)) AS rrf
        FROM (
            SELECT id, rnk, %(w_vec)s::float8 AS w FROM vec
            UNION ALL
            SELECT id, rnk, %(w_lex)s::float8 AS w FROM lex
        ) both_lists
        GROUP BY id
    )
    SELECT c.section, c.title, c.content, f.rrf, v.rnk, l.rnk
    FROM fused f
    JOIN chunks c ON c.id = f.id
    LEFT JOIN vec v ON v.id = f.id
    LEFT JOIN lex l ON l.id = f.id
    ORDER BY f.rrf DESC, c.id
    LIMIT %(limit)s
"""


def search_lexical(
    conn: psycopg.Connection,
    query_text: str,
    limit: int = 5,
    document: str | None = None,
) -> list[RankedChunk]:
    """Keyword search: Postgres full-text search, Portuguese, accent-insensitive."""
    params = {"text": query_text, "limit": limit, "doc": document, "tsq": None}
    rows = conn.execute(_LEXICAL_SQL, params).fetchall()
    return [
        RankedChunk(r[0], r[1], r[2], float(r[3]), vector_rank=None, lexical_rank=i)
        for i, r in enumerate(rows, start=1)
    ]


def _hybrid_params(
    query_text: str,
    query_embedding: list[float],
    limit: int,
    candidates: int,
    rrf_k: int,
    document: str | None,
    lexical_weight: float = 1.0,
    lexical_candidates: int | None = None,
    lexical_terms: list[str] | None = None,
) -> dict:
    return {
        "text": query_text,
        "emb": Vector(query_embedding),
        "n": candidates,
        "n_lex": candidates if lexical_candidates is None else lexical_candidates,
        "k": rrf_k,
        "limit": limit,
        "doc": document,
        "w_vec": 1.0,
        "w_lex": lexical_weight,
        # None = build the query from the question text; a list = use exactly these lexemes
        "tsq": None if lexical_terms is None else " | ".join(_quote(t) for t in lexical_terms),
    }


def _quote(lexeme: str) -> str:
    return "'" + lexeme.replace("'", "''") + "'"


def _to_ranked(r) -> RankedChunk:
    return RankedChunk(r[0], r[1], r[2], float(r[3]), r[4], r[5])


def search_hybrid(
    conn: psycopg.Connection,
    query_text: str,
    query_embedding: list[float],
    limit: int = 5,
    candidates: int = 20,
    rrf_k: int = 60,
    document: str | None = None,
    lexical_weight: float = 1.0,
    lexical_candidates: int | None = None,
    lexical_terms: list[str] | None = None,
) -> list[RankedChunk]:
    """Vector search + keyword search, fused with Reciprocal Rank Fusion.

    Each search contributes its top `candidates`; chunks found by both rise to the top.
    Tuning knobs (defaults reproduce the plain, equal-weight fusion):
      lexical_weight     weight of the keyword list in the fusion (vector is 1.0)
      lexical_candidates how many keyword results enter the fusion (default: `candidates`)
      lexical_terms      stemmed lexemes to search instead of the whole question
    """
    params = _hybrid_params(
        query_text, query_embedding, limit, candidates, rrf_k, document,
        lexical_weight, lexical_candidates, lexical_terms,
    )  # fmt: skip
    return [_to_ranked(r) for r in conn.execute(_HYBRID_SQL, params).fetchall()]


async def asearch_hybrid(
    conn: psycopg.AsyncConnection,
    query_text: str,
    query_embedding: list[float],
    limit: int = 5,
    candidates: int = 20,
    rrf_k: int = 60,
    document: str | None = None,
) -> list[RankedChunk]:
    """Async version of search_hybrid, for the API (connection from the pool)."""
    params = _hybrid_params(
        query_text, query_embedding, limit, candidates, rrf_k, document
    )
    cur = await conn.execute(_HYBRID_SQL, params)
    return [_to_ranked(r) for r in await cur.fetchall()]


# --- Documents -------------------------------------------------------------------------
async def adocument_exists(conn: psycopg.AsyncConnection, name: str) -> bool:
    cur = await conn.execute("SELECT 1 FROM documents WHERE name = %s", (name,))
    return await cur.fetchone() is not None


async def alist_documents(conn: psycopg.AsyncConnection) -> list[tuple[str, int]]:
    """(name, number of chunks) of every document, by name."""
    cur = await conn.execute(
        """
        SELECT d.name, count(c.id)
        FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
        GROUP BY d.name ORDER BY d.name
        """
    )
    return [(r[0], r[1]) for r in await cur.fetchall()]


# --- Helpers to tune the keyword side ---------------------------------------------------
def query_lexemes(conn: psycopg.Connection, text: str) -> list[str]:
    """The question as stemmed, unaccented lexemes with stopwords removed."""
    row = conn.execute(
        "SELECT tsvector_to_array(to_tsvector('portuguese', immutable_unaccent(%s)))", (text,)
    ).fetchone()
    return list(row[0])


def document_frequencies(
    conn: psycopg.Connection, document: str | None = None
) -> tuple[int, dict[str, int]]:
    """(number of chunks, lexeme -> in how many chunks it appears). ts_rank ignores this
    information; the experiments use it to drop terms that appear almost everywhere."""
    rows = conn.execute(
        """
        SELECT tsvector_to_array(c.tsv)
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE (%(doc)s::text IS NULL OR d.name = %(doc)s)
        """,
        {"doc": document},
    ).fetchall()
    counts: dict[str, int] = {}
    for (lexemes,) in rows:
        for lexeme in lexemes:
            counts[lexeme] = counts.get(lexeme, 0) + 1
    return len(rows), counts
