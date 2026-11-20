"""SQL for documents and chunks. Only data access here: no splitting, no embedding."""

import psycopg
from pgvector import Vector

from app.schemas import EmbeddedChunk, SearchResult


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


def search_chunks(
    conn: psycopg.Connection, query_embedding: list[float], k: int = 5
) -> list[SearchResult]:
    """Top-k chunks closest to the query by cosine distance.

    `<=>` is cosine DISTANCE (0 = identical direction), so ORDER BY ascending puts the
    closest first, and similarity = 1 - distance.
    """
    vec = Vector(query_embedding)
    rows = conn.execute(_SEARCH_SQL, (vec, vec, k)).fetchall()
    return [_to_result(r) for r in rows]


async def asearch_chunks(
    conn: psycopg.AsyncConnection, query_embedding: list[float], k: int = 5
) -> list[SearchResult]:
    """Async version of search_chunks, for the API (the connection comes from the pool)."""
    vec = Vector(query_embedding)
    cur = await conn.execute(_SEARCH_SQL, (vec, vec, k))
    return [_to_result(r) for r in await cur.fetchall()]


_SEARCH_SQL = """
    SELECT section, title, content, 1 - (embedding <=> %s) AS similarity
    FROM chunks
    ORDER BY embedding <=> %s
    LIMIT %s
"""


def _to_result(row) -> SearchResult:
    return SearchResult(
        section=row[0], title=row[1], content=row[2], similarity=float(row[3])
    )
