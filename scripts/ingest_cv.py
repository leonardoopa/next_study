"""Day 2: ingest the CV. PDF -> text -> chunks -> embeddings -> pgvector.

Run (inside the api container):
    docker compose exec api python scripts/ingest_cv.py

Idempotent: run it again and nothing is duplicated and no embedding call is made for
chunks that did not change.
"""

import hashlib

import psycopg
from pgvector.psycopg import register_vector

from app import repository
from app.config import get_settings
from app.embeddings import embed_documents, format_document
from app.schemas import EmbeddedChunk
from extract_cv import CV_PATH, extract_text
from split_cv import build_chunks

DOCUMENT_NAME = "cv"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    chunks = build_chunks(extract_text(CV_PATH))
    # Hash of the exact text that gets embedded: if prefix or content changes, hash changes.
    hashes = [sha256(format_document(c)) for c in chunks]

    # autocommit=True: reads run outside a transaction, so no transaction stays open
    # during the (slow) network call to the embedding API. Writes use an explicit one.
    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        register_vector(conn)
        with conn.transaction():
            document_id = repository.upsert_document(conn, DOCUMENT_NAME, str(CV_PATH))
        known = repository.existing_hashes(conn, document_id)

        new = [(i, c, h) for i, (c, h) in enumerate(zip(chunks, hashes, strict=True)) if h not in known]
        # Only new chunks cost an API call.
        vectors = embed_documents([c for _, c, _ in new])
        items = [
            EmbeddedChunk(position=i, chunk=c, content_hash=h, embedding=v)
            for (i, c, h), v in zip(new, vectors, strict=True)
        ]

        # All writes together: either everything lands or nothing does.
        with conn.transaction():
            repository.insert_chunks(conn, document_id, items)
            repository.update_positions(
                conn, document_id, {h: i for i, h in enumerate(hashes) if h in known}
            )
            removed = repository.delete_stale(conn, document_id, set(hashes))

    print(
        f"chunks: {len(chunks)} | novos: {len(new)} | ja existiam: {len(chunks) - len(new)} "
        f"| removidos: {removed}"
    )


if __name__ == "__main__":
    main()
