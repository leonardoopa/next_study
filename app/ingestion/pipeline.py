"""Ingestion pipeline: PDF -> text -> chunks -> embeddings -> pgvector.

Idempotent: running it again duplicates nothing and makes no embedding call for chunks
whose text did not change.
"""

import hashlib
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from app import repository
from app.config import get_settings
from app.embeddings import embed_documents, format_document
from app.ingestion.cv_splitter import build_chunks
from app.ingestion.extract import extract_text
from app.schemas import EmbeddedChunk, IngestSummary


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_cv(path: Path, name: str) -> IngestSummary:
    chunks = build_chunks(extract_text(path))
    # Hash of the exact text that gets embedded: if prefix or content changes, hash changes.
    hashes = [_sha256(format_document(c)) for c in chunks]

    # autocommit=True: reads run outside a transaction, so no transaction stays open
    # during the (slow) network call to the embedding API. Writes use an explicit one.
    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        register_vector(conn)
        with conn.transaction():
            document_id = repository.upsert_document(conn, name, str(path))
        known = repository.existing_hashes(conn, document_id)

        new = [
            (i, c, h)
            for i, (c, h) in enumerate(zip(chunks, hashes, strict=True))
            if h not in known
        ]
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

    return IngestSummary(
        total=len(chunks),
        new=len(new),
        existing=len(chunks) - len(new),
        removed=removed,
    )
