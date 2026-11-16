-- Schema for ingestion. Idempotent (IF NOT EXISTS), so it can be re-run safely.
-- Docker only runs *.sql from this folder on first init of an empty volume; on an
-- existing database apply it by hand:
--   docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
--     -v ON_ERROR_STOP=1 -f /docker-entrypoint-initdb.d/002_schema.sql'

CREATE TABLE IF NOT EXISTS documents (
    id          bigserial PRIMARY KEY,
    name        text NOT NULL UNIQUE,          -- e.g. 'cv'; re-ingesting the same name is detected
    source_path text,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    document_id  bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    position     integer NOT NULL,             -- order inside the document
    section      text NOT NULL,                -- e.g. 'EXPERIÊNCIA PROFISSIONAL'
    title        text NOT NULL,                -- e.g. job or project name
    content      text NOT NULL,                -- text shown to the LLM and cited
    content_hash text NOT NULL,                -- sha256 of the embedded text, for idempotency
    embedding    vector(768) NOT NULL,         -- gemini-embedding-2 with output_dimensionality=768
    created_at   timestamptz NOT NULL DEFAULT now(),
    -- Same chunk twice in the same document is rejected: ingestion can be re-run safely.
    UNIQUE (document_id, content_hash)
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);

-- No vector index yet: with ~9 rows an exact scan is instant. HNSW comes with real data:
--   CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
