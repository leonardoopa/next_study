-- Lexical (keyword) search for hybrid retrieval. Idempotent, like 002_schema.sql.
-- Docker runs *.sql here only on first init of an empty volume; on an existing database:
--   docker compose exec db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
--     -v ON_ERROR_STOP=1 -f /docker-entrypoint-initdb.d/003_fulltext.sql'

CREATE EXTENSION IF NOT EXISTS unaccent;

-- unaccent() is only STABLE, but a generated column needs an IMMUTABLE expression.
-- This wrapper pins the dictionary, which makes it safe to declare IMMUTABLE.
CREATE OR REPLACE FUNCTION immutable_unaccent(text) RETURNS text
    LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT
    AS $$ SELECT public.unaccent('public.unaccent', $1) $$;

-- tsvector = the text reduced to searchable lexemes: lower-cased, unaccented, stopwords
-- removed and words stemmed by the Portuguese dictionary ("estagiários" -> "estagiari").
-- STORED + GENERATED: Postgres keeps it in sync with title/content, no code to maintain.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector('portuguese', immutable_unaccent(title || ' ' || content))
    ) STORED;

-- GIN index: maps each lexeme to the rows that contain it (an inverted index).
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
