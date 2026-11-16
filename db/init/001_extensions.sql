-- Runs once, when the Postgres volume is first created.
-- The schema (documents, chunks) comes on Day 2, after the embedding model is chosen,
-- because the vector dimension depends on the model.
CREATE EXTENSION IF NOT EXISTS vector;
