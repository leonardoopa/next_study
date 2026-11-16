"""Ask a question, get the closest chunks from pgvector.

Run (inside the api container):
    docker compose exec api python scripts/search.py "Quantos estagiários ele mentorou?" -k 3
"""

import argparse

import psycopg
from pgvector.psycopg import register_vector

from app import repository
from app.config import get_settings
from app.embeddings import embed_query

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=3, help="how many chunks to return")
    args = parser.parse_args()

    query_embedding = embed_query(args.question)
    with psycopg.connect(get_settings().database_url) as conn:
        register_vector(conn)
        results = repository.search_chunks(conn, query_embedding, args.k)

    print(args.question)
    for r in results:
        print(f"  {r.similarity:.3f}  {r.title[:70]}")
