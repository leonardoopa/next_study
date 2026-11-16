"""Entry point: ingest the CV into pgvector. All logic lives in app/ingestion/.

Run (inside the api container):
    docker compose exec api python scripts/ingest_cv.py
"""

from pathlib import Path

from app.ingestion.pipeline import ingest_cv

if __name__ == "__main__":
    s = ingest_cv(Path("data/cv.pdf"), name="cv")
    print(f"chunks: {s.total} | novos: {s.new} | ja existiam: {s.existing} | removidos: {s.removed}")
