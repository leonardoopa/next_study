"""Entry point: ingest the LGPD (Lei 13.709/2018) as the document "lgpd".

Run (inside the api container), after saving the law as data/lgpd/lei-13709.html:
    docker compose exec api python scripts/ingest_lgpd.py
"""

from pathlib import Path

from app.ingestion.pipeline import ingest_lgpd

if __name__ == "__main__":
    s = ingest_lgpd(Path("data/lgpd/lei-13709.html"))
    print(
        f"chunks: {s.total} | novos: {s.new} | ja existiam: {s.existing} | removidos: {s.removed}"
    )
