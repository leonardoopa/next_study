"""Sanity checks for a golden set, against the documents stored in the database.

Run (inside the api container):
    docker compose exec api python scripts/check_golden.py eval/lgpd_golden.jsonl lgpd

It checks that every expected article exists in the database, and measures how much each
question COPIES words from its expected chunks (word overlap). A question that copies the
article hands the win to keyword search; paraphrases should overlap little.
"""

import json
import re
import statistics
import sys
import unicodedata
from collections import defaultdict

import psycopg

from app.config import get_settings

# Frequent words that say nothing about the topic (too short or too generic to count).
STOPWORDS = {
    "para", "como", "quais", "qual", "quando", "esta", "este", "essa", "esse", "mais",
    "pode", "podem", "uma", "uns", "dos", "das", "que", "com", "sem", "por", "sobre",
    "seus", "suas", "dele", "dela", "deve", "devem", "ser", "sao", "foi", "tem", "ter",
    "lei", "caso", "casos", "quem", "onde", "quanto", "quantos", "quantas", "entre",
}  # fmt: skip


def words(text: str) -> set[str]:
    """Lower-cased, unaccented words with 5+ letters, cut to a 5-letter stem."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return {w[:5] for w in re.findall(r"[a-z]{5,}", text) if w not in STOPWORDS}


def overlap(question: str, chunk_text: str) -> float:
    q = words(question)
    return len(q & words(chunk_text)) / len(q) if q else 0.0


def label(title: str) -> str:
    return title.split(" (")[0]  # "Art. 18 (parte 1/2)" -> "Art. 18"


if __name__ == "__main__":
    path, document = sys.argv[1], sys.argv[2]
    items = [json.loads(line) for line in open(path, encoding="utf-8")]

    with psycopg.connect(get_settings().database_url) as conn:
        rows = conn.execute(
            "SELECT c.title, c.content FROM chunks c JOIN documents d ON d.id = c.document_id"
            " WHERE d.name = %s",
            (document,),
        ).fetchall()
    text_by_label: dict[str, str] = defaultdict(str)
    for title, content in rows:
        text_by_label[label(title)] += "\n" + content

    problems = []
    ids = [q["id"] for q in items]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")
    by_category: dict[str, list[float]] = defaultdict(list)
    for q in items:
        missing = [e for e in q["expected"] if e not in text_by_label]
        if missing:
            problems.append(f"{q['id']}: expected article not in the database: {missing}")
        if q["answerable"] == (not q["expected"]):
            problems.append(f"{q['id']}: answerable and expected disagree")
        if q["expected"]:
            best = max(overlap(q["question"], text_by_label[e]) for e in q["expected"] if e in text_by_label)
            by_category[q["category"]].append(best)
            q["_overlap"] = best

    print(f"{len(items)} questions | documents in db: {len(text_by_label)} articles")
    print("\ncategory      n   overlap with the expected article (mean / max)")
    for cat, vals in by_category.items():
        print(f"  {cat:<12} {len(vals):>2}   {statistics.mean(vals):.2f} / {max(vals):.2f}")
    leaks = [q for q in items if q["category"] == "parafrase" and q.get("_overlap", 0) > 0.5]
    print("\nparaphrases that copy too much (> 0.50):", [(q["id"], round(q["_overlap"], 2)) for q in leaks] or "none")
    print("\nproblems:", problems or "none")
    sys.exit(1 if problems else 0)
