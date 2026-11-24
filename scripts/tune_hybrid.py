"""Try variants of the hybrid search on the golden set, with a dev/test split.

Run (inside the api container):
    docker compose exec api python scripts/tune_hybrid.py eval/lgpd_golden.jsonl --document lgpd

Why a split: if you pick the best variant on the same questions you report, the score is
optimistic (the variant was chosen BECAUSE it fits those questions). Here the variants are
ranked on the DEV half and the winner is then scored on the TEST half, which played no part
in the choice. Within each category, questions alternate dev, test, dev, test...

Database only after the first run: question embeddings are cached in data/ (gitignored).
"""

import argparse
import json
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from app import repository
from app.config import get_settings
from app.embeddings import embed_query
from app.evaluation.golden import GoldenItem, load_golden
from app.evaluation.metrics import (
    compare_ranks,
    first_rank,
    hit_at_k,
    ranked_articles,
    reciprocal_rank,
    sign_test_p,
)

# name, weight of the keyword list, keyword candidates, max document frequency (None = keep all)
VARIANTS = [
    ("baseline (w1.0)", 1.0, 20, None),
    ("w0.5", 0.5, 20, None),
    ("w0.3", 0.3, 20, None),
    ("w0.15", 0.15, 20, None),
    ("w0.05", 0.05, 20, None),
    ("lex top-5", 1.0, 5, None),
    ("lex top-3", 1.0, 3, None),
    ("idf<=0.30", 1.0, 20, 0.30),
    ("idf<=0.20", 1.0, 20, 0.20),
    ("idf<=0.10", 1.0, 20, 0.10),
    ("idf<=0.20 w0.5", 0.5, 20, 0.20),
    ("idf<=0.20 w0.3", 0.3, 20, 0.20),
    ("idf<=0.10 w0.5", 0.5, 20, 0.10),
    ("idf<=0.20 top-5", 1.0, 5, 0.20),
    ("w0 (vector only)", 0.0, 20, None),
]  # fmt: skip


def split(items: list[GoldenItem]) -> tuple[set[str], set[str]]:
    dev: set[str] = set()
    test: set[str] = set()
    seen: dict[str, int] = defaultdict(int)
    for item in sorted(items, key=lambda i: i.id):
        (dev if seen[item.category] % 2 == 0 else test).add(item.id)
        seen[item.category] += 1
    return dev, test


def embeddings_for(items: list[GoldenItem], document: str) -> dict[str, list[float]]:
    cache_path = Path(f"data/eval_embeddings_{document}.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    missing = [i for i in items if i.question not in cache]
    for item in missing:
        cache[item.question] = embed_query(item.question)
    if missing:
        cache_path.parent.mkdir(exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return cache


def metrics(ranks: list[int | None]) -> dict:
    return {
        "n": len(ranks),
        "hit1": sum(hit_at_k(r, 1) for r in ranks) / len(ranks),
        "hit3": sum(hit_at_k(r, 3) for r in ranks) / len(ranks),
        "mrr": statistics.mean(reciprocal_rank(r) for r in ranks),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("--document", required=True)
    parser.add_argument("--depth", type=int, default=10)
    args = parser.parse_args()

    items = [i for i in load_golden(Path(args.golden)) if i.answerable]
    dev_ids, test_ids = split(items)
    emb = embeddings_for(items, args.document)

    results: dict[str, dict[str, int | None]] = {}  # variant -> question id -> rank
    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        register_vector(conn)
        n_chunks, frequency = repository.document_frequencies(conn, args.document)

        vector: dict[str, int | None] = {}
        for item in items:
            rows = repository.search_chunks(conn, emb[item.question], args.depth, args.document)
            vector[item.id] = first_rank(ranked_articles([r.title for r in rows]), item.expected)

        for name, weight, n_lex, idf_cut in VARIANTS:
            ranks = {}
            for item in items:
                terms = None
                if idf_cut is not None:
                    lexemes = repository.query_lexemes(conn, item.question)
                    terms = [t for t in lexemes if frequency.get(t, 0) / n_chunks <= idf_cut]
                rows = repository.search_hybrid(
                    conn, item.question, emb[item.question], limit=args.depth,
                    document=args.document, lexical_weight=weight,
                    lexical_candidates=n_lex, lexical_terms=terms,
                )  # fmt: skip
                ranks[item.id] = first_rank(
                    ranked_articles([r.title for r in rows]), item.expected
                )
            results[name] = ranks

    def pick(ranks: dict, ids: set[str]) -> list[int | None]:
        return [ranks[i.id] for i in items if i.id in ids]

    all_ids = {i.id for i in items}
    para = {i.id for i in items if i.category == "parafrase"}
    print(f"{len(items)} answerable questions | dev {len(dev_ids)} / test {len(test_ids)}")
    print(f"\n{'variant':<20}{'dev hit@3':>10}{'dev MRR':>9}{'test hit@3':>11}{'test MRR':>9}"
          f"{'all MRR':>9}{'para hit@3':>11}{'vs vector (W/L)':>17}")  # fmt: skip
    rows = []
    for name, ranks in [("vector (reference)", vector), *results.items()]:
        d, t, a = (metrics(pick(ranks, ids)) for ids in (dev_ids, test_ids, all_ids))
        p = metrics(pick(ranks, para))
        w, l_, _ = compare_ranks(pick(ranks, all_ids), pick(vector, all_ids))
        rows.append((name, d, t, a, p, w, l_))
        print(f"{name:<20}{d['hit3']:>10.2f}{d['mrr']:>9.3f}{t['hit3']:>11.2f}{t['mrr']:>9.3f}"
              f"{a['mrr']:>9.3f}{p['hit3']:>11.2f}{f'{w}/{l_}':>17}")  # fmt: skip

    hybrids = [r for r in rows if r[0] not in ("vector (reference)", "w0 (vector only)")]
    best = max(hybrids, key=lambda r: r[1]["mrr"])  # chosen on DEV only
    base = next(r for r in rows if r[0] == "baseline (w1.0)")
    vec = rows[0]
    print(f"\nChosen on the DEV half: {best[0]}  (dev MRR {best[1]['mrr']:.3f})")
    print("Scored on the TEST half, which did not take part in the choice:")
    for label, r in (("  chosen variant", best), ("  baseline hybrid", base), ("  vector only", vec)):
        print(f"{label:<18} hit@1 {r[2]['hit1']:.2f}  hit@3 {r[2]['hit3']:.2f}  MRR {r[2]['mrr']:.3f}  (n={r[2]['n']})")
    wt, lt, _ = compare_ranks(pick(results[best[0]], test_ids), pick(vector, test_ids))
    print(f"  chosen vs vector on TEST: {wt} wins, {lt} losses | sign test p={sign_test_p(wt, lt):.2f}")

    out = Path(f"eval/results/{date.today().isoformat()}-{args.document}-tuning.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": {"dev": sorted(dev_ids), "test": sorted(test_ids)},
        "ranks": {"vector": vector, **results},
        "chosen_on_dev": best[0],
    }, ensure_ascii=False, indent=1))  # fmt: skip
    print(f"\nraw ranks: {out}")
