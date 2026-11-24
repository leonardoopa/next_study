"""Evaluate retrieval (vector, lexical, hybrid, hybrid+rerank) on a golden set.

Run (inside the api container):
    docker compose exec api python scripts/eval_retrieval.py eval/lgpd_golden.jsonl --document lgpd

Options: --depth N (how many results each mode returns, default 10), --modes a,b,c,
--json PATH (raw per-question results; default eval/results/<date>-<document>.json).
Retrieval only: it never generates an answer. Cost: one embedding per question and, for the
rerank mode, one model call per answerable question.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from app.config import get_settings
from app.evaluation.golden import load_golden
from app.evaluation.report import format_text, summarize, to_json
from app.evaluation.runner import MODES, evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("golden")
    parser.add_argument("--document", required=True)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    modes = tuple(args.modes.split(","))
    unknown = set(modes) - set(MODES)
    if unknown:
        sys.exit(f"unknown modes: {sorted(unknown)}; choose from {MODES}")

    settings = get_settings()
    items = load_golden(Path(args.golden))
    meta = {
        "date": date.today().isoformat(),
        "document": args.document,
        "depth": args.depth,
        "golden": args.golden,
        "embedding_model": settings.embedding_model,
        "rerank_model": settings.rerank_model,
        "rerank_timeout_s": settings.rerank_timeout_s,
    }

    def progress(n, total, item):
        print(f"\r  {n}/{total} {item.id}", end="", file=sys.stderr, flush=True)

    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        register_vector(conn)
        results = evaluate(conn, items, args.document, args.depth, modes, progress)
    print("\r" + " " * 30, end="\r", file=sys.stderr)

    print(format_text(results, modes, meta))
    if "hybrid_rerank" in modes:
        s = summarize(results, "hybrid_rerank")
        if s["fallbacks"] > 0.1 * s["n"]:
            print(
                f"\nWARNING: the rerank fell back to the hybrid order in {s['fallbacks']} of "
                f"{s['n']} questions (provider errors or timeouts). Its numbers are NOT a "
                "valid measure of reranking; run again when the model is stable."
            )

    out = Path(args.json or f"eval/results/{meta['date']}-{args.document}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_json(results, modes, meta), ensure_ascii=False, indent=2))
    print(f"\nraw results: {out}")
