"""Turn the raw results into tables: per mode, per category, and mode against mode."""

import statistics
from collections.abc import Sequence

from app.evaluation.metrics import (
    compare_ranks,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    sign_test_p,
)
from app.evaluation.runner import QuestionResult

KS = (1, 3, 5)


def answerable(results: Sequence[QuestionResult]) -> list[QuestionResult]:
    return [r for r in results if r.item.answerable]


def _percentile(values: list[int], pct: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(pct * len(ordered)))]


def summarize(results: Sequence[QuestionResult], mode: str) -> dict:
    rows = [r for r in answerable(results) if mode in r.outcomes]
    n = len(rows)
    summary: dict = {"mode": mode, "n": n}
    for k in KS:
        summary[f"hit@{k}"] = sum(hit_at_k(r.outcomes[mode].rank, k) for r in rows) / n
        summary[f"recall@{k}"] = statistics.mean(
            recall_at_k(r.outcomes[mode].ranked, r.item.expected, k) for r in rows
        )
    summary["mrr"] = statistics.mean(reciprocal_rank(r.outcomes[mode].rank) for r in rows)
    times = [r.outcomes[mode].ms for r in rows]
    summary["ms_p50"], summary["ms_p90"] = _percentile(times, 0.5), _percentile(times, 0.9)
    summary["fallbacks"] = sum(r.outcomes[mode].fallback for r in rows)
    return summary


def by_category(results: Sequence[QuestionResult], modes: Sequence[str]) -> list[dict]:
    categories = list(dict.fromkeys(r.item.category for r in answerable(results)))
    table = []
    for category in categories:
        rows = [r for r in answerable(results) if r.item.category == category]
        entry: dict = {"category": category, "n": len(rows)}
        for mode in modes:
            have = [r for r in rows if mode in r.outcomes]
            entry[mode] = {
                "hit@3": sum(hit_at_k(r.outcomes[mode].rank, 3) for r in have) / len(have),
                "mrr": statistics.mean(reciprocal_rank(r.outcomes[mode].rank) for r in have),
            }
        table.append(entry)
    return table


def pairwise(results: Sequence[QuestionResult], a: str, b: str) -> dict:
    rows = [r for r in answerable(results) if a in r.outcomes and b in r.outcomes]
    wins, losses, ties = compare_ranks(
        [r.outcomes[a].rank for r in rows], [r.outcomes[b].rank for r in rows]
    )
    return {"a": a, "b": b, "wins": wins, "losses": losses, "ties": ties,
            "p": sign_test_p(wins, losses)}  # fmt: skip


def disagreements(results: Sequence[QuestionResult], modes: Sequence[str]) -> list[QuestionResult]:
    """Questions where the modes do not all put the answer at the same position."""
    return [
        r for r in answerable(results)
        if len({r.outcomes[m].rank for m in modes if m in r.outcomes}) > 1
    ]  # fmt: skip


def similarity_split(results: Sequence[QuestionResult]) -> dict:
    """Cosine of the best chunk, for questions the document answers and for those it does not."""
    def tops(flag: bool) -> list[float]:
        return [
            r.outcomes["vector"].top_similarity
            for r in results
            if r.item.answerable == flag and "vector" in r.outcomes
            and r.outcomes["vector"].top_similarity is not None
        ]  # fmt: skip

    yes, no = tops(True), tops(False)
    return {
        "answerable": statistics.mean(yes) if yes else None,
        "unanswerable": statistics.mean(no) if no else None,
        "n_yes": len(yes),
        "n_no": len(no),
    }


def _rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def format_text(results: Sequence[QuestionResult], modes: Sequence[str], meta: dict) -> str:
    ans = answerable(results)
    out = [
        f"Retrieval evaluation | document={meta['document']} | {len(ans)} answerable "
        f"+ {len(results) - len(ans)} unanswerable | depth {meta['depth']}",
        "",
        f"{'mode':<15}{'hit@1':>7}{'hit@3':>7}{'hit@5':>7}{'rec@5':>7}{'MRR':>7}"
        f"{'ms p50':>8}{'ms p90':>8}{'fallback':>9}",
    ]
    for mode in modes:
        s = summarize(results, mode)
        out.append(
            f"{mode:<15}{s['hit@1']:>7.2f}{s['hit@3']:>7.2f}{s['hit@5']:>7.2f}"
            f"{s['recall@5']:>7.2f}{s['mrr']:>7.3f}{s['ms_p50']:>8}{s['ms_p90']:>8}"
            f"{s['fallbacks']:>9}"
        )
    out += ["", "By category (hit@3 / MRR):",
            f"  {'category':<13}{'n':>3}  " + "".join(f"{m:>17}" for m in modes)]  # fmt: skip
    for row in by_category(results, modes):
        cells = "".join(
            f"{row[m]['hit@3']:>9.2f} /{row[m]['mrr']:>6.2f}" if m in row else f"{'':>17}"
            for m in modes
        )
        out.append(f"  {row['category']:<13}{row['n']:>3}  {cells}")

    out += ["", "Mode against mode (rank of the answer; wins = first mode ranks it better):"]
    pairs = [(a, b) for a, b in (("hybrid", "vector"), ("lexical", "vector"),
                                 ("hybrid_rerank", "hybrid"), ("hybrid_rerank", "vector"))
             if a in modes and b in modes]  # fmt: skip
    for a, b in pairs:
        p = pairwise(results, a, b)
        verdict = "inconclusive" if p["p"] >= 0.05 else "likely real"
        out.append(
            f"  {a} vs {b}: {p['wins']} wins, {p['losses']} losses, {p['ties']} ties"
            f" | sign test p={p['p']:.2f} ({verdict})"
        )

    out += ["", "Where the modes disagree (answer position per mode, '-' = not found):"]
    for r in disagreements(results, modes):
        ranks = "  ".join(f"{m}={_rank(r.outcomes[m].rank)}" for m in modes if m in r.outcomes)
        out.append(f"  {r.item.id} {r.item.category:<10} {ranks}  | {r.item.question[:58]}")

    sim = similarity_split(results)
    if sim["answerable"] is not None and sim["unanswerable"] is not None:
        out += ["", f"Best-chunk cosine (vector): answerable {sim['answerable']:.3f} "
                    f"(n={sim['n_yes']}) vs unanswerable {sim['unanswerable']:.3f} "
                    f"(n={sim['n_no']})"]  # fmt: skip
    return "\n".join(out)


def to_json(results: Sequence[QuestionResult], modes: Sequence[str], meta: dict) -> dict:
    return {
        "meta": meta,
        "summary": [summarize(results, m) for m in modes],
        "by_category": by_category(results, modes),
        "questions": [
            {
                "id": r.item.id,
                "category": r.item.category,
                "question": r.item.question,
                "expected": list(r.item.expected),
                "answerable": r.item.answerable,
                "modes": {
                    m: {"rank": o.rank, "top5": o.ranked[:5], "ms": o.ms, "fallback": o.fallback}
                    for m, o in r.outcomes.items()
                },
            }
            for r in results
        ],
    }
