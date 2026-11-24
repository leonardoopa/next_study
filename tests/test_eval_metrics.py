"""Retrieval metrics and report tables (pure functions, no database or model)."""

import pytest

from app.evaluation.golden import GoldenItem, load_golden
from app.evaluation.metrics import (
    article_label,
    compare_ranks,
    first_rank,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    sign_test_p,
    ranked_articles,
)
from app.evaluation.report import (
    by_category,
    disagreements,
    pairwise,
    similarity_split,
    summarize,
)
from app.evaluation.runner import ModeOutcome, QuestionResult


def test_article_label_drops_the_part_suffix():
    assert article_label("Art. 18 (parte 1/2)") == "Art. 18"
    assert article_label("Art. 55-A") == "Art. 55-A"


def test_ranked_articles_keeps_each_article_once_in_first_seen_order():
    titles = ["Art. 18 (parte 1/2)", "Art. 18 (parte 2/2)", "Art. 7", "Art. 18 (parte 1/2)"]
    assert ranked_articles(titles) == ["Art. 18", "Art. 7"]


def test_two_parts_of_one_article_do_not_push_the_next_article_down():
    # chunk positions 1,2 are the same article, so Art. 7 is the SECOND article, not the third
    ranked = ranked_articles(["Art. 18 (parte 1/2)", "Art. 18 (parte 2/2)", "Art. 7"])
    assert first_rank(ranked, ["Art. 7"]) == 2


def test_first_rank_any_expected_article_counts_and_missing_is_none():
    assert first_rank(["Art. 1", "Art. 8", "Art. 7"], ["Art. 7", "Art. 8"]) == 2
    assert first_rank(["Art. 1", "Art. 2"], ["Art. 7"]) is None


def test_hit_at_k():
    assert hit_at_k(3, 3) is True
    assert hit_at_k(4, 3) is False
    assert hit_at_k(None, 5) is False


def test_recall_at_k_counts_the_share_of_expected_articles_found():
    ranked = ["Art. 7", "Art. 1", "Art. 8", "Art. 2"]
    assert recall_at_k(ranked, ["Art. 7", "Art. 8"], k=1) == 0.5
    assert recall_at_k(ranked, ["Art. 7", "Art. 8"], k=3) == 1.0
    assert recall_at_k(ranked, ["Art. 99"], k=4) == 0.0


def test_mrr_matches_the_hand_computed_value_from_the_lesson():
    # answer in positions 2, 1, missing, 3: MRR = (1/2 + 1 + 0 + 1/3) / 4 = 11/24
    ranks = [2, 1, None, 3]
    assert sum(reciprocal_rank(r) for r in ranks) / 4 == pytest.approx(11 / 24)
    assert sum(hit_at_k(r, 3) for r in ranks) / 4 == 0.75


def test_compare_ranks_counts_wins_losses_and_ties_and_none_is_worst():
    a = [1, 3, None, 2, None]
    b = [2, 3, 5, 1, None]
    assert compare_ranks(a, b) == (1, 2, 2)  # win, tie, loss, loss, tie


def test_sign_test_reproduces_the_numbers_from_the_lesson():
    assert sign_test_p(3, 1) == pytest.approx(0.625)  # 3 wins, 1 loss: pure chance
    assert sign_test_p(6, 0) == pytest.approx(2 / 64)  # about 3%
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 5) == 1.0


def _result(item_id, category, answerable, **ranks):
    item = GoldenItem(item_id, category, "pergunta?", ("Art. 1",) if answerable else (), answerable)
    outcomes = {
        mode: ModeOutcome(titles=[], ranked=["Art. 1"] if rank == 1 else [], rank=rank, ms=10 * (i + 1))
        for i, (mode, rank) in enumerate(ranks.items())
    }
    return QuestionResult(item, outcomes)


RESULTS = [
    _result("q1", "exato", True, vector=1, hybrid=1),
    _result("q2", "exato", True, vector=3, hybrid=1),
    _result("q3", "parafrase", True, vector=1, hybrid=None),
    _result("q4", "parafrase", True, vector=2, hybrid=2),
    _result("q5", "sem_resposta", False, vector=None),
]


def test_summary_ignores_unanswerable_questions():
    s = summarize(RESULTS, "vector")
    assert s["n"] == 4
    assert s["hit@1"] == 0.5 and s["hit@3"] == 1.0
    assert s["mrr"] == pytest.approx((1 + 1 / 3 + 1 + 1 / 2) / 4)


def test_summary_of_a_mode_with_misses():
    s = summarize(RESULTS, "hybrid")
    assert s["hit@5"] == 0.75 and s["mrr"] == pytest.approx((1 + 1 + 0 + 1 / 2) / 4)


def test_by_category_separates_the_modes():
    rows = {r["category"]: r for r in by_category(RESULTS, ["vector", "hybrid"])}
    assert rows["exato"]["hybrid"]["hit@3"] == 1.0
    assert rows["parafrase"]["hybrid"]["hit@3"] == 0.5  # q3 missed, q4 found
    assert rows["parafrase"]["vector"]["hit@3"] == 1.0
    assert rows["parafrase"]["n"] == 2


def test_pairwise_and_disagreements():
    p = pairwise(RESULTS, "hybrid", "vector")
    assert (p["wins"], p["losses"], p["ties"]) == (1, 1, 2)
    assert [r.item.id for r in disagreements(RESULTS, ["vector", "hybrid"])] == ["q2", "q3"]


def test_similarity_split_compares_answerable_with_unanswerable():
    results = [_result("a", "x", True, vector=1), _result("b", "y", False, vector=None)]
    results[0].outcomes["vector"].top_similarity = 0.8
    results[1].outcomes["vector"].top_similarity = 0.6
    s = similarity_split(results)
    assert s["answerable"] == pytest.approx(0.8) and s["unanswerable"] == pytest.approx(0.6)


def test_load_golden_reads_the_real_golden_set():
    items = load_golden("eval/lgpd_golden.jsonl")
    assert len(items) == 40
    assert items[0].expected == ("Art. 18",)
    assert sum(not i.answerable for i in items) == 5
