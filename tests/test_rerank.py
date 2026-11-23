"""Rerank tests: pure functions plus the fallback, with the model call replaced."""

import asyncio

import pytest
from google.genai import errors

from app import rerank as rerank_module
from app.rerank import PassageScore, apply_scores, build_rerank_prompt, rerank
from app.schemas import RankedChunk


def _c(title: str) -> RankedChunk:
    return RankedChunk("S", title, f"content {title}", 0.1, vector_rank=1, lexical_rank=1)


CANDIDATES = [_c("A"), _c("B"), _c("C"), _c("D")]


def test_prompt_numbers_passages_from_one_and_hides_the_retrieval_order():
    prompt = build_rerank_prompt("Pergunta?", CANDIDATES[:2])
    assert "[1] S > A\ncontent A" in prompt
    assert "[2] S > B\ncontent B" in prompt
    assert "rank" not in prompt.lower().split("question:")[0]
    assert prompt.endswith("Question: Pergunta?")


def test_sorts_by_score_and_remembers_the_original_position():
    scores = [PassageScore(id=1, score=2), PassageScore(id=2, score=9), PassageScore(id=3, score=5)]
    result = apply_scores(CANDIDATES[:3], scores, top_k=3)
    assert [(r.title, r.original_rank) for r in result] == [("B", 2), ("C", 3), ("A", 1)]


def test_ties_keep_the_original_order():
    scores = [PassageScore(id=i, score=7) for i in (1, 2, 3, 4)]
    assert [r.title for r in apply_scores(CANDIDATES, scores, top_k=4)] == ["A", "B", "C", "D"]


def test_top_k_cuts_the_list():
    scores = [PassageScore(id=i, score=i) for i in (1, 2, 3, 4)]
    assert [r.title for r in apply_scores(CANDIDATES, scores, top_k=2)] == ["D", "C"]


def test_ignores_invented_ids_repeated_ids_and_clamps_scores():
    scores = [
        PassageScore(id=9, score=10),  # does not exist
        PassageScore(id=0, score=10),  # does not exist
        PassageScore(id=2, score=99),  # clamped to 10
        PassageScore(id=2, score=1),  # repeated: ignored
        PassageScore(id=1, score=-4),  # clamped to 0
    ]
    result = apply_scores(CANDIDATES[:3], scores, top_k=3)
    assert [(r.title, r.rerank_score) for r in result] == [("B", 10.0), ("A", 0.0), ("C", 0.0)]


def test_raises_when_no_score_is_usable():
    with pytest.raises(ValueError):
        apply_scores(CANDIDATES, [PassageScore(id=50, score=5)], top_k=3)


def test_falls_back_to_the_original_order_when_the_model_fails(monkeypatch):
    async def broken(_question, _candidates):
        raise errors.APIError(500, {"error": {"message": "boom"}})

    monkeypatch.setattr(rerank_module, "_score", broken)
    result = asyncio.run(rerank("q", CANDIDATES, top_k=3))
    assert result.used_fallback is True
    assert [r.title for r in result.chunks] == ["A", "B", "C"]
    assert all(r.rerank_score is None for r in result.chunks)


def test_falls_back_on_unusable_output(monkeypatch):
    async def useless(_question, _candidates):
        return [PassageScore(id=99, score=5)]

    monkeypatch.setattr(rerank_module, "_score", useless)
    assert asyncio.run(rerank("q", CANDIDATES, top_k=2)).used_fallback is True


def test_no_candidates_is_not_a_failure():
    result = asyncio.run(rerank("q", [], top_k=3))
    assert result.chunks == [] and result.used_fallback is False
