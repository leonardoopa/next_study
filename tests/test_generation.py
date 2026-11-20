"""Pure-function tests: no network, no database."""

from app.generation import NOT_FOUND, ModelAnswer, build_prompt, validate_citations
from app.schemas import SearchResult


def _result(title: str) -> SearchResult:
    return SearchResult(
        section="S", title=title, content=f"content of {title}", similarity=0.5
    )


def test_prompt_numbers_sources_from_one():
    prompt = build_prompt("Pergunta?", [_result("A"), _result("B")])
    assert "[1] S > A\ncontent of A" in prompt
    assert "[2] S > B\ncontent of B" in prompt
    assert prompt.endswith("Question: Pergunta?")


def test_valid_citations_are_kept_sorted_and_deduplicated():
    raw = ModelAnswer(answerable=True, answer="x [2][1]", cited_ids=[2, 1, 2])
    answer = validate_citations(raw, n_sources=3)
    assert answer.answerable is True
    assert answer.cited_ids == [1, 2]


def test_out_of_range_citations_are_dropped():
    raw = ModelAnswer(answerable=True, answer="x [1][9]", cited_ids=[1, 9, 0, -1])
    assert validate_citations(raw, n_sources=3).cited_ids == [1]


def test_answer_with_only_invented_citations_is_refused():
    raw = ModelAnswer(answerable=True, answer="invented [7]", cited_ids=[7])
    answer = validate_citations(raw, n_sources=3)
    assert answer.answerable is False
    assert answer.text == NOT_FOUND
    assert answer.cited_ids == []


def test_unanswerable_keeps_the_model_message_and_no_citations():
    raw = ModelAnswer(
        answerable=False, answer="O currículo não menciona.", cited_ids=[1]
    )
    answer = validate_citations(raw, n_sources=3)
    assert answer.answerable is False
    assert answer.cited_ids == []
    assert answer.text == "O currículo não menciona."
