"""Pure-function tests for the streaming helpers."""

import json

from app.generation import extract_citations
from app.sse import sse_event


def test_sse_event_format():
    assert sse_event("token", {"text": "Ele"}) == 'event: token\ndata: {"text": "Ele"}\n\n'


def test_sse_event_keeps_accents_and_escapes_newlines():
    out = sse_event("token", {"text": "mentorou\n10 estagiários"})
    assert "estagiários" in out  # ensure_ascii=False
    data_line = out.split("\n")[1]
    assert data_line.startswith("data: ")
    assert json.loads(data_line.removeprefix("data: ")) == {"text": "mentorou\n10 estagiários"}
    assert out.endswith("\n\n") and out.count("\n\n") == 1  # only the final blank line


def test_extract_citations_sorted_deduplicated_and_in_range():
    assert extract_citations("a [2][1] b [2] c [9] d [0]", n_sources=3) == [1, 2]


def test_extract_citations_none_found():
    assert extract_citations("Não encontrei essa informação no currículo.", n_sources=3) == []
