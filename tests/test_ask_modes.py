"""The /ask retrieval modes, with every external call replaced (no database, no Gemini)."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import ask as ask_module
from app.schemas import Answer, RankedChunk, RerankedChunk, RerankResult, SearchResult


class FakePool:
    @asynccontextmanager
    async def connection(self, timeout=None):
        yield object()


def _ranked(title: str, vec: int | None, lex: int | None) -> RankedChunk:
    return RankedChunk("S", title, f"content {title}", 0.03, vector_rank=vec, lexical_rank=lex)


HYBRID = [_ranked("A", 1, None), _ranked("B", 2, 1), _ranked("C", 3, 2)]


@pytest.fixture
def env(monkeypatch):
    calls = SimpleNamespace(
        hybrid_limit=None, rerank_top_k=None, prompt_titles=None, vector=0, document="unset"
    )
    state = SimpleNamespace(fallback=False)

    async def embed(_q):
        return [0.0]

    async def vector(_conn, _emb, k, document=None):
        calls.vector += 1
        calls.document = document
        return [SearchResult("S", "V1", "content V1", 0.8)][:k]

    async def hybrid(_conn, _text, _emb, limit=5, document=None, **_kw):
        calls.hybrid_limit = limit
        calls.document = document
        return HYBRID[:limit]

    async def fake_rerank(_q, candidates, top_k=5):
        calls.rerank_top_k = top_k
        if state.fallback:  # original order
            chunks = [RerankedChunk("S", c.title, c.content, i, None)
                      for i, c in enumerate(candidates[:top_k], 1)]
            return RerankResult(chunks, used_fallback=True)
        best = RerankedChunk("S", "C", "content C", original_rank=3, rerank_score=9.0)
        other = RerankedChunk("S", "A", "content A", original_rank=1, rerank_score=2.0)
        return RerankResult([best, other][:top_k], used_fallback=False)

    async def answer(_q, chunks):
        calls.prompt_titles = [c.title for c in chunks]
        return Answer("ok [1]", True, [1])

    async def open_stream(_q, chunks):
        calls.prompt_titles = [c.title for c in chunks]

        async def gen():
            yield SimpleNamespace(text="Ele mentorou ")
            yield SimpleNamespace(text="[1].")

        return gen()

    async def exists(_conn, name):
        return name in {"cv", "lgpd"}

    async def list_docs(_conn):
        return [("cv", 9), ("lgpd", 87)]

    monkeypatch.setattr(ask_module.repository, "adocument_exists", exists)
    monkeypatch.setattr(ask_module.repository, "alist_documents", list_docs)
    monkeypatch.setattr(ask_module, "aembed_query", embed)
    monkeypatch.setattr(ask_module.repository, "asearch_chunks", vector)
    monkeypatch.setattr(ask_module.repository, "asearch_hybrid", hybrid)
    monkeypatch.setattr(ask_module, "rerank", fake_rerank)
    monkeypatch.setattr(ask_module, "generate_answer", answer)
    monkeypatch.setattr(ask_module, "open_stream", open_stream)

    app = FastAPI()
    app.include_router(ask_module.router)
    app.state.pool = FakePool()
    return SimpleNamespace(client=TestClient(app), calls=calls, state=state)


def _post(env, **body):
    return env.client.post("/ask", json={"question": "pergunta de teste", **body})


def test_default_mode_is_vector_and_keeps_the_old_behavior(env):
    data = _post(env).json()
    assert data["mode"] == "vector"
    assert env.calls.hybrid_limit is None  # hybrid search never ran
    assert data["sources"][0]["similarity"] == 0.8
    assert data["sources"][0]["vector_rank"] is None


def test_hybrid_mode_asks_for_k_chunks_and_exposes_both_ranks(env):
    data = _post(env, mode="hybrid", k=2).json()
    assert env.calls.hybrid_limit == 2
    assert env.calls.rerank_top_k is None  # no rerank in this mode
    assert env.calls.prompt_titles == ["A", "B"]
    source = data["sources"][0]  # cited id 1 = chunk A
    assert (source["vector_rank"], source["lexical_rank"], source["similarity"]) == (1, None, None)


def test_hybrid_rerank_fetches_a_pool_then_reorders_down_to_k(env):
    data = _post(env, mode="hybrid_rerank", k=2).json()
    assert env.calls.hybrid_limit == 10  # RERANK_POOL, bigger than k
    assert env.calls.rerank_top_k == 2
    assert env.calls.prompt_titles == ["C", "A"]  # the reranker's order reached the prompt
    source = data["sources"][0]  # id 1 = C, the reranker's favourite
    assert source["title"] == "C"
    assert source["rerank_score"] == 9.0
    assert (source["vector_rank"], source["lexical_rank"]) == (3, 2)  # provenance kept
    assert data["rerank_fallback"] is False


def test_rerank_fallback_is_reported_and_the_old_order_is_used(env):
    env.state.fallback = True
    data = _post(env, mode="hybrid_rerank", k=2).json()
    assert data["rerank_fallback"] is True
    assert env.calls.prompt_titles == ["A", "B"]
    assert data["sources"][0]["rerank_score"] is None


def test_unknown_mode_is_rejected(env):
    assert _post(env, mode="magic").status_code == 422


def test_stream_in_rerank_mode_sends_sources_with_provenance(env):
    resp = env.client.post(
        "/ask/stream", json={"question": "pergunta de teste", "mode": "hybrid_rerank", "k": 2}
    )
    assert resp.status_code == 200
    events = [e for e in resp.text.split("\n\n") if e.strip()]
    names = [e.split("\n")[0].removeprefix("event: ") for e in events]
    assert names == ["token", "token", "sources", "done"]
    sources = json.loads(events[2].split("\n")[1].removeprefix("data: "))
    assert sources["mode"] == "hybrid_rerank"
    assert sources["answerable"] is True
    assert sources["sources"][0]["title"] == "C"
    assert sources["sources"][0]["rerank_score"] == 9.0


def test_document_filter_reaches_the_vector_search(env):
    _post(env, document="lgpd")
    assert env.calls.document == "lgpd"


def test_document_filter_reaches_the_hybrid_search(env):
    _post(env, mode="hybrid", document="cv")
    assert env.calls.document == "cv"


def test_no_document_means_search_everything(env):
    _post(env, mode="hybrid")
    assert env.calls.document is None


def test_unknown_document_is_a_404_and_nothing_is_searched(env):
    resp = _post(env, document="nao-existe")
    assert resp.status_code == 404
    assert "nao-existe" in resp.json()["detail"]
    assert env.calls.vector == 0 and env.calls.hybrid_limit is None


def test_unknown_document_is_a_404_on_the_stream_too(env):
    resp = env.client.post(
        "/ask/stream", json={"question": "pergunta de teste", "document": "nao-existe"}
    )
    assert resp.status_code == 404


def test_documents_endpoint_lists_names_and_chunk_counts(env):
    resp = env.client.get("/documents")
    assert resp.status_code == 200
    assert resp.json() == [{"name": "cv", "chunks": 9}, {"name": "lgpd", "chunks": 87}]
