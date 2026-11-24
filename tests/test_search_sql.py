"""Integration tests for lexical and hybrid SQL search.

They need the compose database (run inside the api container) and are skipped without
it. Everything happens in a transaction that is rolled back, so no data is left behind.
Test chunks use made-up words so they cannot collide with real chunks in the table.
"""

import psycopg
import pytest
from pgvector import Vector
from pgvector.psycopg import register_vector

from app import repository
from app.config import get_settings

DIM = 768


def one_hot(i: int) -> list[float]:
    """A vector with a single 1.0: identical to itself (distance 0), orthogonal to others."""
    v = [0.0] * DIM
    v[i] = 1.0
    return v


@pytest.fixture
def conn():
    try:
        connection = psycopg.connect(get_settings().database_url, connect_timeout=2)
    except psycopg.OperationalError:
        pytest.skip("database not reachable")
    register_vector(connection)
    doc_id = connection.execute(
        "INSERT INTO documents (name) VALUES ('__test__') RETURNING id"
    ).fetchone()[0]

    def add(title: str, content: str, hot: int) -> None:
        connection.execute(
            "INSERT INTO chunks (document_id, position, section, title, content,"
            " content_hash, embedding) VALUES (%s, 0, 'S', %s, %s, %s, %s)",
            (doc_id, title, content, f"hash-{title}", Vector(one_hot(hot))),
        )

    add("Z-both", "usamos quokkamq e zebrinhas nas docas", hot=2)  # keyword match AND vector match
    add("Y-lexical", "quokkamq tambem aparece aqui", hot=4)  # keyword match only
    add("X-vector", "texto sobre banana e laranja", hot=3)  # no keyword match
    add("A-accent", "receita de açaíforme gelado", hot=5)
    yield connection
    connection.rollback()
    connection.close()


def test_lexical_finds_a_rare_exact_term(conn):
    titles = [r.title for r in repository.search_lexical(conn, "quokkamq")]
    assert set(titles) == {"Z-both", "Y-lexical"}


def test_lexical_ignores_accents_in_the_question(conn):
    results = repository.search_lexical(conn, "acaiforme")  # no accent typed
    assert [r.title for r in results] == ["A-accent"]


def test_lexical_stems_words(conn):
    # "zebrinhas" in the text, "zebrinha" in the question: same stem
    assert [r.title for r in repository.search_lexical(conn, "zebrinha")] == ["Z-both"]


def test_lexical_returns_nothing_for_unknown_or_stopword_only_queries(conn):
    assert repository.search_lexical(conn, "zzzzzzzz") == []
    assert repository.search_lexical(conn, "de a o") == []  # only stopwords


def test_lexical_reports_its_own_rank_and_no_vector_rank(conn):
    first = repository.search_lexical(conn, "quokkamq")[0]
    assert first.lexical_rank == 1
    assert first.vector_rank is None


def test_hybrid_puts_the_chunk_found_by_both_searches_first(conn):
    results = repository.search_hybrid(conn, "quokkamq", one_hot(2), limit=5)
    top = results[0]
    assert top.title == "Z-both"
    assert top.vector_rank == 1  # identical vector
    assert top.lexical_rank is not None  # also matched the keyword


def test_hybrid_still_returns_a_chunk_only_the_keyword_search_finds(conn):
    results = repository.search_hybrid(conn, "quokkamq", one_hot(2), limit=10)
    y = next(r for r in results if r.title == "Y-lexical")
    assert y.lexical_rank is not None


def test_hybrid_score_is_the_sum_of_reciprocal_ranks(conn):
    top = repository.search_hybrid(conn, "quokkamq", one_hot(2), limit=1, rrf_k=60)[0]
    expected = 1 / (60 + top.vector_rank) + 1 / (60 + top.lexical_rank)
    assert top.score == pytest.approx(expected)


def test_document_filter_limits_every_kind_of_search(conn):
    own = {"Z-both", "Y-lexical", "X-vector", "A-accent"}

    vector = repository.search_chunks(conn, one_hot(2), k=20, document="__test__")
    assert {r.title for r in vector} == own  # only this document, not the real ones

    lexical = repository.search_lexical(conn, "quokkamq", document="__test__")
    assert {r.title for r in lexical} == {"Z-both", "Y-lexical"}

    hybrid = repository.search_hybrid(conn, "quokkamq", one_hot(2), limit=20, document="__test__")
    assert {r.title for r in hybrid} <= own
    assert hybrid[0].title == "Z-both"


def test_an_unknown_document_returns_nothing(conn):
    assert repository.search_chunks(conn, one_hot(2), k=5, document="nao-existe") == []
    assert repository.search_lexical(conn, "quokkamq", document="nao-existe") == []
    assert repository.search_hybrid(conn, "quokkamq", one_hot(2), document="nao-existe") == []
