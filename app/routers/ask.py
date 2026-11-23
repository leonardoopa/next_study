import logging
import time

import psycopg
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app import repository
from app.api_models import AskRequest, AskResponse, Source
from app.embeddings import aembed_query
from app.gemini import PROVIDER_ERRORS, classify
from app.generation import (
    ModelOutputError,
    extract_citations,
    generate_answer,
    open_stream,
)
from app.rerank import rerank
from app.schemas import RankedChunk, Retrieval, RetrievedChunk
from app.sse import sse_event

router = APIRouter()
logger = logging.getLogger("app.ask")

RETRY_AFTER_SECONDS = "30"
RERANK_POOL = 10  # candidates handed to the reranker (it reads each one: keep it small)


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _sources(chunks: list[RetrievedChunk], ids: list[int]) -> list[Source]:
    """Build the Source objects for 1-based ids that were already validated."""
    return [
        Source(
            id=i,
            section=chunks[i - 1].section,
            title=chunks[i - 1].title,
            content=chunks[i - 1].content,
            similarity=chunks[i - 1].similarity,
            vector_rank=chunks[i - 1].vector_rank,
            lexical_rank=chunks[i - 1].lexical_rank,
            rerank_score=chunks[i - 1].rerank_score,
        )
        for i in ids
    ]


def _from_ranked(r: RankedChunk, rerank_score: float | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        r.section,
        r.title,
        r.content,
        vector_rank=r.vector_rank,
        lexical_rank=r.lexical_rank,
        rerank_score=rerank_score,
    )


async def _retrieve(
    request: Request, body: AskRequest, timings: dict[str, int]
) -> Retrieval:
    """Embed the question and retrieve according to `body.mode`.

    Records embed_ms, db_ms and (when it runs) rerank_ms in `timings`.
    """
    t = time.perf_counter()
    query_embedding = await aembed_query(body.question)
    timings["embed_ms"] = _ms(t)

    t = time.perf_counter()
    async with request.app.state.pool.connection(timeout=2) as conn:
        if body.mode == "vector":
            rows = await repository.asearch_chunks(conn, query_embedding, body.k)
            timings["db_ms"] = _ms(t)
            return Retrieval(
                [
                    RetrievedChunk(r.section, r.title, r.content, similarity=r.similarity)
                    for r in rows
                ]
            )

        # The reranker needs more candidates than the final k to have something to reorder.
        limit = body.k if body.mode == "hybrid" else max(body.k, RERANK_POOL)
        ranked = await repository.asearch_hybrid(
            conn, body.question, query_embedding, limit=limit
        )
    timings["db_ms"] = _ms(t)

    if body.mode == "hybrid":
        return Retrieval([_from_ranked(r) for r in ranked])

    # hybrid_rerank. No database connection is held while the model works.
    t = time.perf_counter()
    result = await rerank(body.question, ranked, top_k=body.k)
    timings["rerank_ms"] = _ms(t)
    return Retrieval(
        [
            _from_ranked(ranked[c.original_rank - 1], rerank_score=c.rerank_score)
            for c in result.chunks
        ],
        rerank_fallback=result.used_fallback,
    )


def _provider_http_error(exc: Exception) -> HTTPException:
    failure = classify(exc)
    logger.warning(
        "provider error status=%d retryable=%s", failure.status, failure.retryable
    )
    headers = {"Retry-After": RETRY_AFTER_SECONDS} if failure.retryable else None
    return HTTPException(
        status_code=failure.status, detail=failure.detail, headers=headers
    )


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """Retrieve chunks (vector, hybrid or hybrid+rerank), answer from them, cite sources."""
    timings: dict[str, int] = {}
    started = time.perf_counter()
    try:
        retrieval = await _retrieve(request, body, timings)
        t = time.perf_counter()
        answer = await generate_answer(body.question, retrieval.chunks)
        timings["generate_ms"] = _ms(t)
    except (psycopg.Error, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    except PROVIDER_ERRORS as exc:
        raise _provider_http_error(exc) from exc
    except ModelOutputError as exc:
        raise HTTPException(status_code=502, detail="model provider error") from exc

    logger.info(
        "ask mode=%s %s total_ms=%d answerable=%s rerank_fallback=%s",
        body.mode,
        timings,
        _ms(started),
        answer.answerable,
        retrieval.rerank_fallback,
    )
    return AskResponse(
        answer=answer.text,
        answerable=answer.answerable,
        sources=_sources(retrieval.chunks, answer.cited_ids),
        mode=body.mode,
        rerank_fallback=retrieval.rerank_fallback,
    )


@router.post("/ask/stream")
async def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    """Same as /ask, but the answer arrives token by token (Server-Sent Events).

    Events: `token` {text} (many), `sources` {answerable, sources, mode, rerank_fallback},
    `done`, or `error`.

    Everything that can fail BEFORE the first byte (embedding, database, rerank, and
    opening the model stream including its first chunk) runs here, so those failures
    still return a proper 503/502/504. After the response starts the status line is
    already sent as 200, so errors become an `error` event.
    """
    timings: dict[str, int] = {}
    started = time.perf_counter()
    try:
        retrieval = await _retrieve(request, body, timings)
        chunks = retrieval.chunks
        t = time.perf_counter()
        stream = await open_stream(body.question, chunks) if chunks else None
        # Includes the wait for the first chunk and any retry: this is the model's part
        # of the time to first token.
        timings["first_chunk_ms"] = _ms(t)
    except (psycopg.Error, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    except PROVIDER_ERRORS as exc:
        raise _provider_http_error(exc) from exc

    async def events():
        parts: list[str] = []
        try:
            if stream is not None:
                async for chunk in stream:
                    if chunk.text:
                        parts.append(chunk.text)
                        yield sse_event("token", {"text": chunk.text})
        except PROVIDER_ERRORS as exc:
            failure = classify(exc)
            logger.warning(
                "stream error mid-way status=%d error=%s code=%s",
                failure.status,
                type(exc).__name__,
                getattr(exc, "code", None),
            )
            yield sse_event(
                "error", {"detail": failure.detail, "retryable": failure.retryable}
            )
            return

        # Text is already on the user's screen, so it cannot be withdrawn. What we can do
        # is tell the client whether it is backed by real sources.
        cited = extract_citations("".join(parts), len(chunks))
        yield sse_event(
            "sources",
            {
                "answerable": bool(cited),
                "sources": [s.model_dump() for s in _sources(chunks, cited)],
                "mode": body.mode,
                "rerank_fallback": retrieval.rerank_fallback,
            },
        )
        yield sse_event("done", {})
        logger.info(
            "ask_stream mode=%s %s total_ms=%d chunks=%d answerable=%s rerank_fallback=%s",
            body.mode,
            timings,
            _ms(started),
            len(parts),
            bool(cited),
            retrieval.rerank_fallback,
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # no-cache: never reuse a stream. X-Accel-Buffering: tell nginx-style proxies not
        # to hold the chunks until the response ends (that would kill the streaming).
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
