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
from app.schemas import SearchResult
from app.sse import sse_event

router = APIRouter()
logger = logging.getLogger("app.ask")

RETRY_AFTER_SECONDS = "30"


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _sources(results: list[SearchResult], ids: list[int]) -> list[Source]:
    """Build the Source objects for 1-based ids that were already validated."""
    return [
        Source(
            id=i,
            section=results[i - 1].section,
            title=results[i - 1].title,
            content=results[i - 1].content,
            similarity=results[i - 1].similarity,
        )
        for i in ids
    ]


async def _retrieve(
    request: Request, question: str, k: int, timings: dict[str, int]
) -> list[SearchResult]:
    """Embed the question and search. Records embed_ms and db_ms in `timings`."""
    t = time.perf_counter()
    query_embedding = await aembed_query(question)
    timings["embed_ms"] = _ms(t)

    t = time.perf_counter()
    async with request.app.state.pool.connection(timeout=2) as conn:
        results = await repository.asearch_chunks(conn, query_embedding, k)
    timings["db_ms"] = _ms(t)
    return results


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
    """Retrieve the closest chunks, answer from them only, and cite the sources used."""
    timings: dict[str, int] = {}
    started = time.perf_counter()
    try:
        results = await _retrieve(request, body.question, body.k, timings)
        t = time.perf_counter()
        answer = await generate_answer(body.question, results)
        timings["generate_ms"] = _ms(t)
    except (psycopg.Error, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    except PROVIDER_ERRORS as exc:
        raise _provider_http_error(exc) from exc
    except ModelOutputError as exc:
        raise HTTPException(status_code=502, detail="model provider error") from exc

    logger.info(
        "ask %s total_ms=%d answerable=%s", timings, _ms(started), answer.answerable
    )
    return AskResponse(
        answer=answer.text,
        answerable=answer.answerable,
        sources=_sources(results, answer.cited_ids),
    )


@router.post("/ask/stream")
async def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    """Same as /ask, but the answer arrives token by token (Server-Sent Events).

    Events: `token` {text} (many), `sources` {answerable, sources}, `done`, or `error`.

    Everything that can fail BEFORE the first byte (embedding, database, and opening the
    model stream including its first chunk) runs here, so those failures still return a
    proper 503/502/504. After the response starts the status line is already sent as
    200, so errors become an `error` event.
    """
    timings: dict[str, int] = {}
    started = time.perf_counter()
    try:
        results = await _retrieve(request, body.question, body.k, timings)
        t = time.perf_counter()
        stream = await open_stream(body.question, results) if results else None
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
        cited = extract_citations("".join(parts), len(results))
        yield sse_event(
            "sources",
            {
                "answerable": bool(cited),
                "sources": [s.model_dump() for s in _sources(results, cited)],
            },
        )
        yield sse_event("done", {})
        logger.info(
            "ask_stream %s total_ms=%d chunks=%d answerable=%s",
            timings,
            _ms(started),
            len(parts),
            bool(cited),
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # no-cache: never reuse a stream. X-Accel-Buffering: tell nginx-style proxies not
        # to hold the chunks until the response ends (that would kill the streaming).
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
