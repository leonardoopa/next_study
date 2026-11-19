import psycopg
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from google.genai import errors

from app import repository
from app.api_models import AskRequest, AskResponse, Source
from app.embeddings import aembed_query
from app.generation import (
    ModelOutputError,
    extract_citations,
    generate_answer,
    open_stream,
)
from app.schemas import SearchResult
from app.sse import sse_event

router = APIRouter()


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


async def _retrieve(request: Request, question: str, k: int) -> list[SearchResult]:
    query_embedding = await aembed_query(question)
    async with request.app.state.pool.connection(timeout=2) as conn:
        return await repository.asearch_chunks(conn, query_embedding, k)


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """Retrieve the closest chunks, answer from them only, and cite the sources used."""
    try:
        results = await _retrieve(request, body.question, body.k)
        answer = await generate_answer(body.question, results)
    except (psycopg.Error, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    except (errors.APIError, ModelOutputError) as exc:
        # 502 = the upstream (model provider) failed, not our code and not the client.
        raise HTTPException(status_code=502, detail="model provider error") from exc

    return AskResponse(
        answer=answer.text,
        answerable=answer.answerable,
        sources=_sources(results, answer.cited_ids),
    )


@router.post("/ask/stream")
async def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    """Same as /ask, but the answer arrives token by token (Server-Sent Events).

    Events: `token` {text} (many), `sources` {answerable, sources}, `done`, or `error`.

    Everything that can fail BEFORE the first byte (embedding, database, opening the
    model stream) runs here, so those failures still return a proper 503/502. After the
    response starts the status line is already sent as 200, so errors become an
    `error` event.
    """
    try:
        results = await _retrieve(request, body.question, body.k)
        stream = await open_stream(body.question, results) if results else None
    except (psycopg.Error, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    except errors.APIError as exc:
        raise HTTPException(status_code=502, detail="model provider error") from exc

    async def events():
        parts: list[str] = []
        try:
            if stream is not None:
                async for chunk in stream:
                    if chunk.text:
                        parts.append(chunk.text)
                        yield sse_event("token", {"text": chunk.text})
        except errors.APIError:
            yield sse_event("error", {"detail": "model provider error"})
            return

        # Text is already on the user's screen, so it cannot be withdrawn. What we can do
        # is tell the client whether it is backed by real sources.
        cited = extract_citations("".join(parts), len(results))
        payload = {
            "answerable": bool(cited),
            "sources": [s.model_dump() for s in _sources(results, cited)],
        }
        yield sse_event("sources", payload)
        yield sse_event("done", {})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # no-cache: never reuse a stream. X-Accel-Buffering: tell nginx-style proxies not
        # to hold the chunks until the response ends (that would kill the streaming).
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
