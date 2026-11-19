import psycopg
from fastapi import APIRouter, HTTPException, Request
from google.genai import errors

from app import repository
from app.api_models import AskRequest, AskResponse, Source
from app.embeddings import aembed_query
from app.generation import ModelOutputError, generate_answer

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """Retrieve the closest chunks, answer from them only, and cite the sources used."""
    try:
        query_embedding = await aembed_query(body.question)
        async with request.app.state.pool.connection(timeout=2) as conn:
            results = await repository.asearch_chunks(conn, query_embedding, body.k)
        answer = await generate_answer(body.question, results)
    except (psycopg.Error, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    except (errors.APIError, ModelOutputError) as exc:
        # 502 = the upstream (model provider) failed, not our code and not the client.
        raise HTTPException(status_code=502, detail="model provider error") from exc

    return AskResponse(
        answer=answer.text,
        answerable=answer.answerable,
        sources=[
            Source(
                id=i,
                section=results[i - 1].section,
                title=results[i - 1].title,
                content=results[i - 1].content,
                similarity=results[i - 1].similarity,
            )
            for i in answer.cited_ids
        ],
    )
