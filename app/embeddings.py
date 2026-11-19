"""Embeddings with the Gemini API (gemini-embedding-2, 768 dimensions).

Documents and questions are embedded differently, so there are separate functions:
    embed_documents(chunks) -> one vector per chunk (sync, used by ingestion)
    embed_query(text)       -> one vector for a question (sync)
    aembed_query(text)      -> same, async (used by the API)
"""

import asyncio
import time

from google.genai import errors, types

from app.config import get_settings
from app.gemini import MAX_RETRIES, client, retry_delay
from app.schemas import Chunk

BATCH_SIZE = 50  # texts per API call


def format_document(chunk: Chunk) -> str:
    """gemini-embedding-2 has no task_type: the task goes in a text prefix instead."""
    return f"title: {chunk.section} > {chunk.title} | text: {chunk.text}"


def format_query(query: str) -> str:
    return f"task: search result | query: {query}"


def _request(texts: list[str]) -> dict:
    settings = get_settings()
    # One Content per text. Passing a plain list of strings could be read as several parts
    # of ONE input (multimodal model) and return a single embedding.
    return {
        "model": settings.embedding_model,
        "contents": [types.Content(parts=[types.Part(text=t)]) for t in texts],
        "config": types.EmbedContentConfig(output_dimensionality=settings.embedding_dim),
    }


def _vectors(result, expected: int) -> list[list[float]]:
    vectors = [list(e.values) for e in result.embeddings]
    if len(vectors) != expected:
        raise RuntimeError(f"asked for {expected} embeddings, got {len(vectors)}")
    return vectors


def _embed_batch(texts: list[str]) -> list[list[float]]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = client().models.embed_content(**_request(texts))
            return _vectors(result, len(texts))
        except errors.APIError as exc:
            delay = retry_delay(exc, attempt)
            if delay is None:
                raise
            time.sleep(delay)
    raise AssertionError("unreachable")


async def _aembed_batch(texts: list[str]) -> list[list[float]]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await client().aio.models.embed_content(**_request(texts))
            return _vectors(result, len(texts))
        except errors.APIError as exc:
            delay = retry_delay(exc, attempt)
            if delay is None:
                raise
            await asyncio.sleep(delay)  # does not block the event loop
    raise AssertionError("unreachable")


def _embed_all(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        vectors.extend(_embed_batch(texts[start : start + BATCH_SIZE]))
    return vectors


def embed_documents(chunks: list[Chunk]) -> list[list[float]]:
    return _embed_all([format_document(c) for c in chunks])


def embed_query(query: str) -> list[float]:
    return _embed_batch([format_query(query)])[0]


async def aembed_query(query: str) -> list[float]:
    return (await _aembed_batch([format_query(query)]))[0]
