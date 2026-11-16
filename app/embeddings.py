"""Embeddings with the Gemini API (gemini-embedding-2, 768 dimensions).

Two public functions, because documents and questions are embedded differently:
    embed_documents(chunks) -> one vector per chunk
    embed_query(text)       -> one vector for a question
"""

import random
import time
from functools import lru_cache

from google import genai
from google.genai import errors, types

from app.config import get_settings
from app.schemas import Chunk

BATCH_SIZE = 50  # texts per API call
MAX_RETRIES = 5


@lru_cache
def _client() -> genai.Client:
    return genai.Client(api_key=get_settings().gemini_api_key.get_secret_value())


def format_document(chunk: Chunk) -> str:
    """gemini-embedding-2 has no task_type: the task goes in a text prefix instead."""
    return f"title: {chunk.section} > {chunk.title} | text: {chunk.text}"


def format_query(query: str) -> str:
    return f"task: search result | query: {query}"


def _embed_batch(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    # One Content per text. Passing a plain list of strings could be read as several parts
    # of ONE input (multimodal model) and return a single embedding.
    contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
    config = types.EmbedContentConfig(output_dimensionality=settings.embedding_dim)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = _client().models.embed_content(
                model=settings.embedding_model, contents=contents, config=config
            )
            break
        except errors.APIError as exc:
            # 429 = rate limit, 5xx = server trouble: worth retrying. Anything else
            # (400 bad request, 401/403 key problems) will fail again, so raise now.
            code = exc.code or 0
            if not (code == 429 or code >= 500) or attempt == MAX_RETRIES:
                raise
            # Exponential backoff (2, 4, 8, ... s, max 30) plus jitter so parallel
            # callers do not retry at the same instant.
            time.sleep(min(2**attempt, 30) + random.random())

    vectors = [list(e.values) for e in result.embeddings]
    if len(vectors) != len(texts):
        raise RuntimeError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
    return vectors


def _embed_all(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        vectors.extend(_embed_batch(texts[start : start + BATCH_SIZE]))
    return vectors


def embed_documents(chunks: list[Chunk]) -> list[list[float]]:
    return _embed_all([format_document(c) for c in chunks])


def embed_query(query: str) -> list[float]:
    return _embed_batch([format_query(query)])[0]
