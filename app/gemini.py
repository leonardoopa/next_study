"""Shared Gemini plumbing: one client and one retry policy for embeddings and generation."""

import random
from functools import lru_cache

from google import genai
from google.genai import errors

from app.config import get_settings

MAX_RETRIES = 5


@lru_cache
def client() -> genai.Client:
    return genai.Client(api_key=get_settings().gemini_api_key.get_secret_value())


def retry_delay(exc: errors.APIError, attempt: int) -> float | None:
    """Seconds to wait before retrying, or None when the error should be raised now.

    429 (rate limit) and 5xx (server trouble) are worth retrying. Anything else
    (400 bad request, 401/403 key problems) would fail again, so raise at once.
    Exponential backoff (2, 4, 8, ... s, max 30) plus jitter, so parallel callers do not
    retry at the same instant.
    """
    code = exc.code or 0
    if not (code == 429 or code >= 500) or attempt >= MAX_RETRIES:
        return None
    return min(2**attempt, 30) + random.random()
