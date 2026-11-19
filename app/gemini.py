"""Shared Gemini plumbing: one client, one retry policy, one error classification."""

import logging
import random
from dataclasses import dataclass
from functools import lru_cache

from google import genai
from google.genai import errors, types

try:  # newer google-genai versions vendor httpx under this name
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

from app.config import get_settings

logger = logging.getLogger("app.gemini")

MAX_RETRIES = 5

# Everything a Gemini call can raise that we handle: API errors (4xx/5xx) and timeouts.
PROVIDER_ERRORS = (errors.APIError, httpx.TimeoutException)


@lru_cache
def client() -> genai.Client:
    settings = get_settings()
    return genai.Client(
        api_key=settings.gemini_api_key.get_secret_value(),
        # Without a timeout a slow call can hang for a very long time.
        http_options=types.HttpOptions(timeout=settings.gemini_timeout_ms),
    )


def retry_delay(exc: Exception, attempt: int) -> float | None:
    """Seconds to wait before retrying, or None when the error should be raised now.

    Retry: 429 (rate limit), 5xx (server trouble) and connect timeouts (the request
    never reached the provider). Anything else (400 bad request, 401/403 key problems,
    a read timeout after the provider started working) would fail again or cost another
    full timeout, so raise at once. Exponential backoff (2, 4, 8, ... s, max 30) plus
    jitter, so parallel callers do not retry at the same instant.
    """
    code = getattr(exc, "code", None) or 0
    retryable = code == 429 or code >= 500 or isinstance(exc, httpx.ConnectTimeout)
    if not retryable or attempt >= MAX_RETRIES:
        return None
    delay = min(2**attempt, 30) + random.random()
    logger.warning(
        "gemini retry attempt=%d error=%s code=%s wait=%.1fs",
        attempt, type(exc).__name__, code or "-", delay,
    )
    return delay


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """How a provider error should look to our own clients."""

    status: int
    detail: str
    retryable: bool


def classify(exc: Exception) -> ProviderFailure:
    if isinstance(exc, httpx.TimeoutException):
        return ProviderFailure(504, "model provider timed out", retryable=True)
    code = getattr(exc, "code", None) or 0
    if code == 429:
        # Our quota with the provider is exhausted: temporary, from the client's view.
        return ProviderFailure(503, "model quota exceeded, try again shortly", retryable=True)
    if code >= 500:
        return ProviderFailure(502, "model provider error", retryable=True)
    return ProviderFailure(502, "model provider error", retryable=False)
