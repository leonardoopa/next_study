"""Error classification and retry policy (pure functions, no network)."""

from google.genai import errors

from app import gemini
from app.gemini import MAX_RETRIES, classify, httpx, retry_delay


def _api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"message": "x"}})


def test_quota_error_is_503_and_retryable():
    failure = classify(_api_error(429))
    assert (failure.status, failure.retryable) == (503, True)


def test_server_error_is_502_and_retryable():
    failure = classify(_api_error(500))
    assert (failure.status, failure.retryable) == (502, True)


def test_client_error_is_502_and_not_retryable():
    failure = classify(_api_error(400))
    assert (failure.status, failure.retryable) == (502, False)


def test_timeout_is_504_and_retryable():
    failure = classify(httpx.ReadTimeout("slow"))
    assert (failure.status, failure.retryable) == (504, True)


def test_retry_on_429_and_5xx_but_not_on_400_401():
    assert retry_delay(_api_error(429), attempt=1) is not None
    assert retry_delay(_api_error(503), attempt=1) is not None
    assert retry_delay(_api_error(400), attempt=1) is None
    assert retry_delay(_api_error(401), attempt=1) is None


def test_connect_timeout_is_retried_but_read_timeout_is_not():
    assert retry_delay(httpx.ConnectTimeout("no route"), attempt=1) is not None
    assert retry_delay(httpx.ReadTimeout("slow"), attempt=1) is None


def test_gives_up_on_the_last_attempt():
    assert retry_delay(_api_error(429), attempt=MAX_RETRIES) is None


def test_backoff_grows_and_is_capped(monkeypatch):
    monkeypatch.setattr(gemini.random, "random", lambda: 0.0)
    delays = [retry_delay(_api_error(429), attempt=a) for a in range(1, MAX_RETRIES)]
    assert delays == [2, 4, 8, 16]
