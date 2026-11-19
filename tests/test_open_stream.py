"""open_stream must retry errors that only appear when the stream is first READ."""

import asyncio
from types import SimpleNamespace

import pytest
from google.genai import errors

from app import generation
from app.gemini import MAX_RETRIES
from app.schemas import SearchResult

RESULTS = [SearchResult(section="S", title="T", content="c", similarity=0.5)]


def _fake_client(behaviors):
    """Each call to generate_content_stream uses the next behavior: an Exception raised on
    the first read (like the real lazy stream), or a list of texts to stream."""
    state = SimpleNamespace(calls=0)

    async def generate_content_stream(**_kwargs):
        behavior = behaviors[min(state.calls, len(behaviors) - 1)]
        state.calls += 1

        async def gen():
            if isinstance(behavior, Exception):
                raise behavior  # raised on first read, NOT when the stream is created
            for text in behavior:
                yield SimpleNamespace(text=text)

        return gen()

    fake = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
        generate_content_stream=generate_content_stream)))
    return fake, state


@pytest.fixture(autouse=True)
def _no_real_waiting(monkeypatch):
    async def instant(_delay):
        return None

    monkeypatch.setattr(generation.asyncio, "sleep", instant)
    monkeypatch.setattr(generation, "get_settings", lambda: SimpleNamespace(generation_model="m"))


def _read_all():
    """Open and read in ONE event loop: asyncio.run() closes async generators that are
    still alive when it ends, so a stream must not outlive the loop that created it."""

    async def run():
        stream = await generation.open_stream("q", RESULTS)
        return [c.text async for c in stream]

    return asyncio.run(run())


def _open(monkeypatch, behaviors):
    fake, state = _fake_client(behaviors)
    monkeypatch.setattr(generation, "client", lambda: fake)
    return state


def test_retries_a_429_raised_on_first_read(monkeypatch):
    quota = errors.APIError(429, {"error": {"message": "quota"}})
    state = _open(monkeypatch, [quota, quota, ["Ele ", "mentorou [2]."]])

    assert _read_all() == ["Ele ", "mentorou [2]."]  # first chunk was not lost
    assert state.calls == 3


def test_does_not_retry_a_client_error(monkeypatch):
    state = _open(monkeypatch, [errors.APIError(400, {"error": {"message": "bad"}})])
    with pytest.raises(errors.APIError):
        _read_all()
    assert state.calls == 1


def test_gives_up_after_max_retries(monkeypatch):
    state = _open(monkeypatch, [errors.APIError(429, {"error": {"message": "quota"}})])
    with pytest.raises(errors.APIError):
        _read_all()
    assert state.calls == MAX_RETRIES


def test_empty_stream_gives_an_empty_iterator(monkeypatch):
    _open(monkeypatch, [[]])
    assert _read_all() == []
