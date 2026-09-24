"""Unit tests for phrase_pipeline.synthesize_phrase: verifies Kokoro
synthesis and reaction classification are actually run in parallel (not
sequentially -- that parallelism is the whole point of this module, see
its docstring) and that a failed/slow classify call degrades to neutral
instead of losing the phrase's audio.

Uses tiny hand-written fakes for aiohttp.ClientSession rather than a mocking
library -- no new dependency, and the ClientSession surface this module
actually uses (post() as an async context manager) is small enough to fake
directly.
"""

from __future__ import annotations

import asyncio
import io
import struct
import wave

import aiohttp
import pytest

from phrase_pipeline import NEUTRAL_REACTION, synthesize_phrase


def _make_wav(*, duration_s: float = 0.5, sample_rate: int = 24000) -> bytes:
    n = int(sample_rate * duration_s)
    samples = [0] * n
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, *, json_body=None, read_body=None, delay=0.0, raise_exc=None):
        self._json_body = json_body
        self._read_body = read_body
        self._delay = delay
        self._raise_exc = raise_exc

    def raise_for_status(self) -> None:
        if self._raise_exc:
            raise self._raise_exc

    async def read(self) -> bytes:
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._read_body

    async def json(self) -> dict:
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._json_body

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeSession:
    def __init__(self, *, kokoro_response: _FakeResponse, affect_response: _FakeResponse):
        self._kokoro_response = kokoro_response
        self._affect_response = affect_response
        self.call_order: list[str] = []

    def post(self, url: str, **_kwargs: object) -> _FakeResponse:
        if "synthesize" in url:
            self.call_order.append("kokoro_start")
            return self._kokoro_response
        self.call_order.append("affect_start")
        return self._affect_response


@pytest.mark.asyncio
async def test_kokoro_and_classify_run_in_parallel_not_sequentially():
    # If these ran sequentially (classify awaited only after Kokoro
    # finishes, e.g.), total wall time would be >= the sum of both delays.
    # Run in parallel (asyncio.gather, as synthesize_phrase does), it's
    # bounded by the slower one alone.
    wav_bytes = _make_wav()
    session = _FakeSession(
        kokoro_response=_FakeResponse(read_body=wav_bytes, delay=0.15),
        affect_response=_FakeResponse(json_body={"reaction": "amusement", "strength": 0.5}, delay=0.15),
    )
    start = asyncio.get_event_loop().time()
    await synthesize_phrase(
        session, kokoro_url="http://kokoro", affect_url="http://affect",
        phrase="that is funny", voice="af_bella", speed=1.0, previous_reaction="neutral",
    )
    elapsed = asyncio.get_event_loop().time() - start
    assert elapsed < 0.25, f"expected parallel calls (~0.15s), took {elapsed:.3f}s -- looks sequential"


@pytest.mark.asyncio
async def test_synthesize_phrase_fuses_classify_result_into_reaction_plan():
    wav_bytes = _make_wav()
    session = _FakeSession(
        kokoro_response=_FakeResponse(read_body=wav_bytes),
        affect_response=_FakeResponse(json_body={"reaction": "amusement", "strength": 0.5}),
    )
    media = await synthesize_phrase(
        session, kokoro_url="http://kokoro", affect_url="http://affect",
        phrase="that is funny", voice="af_bella", speed=1.0, previous_reaction="neutral",
    )
    assert media.wav_bytes == wav_bytes
    assert media.reaction["reaction"] == "amusement"
    assert 0.0 < media.reaction["strength"] <= 0.45  # reaction_plan's own damping cap


@pytest.mark.asyncio
async def test_classify_failure_degrades_to_neutral_without_losing_audio():
    wav_bytes = _make_wav()
    session = _FakeSession(
        kokoro_response=_FakeResponse(read_body=wav_bytes),
        affect_response=_FakeResponse(raise_exc=aiohttp.ClientError("classify down")),
    )
    media = await synthesize_phrase(
        session, kokoro_url="http://kokoro", affect_url="http://affect",
        phrase="hello", voice="af_bella", speed=1.0, previous_reaction="neutral",
    )
    assert media.wav_bytes == wav_bytes
    assert media.reaction["reaction"] == "neutral"
    assert media.reaction["strength"] == 0.0


@pytest.mark.asyncio
async def test_kokoro_failure_raises_speech_is_not_optional():
    session = _FakeSession(
        kokoro_response=_FakeResponse(raise_exc=aiohttp.ClientError("kokoro down")),
        affect_response=_FakeResponse(json_body=dict(NEUTRAL_REACTION)),
    )
    with pytest.raises(aiohttp.ClientError):
        await synthesize_phrase(
            session, kokoro_url="http://kokoro", affect_url="http://affect",
            phrase="hello", voice="af_bella", speed=1.0, previous_reaction="neutral",
        )
