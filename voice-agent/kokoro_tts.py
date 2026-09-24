"""Minimal livekit-agents TTS plugin wrapping the resident Kokoro service.

Kokoro's own /synthesize endpoint (musetalk-volta's kokoro/app.py) is a
plain, non-streaming call: POST {text, voice, speed} -> a complete WAV. That
doesn't fit livekit.plugins.openai.TTS, which drives OpenAI's newer
server-sent-events streaming /v1/audio/speech contract (verified against its
actual source: stream_format="sse" for any non-realtime model) -- faking an
SSE stream on top of a backend that only ever returns one complete WAV would
be more work than implementing livekit-agents' own TTS interface directly
against the ChunkedStream/AudioEmitter contract it already documents for
non-streaming providers.
"""

from __future__ import annotations

import wave
from io import BytesIO

import aiohttp
from livekit.agents import APIConnectionError, APIConnectOptions, utils
from livekit.agents.tts import TTS, ChunkedStream, TTSCapabilities
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

SAMPLE_RATE = 24000  # kokoro_onnx.SAMPLE_RATE, matched in kokoro/app.py


def _wav_to_pcm16_mono(wav_bytes: bytes) -> bytes:
    with wave.open(BytesIO(wav_bytes), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError(
                f"expected 16-bit mono PCM WAV, got sampwidth={handle.getsampwidth()} "
                f"channels={handle.getnchannels()}"
            )
        return handle.readframes(handle.getnframes())


class KokoroTTS(TTS):
    def __init__(self, *, base_url: str, voice: str = "af_bella", speed: float = 1.0) -> None:
        super().__init__(
            capabilities=TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        self._base_url = base_url.rstrip("/")
        self._voice = voice
        self._speed = speed

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "KokoroChunkedStream":
        return KokoroChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            base_url=self._base_url,
            voice=self._voice,
            speed=self._speed,
        )


class KokoroChunkedStream(ChunkedStream):
    def __init__(
        self,
        *,
        tts: KokoroTTS,
        input_text: str,
        conn_options: APIConnectOptions,
        base_url: str,
        voice: str,
        speed: float,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._base_url = base_url
        self._voice = voice
        self._speed = speed

    async def _run(self, output_emitter) -> None:
        timeout = aiohttp.ClientTimeout(total=self._conn_options.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self._base_url}/synthesize",
                    json={"text": self.input_text, "voice": self._voice, "speed": self._speed},
                ) as response:
                    response.raise_for_status()
                    wav_bytes = await response.read()
        except aiohttp.ClientError as exc:
            raise APIConnectionError(f"Kokoro synthesis request failed: {exc}") from exc

        pcm = _wav_to_pcm16_mono(wav_bytes)
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=False,
        )
        output_emitter.push(pcm)
        output_emitter.end_input()
