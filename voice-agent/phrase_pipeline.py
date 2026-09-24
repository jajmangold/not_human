"""Phrase-level TTS + reaction fusion for the avatar pipeline
(internal issue #45 latency fix).

Ported *architecture*, not code, from musetalk-volta's own `/ws/live/` demo
(web/app.py's produce_qwen/synthesize_phrases/render_phrases three-stage
pipeline) -- confirmed to feel near-instant next to this agent's original
whole-turn-buffered behavior. The key property that gives it that feel:
each phrase's Kokoro synthesis and reaction classification happen in
parallel (asyncio.gather), and phrase N+1 starts synthesizing while phrase
N is still being rendered/spoken -- see agent.py's llm_node/_consume_phrases
for the queueing that gets that overlap.

This also removes the extra LLM tool-call round-trip the old `set_reaction`
tool required: reaction is now inferred locally from each phrase's own text
as a byproduct of speaking it (matching the old demo exactly), not
requested from the model. Real usage measured that round-trip costing up to
~80s with gemma4-26b re-invoking the tool repeatedly before ever producing
audio.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import aiohttp

from reaction_plan import reaction_plan, wav_duration, wav_prosody

logger = logging.getLogger("phrase-pipeline")

NEUTRAL_REACTION: dict[str, object] = {"reaction": "neutral", "strength": 0.0}
CLASSIFY_TIMEOUT_S = 1.5


@dataclass(frozen=True)
class PhraseMedia:
    phrase: str
    wav_bytes: bytes
    reaction: dict[str, object]
    # Exposed rather than left for the caller to recompute: NaturalMotion-
    # Scheduler.schedule() (agent.py's _consume_phrases) needs both to place
    # blink/gaze/head events correctly, and they're already computed here as
    # a side effect of building `reaction` above.
    duration: float
    prosody: dict[str, float]


async def _fetch_audio(
    http: aiohttp.ClientSession, *, kokoro_url: str, phrase: str, voice: str, speed: float
) -> bytes:
    async with http.post(
        f"{kokoro_url}/synthesize", json={"text": phrase, "voice": voice, "speed": speed}
    ) as response:
        response.raise_for_status()
        return await response.read()


async def _fetch_reaction(
    http: aiohttp.ClientSession, *, affect_url: str, phrase: str
) -> dict[str, object]:
    # Reaction is an enhancement, not a requirement for speech -- a slow or
    # failed classify call degrades to neutral rather than losing the
    # phrase's audio (matching the old demo's own `optional_json` fallback).
    try:
        async with http.post(
            f"{affect_url}/classify",
            json={"text": phrase},
            timeout=aiohttp.ClientTimeout(total=CLASSIFY_TIMEOUT_S),
        ) as response:
            response.raise_for_status()
            return await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.warning("affect classify failed for phrase %r, defaulting to neutral: %s", phrase, exc)
        return dict(NEUTRAL_REACTION)


async def synthesize_phrase(
    http: aiohttp.ClientSession,
    *,
    kokoro_url: str,
    affect_url: str,
    phrase: str,
    voice: str,
    speed: float,
    previous_reaction: str,
) -> PhraseMedia:
    """Synthesize one phrase's audio and classify its reaction in parallel,
    then fuse them into one bounded reaction plan (see reaction_plan.py).

    Raises whatever the Kokoro call raises (speech is not optional); a
    failed reaction classify call is swallowed internally, see
    ``_fetch_reaction``.
    """
    wav_bytes, sampled_reaction = await asyncio.gather(
        _fetch_audio(http, kokoro_url=kokoro_url, phrase=phrase, voice=voice, speed=speed),
        _fetch_reaction(http, affect_url=affect_url, phrase=phrase),
    )
    prosody = wav_prosody(wav_bytes)
    repeated = str(sampled_reaction.get("reaction", "neutral")) == previous_reaction
    plan = reaction_plan(sampled_reaction, prosody, repeated=repeated)
    return PhraseMedia(
        phrase=phrase, wav_bytes=wav_bytes, reaction=plan,
        duration=wav_duration(wav_bytes), prosody=prosody,
    )
