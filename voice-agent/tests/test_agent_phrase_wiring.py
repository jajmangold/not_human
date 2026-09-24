"""Integration-ish test for MuseTalkVoiceAgent.llm_node's phrase pipeline
wiring (internal issue #45 latency fix) -- the part of agent.py most at
risk of being wrong, since it's new: streamed LLM text -> phrase boundaries
-> synthesize_phrase -> video_gen.set_next_reaction()/push_audio().

Fakes out Agent.default.llm_node (the framework's real LLM call) and
phrase_pipeline.synthesize_phrase (the real Kokoro/affect HTTP calls) so
this runs with no network, GPU, or LiveKit connection -- everything else
(release_speech_phrases boundary detection, the actual queueing/consumer
loop in agent.py, the AudioFrame/AudioSegmentEnd hand-off to video_gen) is
real.
"""

from __future__ import annotations

import asyncio

import pytest
from livekit.agents import Agent

import agent as agent_module
from agent import AvatarHandle, MuseTalkVoiceAgent
from livekit.agents.voice.avatar import AudioSegmentEnd
from phrase_pipeline import PhraseMedia


def _make_wav_bytes() -> bytes:
    import io
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(struct.pack("<4h", 0, 0, 0, 0))
    return buf.getvalue()


class _FakeVideoGen:
    def __init__(self) -> None:
        self.set_reactions: list[dict | None] = []
        self.pushed: list[object] = []
        self.cleared = False

    def set_next_reaction(self, reaction: dict | None) -> None:
        self.set_reactions.append(reaction)

    async def push_audio(self, frame: object) -> None:
        self.pushed.append(frame)

    def clear_buffer(self) -> None:
        self.cleared = True


class _ReadyAvatarHandle(AvatarHandle):
    """A real AvatarHandle, pre-populated -- exercises the same wait_ready()
    codepath _consume_phrases actually calls, just without a real startup
    delay."""

    def __init__(self, video_gen: _FakeVideoGen) -> None:
        super().__init__()
        self.set_ready(video_gen)  # type: ignore[arg-type]


async def _collect(agent_obj: MuseTalkVoiceAgent) -> list[str]:
    collected: list[str] = []
    async for chunk in agent_obj.llm_node(chat_ctx=None, tools=[], model_settings=None):  # type: ignore[arg-type]
        assert isinstance(chunk, str)
        collected.append(chunk)
    return collected


@pytest.mark.asyncio
async def test_completed_phrase_reaches_video_gen_with_reaction(monkeypatch):
    async def fake_default_llm_node(_self, _chat_ctx, _tools, _model_settings):
        for piece in ("That is really ", "funny! "):
            yield piece

    async def fake_synthesize_phrase(_http, *, phrase, previous_reaction, **_kwargs):
        return PhraseMedia(
            phrase=phrase,
            wav_bytes=_make_wav_bytes(),
            reaction={"reaction": "amusement", "strength": 0.3},
            duration=1.0,
            prosody={"energy": 0.5, "pause_ratio": 0.2, "onset": 0.3},
        )

    monkeypatch.setattr(Agent.default, "llm_node", fake_default_llm_node)
    monkeypatch.setattr(agent_module, "synthesize_phrase", fake_synthesize_phrase)

    video_gen = _FakeVideoGen()
    handle = _ReadyAvatarHandle(video_gen)
    voice_agent = MuseTalkVoiceAgent(avatar_handle=handle)

    collected = await _collect(voice_agent)

    assert "".join(collected).strip() == "That is really funny!"
    assert len(video_gen.set_reactions) == 1
    reaction = video_gen.set_reactions[0]
    # NaturalMotionScheduler's blink/gaze/head timing must be merged in
    # alongside the classified facial-expression reaction (internal
    # volta#45's "why is it still crappy" follow-up: the resident's
    # compositor reads blinks/gaze_events/head_events off this same dict).
    assert reaction["reaction"] == "amusement"
    assert reaction["strength"] == 0.3
    assert set(reaction) >= {"blinks", "gaze_events", "head_events"}
    assert len(video_gen.pushed) == 2
    assert isinstance(video_gen.pushed[1], AudioSegmentEnd)
    assert voice_agent._previous_reaction == "amusement"


@pytest.mark.asyncio
async def test_neutral_reaction_still_carries_blink_and_gaze_motion(monkeypatch):
    # A classified-neutral phrase must NOT be dropped to None -- blinking
    # and gaze/head micro-motion are independent of the facial-expression
    # reaction (people blink regardless of what they're feeling) and must
    # keep reaching the avatar on every phrase, not just non-neutral ones.
    # This was a real bug: the original wiring dropped the whole reaction
    # dict (including motion data) to None whenever reaction == "neutral",
    # which is most conversational speech -- silently disabling blinking
    # for nearly the entire conversation.
    async def fake_default_llm_node(_self, _chat_ctx, _tools, _model_settings):
        yield "That is a fact. "

    async def fake_synthesize_phrase(_http, *, phrase, previous_reaction, **_kwargs):
        return PhraseMedia(
            phrase=phrase, wav_bytes=_make_wav_bytes(),
            reaction={"reaction": "neutral", "strength": 0.0},
            duration=1.0, prosody={"energy": 0.2, "pause_ratio": 0.5, "onset": 0.1},
        )

    monkeypatch.setattr(Agent.default, "llm_node", fake_default_llm_node)
    monkeypatch.setattr(agent_module, "synthesize_phrase", fake_synthesize_phrase)

    video_gen = _FakeVideoGen()
    voice_agent = MuseTalkVoiceAgent(avatar_handle=_ReadyAvatarHandle(video_gen))
    await _collect(voice_agent)

    assert len(video_gen.set_reactions) == 1
    reaction = video_gen.set_reactions[0]
    assert reaction is not None, "neutral reactions must still carry blink/gaze motion, not be dropped to None"
    assert reaction["reaction"] == "neutral"
    assert set(reaction) >= {"blinks", "gaze_events", "head_events"}


class _FakeSchedulerAlwaysProducesHeadEvents:
    """Deterministic stand-in for NaturalMotionScheduler: the real one only
    produces head_events on a random schedule, which would make a test
    relying on one flaky. Always returns a head_event so the gating logic
    under test doesn't depend on hitting the right random seed."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def schedule(self, *_args, **_kwargs) -> dict[str, list]:
        return {
            "blinks": [],
            "gaze_events": [],
            "head_events": [{
                "axis": "yaw", "direction": 1, "at_ms": 100,
                "move_ms": 200, "hold_ms": 200, "return_ms": 300,
            }],
        }


@pytest.mark.asyncio
async def test_head_events_omitted_when_avatar_bank_lacks_head_pose_slots(monkeypatch):
    # Regression test for the reported "lips blocking/glitching":
    # muse_server.py's _render_sources() slices the prepared bank by
    # bank_start_frame before indexing HEAD_POSE_SLOTS (needs indices up to
    # 19); this identity's bank (24 frames, bank_start_frame=8) only has 16
    # usable frames left, and sending head_events anyway crashed the
    # resident mid-job (confirmed live: "IndexError: list index out of
    # range" in _render_sources, job fails partway through and freezes on
    # its last rendered frame). MUSE_AVATAR_SUPPORTS_HEAD_POSE must stay
    # False for this identity regardless of what NaturalMotionScheduler
    # itself produces.
    monkeypatch.setattr(agent_module, "MUSE_AVATAR_SUPPORTS_HEAD_POSE", False)
    monkeypatch.setattr(agent_module, "NaturalMotionScheduler", _FakeSchedulerAlwaysProducesHeadEvents)

    async def fake_default_llm_node(_self, _chat_ctx, _tools, _model_settings):
        yield "Turning to look at that. "

    async def fake_synthesize_phrase(_http, *, phrase, previous_reaction, **_kwargs):
        return PhraseMedia(
            phrase=phrase, wav_bytes=_make_wav_bytes(),
            reaction={"reaction": "interest", "strength": 0.4},
            duration=4.0, prosody={"energy": 0.6, "pause_ratio": 0.1, "onset": 0.4},
        )

    monkeypatch.setattr(Agent.default, "llm_node", fake_default_llm_node)
    monkeypatch.setattr(agent_module, "synthesize_phrase", fake_synthesize_phrase)

    video_gen = _FakeVideoGen()
    voice_agent = MuseTalkVoiceAgent(avatar_handle=_ReadyAvatarHandle(video_gen))
    await _collect(voice_agent)

    assert len(video_gen.set_reactions) == 1
    assert video_gen.set_reactions[0]["head_events"] == []


@pytest.mark.asyncio
async def test_head_events_pass_through_when_avatar_supports_head_pose(monkeypatch):
    # The flip side: a future identity prepared with a large enough bank
    # (the old demo's current ALP profile uses 28 frames specifically for
    # this headroom) should actually get head_events once the capability
    # flag is turned on for it.
    monkeypatch.setattr(agent_module, "MUSE_AVATAR_SUPPORTS_HEAD_POSE", True)
    monkeypatch.setattr(agent_module, "NaturalMotionScheduler", _FakeSchedulerAlwaysProducesHeadEvents)

    async def fake_default_llm_node(_self, _chat_ctx, _tools, _model_settings):
        yield "Turning to look at that. "

    async def fake_synthesize_phrase(_http, *, phrase, previous_reaction, **_kwargs):
        return PhraseMedia(
            phrase=phrase, wav_bytes=_make_wav_bytes(),
            reaction={"reaction": "interest", "strength": 0.4},
            duration=4.0, prosody={"energy": 0.6, "pause_ratio": 0.1, "onset": 0.4},
        )

    monkeypatch.setattr(Agent.default, "llm_node", fake_default_llm_node)
    monkeypatch.setattr(agent_module, "synthesize_phrase", fake_synthesize_phrase)

    video_gen = _FakeVideoGen()
    voice_agent = MuseTalkVoiceAgent(avatar_handle=_ReadyAvatarHandle(video_gen))
    await _collect(voice_agent)

    assert len(video_gen.set_reactions) == 1
    assert video_gen.set_reactions[0]["head_events"] != []


@pytest.mark.asyncio
async def test_cancellation_mid_reply_clears_the_avatar_buffer(monkeypatch):
    started = asyncio.Event()

    async def fake_default_llm_node(_self, _chat_ctx, _tools, _model_settings):
        yield "Hold on, "
        started.set()
        await asyncio.sleep(10)  # never resolves before the test cancels it
        yield "never gets here."

    async def fake_synthesize_phrase(*_args, **_kwargs):
        raise AssertionError("first phrase is intentionally too short to release before cancellation")

    monkeypatch.setattr(Agent.default, "llm_node", fake_default_llm_node)
    monkeypatch.setattr(agent_module, "synthesize_phrase", fake_synthesize_phrase)

    video_gen = _FakeVideoGen()
    voice_agent = MuseTalkVoiceAgent(avatar_handle=_ReadyAvatarHandle(video_gen))

    gen = voice_agent.llm_node(chat_ctx=None, tools=[], model_settings=None)  # type: ignore[arg-type]
    task = asyncio.create_task(gen.__anext__())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await gen.aclose()

    assert video_gen.cleared is True
