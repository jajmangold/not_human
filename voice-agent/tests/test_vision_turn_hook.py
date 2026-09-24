"""Unit tests for MuseTalkVoiceAgent.on_user_turn_completed -- the
STT-artifact filter (MiniCPM-V via the `vision` sidecar) and scene-caption
injection added alongside the live-perception feature. See
vision_client.py's module docstring and agent.py's on_user_turn_completed
docstring for the full rationale.

Fakes out VisionClient entirely (no network) -- real per-call HTTP
behavior is covered by test_vision_client.py; this file is about whether
agent.py wires that client into the chat-context lifecycle correctly.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import pytest
from livekit.agents import StopResponse, llm

import agent as agent_module
from agent import MuseTalkVoiceAgent, _maybe_react_to_visual_event, _run_scene_narrator
from vision_client import SceneNarrative, SceneState


class _FakeSession:
    def __init__(self, *, agent_state: str = "idle", user_state: str = "listening") -> None:
        self.agent_state = agent_state
        self.user_state = user_state
        self.generate_reply_calls: list[dict] = []

    def generate_reply(self, **kwargs: object) -> None:
        self.generate_reply_calls.append(kwargs)


class _FakeVisionClient:
    def __init__(
        self,
        *,
        is_real_speech: bool = True,
        visual_answer: str | None = "there are three fingers up",
        narrative: str | None = "The person is at their desk, and a bird is nearby.",
    ) -> None:
        self.is_real_speech = is_real_speech
        self.visual_answer = visual_answer
        self.narrative = narrative
        self.classified: list[str] = []
        self.visual_questions: list[tuple[bytes, str]] = []
        self.narrate_calls: list[tuple[bytes, str, str | None]] = []

    async def classify_transcript(self, text: str) -> bool:
        self.classified.append(text)
        return self.is_real_speech

    async def answer_visual_question(self, jpeg_bytes: bytes, question: str) -> str | None:
        self.visual_questions.append((jpeg_bytes, question))
        return self.visual_answer

    async def narrate(self, jpeg_bytes: bytes, raw_log: str, object_timeline: str | None) -> str | None:
        self.narrate_calls.append((jpeg_bytes, raw_log, object_timeline))
        return self.narrative


def _turn_ctx_with_message(text: str) -> tuple[llm.ChatContext, llm.ChatMessage]:
    ctx = llm.ChatContext.empty()
    message = ctx.add_message(role="user", content=text)
    return ctx, message


@pytest.mark.asyncio
async def test_vision_disabled_is_a_no_op():
    # vision_client=None is the default (VISION_ENABLED=0) -- must never
    # raise or otherwise change behavior when the feature is off.
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None)
    ctx, message = _turn_ctx_with_message("hello there")

    await voice_agent.on_user_turn_completed(ctx, message)  # must not raise

    assert len(ctx.items) == 1  # nothing injected


@pytest.mark.asyncio
async def test_real_speech_passes_through_with_no_scene_note_when_stale():
    fake_client = _FakeVisionClient(is_real_speech=True)
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    ctx, message = _turn_ctx_with_message("What's the weather like today?")

    await voice_agent.on_user_turn_completed(ctx, message)

    assert fake_client.classified == ["What's the weather like today?"]
    assert len(ctx.items) == 1  # scene_state is at its empty/stale default -- nothing injected


@pytest.mark.asyncio
async def test_artifact_is_logged_but_not_dropped_by_default(monkeypatch):
    # Log-only by default as of 2026-09-10 -- a real live test found the
    # classifier can misjudge a real question stitched to a hallucinated
    # tail as a whole artifact; a wrongly dropped real utterance is worse
    # than an occasional artifact slipping through. See
    # VISION_STT_FILTER_ENFORCE's comment in agent.py.
    monkeypatch.setattr(agent_module, "VISION_STT_FILTER_ENFORCE", False)
    fake_client = _FakeVisionClient(is_real_speech=False)
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    ctx, message = _turn_ctx_with_message("mmm-hmm")

    await voice_agent.on_user_turn_completed(ctx, message)  # must not raise

    assert fake_client.classified == ["mmm-hmm"]  # still classified/logged


@pytest.mark.asyncio
async def test_artifact_raises_stop_response_when_enforce_is_enabled(monkeypatch):
    monkeypatch.setattr(agent_module, "VISION_STT_FILTER_ENFORCE", True)
    fake_client = _FakeVisionClient(is_real_speech=False)
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    ctx, message = _turn_ctx_with_message("mmm-hmm")

    with pytest.raises(StopResponse):
        await voice_agent.on_user_turn_completed(ctx, message)


@pytest.mark.asyncio
async def test_fresh_scene_caption_is_injected_as_system_message():
    fake_client = _FakeVisionClient(is_real_speech=True)
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    voice_agent.scene_state = SceneState(caption="a person is waving", updated_at=time.monotonic())
    voice_agent.scene_log.observe("a person is waving")
    # Deliberately NOT a visual question (no keyword match, no
    # last_frame_jpeg set) -- isolates the ambient-note path from the
    # targeted-query path covered separately below.
    ctx, message = _turn_ctx_with_message("What's for dinner tonight?")

    await voice_agent.on_user_turn_completed(ctx, message)

    assert fake_client.visual_questions == []  # targeted path never triggered
    assert len(ctx.items) == 2
    injected = ctx.items[-1]
    assert injected.role == "system"
    assert "a person is waving" in injected.text_content


@pytest.mark.asyncio
async def test_stale_scene_caption_is_not_injected():
    fake_client = _FakeVisionClient(is_real_speech=True)
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    voice_agent.scene_state = SceneState(caption="a person is waving", updated_at=time.monotonic() - 60.0)
    ctx, message = _turn_ctx_with_message("What's for dinner tonight?")

    await voice_agent.on_user_turn_completed(ctx, message)

    assert len(ctx.items) == 1  # too stale -- not injected


@pytest.mark.asyncio
async def test_visual_question_with_fresh_frame_triggers_targeted_query():
    fake_client = _FakeVisionClient(is_real_speech=True, visual_answer="you're holding up three fingers")
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    # Ambient caption present too, to confirm the targeted answer takes
    # priority over it rather than both (or neither) being injected.
    voice_agent.scene_state = SceneState(caption="a person is sitting at a desk", updated_at=time.monotonic())
    voice_agent.scene_log.observe("a person is sitting at a desk")
    voice_agent.last_frame_jpeg = b"fake-jpeg-bytes"
    voice_agent.last_frame_at = time.monotonic()
    ctx, message = _turn_ctx_with_message("How many fingers am I holding up?")

    await voice_agent.on_user_turn_completed(ctx, message)

    assert fake_client.visual_questions == [(b"fake-jpeg-bytes", "How many fingers am I holding up?")]
    assert len(ctx.items) == 2
    injected = ctx.items[-1]
    assert injected.role == "system"
    assert "three fingers" in injected.text_content
    assert "sitting at a desk" not in injected.text_content  # targeted note wins, not both


@pytest.mark.asyncio
async def test_visual_question_with_no_frame_yet_falls_back_to_ambient_note():
    fake_client = _FakeVisionClient(is_real_speech=True)
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    voice_agent.scene_state = SceneState(caption="a person is sitting at a desk", updated_at=time.monotonic())
    voice_agent.scene_log.observe("a person is sitting at a desk")
    # last_frame_jpeg stays at its default None -- no frame has arrived yet.
    ctx, message = _turn_ctx_with_message("How many fingers am I holding up?")

    await voice_agent.on_user_turn_completed(ctx, message)

    assert fake_client.visual_questions == []  # never attempted -- nothing to query against
    injected = ctx.items[-1]
    assert "sitting at a desk" in injected.text_content


@pytest.mark.asyncio
async def test_non_visual_question_never_triggers_targeted_query():
    fake_client = _FakeVisionClient(is_real_speech=True)
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    voice_agent.last_frame_jpeg = b"fake-jpeg-bytes"
    voice_agent.last_frame_at = time.monotonic()
    ctx, message = _turn_ctx_with_message("What's the capital of France?")

    await voice_agent.on_user_turn_completed(ctx, message)

    assert fake_client.visual_questions == []  # non-visual, never triggered


@pytest.mark.asyncio
async def test_empty_transcript_skips_the_classifier_entirely():
    fake_client = _FakeVisionClient(is_real_speech=False)  # would drop the turn if called
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    ctx, message = _turn_ctx_with_message("")

    await voice_agent.on_user_turn_completed(ctx, message)  # must not raise StopResponse

    assert fake_client.classified == []  # never even called


@pytest.mark.asyncio
async def test_followup_within_window_gets_a_fresh_check_without_keyword_match():
    # Found live (2026-09-10): "how many fingers" matched and got a real
    # answer; the finger count then changed and the follow-up ask used
    # different wording that matched no keyword at all -- no fresh camera
    # check happened, and the model just repeated its first answer from
    # memory. Fix: a genuine visual answer keeps the exchange "sticky" for
    # VISUAL_FOLLOWUP_WINDOW_S, so a same-topic follow-up with no keyword
    # match still gets re-checked.
    fake_client = _FakeVisionClient(is_real_speech=True, visual_answer="four fingers")
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    voice_agent.last_frame_jpeg = b"frame-1"
    voice_agent.last_frame_at = time.monotonic()
    ctx, message = _turn_ctx_with_message("How many fingers am I holding up?")
    await voice_agent.on_user_turn_completed(ctx, message)
    assert len(fake_client.visual_questions) == 1  # the keyword match itself

    # A brand new turn, no keyword match, but within the follow-up window
    # and a fresh frame (the count changed) -- must still trigger a check.
    fake_client.visual_answer = "one finger"
    voice_agent.last_frame_jpeg = b"frame-2"
    voice_agent.last_frame_at = time.monotonic()
    ctx2, message2 = _turn_ctx_with_message("What about now?")

    await voice_agent.on_user_turn_completed(ctx2, message2)

    assert len(fake_client.visual_questions) == 2
    assert fake_client.visual_questions[-1] == (b"frame-2", "What about now?")
    injected = ctx2.items[-1]
    assert "one finger" in injected.text_content


@pytest.mark.asyncio
async def test_followup_after_window_expires_does_not_get_a_fresh_check():
    fake_client = _FakeVisionClient(is_real_speech=True, visual_answer="four fingers")
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    voice_agent.last_frame_jpeg = b"frame-1"
    voice_agent.last_frame_at = time.monotonic()
    ctx, message = _turn_ctx_with_message("How many fingers am I holding up?")
    await voice_agent.on_user_turn_completed(ctx, message)
    assert len(fake_client.visual_questions) == 1

    # Simulate the sticky window having expired.
    voice_agent._last_visual_question_at = time.monotonic() - (agent_module.VISUAL_FOLLOWUP_WINDOW_S + 5.0)
    voice_agent.last_frame_jpeg = b"frame-2"
    voice_agent.last_frame_at = time.monotonic()
    ctx2, message2 = _turn_ctx_with_message("What about now?")

    await voice_agent.on_user_turn_completed(ctx2, message2)

    assert len(fake_client.visual_questions) == 1  # not re-triggered -- window expired, no keyword either


# -- YOLO cross-check (parrot-hat misattribution fix) ------------------------


def test_render_yolo_objects_counts_repeats_and_preserves_order():
    from agent import _render_yolo_objects

    objects = [
        {"label": "person", "confidence": 0.9},
        {"label": "bird", "confidence": 0.7},
        {"label": "person", "confidence": 0.6},
    ]
    assert _render_yolo_objects(objects) == "YOLO detected: person x2, bird"


def test_render_yolo_objects_empty_list_returns_none():
    from agent import _render_yolo_objects

    assert _render_yolo_objects([]) is None


@pytest.mark.asyncio
async def test_ambient_note_includes_yolo_objects_alongside_changelog():
    fake_client = _FakeVisionClient(is_real_speech=True)
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    voice_agent.scene_state = SceneState(
        caption="a person is holding a bird",
        objects=[{"label": "person", "confidence": 0.9}, {"label": "bird", "confidence": 0.7}],
        updated_at=time.monotonic(),
    )
    voice_agent.scene_log.observe("a person is holding a bird")
    ctx, message = _turn_ctx_with_message("What's for dinner tonight?")

    await voice_agent.on_user_turn_completed(ctx, message)

    injected = ctx.items[-1]
    assert "a person is holding a bird" in injected.text_content
    assert "YOLO detected: person, bird" in injected.text_content


@pytest.mark.asyncio
async def test_targeted_note_includes_yolo_objects_too():
    # This is exactly the path that produced the real parrot-hat
    # misattribution: a targeted visual answer with no YOLO cross-check
    # available at all, so gemma had nothing to weigh it against.
    fake_client = _FakeVisionClient(is_real_speech=True, visual_answer="you're holding a bird wearing a hat")
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    voice_agent.scene_state = SceneState(objects=[{"label": "person", "confidence": 0.9}, {"label": "bird", "confidence": 0.7}])
    voice_agent.last_frame_jpeg = b"fake-jpeg-bytes"
    voice_agent.last_frame_at = time.monotonic()
    ctx, message = _turn_ctx_with_message("What am I holding?")

    await voice_agent.on_user_turn_completed(ctx, message)

    injected = ctx.items[-1]
    assert "you're holding a bird wearing a hat" in injected.text_content
    assert "YOLO detected: person, bird" in injected.text_content


@pytest.mark.asyncio
async def test_note_omits_yolo_line_when_no_objects_detected():
    fake_client = _FakeVisionClient(is_real_speech=True)
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    voice_agent.scene_state = SceneState(caption="an empty room", objects=[], updated_at=time.monotonic())
    voice_agent.scene_log.observe("an empty room")
    ctx, message = _turn_ctx_with_message("What's for dinner tonight?")

    await voice_agent.on_user_turn_completed(ctx, message)

    injected = ctx.items[-1]
    assert "an empty room" in injected.text_content
    assert "YOLO" not in injected.text_content


# -- SceneNarrator (background consolidation) --------------------------------


@pytest.mark.asyncio
async def test_ambient_note_prefers_fresh_narrative_over_raw_changelog():
    fake_client = _FakeVisionClient(is_real_speech=True)
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    voice_agent.scene_state = SceneState(caption="a person is holding a mug", updated_at=time.monotonic())
    voice_agent.scene_log.observe("a person is holding a mug")
    voice_agent.scene_narrative = SceneNarrative(
        text="The person has been at their desk drinking coffee for the last minute.", updated_at=time.monotonic()
    )
    ctx, message = _turn_ctx_with_message("What's for dinner tonight?")

    await voice_agent.on_user_turn_completed(ctx, message)

    injected = ctx.items[-1]
    assert "drinking coffee" in injected.text_content
    assert "a person is holding a mug" not in injected.text_content  # narrative wins, not the raw changelog


@pytest.mark.asyncio
async def test_ambient_note_falls_back_to_raw_changelog_when_narrative_stale():
    fake_client = _FakeVisionClient(is_real_speech=True)
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    voice_agent.scene_state = SceneState(caption="a person is holding a mug", updated_at=time.monotonic())
    voice_agent.scene_log.observe("a person is holding a mug")
    # Default SceneNarrative() -- never populated (narrator hasn't run
    # yet, or lfm2vl-narrate has been down) -- must fall back rather than
    # inject nothing.
    ctx, message = _turn_ctx_with_message("What's for dinner tonight?")

    await voice_agent.on_user_turn_completed(ctx, message)

    injected = ctx.items[-1]
    assert "a person is holding a mug" in injected.text_content


@pytest.mark.asyncio
async def test_run_scene_narrator_updates_scene_narrative_from_log_and_frame(monkeypatch):
    monkeypatch.setattr(agent_module, "NARRATE_INTERVAL_S", 0.0)
    fake_client = _FakeVisionClient(narrative="A person is waving near a bird.")
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    voice_agent.last_frame_jpeg = b"fake-jpeg-bytes"
    voice_agent.scene_log.observe("a person is waving")
    voice_agent.object_presence.observe([{"label": "bird", "confidence": 0.7}])

    task = asyncio.create_task(_run_scene_narrator(voice_agent, fake_client))
    try:
        await asyncio.sleep(0.05)  # let a handful of (near-instant) loop iterations run
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert fake_client.narrate_calls
    jpeg_bytes, raw_log, object_timeline = fake_client.narrate_calls[0]
    assert jpeg_bytes == b"fake-jpeg-bytes"
    assert "a person is waving" in raw_log
    assert object_timeline is not None and "bird" in object_timeline
    assert voice_agent.scene_narrative.text == "A person is waving near a bird."
    assert voice_agent.scene_narrative.updated_at > 0


@pytest.mark.asyncio
async def test_run_scene_narrator_skips_when_no_frame_yet(monkeypatch):
    monkeypatch.setattr(agent_module, "NARRATE_INTERVAL_S", 0.0)
    fake_client = _FakeVisionClient()
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    voice_agent.scene_log.observe("a person is waving")  # log has content, but no frame yet

    task = asyncio.create_task(_run_scene_narrator(voice_agent, fake_client))
    try:
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert fake_client.narrate_calls == []
    assert voice_agent.scene_narrative.text is None


@pytest.mark.asyncio
async def test_run_scene_narrator_skips_when_log_is_empty(monkeypatch):
    monkeypatch.setattr(agent_module, "NARRATE_INTERVAL_S", 0.0)
    fake_client = _FakeVisionClient()
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None, vision_client=fake_client)
    voice_agent.last_frame_jpeg = b"fake-jpeg-bytes"  # frame present, but nothing observed yet

    task = asyncio.create_task(_run_scene_narrator(voice_agent, fake_client))
    try:
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert fake_client.narrate_calls == []


# -- _maybe_react_to_visual_event (proactive wave/new-object reactions) ------


@pytest.mark.asyncio
async def test_maybe_react_fires_for_wave_when_idle_and_listening(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr(agent_module.time, "monotonic", lambda: fake_now[0])
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None)
    reset_calls = []
    voice_agent.wave_detector.reset = lambda: reset_calls.append(1)
    session = _FakeSession(agent_state="idle", user_state="listening")

    await _maybe_react_to_visual_event(session, voice_agent, kind="wave")

    assert len(session.generate_reply_calls) == 1
    assert "waved" in session.generate_reply_calls[0]["instructions"]
    assert reset_calls == [1]  # detector reset so the same continued wave can't refire immediately
    assert voice_agent._last_proactive_reaction_at == fake_now[0]


@pytest.mark.asyncio
async def test_maybe_react_fires_for_new_object_with_its_label_in_the_instructions():
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None)
    session = _FakeSession(agent_state="idle", user_state="listening")

    await _maybe_react_to_visual_event(session, voice_agent, kind="object", detail="bird")

    assert len(session.generate_reply_calls) == 1
    assert "bird" in session.generate_reply_calls[0]["instructions"]


@pytest.mark.asyncio
async def test_maybe_react_does_not_reset_wave_detector_for_an_object_trigger():
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None)
    reset_calls = []
    voice_agent.wave_detector.reset = lambda: reset_calls.append(1)
    session = _FakeSession(agent_state="idle", user_state="listening")

    await _maybe_react_to_visual_event(session, voice_agent, kind="object", detail="cup")

    assert reset_calls == []


@pytest.mark.asyncio
async def test_maybe_react_respects_cooldown():
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None)
    session = _FakeSession(agent_state="idle", user_state="listening")

    await _maybe_react_to_visual_event(session, voice_agent, kind="wave")
    await _maybe_react_to_visual_event(session, voice_agent, kind="wave")

    assert len(session.generate_reply_calls) == 1  # second call suppressed by cooldown


@pytest.mark.asyncio
async def test_maybe_react_suppressed_while_agent_is_speaking():
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None)
    session = _FakeSession(agent_state="speaking", user_state="listening")

    await _maybe_react_to_visual_event(session, voice_agent, kind="wave")

    assert session.generate_reply_calls == []


@pytest.mark.asyncio
async def test_maybe_react_suppressed_while_user_is_speaking():
    voice_agent = MuseTalkVoiceAgent(avatar_handle=None)
    session = _FakeSession(agent_state="idle", user_state="speaking")

    await _maybe_react_to_visual_event(session, voice_agent, kind="wave")

    assert session.generate_reply_calls == []
