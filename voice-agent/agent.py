"""LiveKit voice agent (internal issue #45): self-hosted STT -> qwen27b
LLM -> self-hosted Kokoro TTS, streamed phrase-by-phrase into a MuseTalk
avatar with an automatically classified reaction per phrase (see
phrase_pipeline.py -- no LLM tool-call round-trip is involved in that
anymore, matching musetalk-volta's own low-latency `/ws/live/` demo).

Run from this directory:

    uv sync
    uv run python agent.py console   # local terminal test, no LiveKit server/room needed
    uv run python agent.py dev       # connects to the real LiveKit room
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import time
from pathlib import Path
from typing import AsyncIterable

import aiohttp
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomOutputOptions,
    StopResponse,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents import voice
from livekit.agents.types import FlushSentinel
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.avatar import AudioSegmentEnd
from livekit.plugins import openai, silero

from avatar_bridge import AVATAR_IDENTITY, start_avatar_worker
from kokoro_tts import SAMPLE_RATE as KOKORO_SAMPLE_RATE
from kokoro_tts import KokoroTTS, _wav_to_pcm16_mono
from muse_video_generator import MuseTalkVideoGenerator
from natural_motion import NaturalMotionScheduler
from phrase_pipeline import synthesize_phrase
from speech_phrases import release_speech_phrases
from text_sanitizer import LeakedToolCallSanitizer
from system_prompt import SYSTEM_PROMPT
from vision_client import (
    BUFFER_FRAME_INTERVAL_S,
    NARRATE_INTERVAL_S,
    FrameThrottle,
    ObjectPresenceTracker,
    SceneChangeLog,
    SceneNarrative,
    SceneState,
    VisionClient,
    WaveGestureDetector,
    frame_to_jpeg,
)

# Share the same .env file as the Next.js frontend.
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# The agents SDK expects LIVEKIT_URL; the frontend uses NEXT_PUBLIC_LIVEKIT_URL.
if not os.getenv("LIVEKIT_URL") and os.getenv("NEXT_PUBLIC_LIVEKIT_URL"):
    os.environ["LIVEKIT_URL"] = os.environ["NEXT_PUBLIC_LIVEKIT_URL"]

WHISPER_STT_URL = os.environ.get("WHISPER_STT_URL", "http://127.0.0.1:18097")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen27b")
KOKORO_URL = os.environ.get("KOKORO_URL", "http://127.0.0.1:18091")
AFFECT_URL = os.environ.get("AFFECT_URL", "http://127.0.0.1:18094")
# Live perception of the human participant's own camera (not the avatar's
# driving frames -- see vision_client.py's module docstring). Off by
# default: it's new, not yet exercised against a real camera anywhere in
# this fleet (see musetalk-volta/docs/VISION.md), and depends on a sidecar
# that loads two GPU-resident models at startup -- opt in once that's
# actually deployed and healthy, don't silently require it for every
# session.
VISION_URL = os.environ.get("VISION_URL", "http://127.0.0.1:18098")
VISION_ENABLED = os.environ.get("VISION_ENABLED", "0") == "1"
# How stale a scene caption can be before on_user_turn_completed stops
# bothering to inject it -- no point telling gemma4-26b what the camera
# saw several turns ago as if it were current.
VISION_SCENE_MAX_AGE_S = float(os.environ.get("VISION_SCENE_MAX_AGE_S", "10.0"))
# Log-only again (2026-09-10): v2's prompt fixed the first false positive
# ("What do I look like?  Let's get you a change.  Thank you." -> REAL),
# so enforcement was turned back on -- but a longer live test found a
# 67% false-positive rate on the turns it actually dropped, including
# clearly real, substantive content ("Talk like a futuristic cyborg robot
# from now on.", "No toothbrush, just the pliers.", "It's the same
# parrot, it's just backwards.", "Yep."). This classifier (MiniCPM-V-4.6,
# CLASSIFY_PROMPT_TEMPLATE in minicpmv_runtime.py) is not reliable enough
# for hard enforcement in real conversation right now -- the risk of
# dropping real user turns outweighs catching the stray noise it's meant
# to filter, most of which the VAD tuning in this AgentSession already
# handles anyway (trailing "thank you"/"bye", short coughs). Back to
# log-only (verdicts still logged for future prompt tuning, never acted
# on) until the classifier's real-world accuracy actually improves --
# don't flip this back to "1" without a real before/after test against a
# live session, not just the synthetic transcript list this prompt was
# originally tuned against.
VISION_STT_FILTER_ENFORCE = os.environ.get("VISION_STT_FILTER_ENFORCE", "0") == "1"

# Appended to SYSTEM_PROMPT only when a vision_client is actually wired in
# (see MuseTalkVoiceAgent.__init__) -- states the camera capability
# up front rather than relying on a passive per-turn note to convince the
# model it has one. See that __init__'s comment for why: a live test
# found the model denies having a camera even when a fresh caption WAS
# injected that same turn.
# v2 (2026-09-10): v1 said to "trust it completely," which a live test
# showed was too strong -- shown a toy parrot wearing a tiny hat, the 450M
# vision model correctly said "parrot" but wrongly attributed the hat to
# the PERSON ("you put on the hat"), and gemma repeated that as fact
# because it had been told to trust the note completely. The fix isn't
# just wording -- it's giving gemma something to cross-check specific
# claims against: YOLO's object list (a separate, more reliable detector
# for coarse "what's in frame," even though it can't describe attributes
# or actions) now rides alongside every "[Camera" note for exactly that
# reason. See _scene_note()/_targeted_note() in on_user_turn_completed.
VISION_CAPABILITY_CLAUSE = """

You have a live camera feed of the person you're talking to -- this is real,
not hypothetical, and you should never claim you have no camera or can't
see. But be clear-eyed about what's reliable in it and what isn't:

Any line starting with "[Camera" is a live read of that feed, not
transcribed speech or invented -- trust that the FEED is real and current.
Do NOT extend that same trust to every DETAIL claimed in it. The caption
and targeted-answer text come from a small, fast 450M-parameter vision
model: reliably fine at coarse "who/what is roughly in frame," genuinely
unreliable on specifics like which object has which attribute or exactly
who did what to what -- it can confidently state a wrong detail with no
sign of hesitation. A separate "YOLO detected: ..." line, when present,
comes from a dedicated object detector and is more reliable for "is X
actually in frame" than the caption text, though it can't describe
actions or attributes at all. Cross-check: if the caption claims something
specific and detailed (an action, who's wearing/holding/doing what) that
isn't corroborated by the YOLO list or doesn't fit the rest of what you
know, don't just repeat it as fact -- say what you're reasonably confident
of and hold the specific detail more loosely, or ask rather than assert.
If no "[Camera" line appears on a given turn, the feed hasn't produced
anything new that moment -- say you're not sure what you're seeing right
now rather than denying the camera exists.

Crucial: all of that is for YOUR reasoning only, never for your words. You
are the one looking, not a system relaying reports from one -- speak
entirely in first person, as if the seeing is yours ("I think I'm seeing
...", "it looks like ..."), and never say "the feed", "the camera", "the
caption", "the note", "the model", or anything else that names the
machinery behind what you see. A live test produced exactly the failure to
avoid: told the same person looked like they had a mustache after an
earlier read said beard, it said out loud "though the feed just mentioned
a beard a second ago" -- correct to notice the conflict and hedge, wrong to
narrate where the conflicting read came from. Handle that same situation by
just describing your own uncertainty instead: "I think I'm seeing a
mustache now, though I may have misread it a moment ago."
"""

# A rolling ambient caption (one generic sentence, refreshed every ~2s) can't
# answer a SPECIFIC visual question -- found live: asked to count fingers,
# got "there's no camera" instead of an attempt, because nothing told
# MiniCPM-V that finger-counting was the actual question. This heuristic
# flags a turn as worth a fresh, targeted answer_visual_question() call
# (the user's own words become the vision model's prompt, run against the
# most recent raw frame) instead of just the passive ambient note. A plain
# keyword list is crude -- it will both miss some visual questions and
# occasionally fire on non-visual ones ("what do you see in this plan") --
# but the targeted call is extra latency, not a correctness risk (it just
# adds one more grounded note), so a false-positive match costs a couple
# of seconds, not a wrong answer.
VISUAL_QUESTION_KEYWORDS = (
    "see me", "look at", "looking at", "hold up", "holding up", "how many finger",
    "count my", "what am i", "what am i wearing", "what color", "what colour",
    "am i wearing", "do i look", "what do i look", "can you see", "what's behind me",
    "what is behind me", "what's in my hand", "what is in my hand",
)


def _looks_like_visual_question(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in VISUAL_QUESTION_KEYWORDS)


def _render_yolo_objects(objects: list[dict[str, object]]) -> str | None:
    """Renders scene_state.objects (YOLO26n's detections, already
    confidence-thresholded server-side -- see vision/yolo_runtime.py) as a
    compact label list, e.g. "person x2, bird" -- the cross-check
    VISION_CAPABILITY_CLAUSE tells gemma to weigh a specific caption claim
    against. Counts repeats of the same label rather than listing each
    detection separately (box coordinates aren't useful to a spoken
    conversation); order follows the service's own confidence-descending
    sort, so the first-mentioned label is also the most confident one."""
    if not objects:
        return None
    counts: dict[str, int] = {}
    for obj in objects:
        label = obj.get("label")
        if isinstance(label, str):
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return None
    parts = [f"{label} x{count}" if count > 1 else label for label, count in counts.items()]
    return "YOLO detected: " + ", ".join(parts)


# Found live (2026-09-10): asked "how many fingers" (matched, got a real
# targeted answer, correct), then changed the finger count and asked
# again with different phrasing -- that second ask didn't match any
# keyword, so no fresh camera check happened at all, and the model just
# repeated its FIRST answer from conversation memory instead of admitting
# it hadn't looked again. A bigger keyword list is a losing game against
# natural follow-up phrasing ("what about now", "and this one") -- the
# real fix is treating a visual exchange as "sticky" for a while: once a
# real visual question fires, keep re-checking the camera on every turn
# for this window even if later turns don't match any keyword. Same
# cost/risk framing as the keyword list itself -- an unnecessary check
# costs a couple of seconds, not correctness.
VISUAL_FOLLOWUP_WINDOW_S = 20.0

# How long to wait between proactive, unprompted reactions to something
# purely visual (a wave, a new object arriving) -- see
# _maybe_react_to_visual_event(). Shared across every trigger type on
# purpose: a wave followed moments later by "oh, a new mug appeared" reads
# as attentive once, erratic/chatty if it keeps interjecting. Long enough
# that it never competes with the conversation itself, short enough that
# a genuinely new event minutes later still gets a fresh reaction.
PROACTIVE_REACTION_COOLDOWN_S = 20.0

# Avatar video (#45 milestone 2). MUSE_TALK_IO_DIR is the *host* path to
# the shared directory MuseTalk's container mounts as /io; the container
# paths below are what job JSON files must reference.
# Off by default: the avatar video path had an unbounded-queue memory leak whose fix has
# not been confirmed in a live session (docs/lab-notebook/05-dispatch-bound.md).
MUSE_AVATAR_ENABLED = os.environ.get("MUSE_AVATAR_ENABLED", "0") == "1"
MUSE_TALK_IO_DIR = Path(os.environ.get(
    "MUSE_TALK_IO_DIR",
    "./runtime",
))
# Plain uploaded-portrait image -- used only for the room video track's
# width/height and the initial idle frame (see avatar_bridge.py), not for
# the MuseTalk job itself.
MUSE_AVATAR_PORTRAIT_CONTAINER_PATH = os.environ.get(
    "MUSE_AVATAR_PORTRAIT_CONTAINER_PATH",
    "/io/web_demo/uploads/portrait-1e4285741243ef20d9f6.png",
)
# The actual job target: a prepared ALP semantic-bank *motion* video, not
# the plain portrait above. Originally this agent pointed MuseTalk jobs at
# the bare portrait directly (bank_size=1) -- confirmed live, via
# muse_server.py's own "Total frame:「1」" log line, that this silently
# disabled *all* reaction/blink/gaze/head-pose rendering, since
# muse_server.py's _render_sources() only engages REACTION_SLOTS-based
# blending for bank_size >= 16. This identity (bank_size=24, same source
# portrait, prepared via the same /api/live/start ALP pipeline
# musetalk-volta's own web demo uses) is the one already validated live in
# this repo's reaction-render tests. bank_start_frame=8/source_start_frame=0
# match how it was actually prepared -- these are part of the resident's
# avatar cache key (muse_server.py) *and* the offset REACTION_SLOTS'
# indices are relative to, so they must match the prepared identity exactly
# or reactions render against the wrong frames (or the resident silently
# re-prepares from scratch under the same avatar_id). Override all four via
# env if pointing at a different prepared identity.
MUSE_AVATAR_VIDEO_CONTAINER_PATH = os.environ.get(
    "MUSE_AVATAR_VIDEO_CONTAINER_PATH",
    "/io/web_demo/motion/motion-1e4285741243ef20d9f6-semantic-bank-v9-kiss-f24-3bba7329.mp4",
)
MUSE_AVATAR_ID = os.environ.get(
    "MUSE_AVATAR_ID", "web-1e4285741243ef20d9f6-alp-semantic-bank-v9-kiss-settled-f24-3bba7329"
)
MUSE_AVATAR_BANK_START_FRAME = int(os.environ.get("MUSE_AVATAR_BANK_START_FRAME", "8"))
MUSE_AVATAR_SOURCE_START_FRAME = int(os.environ.get("MUSE_AVATAR_SOURCE_START_FRAME", "0"))
# muse_server.py's _render_sources() slices the prepared bank by
# bank_start_frame before indexing into it (frame_cycle =
# self.frame_list_cycle[bank_start_frame:]), so a bank's *usable* size for
# this purpose is (total frames - bank_start_frame), not the raw total.
# reaction_schedule.py's HEAD_POSE_SLOTS needs indices up to 19, i.e. a
# usable size >= 20. This identity has 24 total frames and
# bank_start_frame=8 -> usable size 16, which supports REACTION_SLOTS/
# QUIET_SLOTS/BLINK_SLOT (all <= 15) but NOT head-pose slots. Confirmed
# live: enabling head_events for this identity crashed the resident
# ("IndexError: list index out of range" indexing coord_cycle[head_index]
# in _render_sources) mid-job, which is what actually produced the
# reported "lips blocking/glitching" -- the job failed partway through,
# freezing on its last successfully rendered frame. A future identity
# prepared with a larger bank (the old demo's current ALP profile uses 28
# frames specifically for this headroom: 28-8=20, exactly enough) can flip
# this on.
MUSE_AVATAR_SUPPORTS_HEAD_POSE = os.environ.get("MUSE_AVATAR_SUPPORTS_HEAD_POSE", "0") == "1"

# Larger than release_speech_phrases' own defaults (50/220, the old demo's
# tuning) -- deliberately. Live-tested against a real 15-phrase multi-turn
# conversation: MuseTalk's resident spent ~75s of GPU/processing time on
# ~82s of audio, of which ~20s (27%) was roughly-fixed per-job overhead
# (~1.3-2s/job, nearly independent of length) rather than actual frame
# generation -- confirmed via [PROFILE] GPU gen loop vs MUSE JOB DONE total
# time in the resident's own logs. That left no margin: bursts of several
# short phrases arriving without a natural pause between them measured the
# LiveKit-side pacer up to ~29s behind schedule before catching back up.
# Fewer, longer phrases pay that fixed tax less often for the same amount
# of speech, buying real throughput headroom at the cost of a somewhat
# later first-audio moment for a long reply (still far faster than this
# agent's original whole-turn-buffered behavior). Chosen empirically, not
# just doubled on paper: a naive bump (e.g. 120) barely merges realistic
# LLM prose once individual sentences already exceed it alone, since each
# one then releases immediately rather than waiting to combine with its
# neighbor -- 200/420 was verified to roughly halve the phrase count on
# actual multi-sentence replies (see test_speech_phrases.py).
PHRASE_MIN_CHARS = int(os.environ.get("PHRASE_MIN_CHARS", "200"))
PHRASE_MAX_CHARS = int(os.environ.get("PHRASE_MAX_CHARS", "420"))

# The target fps requested from MuseTalk *and* the AVSynchronizer's pacing
# target -- these must match, since MuseTalk's own audio-to-frame-count
# chunking is fps-parameterized (a lower fps means fewer, proportionally
# longer frames for the same audio, not just slower playback of the same
# frames). Reported as "glitchy, speed all over the place" -- measured from
# musetalk-volta's own [PROFILE] GPU gen loop log lines across 11 real
# segments: sustained generation throughput ranged 14.6-24.2 fps, i.e. it
# never once reached the previous default of 25 even at its best, and swings
# by nearly 2x sample to sample (GPU is shared with kokoro/ALP/other
# musetalk-volta services). LiveKit's AVSynchronizer (verified against its
# actual source) paces output at a steady rate once frames arrive -- that
# part already matches its documented usage pattern correctly. The jitter is
# entirely upstream: whenever real generation throughput dips below the
# requested target, the synchronizer's queue runs dry, it logs "Frame
# capture was behind schedule", and resets its pacing clock -- repeatedly,
# throughout playback, not just once at segment start. Targeting a rate
# safely under the observed floor (rather than an aspirational one no
# sample ever sustained) trades peak smoothness for actually-steady
# playback. Override via MUSE_AVATAR_FPS if a real session shows headroom
# for higher.
MUSE_AVATAR_FPS = int(os.environ.get("MUSE_AVATAR_FPS", "12"))

logger = logging.getLogger("musetalk-voice-agent")

DEFAULT_VOICE = os.environ.get("KOKORO_VOICE", "af_bella")


class AvatarHandle:
    """Shared handle letting llm_node's phrase pipeline reach the avatar's
    MuseTalkVideoGenerator once avatar startup actually finishes.

    Avatar startup (~10s: joining a second room connection, loading the
    portrait, etc.) runs as a background task *after* session.start() (see
    entrypoint below -- fixed separately, so mic listening doesn't wait on
    it either). That means the very first reply's phrases can arrive before
    the avatar is ready; wait_ready() bounds how long a phrase's pipeline
    will wait for it before giving up and speaking text-only for that turn,
    rather than hanging indefinitely.
    """

    def __init__(self) -> None:
        self.video_gen: MuseTalkVideoGenerator | None = None
        self._ready = asyncio.Event()

    def set_ready(self, video_gen: MuseTalkVideoGenerator) -> None:
        self.video_gen = video_gen
        self._ready.set()

    async def wait_ready(self, timeout: float) -> MuseTalkVideoGenerator | None:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return self.video_gen


class MuseTalkVoiceAgent(Agent):
    def __init__(
        self,
        *,
        avatar_handle: AvatarHandle | None,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        vision_client: VisionClient | None = None,
    ) -> None:
        # Told explicitly and unconditionally that a camera exists, not
        # left to infer it from a stray bracketed note mid-conversation --
        # a live test found the model flatly denies having a camera
        # ("count my fingers" -> "there's no camera") even when a fresh
        # scene caption WAS injected that same turn. A passive note isn't
        # enough to override the model's default "I'm voice-only" prior;
        # this clause states the capability up front, in the same
        # instructions block the rest of its behavior comes from.
        instructions = SYSTEM_PROMPT + (VISION_CAPABILITY_CLAUSE if vision_client is not None else "")
        super().__init__(instructions=instructions)
        self._avatar_handle = avatar_handle
        self._voice = voice
        self._speed = speed
        self._previous_reaction = "neutral"
        # One instance for this agent's whole lifetime (one voice session),
        # not one per phrase -- its clock must advance continuously across
        # phrases for blink/gaze timing to look natural instead of
        # restarting (and clustering near phrase starts) every phrase. See
        # natural_motion.py's module docstring for why this was missing
        # entirely before (internal issue #45's "why is it still
        # crappy" follow-up): the avatar never blinked or shifted gaze
        # regardless of how well the facial-expression reaction rendered.
        self._natural_motion = NaturalMotionScheduler(secrets.randbits(64), fps=MUSE_AVATAR_FPS)
        # Live perception of the human participant's camera (see
        # vision_client.py). None when VISION_ENABLED=0 (the default) --
        # every use of it below already treats that as "feature off,"
        # never a startup error. `scene_state` is written from
        # entrypoint()'s video-frame loop, a plain background asyncio
        # task -- not a running-mode/streaming actor with its own
        # lifecycle, so a bare mutable dataclass instance is enough; no
        # lock needed since both the writer (one task) and reader
        # (on_user_turn_completed) run on the same event loop thread and
        # never await mid-mutation of the object itself (whole-object
        # replace, not a partial in-place edit -- see the frame loop).
        self._vision_client = vision_client
        self.scene_state = SceneState()
        # Rolling log of DISTINCT ambient captions over time (see
        # SceneChangeLog's docstring) -- added 2026-09-10 so the model
        # gets a sense of what's happened recently, not just a single
        # current-state snapshot that gets silently overwritten every
        # cycle.
        self.scene_log = SceneChangeLog()
        # Per-label presence timeline from YOLO, independent of caption
        # wording -- see ObjectPresenceTracker's docstring. Fed alongside
        # scene_log in the video-frame loop; consumed by the background
        # SceneNarrator as supporting context, not injected into chat
        # context directly.
        self.object_presence = ObjectPresenceTracker()
        # Background-consolidated storyline (see SceneNarrator below and
        # SceneNarrative's docstring in vision_client.py) -- much
        # slower-updating than scene_log, preferred over the raw changelog
        # in _scene_note() when fresh since it reads as one coherent
        # sentence instead of a list of individually noisy ticks.
        self.scene_narrative = SceneNarrative()
        # Raw frame kept alongside scene_state -- needed for an on-demand
        # answer_visual_question() call (see on_user_turn_completed),
        # independent of whether the ambient captioner that populates
        # scene_state happens to be healthy this cycle.
        self.last_frame_jpeg: bytes | None = None
        self.last_frame_at: float = 0.0
        # 0.0 means "not currently in a visual exchange" -- see
        # VISUAL_FOLLOWUP_WINDOW_S's comment.
        self._last_visual_question_at: float = 0.0
        # Proactive-reaction trigger state -- see _maybe_react_to_visual_event().
        # Pure-Python pose-motion detector (no model calls), fed from the
        # frame loop's fast buffer cadence.
        self.wave_detector = WaveGestureDetector()
        # 0.0 means "never fired" -- shared across trigger types (wave,
        # new object) so a wave and a new object arriving moments apart
        # don't both fire back to back; one proactive interjection at a
        # time reads as attentive, several in a row reads as erratic.
        self._last_proactive_reaction_at: float = 0.0

    async def on_user_turn_completed(self, turn_ctx: "llm.ChatContext", new_message: "llm.ChatMessage") -> None:
        """STT-artifact filter (MiniCPM-V's text backbone, via the
        `vision` sidecar) plus scene-caption injection, both gated on the
        same vision_client instance and run concurrently since they're
        independent -- see vision_client.py's module docstring for why
        each is best-effort/fail-open rather than something that can ever
        block or break a turn.

        Complements, does not replace, the VAD tuning in entrypoint()'s
        AgentSession (min_silence_duration/min_speech_duration): those
        catch trailing-silence and short-non-speech hallucinations before
        STT even runs; this catches semantic garbage that clears VAD and
        comes back as real-looking words (word-salad fragments, coughs
        transcribed as text) -- see vision_client.py's classify_transcript
        docstring for a measured accuracy caveat (it can't reliably tell a
        genuine standalone "Thank you." from a spurious one without
        conversation context; that specific case is what the VAD tuning
        already handles).
        """
        if self._vision_client is None:
            return
        text = new_message.text_content
        if not text:
            return

        async def _scene_note() -> str | None:
            # The changelog (see SceneChangeLog) rather than just the
            # latest single caption -- a short recent history of what's
            # changed reads more naturally to the model than a single
            # current-state line that silently replaces itself every
            # cycle. Still gated on the underlying scene_state's own
            # freshness: if the ambient captioner has gone stale (sidecar
            # down, camera stopped), don't hand the model a changelog that
            # looks current but isn't.
            if self.scene_state.is_stale(VISION_SCENE_MAX_AGE_S):
                return None
            # Prefer the background-consolidated narrative (see
            # SceneNarrator) over the raw changelog when it's fresh enough
            # to trust -- one coherent sentence synthesized from the last
            # ~20s of ticks (by the larger, more accurate 3B model, looking
            # at the actual current frame) reads more naturally and tracks
            # a storyline better than a bare list of per-tick captions.
            # Falls back to the raw changelog (scene_log) when the
            # narrator hasn't produced anything fresh yet -- e.g. right
            # after a session starts, or if lfm2vl-narrate is down --
            # rather than giving the model nothing at all.
            if not self.scene_narrative.is_stale(NARRATE_INTERVAL_S * 2):
                changelog = f"[Camera -- recent scene]\n{self.scene_narrative.text}"
            else:
                changelog = self.scene_log.render()
            if not changelog:
                return None
            # YOLO's object list rides alongside the caption changelog --
            # see VISION_CAPABILITY_CLAUSE and _render_yolo_objects()'s
            # docstring for why: it's what lets gemma cross-check a
            # specific caption claim instead of just repeating it.
            objects_line = _render_yolo_objects(self.scene_state.objects)
            return changelog if not objects_line else f"{changelog}\n{objects_line}"

        now = time.monotonic()
        is_visual_turn = _looks_like_visual_question(text) or (
            self._last_visual_question_at != 0.0 and (now - self._last_visual_question_at) < VISUAL_FOLLOWUP_WINDOW_S
        )

        async def _targeted_note() -> str | None:
            # Fires on a plausible visual question OR a follow-up within
            # VISUAL_FOLLOWUP_WINDOW_S of one -- see that constant's
            # comment for why a keyword match alone isn't enough. Also
            # requires a recent-enough raw frame -- see last_frame_jpeg's
            # comment in __init__.
            if not is_visual_turn:
                return None
            if self.last_frame_jpeg is None or (now - self.last_frame_at) > VISION_SCENE_MAX_AGE_S:
                return None
            answer = await self._vision_client.answer_visual_question(self.last_frame_jpeg, text)
            if not answer:
                return None
            note = f"[Camera -- you just looked, right now, specifically to answer what they asked: {answer}]"
            # Same cross-check as the ambient note -- the targeted answer
            # is still just the small vision model's read, not verified
            # truth, and this is exactly the path that produced the
            # parrot-hat misattribution that motivated adding this at all.
            objects_line = _render_yolo_objects(self.scene_state.objects)
            return note if not objects_line else f"{note}\n{objects_line}"

        is_real_speech, scene_note, targeted_note = await asyncio.gather(
            self._vision_client.classify_transcript(text), _scene_note(), _targeted_note()
        )
        if not is_real_speech:
            if VISION_STT_FILTER_ENFORCE:
                logger.info("dropping STT-artifact turn per MiniCPM-V classifier: %r", text)
                raise StopResponse()
            logger.info("MiniCPM-V classifier would drop this turn as an artifact (log-only, not enforced): %r", text)
        # Prefer the targeted answer over the ambient embedding when both
        # are available -- a direct answer to what they actually asked is
        # more useful than a generic scene description, and giving the
        # model two possibly-differently-phrased "[Camera" lines risks
        # reading as contradictory rather than complementary.
        note = targeted_note or scene_note
        if note:
            turn_ctx.add_message(role="system", content=note)
        if targeted_note is not None:
            # Refresh (not just set once) so a whole back-and-forth chain
            # of follow-ups stays "sticky," each one extending the window
            # from its own turn rather than counting down from only the
            # first question that happened to match a keyword.
            self._last_visual_question_at = now

    async def llm_node(
        self,
        chat_ctx: "llm.ChatContext",
        tools: list["llm.Tool"],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk | str | FlushSentinel]:
        # gemma4-26b (unlike qwen27b) has no native function-calling tokens
        # -- llama.cpp has to *parse* its delimiter-based output into
        # structured tool_calls, and that parsing is documented to be
        # unreliable: confirmed live, it sometimes leaks the raw call
        # syntax straight into visible content instead. Matches a known
        # upstream issue (llama-cpp-python#2227). See text_sanitizer.py
        # (unit-tested in tests/test_text_sanitizer.py) for the full
        # rationale; this is a safety net, not the fix.
        sanitizer = LeakedToolCallSanitizer()

        # internal issue #45's latency fix: in avatar mode, speech
        # synthesis + reaction classification happen phrase-by-phrase as
        # this text streams in (see _consume_phrases/phrase_pipeline.py),
        # not after the whole reply is buffered -- matching the
        # architecture of musetalk-volta's own low-latency `/ws/live/`
        # demo, confirmed to feel near-instant next to this agent's
        # original whole-turn-buffered behavior. tts_node below is
        # short-circuited in avatar mode so Kokoro isn't also called a
        # second time on the same text via the framework's default path.
        phrase_queue: asyncio.Queue[str | None] | None = None
        consumer_task: asyncio.Task[None] | None = None
        if self._avatar_handle is not None:
            phrase_queue = asyncio.Queue(maxsize=4)
            consumer_task = asyncio.create_task(self._consume_phrases(phrase_queue))

        phrase_buffer = ""
        completed_normally = False
        try:
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                # A genuine structured tool call (delta.tool_calls populated)
                # is never itself the leaked-text problem -- pass it through
                # as-is, immediately, rather than delaying it behind the
                # text holdback buffer below.
                if isinstance(chunk, llm.ChatChunk) and chunk.delta and chunk.delta.tool_calls:
                    yield chunk
                    continue

                text = chunk if isinstance(chunk, str) else (chunk.delta.content if chunk.delta else None)
                if not text:
                    yield chunk
                    continue

                to_emit = sanitizer.feed(text)
                if not to_emit:
                    continue
                if isinstance(chunk, str):
                    yield to_emit
                else:
                    yield chunk.model_copy(update={"delta": chunk.delta.model_copy(update={"content": to_emit})})

                if phrase_queue is not None:
                    phrase_buffer += to_emit
                    phrases, phrase_buffer = release_speech_phrases(
                        phrase_buffer, False,
                        min_phrase_chars=PHRASE_MIN_CHARS, max_phrase_chars=PHRASE_MAX_CHARS,
                    )
                    for phrase in phrases:
                        await phrase_queue.put(phrase)

            remaining = sanitizer.flush()
            if remaining:
                yield remaining
                if phrase_queue is not None:
                    phrase_buffer += remaining

            if phrase_queue is not None:
                phrases, _ = release_speech_phrases(
                    phrase_buffer, True,
                    min_phrase_chars=PHRASE_MIN_CHARS, max_phrase_chars=PHRASE_MAX_CHARS,
                )
                for phrase in phrases:
                    await phrase_queue.put(phrase)
                await phrase_queue.put(None)
                await consumer_task
            completed_normally = True
        finally:
            # Reached without completed_normally either because the
            # framework cancelled this generator (a user interruption) or
            # some other exception hit above -- stop feeding phrases that
            # will never be spoken and cut the avatar's audio short instead
            # of letting an in-flight phrase play out after the fact.
            if consumer_task is not None and not completed_normally:
                consumer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await consumer_task
                if self._avatar_handle is not None and self._avatar_handle.video_gen is not None:
                    self._avatar_handle.video_gen.clear_buffer()

    async def _consume_phrases(self, phrase_queue: "asyncio.Queue[str | None]") -> None:
        """Single consumer draining phrase_queue in order -- pipelining
        comes from queue depth (the LLM keeps producing phrases while an
        earlier one is still being synthesized/rendered), not from
        parallel phrases, so playback order is never at risk.
        """
        assert self._avatar_handle is not None
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as http:
            while True:
                phrase = await phrase_queue.get()
                if phrase is None:
                    return
                try:
                    media = await synthesize_phrase(
                        http,
                        kokoro_url=KOKORO_URL,
                        affect_url=AFFECT_URL,
                        phrase=phrase,
                        voice=self._voice,
                        speed=self._speed,
                        previous_reaction=self._previous_reaction,
                    )
                except aiohttp.ClientError as exc:
                    logger.warning("Kokoro synthesis failed for phrase %r, skipping: %s", phrase, exc)
                    continue

                self._previous_reaction = str(media.reaction.get("reaction", "neutral"))
                video_gen = await self._avatar_handle.wait_ready(timeout=15.0)
                if video_gen is None:
                    logger.warning("avatar not ready after 15s, dropping spoken phrase (text-only): %r", phrase)
                    continue

                pcm = _wav_to_pcm16_mono(media.wav_bytes)
                # NaturalMotionScheduler's blink/gaze/head timing is
                # independent of the classified *facial-expression* reaction
                # -- people blink constantly regardless of what they're
                # feeling -- so this must run, and its output must reach the
                # avatar, on every phrase, not just non-neutral ones. Merging
                # it into `reaction` (rather than a separate job field)
                # matches the resident's own contract: reaction_schedule.py's
                # blink_envelope/gaze_envelope/head_pose_envelope all read
                # blinks/gaze_events/head_events off the same "reaction" dict
                # muse_server.py receives.
                controls = media.reaction.get("controls") or {}
                motion_plan = self._natural_motion.schedule(media.duration, media.prosody, controls)
                reaction = dict(media.reaction)
                reaction["blinks"] = motion_plan["blinks"]
                reaction["gaze_events"] = motion_plan["gaze_events"]
                # See MUSE_AVATAR_SUPPORTS_HEAD_POSE's definition -- the
                # configured avatar's bank is too small for head-pose slots;
                # sending head_events anyway crashes the resident mid-job
                # (confirmed live), which is what produced the reported
                # "lips blocking/glitching" (the job fails partway through
                # and freezes on its last rendered frame). The browser's own
                # whole-canvas head fallback doesn't exist in this raw-
                # video-track model either, so there's no equivalent of the
                # old demo's own "if not ALP_ENABLED: head_events = []" path
                # to fall back to -- just omit them outright.
                reaction["head_events"] = motion_plan["head_events"] if MUSE_AVATAR_SUPPORTS_HEAD_POSE else []
                video_gen.set_next_reaction(reaction)
                await video_gen.push_audio(
                    rtc.AudioFrame(
                        data=pcm,
                        sample_rate=KOKORO_SAMPLE_RATE,
                        num_channels=1,
                        samples_per_channel=len(pcm) // 2,
                    )
                )
                await video_gen.push_audio(AudioSegmentEnd())

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        if self._avatar_handle is not None:
            # Speech synthesis for avatar mode is fully handled inside
            # llm_node's phrase pipeline above -- drain the text here
            # without re-synthesizing it, so Kokoro isn't called twice for
            # the same reply (once per phrase above, once per sentence via
            # the framework's own default tts_node if this didn't
            # short-circuit it).
            async for _ in text:
                pass
            return
            yield  # pragma: no cover - never reached; keeps this an async generator
        async for frame in Agent.default.tts_node(self, text, model_settings):
            yield frame


async def _maybe_react_to_visual_event(
    session: voice.AgentSession, agent: MuseTalkVoiceAgent, *, kind: str, detail: str | None = None
) -> None:
    """Proactively makes the agent speak in response to something purely
    visual -- a wave, a new object arriving -- with no user turn involved
    at all. Added 2026-09-10 per "if I wave at the camera it says hi...
    how can we keep it cheap but have an always watching effect, like an
    actual person," generalized per "it needs to work universally for any
    visual, within reason" to cover more than just waving -- see
    WaveGestureDetector and ObjectPresenceTracker.newly_confirmed labels
    for the two trigger sources this currently handles.

    Gated on session state so this can never step on an actual
    conversation: only fires when the agent isn't already speaking/
    thinking and the user isn't currently talking. `session.
    generate_reply()` (not a scripted `session.say()`) so the reaction is
    genuinely in-character and in-context -- it still sees the same chat
    history and system prompt, including VISION_CAPABILITY_CLAUSE's
    trust-calibration and first-person-only instructions, so a proactive
    reaction is held to the same "don't over-trust the small model, speak
    naturally" standard as any other camera-grounded reply.
    """
    now = time.monotonic()
    if (now - agent._last_proactive_reaction_at) < PROACTIVE_REACTION_COOLDOWN_S:
        return
    if session.agent_state not in ("idle", "listening") or session.user_state != "listening":
        return

    if kind == "wave":
        instructions = (
            "The person just waved at you through the camera, unprompted -- you weren't "
            "asked anything. Greet them warmly and briefly, in one short sentence, as if "
            "you just noticed them waving."
        )
    elif kind == "object":
        instructions = (
            f"You just noticed a {detail} appear in the camera view that wasn't there a "
            "moment ago -- nobody said anything about it. If it feels natural to make a "
            "brief, low-key comment, do; keep it to one short sentence and don't make a "
            "big deal of it."
        )
    else:
        return

    agent._last_proactive_reaction_at = now
    if kind == "wave":
        agent.wave_detector.reset()
    try:
        session.generate_reply(instructions=instructions, allow_interruptions=True)
    except Exception:
        logger.warning("failed to generate a proactive visual reaction", exc_info=True)


def _start_vision_video_subscription(
    room: rtc.Room, agent: MuseTalkVoiceAgent, vision_client: VisionClient, session: voice.AgentSession
) -> None:
    """Subscribe to the human participant's video track, if and when one
    gets published, and start a throttled background loop feeding frames
    to the `vision` sidecar (see vision_client.py). Explicitly skips
    AVATAR_IDENTITY's own video publish -- that's this agent's own
    rendered output (see avatar_bridge.py), not something to perceive.

    A caller may never publish video at all (voice-only is the common
    case); this is a no-op until/unless "track_subscribed" actually
    fires, which is the correct behavior, not a fallback path -- scene_state
    simply stays at its empty default and on_user_turn_completed already
    treats that as "no scene note to add."
    """

    def _on_track_subscribed(
        track: rtc.Track, _publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_VIDEO or participant.identity == AVATAR_IDENTITY:
            return
        logger.info("subscribing to video track from participant %s for live perception", participant.identity)
        asyncio.create_task(_consume_video_track(track, agent, vision_client, session))

    room.on("track_subscribed", _on_track_subscribed)


async def _consume_video_track(
    track: rtc.Track, agent: MuseTalkVoiceAgent, vision_client: VisionClient, session: voice.AgentSession
) -> None:
    # Two independent throttles, not one -- added 2026-09-10 after a live
    # report that a targeted visual question ("count my fingers") could
    # answer against a frame up to a full ambient-cadence-interval stale,
    # since it used to reuse whatever frame the last /perceive cycle
    # happened to grab. `buffer_throttle` is cheap (encode only, no model
    # call, no network round-trip) and runs much more often, just to keep
    # agent.last_frame_jpeg fresh for an on-demand question asked at any
    # moment; `perceive_throttle` gates the actual (model-call-backed)
    # ambient caption/changelog update, at its own slower cadence -- see
    # DEFAULT_FRAME_INTERVAL_S/BUFFER_FRAME_INTERVAL_S in vision_client.py.
    buffer_throttle = FrameThrottle(interval_s=BUFFER_FRAME_INTERVAL_S)
    perceive_throttle = FrameThrottle()
    # Guards against overlapping /gesture calls piling up if the sidecar
    # is ever slow -- a dropped tick just means one fewer sample in
    # WaveGestureDetector's window, never a backlog of stale requests.
    gesture_probe_inflight = False

    async def _probe_gesture(jpeg_bytes: bytes) -> None:
        nonlocal gesture_probe_inflight
        gesture_probe_inflight = True
        try:
            pose = await vision_client.observe_pose(jpeg_bytes)
            agent.wave_detector.observe(pose)
            if agent.wave_detector.detect_wave():
                await _maybe_react_to_visual_event(session, agent, kind="wave")
        finally:
            gesture_probe_inflight = False

    video_stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)
    try:
        async for event in video_stream:
            # Both throttles are checked every frame regardless of which
            # one(s) actually fire this iteration -- each call mutates
            # only its own internal timer, so checking one never
            # interferes with the other's cadence.
            should_buffer = buffer_throttle.should_accept()
            should_perceive = perceive_throttle.should_accept()
            if not (should_buffer or should_perceive):
                continue
            try:
                jpeg_bytes = frame_to_jpeg(event.frame)
            except Exception:
                logger.warning("failed to encode video frame for live perception", exc_info=True)
                continue
            # Written on every accepted tick (buffer- or perceive-driven)
            # -- an on-demand visual question needs a recent raw frame to
            # query against, independent of whether the ambient captioner
            # happens to be healthy this cycle.
            agent.last_frame_jpeg = jpeg_bytes
            agent.last_frame_at = time.monotonic()
            # Fired as a background task, not awaited inline -- a /gesture
            # round trip must never delay writing last_frame_jpeg above,
            # which is what keeps an on-demand visual question "instant"
            # regardless of anything else going on in this loop.
            if should_buffer and not gesture_probe_inflight:
                asyncio.create_task(_probe_gesture(jpeg_bytes))
            if should_perceive:
                scene = await vision_client.perceive(jpeg_bytes)
                if scene is not None:
                    agent.scene_state = scene
                    agent.scene_log.observe(scene.caption)
                    newly_confirmed = agent.object_presence.observe(scene.objects)
                    if newly_confirmed:
                        # Only the first -- if several objects were
                        # confirmed on the same tick, one reaction is
                        # plenty; _maybe_react_to_visual_event's own
                        # cooldown would suppress the rest anyway.
                        await _maybe_react_to_visual_event(
                            session, agent, kind="object", detail=newly_confirmed[0]
                        )
    finally:
        await video_stream.aclose()


async def _run_scene_narrator(agent: MuseTalkVoiceAgent, vision_client: VisionClient) -> None:
    """Background task, started alongside the video-frame loop: every
    NARRATE_INTERVAL_S, consolidates the raw per-tick changelog + YOLO
    object-presence timeline into one coherent narrative sentence via
    lfm2vl-narrate (LFM2.5-VL-3B, looking at the actual current frame --
    see VisionClient.narrate()'s docstring) and stores it on
    agent.scene_narrative. See "Is there anything we can do with yolo+the
    running log to boost temporal coherence" -- the raw changelog alone
    reads as a list of disconnected snapshots, not a storyline; this is
    the "something running in the background cleaning up the storyline"
    that request asked for.

    Runs for the lifetime of the room (cancelled via entrypoint's
    add_shutdown_callback, same pattern as every other background task
    here) -- there's no natural "done" state, narration should keep
    consolidating for as long as the conversation does.

    Sleeps first, before ever narrating -- there's nothing worth
    consolidating in the first NARRATE_INTERVAL_S of a session anyway
    (scene_log needs a few real entries to be worth summarizing), and
    starting with a sleep means a slow/stuck vision_client at session
    start can never block anything else here (this task doesn't gate
    on_user_turn_completed regardless, but no reason to race it either).
    """
    while True:
        await asyncio.sleep(NARRATE_INTERVAL_S)
        if agent.last_frame_jpeg is None:
            continue  # no camera frame yet (or ever) -- nothing to narrate against
        if agent.scene_log.is_empty():
            continue  # nothing new has been observed yet this cycle
        raw_log = agent.scene_log.render_for_narration()
        object_timeline = agent.object_presence.render_for_narration()
        narrative = await vision_client.narrate(agent.last_frame_jpeg, raw_log or "", object_timeline)
        if narrative:
            agent.scene_narrative.text = narrative
            agent.scene_narrative.updated_at = time.monotonic()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    session = AgentSession(
        # min_silence_duration default is 0.55s -- that whole trailing-silence
        # window is included in the audio clip sent to STT (VAD needs to see
        # it to decide the turn ended), and 500ms+ of pure silence is a
        # well-known, reliable trigger for Whisper-family models to
        # hallucinate a filler sign-off ("Thank you.", "Bye.") appended to
        # otherwise-correct real speech -- heavily trained on YouTube-style
        # audio where silence often precedes a sign-off. Reported live: real
        # user speech kept getting "thank you"/"bye" tacked on that was never
        # said. Shortening the tail directly reduces the trigger window;
        # traded against a slightly higher (but still reasonable) risk of
        # ending a turn on a longer mid-sentence pause.
        # min_speech_duration default is 0.05s (50ms) -- a cough, click, or
        # breath easily clears that and gets treated as a real utterance,
        # sent whole to STT. Whisper-family models are trained to always
        # produce *some* text, so a short non-speech burst with no real
        # phonetic content typically gets filled with a generic filler
        # ("mmm-hmm", "thank you") instead of correctly coming back empty --
        # a different failure mode than the trailing-silence hallucination
        # above (that one corrupts the tail of an otherwise-real utterance;
        # this one manufactures a whole fake utterance from noise). Reported
        # live: coughs/non-speech sounds kept transcribing as "mmm-hmm"/
        # "thank you". Raised to 0.2s -- clears real coughs and clicks
        # (typically 100-150ms) while still well under a real short word
        # ("no", "hi") at ~200ms+.
        vad=silero.VAD.load(min_silence_duration=0.3, min_speech_duration=0.2),
        # livekit-agents' default interruption handling is "adaptive": it
        # calls out to LiveKit Cloud's hosted inference
        # (agent-gateway.livekit.cloud) on every session, which contradicts
        # this project's tailnet-only/self-hosted posture and isn't
        # configured with any cloud credentials -- confirmed empirically: it
        # retries 3x, gets WSServerHandshakeError 401 each time, then falls
        # back. Forcing "vad" mode here (the plain `turn_detection="vad"`
        # kwarg is deprecated in this version) keeps real interruption
        # working with zero external dependency, just without the adaptive
        # semantic-turn-completion model's extra precision.
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            interruption={"enabled": True, "mode": "vad"},
            # Preemptive generation (on by default) starts a speculative LLM
            # reply as soon as end-of-turn is predicted, before
            # on_user_turn_completed runs -- a real win normally, since it
            # overlaps LLM generation with the tail of STT/endpointing. But
            # when vision is enabled, on_user_turn_completed injects a
            # "[Camera ...]" system message into the chat context on almost
            # every turn (scene_state is fresh at ~1Hz ambient cadence), and
            # the framework invalidates any preemptive generation whose
            # captured chat_ctx no longer matches -- see
            # agent_activity.py's "preemptive generation invalidated after
            # on_user_turn_completed" warning. Confirmed live 2026-09-10: a
            # real test session logged this invalidation on every single
            # turn (9/9), zero successful uses. With the hit rate at 0%,
            # leaving it enabled is strictly a cost, not a speedup -- a full
            # wasted LLM call every turn, competing for the same LLM
            # server's capacity as the real (non-preemptive) generation that
            # has to run anyway. Disabled only when vision is on, since a
            # voice-only session (no camera, nothing ever injected into the
            # chat context) gets the real benefit with no downside.
            preemptive_generation={"enabled": not VISION_ENABLED},
        ),
        stt=openai.STT(
            base_url=f"{WHISPER_STT_URL}/v1",
            api_key="not-required-on-tailnet",
            model="whisper-1",
            language="en",
        ),
        llm=openai.LLM(
            base_url=LLM_BASE_URL,
            api_key=os.environ.get("LLM_API_KEY") or "not-required-on-tailnet",
            model=LLM_MODEL,
        ),
        tts=KokoroTTS(base_url=KOKORO_URL, voice=DEFAULT_VOICE),
    )

    # Real per-turn latency breakdown, logged plainly -- added 2026-09-10
    # chasing "anything we can do to reduce the lag". Without this, the
    # only way to investigate a lag report was log archaeology (e.g.
    # counting "preemptive generation invalidated" lines by hand -- see
    # the comment on preemptive_generation above, found exactly that way).
    # ChatMessage.metrics (session/agent_session.py's replacement for the
    # deprecated metrics_collected event) already carries every number
    # that matters here; this just makes it visible without needing to
    # attach a debugger. Best-effort field access throughout (`.get(...,
    # -1)`) since MetricsReport is a total=False TypedDict -- a field is
    # legitimately absent for turns it doesn't apply to (e.g. no TTS
    # metrics on an interrupted reply with no audio yet).
    def _log_turn_metrics(event: voice.ConversationItemAddedEvent) -> None:
        item = event.item
        if not isinstance(item, llm.ChatMessage):
            return
        m = item.metrics
        if item.role == "user":
            logger.info(
                "turn latency (user): transcription=%.0fms end_of_turn=%.0fms "
                "on_user_turn_completed=%.0fms",
                m.get("transcription_delay", -1) * 1000,
                m.get("end_of_turn_delay", -1) * 1000,
                m.get("on_user_turn_completed_delay", -1) * 1000,
            )
        elif item.role == "assistant":
            logger.info(
                "turn latency (assistant): e2e=%.0fms llm_ttft=%.0fms tts_ttfb=%.0fms",
                m.get("e2e_latency", -1) * 1000,
                m.get("llm_node_ttft", -1) * 1000,
                m.get("tts_node_ttfb", -1) * 1000,
            )

    session.on("conversation_item_added", _log_turn_metrics)

    avatar_handle = AvatarHandle() if MUSE_AVATAR_ENABLED else None
    vision_client: VisionClient | None = None
    if VISION_ENABLED:
        # One session reused for this agent's whole lifetime (both the
        # video-frame loop's perceive() calls and every turn's
        # classify_transcript() call), matching phrase_pipeline.py's
        # session-reuse convention elsewhere in this file. Closed via
        # add_shutdown_callback rather than an `async with`, since it
        # needs to outlive this function's own scope.
        vision_http = aiohttp.ClientSession()
        ctx.add_shutdown_callback(vision_http.close)
        vision_client = VisionClient(VISION_URL, http=vision_http)
    # Constructed here (before session.start(), unlike previous versions
    # of this function) specifically so the video-track subscription below
    # can be wired up as early as possible -- ctx.room already exists
    # right after ctx.connect(), and a participant may publish video
    # before this agent's own session/turn machinery is ready. Agent's
    # __init__ has no dependency on an active session to construct safely
    # (confirmed: it only calls super().__init__(instructions=...)).
    agent = MuseTalkVoiceAgent(avatar_handle=avatar_handle, vision_client=vision_client)
    if vision_client is not None:
        _start_vision_video_subscription(ctx.room, agent, vision_client, session)
        # Independent of any particular video track -- started once per
        # session rather than per-track-subscribed, since it just reads
        # whatever agent.last_frame_jpeg/scene_log/object_presence happen
        # to hold at each tick (see _run_scene_narrator's docstring) and
        # is a no-op until a track actually shows up.
        narrator_task = asyncio.create_task(_run_scene_narrator(agent, vision_client))

        async def _stop_narrator() -> None:
            narrator_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await narrator_task

        ctx.add_shutdown_callback(_stop_narrator)

    async def _start_avatar() -> None:
        assert avatar_handle is not None
        try:
            portrait_host_path = MUSE_TALK_IO_DIR / MUSE_AVATAR_PORTRAIT_CONTAINER_PATH.removeprefix("/io/")
            avatar_room, _runner, video_gen = await start_avatar_worker(
                livekit_url=os.environ["LIVEKIT_URL"],
                api_key=os.environ["LIVEKIT_API_KEY"],
                api_secret=os.environ["LIVEKIT_API_SECRET"],
                room_name=ctx.room.name,
                io_root=MUSE_TALK_IO_DIR,
                portrait_container_path=MUSE_AVATAR_PORTRAIT_CONTAINER_PATH,
                portrait_host_path=portrait_host_path,
                avatar_id=MUSE_AVATAR_ID,
                fps=MUSE_AVATAR_FPS,
                video_container_path=MUSE_AVATAR_VIDEO_CONTAINER_PATH,
                bank_start_frame=MUSE_AVATAR_BANK_START_FRAME,
                source_start_frame=MUSE_AVATAR_SOURCE_START_FRAME,
            )

            async def _close_avatar_room() -> None:
                await avatar_room.disconnect()

            ctx.add_shutdown_callback(_close_avatar_room)
            avatar_handle.set_ready(video_gen)
        except Exception:
            logger.exception("avatar worker failed to start; replies will be text-only")

    # Kicked off as a background task rather than awaited here -- this is
    # the fix for real users reporting "nothing happens when I talk"
    # (confirmed empirically: two real sessions each disconnected after
    # 6-19s, neither ever reaching STT). The avatar worker's own separate
    # room join + muse-video-generator startup takes ~10s; the previous
    # code awaited it before ever calling session.start(), so the
    # AgentSession -- and therefore VAD/STT on the user's mic -- simply
    # didn't exist yet for the first ~10s after "Connected" appeared in the
    # browser. A user who starts talking right away, as any reasonable
    # person would, spoke into a pipeline that wasn't listening, then gave
    # up within seconds of getting no response. Nothing needs to join this
    # task afterward (unlike before this latency fix): avatar_handle above
    # is how the phrase pipeline finds out once it's ready, from inside
    # llm_node, not from anything awaited here in entrypoint.
    if avatar_handle is not None:
        asyncio.create_task(_start_avatar())

    room_output_options = (
        # sync_transcription=False: its default pacing ties published text
        # segments to room-level audio playback events, but in avatar mode
        # the spoken audio never publishes to the room at all -- it goes
        # straight from the phrase pipeline to the avatar worker's
        # MuseTalkVideoGenerator (see MuseTalkVoiceAgent._consume_phrases),
        # which publishes its own (lip-synced) track under a different
        # identity. Left at the default, transcript text for the agent's
        # replies never reaches the frontend at all (confirmed empirically:
        # audio generates and plays back fine, but no assistant text ever
        # appears in the UI transcript). Publishing immediately instead of
        # pacing to nonexistent playback events fixes this; it just means
        # the text can arrive slightly ahead of the spoken audio rather than
        # in lockstep with it.
        #
        # Gated on the *intent* to run in avatar mode (the env flag), not on
        # whether the avatar worker has actually finished starting -- it
        # hasn't, at this point, by design. The trade-off: if avatar startup
        # goes on to fail, there's no longer a graceful fallback to plain
        # room audio (audio_enabled is already False here); that's an
        # accepted, rare, clearly-logged edge case against the alternative
        # of blocking every session's mic input on avatar startup.
        RoomOutputOptions(audio_enabled=False, sync_transcription=False)
        if MUSE_AVATAR_ENABLED
        else RoomOutputOptions()
    )
    await session.start(room=ctx.room, agent=agent, room_output_options=room_output_options)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
