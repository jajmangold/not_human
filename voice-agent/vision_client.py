"""Thin client for musetalk-volta's `vision` sidecar (the stack/ directory,
`vision/app.py`, see `docs/VISION.md`) -- live perception of the human
participant's own camera in the LiveKit room, plus MiniCPM-V's text
backbone used standalone to flag STT artifacts before they reach
gemma4-26b.

Distinct from every other perception path in this codebase: `feedback`/
`upper-body` (proxied by the `vision` service, not called from here
directly) analyze the AVATAR's own driving frames for MuseTalk rendering.
This is the first thing that looks at what the LiveKit agent's own camera
sees.

Every call here is best-effort: a scene caption or a transcript-artifact
verdict is a nice-to-have for the conversation, never something worth
blocking or breaking a turn over if the `vision` sidecar is down, slow, or
still warming up (it loads two GPU-resident models at startup -- see
docs/VISION.md's note on start_period). Callers get `None`/a safe default
on any failure rather than a raised exception, mirroring `_start_avatar`'s
defensive posture in agent.py.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import aiohttp
import cv2
import numpy as np
from livekit import rtc

logger = logging.getLogger("vision_client")

# Reverted to 1Hz (2026-09-10): pushed to 500ms same day based on a
# clean ~190ms /perceive measurement, but that measurement was taken
# against small (384x216) test images, not real webcam frames -- a live
# test at 500ms reported real lag and wrong answers, and the actual
# request logs showed why: real frames were going out at native camera
# resolution (confirmed 1280x720, unresized), costing ~788-806
# tokens/request and 370-450ms, not ~150-185ms. See MAX_FRAME_EDGE_PX
# below for the real fix (bound resolution before encoding, not just
# tune the cadence around whatever resolution happens to show up).
# Reverted the cadence itself back to the last known-good value while
# that fix gets verified against a real frame from the session that
# caught this, rather than assume the resize alone makes 500ms safe
# again without re-measuring.
DEFAULT_FRAME_INTERVAL_S = 1.0
# Separate, shorter interval for agent.py's raw-frame buffer (encode
# only, no model call, no network round-trip) -- this is what keeps an
# on-demand answer_visual_question() call "instant" regardless of the
# ambient caption's own cadence. Found live (2026-09-10): a targeted
# query previously reused whatever frame the last ambient /perceive
# cycle happened to grab, up to a full cadence-interval stale; decoupling
# frame *capture* from frame *analysis* means a question asked right now
# gets a frame that's at most ~250ms old, not up to a full cadence
# interval old.
BUFFER_FRAME_INTERVAL_S = 0.25
JPEG_QUALITY = 85
# Found live (2026-09-10): frames were being sent at the camera's native
# resolution (confirmed 1280x720, unresized) straight through to lfm2vl,
# whose SigLIP2-NaFlex vision encoder scales token count with input
# resolution -- real request logs showed ~788-806 tokens/request and
# 370-450ms, not the ~150-185ms measured earlier against small (384x216)
# test images. That's most of why 500ms cadence felt laggy and answered
# wrong (more raw detail than the model handles well at this size, not
# more useful signal) instead of just fast. Resizing down to a bounded
# longest edge before encoding brings real webcam frames back in line
# with what was actually measured and tuned against.
MAX_FRAME_EDGE_PX = 640

# Short, not the ~4s an actual /perceive call can take end-to-end -- these
# are per-attempt network timeouts against a sidecar that's expected to be
# resident and warmed up already; a slow response here means something is
# actually wrong, not just "the model is thinking."
PERCEIVE_TIMEOUT_S = 6.0
# Deliberately tighter than PERCEIVE_TIMEOUT_S: this one sits in the
# conversational hot path (on_user_turn_completed gates every turn on it),
# so a slow/stuck vision service should degrade to "treat as real speech,
# skip the filter" rather than add unbounded latency to every reply.
# ~500ms was the real single-request measurement (see docs/VISION.md); 2s
# gives real margin above that without risking a multi-second stall.
CLASSIFY_TIMEOUT_S = 2.0
# Also on the conversational hot path, but was sized against MiniCPM-V's
# ~939ms-3.1s single-request cost before the 2026-09-10 swap to `lfm2vl`
# (~150-185ms measured -- see docs/VISION.md). Left generous rather than
# tightened to the new number: this only fires when agent.py's
# visual-question heuristic matches, so the added latency is occasional,
# not on every turn, and a wide timeout costs nothing on the common case
# where the real call finishes in well under a second anyway.
QUESTION_TIMEOUT_S = 5.0
# NOT on the conversational hot path (runs from a slow background task,
# not gated on a turn -- see SceneNarrator in agent.py), so this can
# afford to be generous.
NARRATE_TIMEOUT_S = 8.0
# How often the background narrator consolidates the raw changelog into
# one coherent sentence -- slow relative to the ~1Hz ambient cadence on
# purpose: consolidating on every single tick would have nothing new to
# say most of the time (see SceneChangeLog's own dedup) and would just
# add GPU load for no benefit. Long enough to accumulate a few real
# changelog entries worth synthesizing, short enough to still feel
# current in conversation.
NARRATE_INTERVAL_S = 20.0
# Tight -- /gesture is CPU-only MediaPipe (no GPU, no LLM), meant to be
# called at the frame loop's faster buffer cadence (see
# BUFFER_FRAME_INTERVAL_S), so a slow response here means something is
# actually wrong, not "the model is thinking." A missed sample just means
# one fewer point in WaveGestureDetector's rolling window, not a broken
# feature -- fine to fail fast and move on to the next tick.
GESTURE_TIMEOUT_S = 1.0


@dataclass
class SceneNarrative:
    """The most recent background-consolidated narrative -- see
    SceneNarrator in agent.py. Separate from SceneState/SceneChangeLog:
    those are raw, per-tick, individually noisy observations; this is the
    (much slower-updating) synthesized storyline built from them."""

    text: str | None = None
    updated_at: float = 0.0

    def is_stale(self, max_age_s: float) -> bool:
        return self.updated_at == 0.0 or (time.monotonic() - self.updated_at) > max_age_s


@dataclass
class SceneState:
    """The most recent perception result, held on MuseTalkVoiceAgent and
    written by agent.py's video-frame loop. `caption` is what
    on_user_turn_completed injects into chat context -- see agent.py."""

    caption: str | None = None
    objects: list[dict[str, object]] = field(default_factory=list)
    face_present: bool = False
    pose_present: bool = False
    updated_at: float = 0.0

    def is_stale(self, max_age_s: float) -> bool:
        return self.updated_at == 0.0 or (time.monotonic() - self.updated_at) > max_age_s


@dataclass
class _ChangeLogEntry:
    at: float
    caption: str


class SceneChangeLog:
    """A rolling log of DISTINCT ambient captions over time, not just the
    latest snapshot -- added 2026-09-10 per request ("maybe once a second
    it updates a running visual changelog"). Gives the model a sense of
    what's happened recently ("12s ago: X; 3s ago: Y"), not just what's
    true right now, without re-sending every single caption (most
    consecutive captions at 1Hz are identical or near-identical for a
    mostly-static scene).

    Dedup is exact-string-match only, not semantic -- two captions that
    mean the same thing but are phrased slightly differently (temperature
    is 0/greedy, but genuine scene changes plus minor JPEG/lighting noise
    can still shift the exact wording) will both get logged as if the
    scene changed twice. Cheap and predictable; a real semantic-similarity
    dedup would be more accurate but is real added complexity/latency not
    justified yet -- revisit if the log turns out noisy in practice.
    """

    def __init__(self, max_age_s: float = 60.0, max_entries: int = 8) -> None:
        self._entries: list[_ChangeLogEntry] = []
        self._max_age_s = max_age_s
        self._max_entries = max_entries

    def observe(self, caption: str | None) -> None:
        if not caption:
            return
        if self._entries and self._entries[-1].caption == caption:
            return  # no change since last observation -- don't log a repeat
        now = time.monotonic()
        self._entries.append(_ChangeLogEntry(at=now, caption=caption))
        cutoff = now - self._max_age_s
        self._entries = [e for e in self._entries if e.at >= cutoff][-self._max_entries :]

    def render(self) -> str | None:
        """Returns a "[Camera -- recent changelog]" block listing each
        distinct entry with how long ago it was observed, oldest first, or
        None if nothing has been logged yet."""
        if not self._entries:
            return None
        now = time.monotonic()
        lines = [f"{now - e.at:.0f}s ago: {e.caption}" for e in self._entries]
        return "[Camera -- recent visual changelog, oldest first]\n" + "\n".join(lines)

    def render_for_narration(self) -> str | None:
        """Same content as render(), without the "[Camera" wrapper -- used
        as raw input to VisionClient.narrate() rather than injected
        directly into chat context. See SceneNarrator's docstring."""
        if not self._entries:
            return None
        now = time.monotonic()
        return "\n".join(f"{now - e.at:.0f}s ago: {e.caption}" for e in self._entries)

    def is_empty(self) -> bool:
        return not self._entries


@dataclass
class _PresenceEntry:
    first_seen_at: float
    last_seen_at: float
    # Set once this label has been continuously present for at least
    # NEW_OBJECT_CONFIRM_S -- see observe()'s return value. Prevents a
    # single-frame flicker (a borderline YOLO detection popping in and out)
    # from ever counting as "new," and ensures a label that stays in frame
    # only gets reported as newly-arrived once, not on every tick after.
    confirmed: bool = False


# How long a label must be continuously present (within ObjectPresenceTracker's
# own grace period) before it counts as a genuine arrival rather than a
# single-frame flicker -- see observe()'s docstring. Short enough to still
# feel prompt/"watching," long enough that a borderline single-frame YOLO
# detection (this fleet's spike doc already noted false positives on
# ambiguous single frames) can't trigger a proactive reaction on its own.
NEW_OBJECT_CONFIRM_S = 1.5


class ObjectPresenceTracker:
    """Tracks how long YOLO has continuously seen each object label,
    independent of the vision model's caption wording -- added 2026-09-10
    per request ("is there anything we can do with yolo+the running log
    to boost temporal coherence"). YOLO detections are per-frame and
    label-only (no identity tracking -- it can't tell "the same bird" from
    "a different bird"), but a label's PRESENCE over time is still a
    reliable, cheap signal the noisier caption text doesn't reconstruct on
    its own: "bird has been present for the last 40s" is something the
    caption model would need to independently re-notice and re-describe
    correctly every single tick to convey, and it doesn't always.

    An object is considered "still present" across brief gaps up to
    `grace_s` (a missed detection some ticks isn't necessarily the object
    leaving -- YOLO's own per-frame recall isn't perfect) -- only silently
    dropped from the summary once it hasn't been seen for longer than
    that.

    Also doubles as the GENERAL half of the proactive-reaction trigger
    (see WaveGestureDetector for the specific/high-precision half) --
    added 2026-09-10 per "it needs to work universally for any visual,
    within reason." A hand-coded detector per gesture/event type doesn't
    generalize; YOLO's ~80-class COCO vocabulary is already a broad,
    already-computed-for-free "what showed up" signal, so observe()
    doubles as that trigger's source rather than adding a second detector.
    Bounded by what YOLO26n can actually recognize -- "within reason," not
    universal scene understanding.
    """

    def __init__(self, grace_s: float = 5.0) -> None:
        self._entries: dict[str, _PresenceEntry] = {}
        self._grace_s = grace_s

    def observe(self, objects: list[dict[str, object]]) -> list[str]:
        """Returns labels that just crossed NEW_OBJECT_CONFIRM_S of
        continuous presence this call -- i.e. newly-arrived-and-confirmed,
        not "still here from before" and not "arrived but too recently to
        trust yet." Each label fires at most once per continuous presence
        (see _PresenceEntry.confirmed), not every tick after."""
        now = time.monotonic()
        seen_labels = {obj["label"] for obj in objects if isinstance(obj.get("label"), str)}
        for label in seen_labels:
            entry = self._entries.get(label)
            if entry is None:
                self._entries[label] = _PresenceEntry(first_seen_at=now, last_seen_at=now)
            else:
                entry.last_seen_at = now
        cutoff = now - self._grace_s
        self._entries = {label: e for label, e in self._entries.items() if e.last_seen_at >= cutoff}

        newly_confirmed = []
        for label, entry in self._entries.items():
            if not entry.confirmed and (now - entry.first_seen_at) >= NEW_OBJECT_CONFIRM_S:
                entry.confirmed = True
                newly_confirmed.append(label)
        return newly_confirmed

    def render_for_narration(self) -> str | None:
        if not self._entries:
            return None
        now = time.monotonic()
        lines = [f"{label}: present for the last {now - e.first_seen_at:.0f}s" for label, e in self._entries.items()]
        return "\n".join(lines)


# WaveGestureDetector tuning -- see its docstring for the overall design.
# Needs at least this many samples in the window before a reversal count
# means anything; below this, a real wave and random noise both look like
# "too little data," so just wait for more rather than guess either way.
WAVE_MIN_SAMPLES = 5
WAVE_WINDOW_S = 1.5
# A real wave is several full back-and-forth swings within the window;
# picked from "how many times would a hand crossing back and forth for
# ~1.5s actually reverse direction" (roughly 2 cycles => >=3 reversals),
# not from a specific measurement -- a live test may need to retune this.
WAVE_MIN_REVERSALS = 3
# Movement smaller than this fraction of shoulder width is jitter (pose-
# estimation noise, not real hand motion) and is ignored entirely rather
# than counted as a direction -- otherwise noise alone produces spurious
# reversals every tick.
WAVE_JITTER_FRACTION = 0.03
# A wrist must be at least this far above the shoulder (as a fraction of
# shoulder width, so it scales with how close the person is to the
# camera) to count as "raised" at all -- resting a hand near chest height
# or lower should never contribute to wave detection.
WAVE_RAISED_MARGIN_FRACTION = 0.05
# Below this, MediaPipe is extrapolating a best guess for an occluded/off-
# frame wrist rather than reporting something it actually sees clearly --
# confirmed live (2026-09-10): a real captured frame returned
# leftWristY=1.99 (a wrist estimated well outside the image entirely) at
# visibility 0.12. Feeding low-confidence guesses like that into the
# reversal buffer risks a spurious wave from tracking noise, not motion.
WAVE_MIN_WRIST_VISIBILITY = 0.5


def _count_reversals(samples: list[tuple[float, float]], jitter_threshold: float) -> int:
    if len(samples) < WAVE_MIN_SAMPLES:
        return 0
    xs = [x for _, x in samples]
    deltas = (xs[i + 1] - xs[i] for i in range(len(xs) - 1))
    signs = [1 if d > jitter_threshold else (-1 if d < -jitter_threshold else 0) for d in deltas]
    signs = [s for s in signs if s != 0]  # sub-jitter movement doesn't count as a direction at all
    return sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])


@dataclass
class _WristTrack:
    samples: list[tuple[float, float]] = field(default_factory=list)  # (monotonic time, x)


class WaveGestureDetector:
    """Pure-Python, no model calls -- detects an actual side-to-side wave
    from a rolling buffer of wrist positions, fed from VisionClient.
    observe_pose() (see that method's docstring for why this needs a
    faster, cheaper cadence than the ambient caption). Added 2026-09-10
    for "if I wave at the camera it says hi" -- the specific/high-
    precision half of the proactive-reaction trigger (see
    ObjectPresenceTracker for the general half, "any visual, within
    reason").

    Deliberately requires actual side-to-side motion, not just a raised
    hand -- a raised-hand-only check would false-positive constantly on
    ordinary gestures near the face (adjusting glasses, touching hair,
    resting a chin on a hand), which would make proactive reactions feel
    erratic rather than like a person noticing an actual wave. A hand is
    tracked only while it's raised (see observe()) -- lowering it resets
    the window, so a genuine wave has to be one continuous raised gesture,
    not a raised hand from a while ago stitched to an unrelated later one.
    """

    def __init__(self, window_s: float = WAVE_WINDOW_S) -> None:
        self._window_s = window_s
        self._left = _WristTrack()
        self._right = _WristTrack()
        self._last_shoulder_span: float | None = None

    def observe(self, pose: dict[str, object] | None) -> None:
        """Feed one /gesture result. No-ops on a missing/incomplete pose
        (nothing detected this tick) rather than raising -- a dropped
        sample just means a slightly smaller window, not a broken
        detector."""
        if not pose:
            return
        shoulder_span = pose.get("shoulderSpan")
        shoulder_y = pose.get("shoulderCenterY")
        if not isinstance(shoulder_span, (int, float)) or shoulder_span <= 0:
            return
        if not isinstance(shoulder_y, (int, float)):
            return
        self._last_shoulder_span = float(shoulder_span)
        now = time.monotonic()
        margin = WAVE_RAISED_MARGIN_FRACTION * shoulder_span
        self._update_track(
            self._left, pose.get("leftWristY"), pose.get("leftWristX"), pose.get("leftWristVisibility"), shoulder_y, margin, now
        )
        self._update_track(
            self._right,
            pose.get("rightWristY"),
            pose.get("rightWristX"),
            pose.get("rightWristVisibility"),
            shoulder_y,
            margin,
            now,
        )

    def _update_track(
        self,
        track: _WristTrack,
        wrist_y: object,
        wrist_x: object,
        wrist_visibility: object,
        shoulder_y: float,
        margin: float,
        now: float,
    ) -> None:
        # Image-space y increases downward, so "raised" means a smaller y
        # than the shoulder's. Also requires real confidence in the
        # landmark itself -- see WAVE_MIN_WRIST_VISIBILITY's comment.
        raised = (
            isinstance(wrist_y, (int, float))
            and isinstance(wrist_x, (int, float))
            and isinstance(wrist_visibility, (int, float))
            and wrist_visibility >= WAVE_MIN_WRIST_VISIBILITY
            and wrist_y < shoulder_y - margin
        )
        if not raised:
            track.samples = []  # hand down (or not confidently tracked) -- next raise starts a clean window
            return
        track.samples.append((now, float(wrist_x)))
        cutoff = now - self._window_s
        track.samples = [(t, x) for t, x in track.samples if t >= cutoff]

    def detect_wave(self) -> bool:
        """True if either wrist has swung side-to-side enough times,
        recently enough, to call it a real wave rather than noise or an
        incidentally-raised hand."""
        if self._last_shoulder_span is None:
            return False
        jitter = WAVE_JITTER_FRACTION * self._last_shoulder_span
        return (
            _count_reversals(self._left.samples, jitter) >= WAVE_MIN_REVERSALS
            or _count_reversals(self._right.samples, jitter) >= WAVE_MIN_REVERSALS
        )

    def reset(self) -> None:
        """Call after reacting to a detected wave, so the same continued
        motion (hand still mid-wave when the reaction fires) doesn't
        immediately re-trigger on the very next observe()."""
        self._left = _WristTrack()
        self._right = _WristTrack()


class VisionClient:
    """`http` is an injected aiohttp.ClientSession, owned and closed by the
    caller -- matches phrase_pipeline.py's existing convention in this
    codebase (one session reused across many calls, rather than a fresh
    connection pool stood up per request) and is what makes this testable
    with the same hand-written fakes test_phrase_pipeline.py already
    uses."""

    def __init__(self, base_url: str, http: aiohttp.ClientSession) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http

    async def perceive(self, jpeg_bytes: bytes) -> SceneState | None:
        """POST one JPEG frame to /perceive. Returns None on any failure
        -- callers should treat that as "no update this cycle," not an
        error to surface anywhere."""
        try:
            async with self._http.post(
                f"{self._base_url}/perceive",
                data=jpeg_bytes,
                headers={"Content-Type": "image/jpeg"},
                timeout=aiohttp.ClientTimeout(total=PERCEIVE_TIMEOUT_S),
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()
        except Exception:
            logger.warning("vision /perceive call failed", exc_info=True)
            return None

        return SceneState(
            caption=payload.get("caption"),
            objects=payload.get("objects") or [],
            face_present=payload.get("face") is not None,
            pose_present=payload.get("pose") is not None,
            updated_at=time.monotonic(),
        )

    async def classify_transcript(self, text: str) -> bool:
        """Returns True if `text` should be treated as real speech and
        sent on to the LLM. Fails OPEN (returns True, i.e. "treat as
        real") on any error or timeout -- a broken filter must never
        silently eat real user speech; the worst case of failing open is
        the pre-existing behavior (no filter at all), not a new failure
        mode."""
        try:
            async with self._http.post(
                f"{self._base_url}/classify_transcript",
                json={"text": text},
                timeout=aiohttp.ClientTimeout(total=CLASSIFY_TIMEOUT_S),
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()
                return bool(payload.get("is_real_speech", True))
        except Exception:
            logger.warning("vision /classify_transcript call failed; treating as real speech", exc_info=True)
            return True

    async def answer_visual_question(self, jpeg_bytes: bytes, question: str) -> str | None:
        """POST one JPEG frame + a specific question to /query -- an
        on-demand, targeted look, distinct from perceive()'s fixed ambient
        caption. Added 2026-09-10: the ambient caption alone could not
        answer "how many fingers am I holding up" (a live test asked
        exactly that and the agent denied having a camera at all -- see
        agent.py's VISUAL_QUESTION_KEYWORDS). Returns None on any
        failure/timeout; callers should skip injecting anything rather
        than inject a wrong or stale answer."""
        try:
            async with self._http.post(
                f"{self._base_url}/query",
                params={"question": question},
                data=jpeg_bytes,
                headers={"Content-Type": "image/jpeg"},
                timeout=aiohttp.ClientTimeout(total=QUESTION_TIMEOUT_S),
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()
                return payload.get("answer")
        except Exception:
            logger.warning("vision /query call failed", exc_info=True)
            return None

    async def narrate(self, jpeg_bytes: bytes, raw_log: str, object_timeline: str | None) -> str | None:
        """POST the CURRENT frame + raw changelog + object-presence
        timeline to /narrate for consolidation into one coherent sentence
        -- see SceneNarrator's docstring in agent.py for why this exists
        (the raw per-tick log is individually noisy and doesn't read as a
        storyline) and why the frame is included (lets the larger
        narration model verify/correct against what it actually sees,
        not just synthesize text from what a smaller model already said).
        Not on any conversational hot path -- this runs from a slow
        (~20s) background task, not gated on a turn -- so failure here
        just means the narrative goes stale for a cycle, never blocks or
        breaks anything. Returns None on any failure/timeout."""
        try:
            async with self._http.post(
                f"{self._base_url}/narrate",
                params={"raw_log": raw_log, "object_timeline": object_timeline or ""},
                data=jpeg_bytes,
                headers={"Content-Type": "image/jpeg"},
                timeout=aiohttp.ClientTimeout(total=NARRATE_TIMEOUT_S),
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()
                return payload.get("narrative")
        except Exception:
            logger.warning("vision /narrate call failed", exc_info=True)
            return None

    async def observe_pose(self, jpeg_bytes: bytes) -> dict[str, object] | None:
        """POST one JPEG frame to /gesture -- a cheap, CPU-only pose probe
        (see vision/app.py's /gesture docstring), distinct from perceive()
        which also runs YOLO + lfm2vl (GPU, real cost) on every call.
        Meant to be called much more often than perceive() -- see
        WaveGestureDetector, which needs several samples across ~1-1.5s to
        see a hand actually swing side-to-side, not just be raised once.
        Returns the raw pose metrics dict (or None on no pose detected /
        any failure) -- same shape as SceneState.pose_present's source,
        just with the actual landmark values instead of collapsed to a
        bool."""
        try:
            async with self._http.post(
                f"{self._base_url}/gesture",
                data=jpeg_bytes,
                headers={"Content-Type": "image/jpeg"},
                timeout=aiohttp.ClientTimeout(total=GESTURE_TIMEOUT_S),
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()
                return payload.get("pose")
        except Exception:
            logger.warning("vision /gesture call failed", exc_info=True)
            return None


class FrameThrottle:
    """Gate for "has enough time passed to process another frame" -- the
    cheapest correct throttle for a video track that may deliver frames
    much faster (15-30fps) than the ~0.5Hz this feature actually needs:
    just drop everything that arrives inside the window rather than queue
    or sample-and-hold. No relation to VAD/turn-taking; this only paces
    how often agent.py calls vision_client.perceive()."""

    def __init__(self, interval_s: float = DEFAULT_FRAME_INTERVAL_S) -> None:
        self._interval_s = interval_s
        self._last_accepted_at: float | None = None

    def should_accept(self) -> bool:
        now = time.monotonic()
        if self._last_accepted_at is not None and (now - self._last_accepted_at) < self._interval_s:
            return False
        self._last_accepted_at = now
        return True


def frame_to_jpeg(frame: rtc.VideoFrame) -> bytes:
    """Convert one RGB24 rtc.VideoFrame (as delivered by
    rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)) to JPEG
    bytes for vision_client/vision service calls, which all speak
    image/jpeg like every other sidecar in this codebase (feedback,
    upper-body). Downscaled to MAX_FRAME_EDGE_PX on the longer edge first
    -- see that constant's comment for why this matters beyond just
    saving bandwidth."""
    rgb = np.frombuffer(frame.data, dtype=np.uint8).reshape((frame.height, frame.width, 3))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    longer_edge = max(frame.width, frame.height)
    if longer_edge > MAX_FRAME_EDGE_PX:
        scale = MAX_FRAME_EDGE_PX / longer_edge
        new_size = (round(frame.width * scale), round(frame.height * scale))
        bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("failed to JPEG-encode video frame")
    return encoded.tobytes()
