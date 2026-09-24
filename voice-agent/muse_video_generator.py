"""VideoGenerator bridging Kokoro-synthesized speech to MuseTalk lip-sync.

Implements livekit.agents.voice.avatar.VideoGenerator: buffers the PCM audio
pushed to it, submits it as MuseTalk streaming job(s) (the same file-drop job
protocol scripts/submit_stream_chunk.py uses -- musetalk-volta's resident
MuseTalk has no HTTP API, only a shared-directory job queue), and yields the
resulting lip-synced JPEG frames interleaved with the original audio so
AvatarRunner's AVSynchronizer can publish both in sync.

Two real problems surfaced testing this live (internal issue #45,
reported as "speed is all over the place"):

1. MuseTalk's GPU generation throughput (measured 14.6-24.2 fps across real
   segments -- the GPU is shared with kokoro/ALP/other musetalk-volta
   services) doesn't reliably match any fixed fps we ask for, and swings
   noticeably job to job. Pushing a frame to the room only when a real one
   exists meant playback rate directly tracked that raw, jittery generation
   rate. Fixed with a fixed-clock pacer (see `_pace_one_segment`) that holds
   (repeats) the last decoded frame when the next real one isn't ready --
   the same principle video calls use for network jitter (freeze rather
   than stutter) -- so output pacing is decoupled from generation jitter.

2. The whole reply was submitted as one MuseTalk job, which meant a long,
   multi-sentence reply had zero slack: MuseTalk had to sustain the full
   requested fps for the entire reply's duration with no breathing room,
   accumulating lip-sync lag over a long reply. Real speech has natural
   pauses between sentences/phrases; Kokoro's audio isn't truly streamed to
   us token-by-token (KokoroTTS.synthesize does one blocking HTTP call and
   returns a complete WAV), so by the time push_audio's AudioSegmentEnd
   fires, 100% of the reply's audio already exists. That means every pause
   is known lead time up front: `_detect_speech_spans` finds the real
   silences and each speech-active span becomes its own MuseTalk job,
   submitted immediately (not gated on real-time playback reaching that
   point) so MuseTalk's own serial job queue renders span N+1 the instant
   span N finishes, using span N's own render time plus the pause before
   N+1 is actually due on screen as free lead time.

Since internal issue #45's latency fix, agent.py no longer waits for a
full LLM reply before calling push_audio at all -- it streams one
AudioSegmentEnd-terminated call per breath-sized phrase as the LLM
generates (see phrase_pipeline.py). To get real overlap out of that (phrase
N+1's Kokoro/classify call and MuseTalk job both starting while phrase N is
still being paced out on screen -- the property that makes musetalk-volta's
own `/ws/live/` demo feel instant), job *submission* happens immediately in
push_audio's AudioSegmentEnd branch (cheap: just writes files), while the
real-time frame-pacing loop for each segment runs in a single persistent
background task (`_pace_segments`) that drains a FIFO queue one segment at
a time -- so segments never race each other for the output queue, but their
MuseTalk jobs queue up and start rendering back-to-back rather than one at
a time gated on real-time playback.

V1 scope: a single fixed portrait/avatar_id, no ALP-rendered idle motion or
reaction-driven background (that's musetalk-volta's `motion_path` selection
system -- a real, separately-tuned piece of production logic, not
reimplemented here). This gets a correctly lip-synced avatar on a static
portrait; expression/gaze/head motion is a follow-up.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, VideoGenerator

logger = logging.getLogger("muse-video-generator")

JOB_TIMEOUT_S = float(os.environ.get("MUSE_JOB_TIMEOUT_S", "30"))
POLL_INTERVAL_S = 0.03

# Bounds for _out_queue/_segment_queue -- see their definitions in __init__
# for the root-cause analysis this addresses (the unbounded-queue memory
# leak reported live with MUSE_AVATAR_ENABLED=1: ~80MB -> 3GB+ within ~80s).
OUT_QUEUE_MAX_SECONDS = float(os.environ.get("MUSE_OUT_QUEUE_MAX_SECONDS", "1.0"))
SEGMENT_QUEUE_MAXSIZE = int(os.environ.get("MUSE_SEGMENT_QUEUE_MAXSIZE", "4"))

# Silence-span detection tuning (see _detect_speech_spans). Chosen for
# natural sentence/phrase pauses in TTS speech, not fine phoneme gaps --
# splitting too aggressively just means more, smaller MuseTalk jobs with
# more per-job fixed overhead for no real pipelining benefit.
SILENCE_WINDOW_MS = 20.0
MIN_SILENCE_MS = 180.0
MIN_SPAN_MS = 250.0
SPAN_PAD_MS = 60.0
SILENCE_RMS_RATIO = 0.08


@dataclass(frozen=True)
class _Region:
    start_sample: int
    end_sample: int
    is_speech: bool


@dataclass(frozen=True)
class _QueuedSegment:
    """One phrase's already-submitted MuseTalk job, waiting its turn in
    `_pace_segments`'s FIFO. `pcm`/`duration` are kept alongside the job
    (rather than re-derived from it) since the job itself only knows about
    decoded frames, not the source audio driving frame-count/pacing math."""

    job: "_JobHandle"
    pcm: bytes
    duration: float


class MuseTalkVideoGenerator(VideoGenerator):
    def __init__(
        self,
        *,
        io_root: Path,
        portrait_container_path: str,
        avatar_id: str,
        fps: int = 25,
        sample_rate: int = 24000,
        num_channels: int = 1,
        initial_frame: rtc.VideoFrame | None = None,
        video_container_path: str | None = None,
        bank_start_frame: int = 0,
        source_start_frame: int = 0,
    ) -> None:
        self._io_root = io_root
        self._portrait_container_path = portrait_container_path
        # A prepared ALP semantic-bank identity's actual MuseTalk job
        # points at its rendered *motion* video (many frames: quiet slots,
        # blink slot, 8 reaction slots, head-pose slots -- see
        # reaction_schedule.py's REACTION_SLOTS), not the plain uploaded
        # portrait image `portrait_container_path` still refers to for room
        # video-track sizing and the initial idle frame. Falls back to
        # `portrait_container_path` for a bare single-frame avatar that has
        # no separate motion identity at all.
        self._video_container_path = video_container_path or portrait_container_path
        # bank_start_frame/source_start_frame are part of the resident's
        # avatar cache key (muse_server.py) *and* determine which frames
        # REACTION_SLOTS' indices actually land on within the bank -- they
        # must match whatever the identity was prepared with (confirmed:
        # this avatar's bank was prepared with bank_start_frame=8) or the
        # resident either re-prepares under the same avatar_id with the
        # wrong offset (silently misaligning every reaction slot) or, if
        # different jobs disagree, keeps re-preparing from scratch. Left at
        # 0/0 (MuseTalk's own defaults) for a bare portrait with no bank.
        self._bank_start_frame = bank_start_frame
        self._source_start_frame = source_start_frame
        self._avatar_id = avatar_id
        self._fps = fps
        self._sample_rate = sample_rate
        self._num_channels = num_channels

        self._pending_pcm = bytearray()
        self._pending_reaction: dict | None = None
        # Bounded -- root-caused 2026-09-10 to the live "MUSE_AVATAR_ENABLED=1:
        # ~80MB -> 3GB+ within ~80s" memory leak. Traced the consumer side:
        # AvatarRunner._forward_video (livekit.agents.voice.avatar._runner)
        # does `async for frame in self._video_gen: ... await
        # self._av_sync.push(frame)`, and AVSynchronizer.push (livekit.rtc.
        # synchronizer) puts video frames on its OWN internal queue, bounded
        # to `video_queue_size_ms=100` -- at MUSE_AVATAR_FPS=12 that's ~1
        # frame of headroom. Any downstream stall (the native FFI publish
        # call in VideoSource.capture_frame, GPU/network contention, a
        # WebRTC reconnect -- all independently observed live) blocks that
        # tiny queue, which blocks AVSynchronizer.push, which blocks
        # _forward_video's `async for`, which stops draining THIS queue.
        # Before this fix, this queue had no bound at all, so
        # _pace_one_segment's real-time production loop (which never blocks
        # on an unbounded Queue.put) just kept absorbing the entire backlog
        # as full RGB24 frame buffers -- multi-MB each at real portrait
        # resolution -- explaining the observed growth rate. Bounding it
        # means a downstream stall now applies real backpressure (this
        # generator's own put() blocks) instead of unbounded memory growth;
        # the existing "already behind, don't compound debt" catch-up logic
        # in _pace_one_segment already tolerates the resulting pacing slip.
        self._out_queue: asyncio.Queue[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd] = asyncio.Queue(
            maxsize=max(4, round(self._fps * OUT_QUEUE_MAX_SECONDS))
        )
        if initial_frame is not None:
            # AvatarRunner only publishes its tracks once the *first* frame
            # arrives from this generator (`_lazy_publish`) -- without this,
            # that first frame is whatever MuseTalk renders for the user's
            # first reply, i.e. nothing is visible at all until well after
            # STT+LLM+Kokoro+a MuseTalk job all complete (confirmed live,
            # reported as "takes forever to first come up": the ~10s avatar
            # room-join was never the whole story). Publishing the static
            # portrait immediately means the avatar's face is visible within
            # a couple seconds of connecting, before the user even finishes
            # their first sentence.
            self._out_queue.put_nowait(initial_frame)
        # Segments are queued here as soon as their MuseTalk job is
        # submitted (push_audio's AudioSegmentEnd branch); `_pace_segments`
        # drains them one at a time so real-time output pacing stays
        # strictly ordered even though job *submission* runs ahead of it.
        # `None` is the empty-segment sentinel (a flush with no audio).
        # Bounded (defense in depth alongside _out_queue above, and the
        # explicit backpressure gap this queue specifically was flagged as
        # missing): each queued-but-not-yet-active segment holds its full
        # phrase's raw PCM bytes in memory on top of an already-submitted,
        # already-running MuseTalk job -- if phrases keep arriving faster
        # than MuseTalk can render them (the documented "no safety margin"
        # capacity issue), an unbounded queue here lets that backlog grow
        # without limit instead of throttling job submission itself. Matches
        # phrase_queue's own bound (agent.py, maxsize=4) so this doesn't
        # become the new bottleneck instead.
        self._segment_queue: asyncio.Queue[_QueuedSegment | None] = asyncio.Queue(maxsize=SEGMENT_QUEUE_MAXSIZE)
        self._active_cancel_paths: list[Path] = []
        self._closed = False

        (self._io_root / "agent_video" / "audio").mkdir(parents=True, exist_ok=True)
        (self._io_root / "agent_video" / "stream").mkdir(parents=True, exist_ok=True)
        (self._io_root / "muse_jobs").mkdir(parents=True, exist_ok=True)

        self._pacer_task = asyncio.create_task(self._pace_segments())

    def set_next_reaction(self, reaction: dict | None) -> None:
        """Attach a reaction plan (reaction_plan.reaction_plan's output) to
        the *next* segment this generator renders. Call before the
        AudioSegmentEnd that closes that segment.

        Previously nothing ever called this (no equivalent existed), so
        set_reaction/classify results never reached MuseTalk at all --
        confirmed empirically: the avatar's visible expression never
        changed regardless of what reaction was applied. Wiring this in is
        what actually makes a reaction visible, matching how
        musetalk-volta's own web demo's job payload already does it.
        """
        self._pending_reaction = reaction

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            pcm = bytes(self._pending_pcm)
            self._pending_pcm.clear()
            reaction = self._pending_reaction
            self._pending_reaction = None
            if not pcm:
                # Nothing was actually spoken (e.g. a flush with no audio) --
                # still signal segment end so AvatarRunner's playback
                # bookkeeping doesn't stall.
                await self._segment_queue.put(None)
                return
            # Submit the MuseTalk job *now* -- this is what lets phrase N+1
            # start rendering while phrase N is still being paced out below
            # by _pace_segments (see module docstring). Submission itself is
            # cheap (just writes files); this call does not block on it.
            segment_id = f"agent-{uuid.uuid4().hex[:12]}"
            job = self._submit_job(
                segment_id=segment_id,
                pcm_slice=pcm,
                jobs_root=self._io_root / "muse_jobs",
                audio_root=self._io_root / "agent_video" / "audio",
                stream_root=self._io_root / "agent_video" / "stream",
                reaction=reaction,
            )
            self._active_cancel_paths.append(job.cancel_path)
            bytes_per_sample = 2 * self._num_channels
            duration = (len(pcm) // bytes_per_sample) / self._sample_rate
            await self._segment_queue.put(_QueuedSegment(job=job, pcm=pcm, duration=duration))
            return
        self._pending_pcm.extend(bytes(frame.data))

    def clear_buffer(self) -> None:
        self._pending_pcm.clear()
        for cancel_path in self._active_cancel_paths:
            try:
                cancel_path.write_bytes(b"cancel\n")
            except OSError:
                pass
        self._active_cancel_paths.clear()
        # Drop any segments that were already submitted (queued for pacing)
        # but haven't started playing yet -- an interruption should silence
        # them too, not just whatever is currently on screen.
        while not self._segment_queue.empty():
            try:
                queued = self._segment_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if queued is not None:
                queued.job.cleanup()
        while not self._out_queue.empty():
            try:
                self._out_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _pace_segments(self) -> None:
        """Persistent background task: drain _segment_queue one segment at a
        time, real-time-pacing each to _out_queue. Runs for the generator's
        whole lifetime so segment N+1's job (already submitted by push_audio
        by the time this gets to it) starts rendering immediately rather
        than only once N+1 becomes the "current" segment.
        """
        while True:
            queued = await self._segment_queue.get()
            if queued is None:
                await self._out_queue.put(AudioSegmentEnd())
                continue
            try:
                await self._pace_one_segment(queued.job, queued.pcm, queued.duration)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("segment pacing failed for job %s", queued.job.job_id)

    async def _pace_one_segment(self, job: "_JobHandle", pcm: bytes, segment_duration: float) -> None:
        bytes_per_sample = 2 * self._num_channels
        samples_per_frame = int(round(self._sample_rate / self._fps))
        bytes_per_frame = samples_per_frame * bytes_per_sample
        frame_interval = 1.0 / self._fps

        frame_count = max(1, round(segment_duration * self._fps))
        last_frame: rtc.VideoFrame | None = None
        last_shown_index = -1
        emitted_real = 0
        decoded_total = 0
        next_tick = time.monotonic()
        watch_task = asyncio.create_task(job.watch())

        try:
            for i in range(frame_count):
                sample_index = int(i * self._sample_rate / self._fps)
                start = sample_index * bytes_per_sample
                chunk = pcm[start : start + bytes_per_frame]
                if chunk:
                    if len(chunk) < bytes_per_frame:
                        chunk = chunk + b"\x00" * (bytes_per_frame - len(chunk))
                    await self._out_queue.put(
                        rtc.AudioFrame(
                            data=chunk,
                            sample_rate=self._sample_rate,
                            num_channels=self._num_channels,
                            samples_per_channel=samples_per_frame,
                        )
                    )

                # pop(), not get(): once a decoded frame has been shown it's
                # never looked up again (each index is consumed at most once
                # here or in the late-catch-up loop below) -- popping drops
                # this dict's own reference immediately instead of holding
                # every decoded frame's buffer for the job's whole lifetime.
                # Confirmed via tracemalloc against a real room-connected
                # session: job.decoded held onto ~305KB per real frame
                # (rgb.tobytes() in _decode_jpg_to_video_frame) for far
                # longer than needed, contributing real MB-scale growth per
                # segment -- part of what caused the memory-pressure/GC-
                # stall pattern seen live (a ~1GB burst around avatar
                # startup, intermittent frozen frames, one mid-call WebRTC
                # reconnect).
                frame = job.decoded.pop(i, None)
                if frame is not None:
                    decoded_total += 1
                    last_frame = frame
                    last_shown_index = i
                    emitted_real += 1
                if last_frame is not None:
                    # Held (repeated) when `frame` was None -- last_frame
                    # covers both the real and held case identically.
                    await self._out_queue.put(last_frame)

                next_tick += frame_interval
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                else:
                    # Already behind -- don't compound the debt, resume
                    # pacing from now rather than racing to catch up.
                    next_tick = time.monotonic()

            await watch_task
            if job.failed_msg is not None:
                logger.warning(
                    "MuseTalk job %s ended early (%s), %d/%d real frames",
                    job.job_id, job.failed_msg, decoded_total + len(job.decoded), frame_count,
                )

            # MuseTalk's blend/write threads finish writing a job's frames in
            # one late burst close to job completion rather than
            # progressively (confirmed live: total job time, e.g. "MUSE JOB
            # DONE ... 2.6s", consistently runs a bit longer than the
            # segment's own audio duration the fixed-rate loop above paces
            # against -- so that burst regularly lands just *after* the loop
            # already finished, per index, and would otherwise be decoded
            # into job.decoded but never actually pushed to output). At
            # phrase-level granularity (this refactor) that's not a rare
            # edge case anymore -- it was hitting nearly every phrase in
            # live testing, i.e. real lip-sync frames decoding correctly but
            # never being shown at all, only the held very-first frame for
            # the whole segment. Catch up on them now instead of discarding
            # them: late-but-correct mouth shapes (audio's already finished
            # playing by this point) beat a frozen avatar for the segment's
            # entire duration.
            for i in range(last_shown_index + 1, frame_count):
                frame = job.decoded.pop(i, None)
                if frame is None:
                    continue
                decoded_total += 1
                last_frame = frame
                last_shown_index = i
                emitted_real += 1
                await self._out_queue.put(last_frame)
        finally:
            await self._out_queue.put(AudioSegmentEnd())
            job.cleanup()
            try:
                self._active_cancel_paths.remove(job.cancel_path)
            except ValueError:
                pass  # already removed by clear_buffer (interruption raced us here)
            held = frame_count - emitted_real
            logger.info(
                "segment %s: %.2fs audio -> %d/%d real frames (%d held/repeated)",
                job.job_id, segment_duration, emitted_real, frame_count, held,
            )

    def _submit_job(
        self,
        *,
        segment_id: str,
        pcm_slice: bytes,
        jobs_root: Path,
        audio_root: Path,
        stream_root: Path,
        reaction: dict | None = None,
    ) -> "_JobHandle":
        audio_path = audio_root / f"{segment_id}.wav"
        stream_dir = stream_root / segment_id
        stream_dir.mkdir(parents=True, exist_ok=True)
        with wave.open(str(audio_path), "wb") as handle:
            handle.setnchannels(self._num_channels)
            handle.setsampwidth(2)
            handle.setframerate(self._sample_rate)
            handle.writeframes(pcm_slice)

        job_id = f"live-{segment_id}"
        cancel_path = jobs_root / f"{job_id}.cancel"
        job_payload = {
            "avatar_id": self._avatar_id,
            "video": self._video_container_path,
            "audio": f"/io/agent_video/audio/{segment_id}.wav",
            "stream_dir": f"/io/agent_video/stream/{segment_id}",
            "bbox_shift": 0,
            "source_start_frame": self._source_start_frame,
            "bank_start_frame": self._bank_start_frame,
            "fps": self._fps,
            "stream_format": "jpg",
            "cancel_path": f"/io/muse_jobs/{job_id}.cancel",
        }
        if reaction:
            job_payload["reaction"] = reaction
        _write_atomic(jobs_root / f"{job_id}.json", json.dumps(job_payload).encode())

        return _JobHandle(
            job_id=job_id,
            job_json=jobs_root / f"{job_id}.json",
            done_marker=jobs_root / f"{job_id}.done",
            err_marker=jobs_root / f"{job_id}.err",
            cancel_path=cancel_path,
            stream_dir=stream_dir,
            audio_path=audio_path,
        )

    def __aiter__(self):
        return self

    async def __anext__(self) -> rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd:
        if self._closed and self._out_queue.empty():
            raise StopAsyncIteration
        return await self._out_queue.get()

    async def aclose(self) -> None:
        self._closed = True
        self._pacer_task.cancel()
        try:
            await self._pacer_task
        except (asyncio.CancelledError, Exception):
            pass


@dataclass
class _JobHandle:
    job_id: str
    job_json: Path
    done_marker: Path
    err_marker: Path
    cancel_path: Path
    stream_dir: Path
    audio_path: Path

    def __post_init__(self) -> None:
        self.decoded: dict[int, rtc.VideoFrame] = {}
        self.failed_msg: str | None = None
        self._already_read: set[str] = set()

    async def _decode_new_frames(self) -> None:
        for frame_path in sorted(self.stream_dir.glob("*.jpg")):
            if frame_path.name in self._already_read:
                continue
            self._already_read.add(frame_path.name)
            index = int(frame_path.stem)
            data = await asyncio.to_thread(frame_path.read_bytes)
            video_frame = await asyncio.to_thread(_decode_jpg_to_video_frame, data)
            if video_frame is not None:
                self.decoded[index] = video_frame

    async def watch(self) -> None:
        """Poll for this job's frames/completion until done, failed, or timed out."""
        start = time.monotonic()
        poll_count = 0
        deadline = start + JOB_TIMEOUT_S
        while time.monotonic() < deadline:
            poll_count += 1
            if self.err_marker.is_file():
                self.failed_msg = self.err_marker.read_text(errors="replace")[-500:]
                return
            before = len(self.decoded)
            await self._decode_new_frames()
            if len(self.decoded) != before:
                # DEBUG, not INFO: confirmed (2026-09-07) that frames from
                # the resident's blend-worker threads land in one late burst
                # near job completion rather than progressively -- Python
                # GIL contention between the main GPU loop and the 10
                # blend/write threads in muse_server.py's process_frames,
                # not a bug in this poller. Left in for the next person
                # investigating that resident-side bottleneck.
                logger.debug(
                    "job %s: +%d frames at t=%.2fs (poll #%d, total=%d)",
                    self.job_id, len(self.decoded) - before, time.monotonic() - start,
                    poll_count, len(self.decoded),
                )
            if self.done_marker.is_file():
                for _ in range(2):
                    await asyncio.sleep(0.02)
                    await self._decode_new_frames()
                logger.debug(
                    "job %s: done at t=%.2fs, %d polls, %d frames decoded total",
                    self.job_id, time.monotonic() - start, poll_count, len(self.decoded),
                )
                return
            await asyncio.sleep(POLL_INTERVAL_S)
        self.failed_msg = f"timed out after {JOB_TIMEOUT_S:.1f}s"

    def cleanup(self) -> None:
        for marker in (self.job_json, self.done_marker, self.err_marker, self.cancel_path):
            marker.unlink(missing_ok=True)
        _cleanup_dir(self.stream_dir)
        self.audio_path.unlink(missing_ok=True)


def _partition_regions(
    pcm: bytes, *, total_samples: int, sample_rate: int, num_channels: int
) -> list[_Region]:
    """Split [0, total_samples) into alternating speech/silence regions.

    Silence regions get no MuseTalk job at all -- they're natural pauses,
    paced through by just holding whatever frame speech last showed. See
    _detect_speech_spans for the detection itself.
    """
    spans = _detect_speech_spans(pcm, sample_rate=sample_rate, num_channels=num_channels)
    regions: list[_Region] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            regions.append(_Region(cursor, start, is_speech=False))
        regions.append(_Region(start, end, is_speech=True))
        cursor = end
    if cursor < total_samples:
        regions.append(_Region(cursor, total_samples, is_speech=False))
    return regions


def _detect_speech_spans(
    pcm: bytes,
    *,
    sample_rate: int,
    num_channels: int,
    min_silence_ms: float = MIN_SILENCE_MS,
    min_span_ms: float = MIN_SPAN_MS,
    pad_ms: float = SPAN_PAD_MS,
    window_ms: float = SILENCE_WINDOW_MS,
    silence_rms_ratio: float = SILENCE_RMS_RATIO,
) -> list[tuple[int, int]]:
    """Return (start_sample, end_sample) spans of speech-active audio.

    Splits on genuine pauses (>= min_silence_ms of low-energy audio relative
    to this segment's own peak level) rather than fixed-size chunks, so
    MuseTalk jobs land on natural sentence/phrase boundaries instead of
    mid-word. Falls back to one span covering the whole segment if the
    audio is silent, too short to window, or has no qualifying pause.
    """
    samples = np.frombuffer(pcm, dtype=np.int16)
    if num_channels > 1:
        samples = samples.reshape(-1, num_channels).mean(axis=1)
    total_samples = len(samples)
    if total_samples == 0:
        return []

    window = max(1, int(sample_rate * window_ms / 1000))
    n_windows = total_samples // window
    if n_windows < 2:
        return [(0, total_samples)]

    trimmed = samples[: n_windows * window].astype(np.float64)
    rms = np.sqrt(np.mean(trimmed.reshape(n_windows, window) ** 2, axis=1))
    peak = float(rms.max())
    if peak <= 0:
        return [(0, total_samples)]
    active = rms > (peak * silence_rms_ratio)

    min_silence_windows = max(1, int(min_silence_ms / window_ms))
    min_span_samples = int(sample_rate * min_span_ms / 1000)
    pad_samples = int(sample_rate * pad_ms / 1000)

    raw_spans: list[tuple[int, int]] = []
    span_start: int | None = None
    silence_run = 0
    for i, is_active in enumerate(active):
        if is_active:
            if span_start is None:
                span_start = i
            silence_run = 0
        elif span_start is not None:
            silence_run += 1
            if silence_run >= min_silence_windows:
                span_end_window = i - silence_run + 1
                start_sample = max(0, span_start * window - pad_samples)
                end_sample = min(total_samples, span_end_window * window + pad_samples)
                if end_sample - start_sample >= min_span_samples:
                    raw_spans.append((start_sample, end_sample))
                span_start = None
                silence_run = 0
    if span_start is not None:
        start_sample = max(0, span_start * window - pad_samples)
        raw_spans.append((start_sample, total_samples))

    if not raw_spans:
        return [(0, total_samples)]

    merged: list[tuple[int, int]] = [raw_spans[0]]
    for start, end in raw_spans[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _decode_jpg_to_video_frame(data: bytes) -> rtc.VideoFrame | None:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return rtc.VideoFrame(width, height, rtc.VideoBufferType.RGB24, rgb.tobytes())


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _cleanup_dir(path: Path) -> None:
    try:
        for child in path.iterdir():
            child.unlink(missing_ok=True)
        path.rmdir()
    except OSError:
        pass
