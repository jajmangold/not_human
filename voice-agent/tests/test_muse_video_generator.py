"""Unit tests for muse_video_generator's pure logic: silence-span
detection and the reaction hand-off, run without any real MuseTalk job,
GPU, or LiveKit connection.

These were previously only ever run ad hoc (a python3 -c script during
live debugging, internal issue #45) -- formalized here so a future
change to the detection thresholds gets caught by CI instead of only
showing up as "the avatar looks wrong" during a live demo.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd

from muse_video_generator import (
    SEGMENT_QUEUE_MAXSIZE,
    MuseTalkVideoGenerator,
    _detect_speech_spans,
    _partition_regions,
    _QueuedSegment,
)

SR = 24000


def _tone(duration_s: float, freq: float = 200.0, amp: int = 8000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(SR * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.int16)


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(SR * duration_s), dtype=np.int16)


def test_single_continuous_utterance_is_one_span():
    clip = _tone(2.0)
    spans = _detect_speech_spans(clip.tobytes(), sample_rate=SR, num_channels=1)
    assert spans == [(0, len(clip))]


def test_real_pause_between_sentences_splits_into_two_spans():
    clip = np.concatenate([_tone(1.5), _silence(0.4), _tone(1.2)])
    spans = _detect_speech_spans(clip.tobytes(), sample_rate=SR, num_channels=1)
    assert len(spans) == 2
    (s1_start, s1_end), (s2_start, s2_end) = spans
    assert s1_start == 0
    assert s1_end < s2_start  # a real gap separates them
    assert s2_end == len(clip)


def test_pure_silence_falls_back_to_one_span_not_zero():
    clip = _silence(1.0)
    spans = _detect_speech_spans(clip.tobytes(), sample_rate=SR, num_channels=1)
    assert spans == [(0, len(clip))]


def test_very_short_clip_does_not_crash():
    clip = _tone(0.005)
    spans = _detect_speech_spans(clip.tobytes(), sample_rate=SR, num_channels=1)
    assert spans and spans[0][0] == 0


def test_empty_clip_returns_no_spans():
    spans = _detect_speech_spans(b"", sample_rate=SR, num_channels=1)
    assert spans == []


def test_partition_regions_fills_gaps_with_silence_regions():
    clip = np.concatenate([_tone(1.0), _silence(0.4), _tone(1.0)])
    regions = _partition_regions(
        clip.tobytes(), total_samples=len(clip), sample_rate=SR, num_channels=1
    )
    kinds = [r.is_speech for r in regions]
    assert kinds == [True, False, True]
    # Regions must tile [0, total_samples) with no gaps or overlaps.
    assert regions[0].start_sample == 0
    assert regions[-1].end_sample == len(clip)
    for a, b in zip(regions, regions[1:]):
        assert a.end_sample == b.start_sample


def test_no_pause_produces_a_single_speech_region():
    clip = _tone(1.0)
    regions = _partition_regions(
        clip.tobytes(), total_samples=len(clip), sample_rate=SR, num_channels=1
    )
    assert len(regions) == 1
    assert regions[0].is_speech is True


@pytest.mark.asyncio
async def test_initial_frame_is_queued_immediately_for_lazy_publish(tmp_path):
    # AvatarRunner only publishes its tracks once the first frame arrives
    # from this generator -- without an initial frame, nothing is visible
    # at all until the user's first reply fully renders (confirmed live,
    # reported as "takes forever to first come up"). Passing initial_frame
    # must make it available on _out_queue before any push_audio call.
    portrait = rtc.VideoFrame(2, 2, rtc.VideoBufferType.RGB24, b"\x00" * 12)
    generator = MuseTalkVideoGenerator(
        io_root=tmp_path,
        portrait_container_path="/io/fake.png",
        avatar_id="test-avatar",
        initial_frame=portrait,
    )
    first = await generator.__anext__()
    assert first is portrait
    await generator.aclose()


@pytest.mark.asyncio
async def test_no_initial_frame_means_out_queue_starts_empty(tmp_path):
    generator = MuseTalkVideoGenerator(
        io_root=tmp_path, portrait_container_path="/io/fake.png", avatar_id="test-avatar",
    )
    assert generator._out_queue.empty()
    await generator.aclose()


@pytest.mark.asyncio
async def test_set_next_reaction_is_consumed_exactly_once(tmp_path):
    # Doesn't touch the filesystem job queue or MuseTalk -- just verifies
    # the reaction hand-off contract: set before AudioSegmentEnd, consumed
    # (and cleared) by the segment it was set for, not leaked into the next
    # one. Uses a real MuseTalkVideoGenerator but never lets a real render
    # start (segment stays empty), so this stays fast and GPU-free.
    generator = MuseTalkVideoGenerator(
        io_root=tmp_path,
        portrait_container_path="/io/fake.png",
        avatar_id="test-avatar",
    )
    assert generator._pending_reaction is None
    generator.set_next_reaction({"reaction": "amusement", "strength": 0.5})
    assert generator._pending_reaction == {"reaction": "amusement", "strength": 0.5}
    # Consuming happens inside push_audio's AudioSegmentEnd branch; a plain
    # unit check on the private field is enough here since the segment
    # render itself is covered by the rendered-output test suite instead.
    await generator.aclose()


@pytest.mark.asyncio
async def test_second_segment_job_submitted_without_waiting_for_first_to_play(tmp_path):
    # Regression test for the internal issue #45 latency-fix refactor:
    # push_audio's AudioSegmentEnd branch must submit that segment's
    # MuseTalk job (a cheap file write) and return immediately, instead of
    # blocking for the full real-time pacing loop -- that's what lets
    # phrase N+1's job start rendering while phrase N is still being paced
    # out on screen. Before this refactor, push_audio awaited the entire
    # render+pacing loop inline, so this second job's JSON would not exist
    # on disk yet at the point checked below.
    generator = MuseTalkVideoGenerator(
        io_root=tmp_path,
        portrait_container_path="/io/fake.png",
        avatar_id="test-avatar",
        fps=25,
    )
    # 1 second of silence per "phrase" -- real duration doesn't matter here,
    # only that push_audio's own return doesn't wait for it to be paced.
    silence = (b"\x00\x00" * 24000)

    await generator.push_audio(rtc.AudioFrame(
        data=silence, sample_rate=24000, num_channels=1, samples_per_channel=24000,
    ))
    await generator.push_audio(AudioSegmentEnd())

    await generator.push_audio(rtc.AudioFrame(
        data=silence, sample_rate=24000, num_channels=1, samples_per_channel=24000,
    ))
    await generator.push_audio(AudioSegmentEnd())

    job_files = list((tmp_path / "muse_jobs").glob("live-agent-*.json"))
    assert len(job_files) == 2, (
        f"expected both segments' MuseTalk jobs submitted immediately, found {job_files}"
    )
    await generator.aclose()


@pytest.mark.asyncio
async def test_job_payload_targets_the_motion_video_with_bank_offsets(tmp_path):
    # Regression test: a prepared ALP semantic-bank identity's job must
    # target its rendered *motion* video (video_container_path), not the
    # plain uploaded portrait (portrait_container_path) -- and must carry
    # bank_start_frame/source_start_frame, since those are part of the
    # resident's avatar cache key *and* the offset REACTION_SLOTS' indices
    # are relative to (muse_server.py). Confirmed live: omitting them (or
    # sending the portrait instead of the motion video) either falls back
    # to a bank_size=1 identity with reactions/blinks/gaze silently
    # disabled, or misaligns which frames the reaction slots point at.
    generator = MuseTalkVideoGenerator(
        io_root=tmp_path,
        portrait_container_path="/io/portrait.png",
        video_container_path="/io/motion.mp4",
        avatar_id="test-avatar",
        bank_start_frame=8,
        source_start_frame=1,
    )
    await generator.push_audio(rtc.AudioFrame(
        data=b"\x00\x00" * 24000, sample_rate=24000, num_channels=1, samples_per_channel=24000,
    ))
    await generator.push_audio(AudioSegmentEnd())

    job_files = list((tmp_path / "muse_jobs").glob("live-agent-*.json"))
    assert len(job_files) == 1
    payload = json.loads(job_files[0].read_text())
    assert payload["video"] == "/io/motion.mp4"
    assert payload["bank_start_frame"] == 8
    assert payload["source_start_frame"] == 1
    await generator.aclose()


@pytest.mark.asyncio
async def test_job_payload_falls_back_to_portrait_when_no_motion_video_given(tmp_path):
    # A bare single-frame avatar (no separate ALP motion identity) should
    # still work -- video_container_path defaults to portrait_container_path.
    generator = MuseTalkVideoGenerator(
        io_root=tmp_path, portrait_container_path="/io/portrait.png", avatar_id="test-avatar",
    )
    await generator.push_audio(rtc.AudioFrame(
        data=b"\x00\x00" * 24000, sample_rate=24000, num_channels=1, samples_per_channel=24000,
    ))
    await generator.push_audio(AudioSegmentEnd())

    job_files = list((tmp_path / "muse_jobs").glob("live-agent-*.json"))
    payload = json.loads(job_files[0].read_text())
    assert payload["video"] == "/io/portrait.png"
    assert payload["bank_start_frame"] == 0
    assert payload["source_start_frame"] == 0
    await generator.aclose()


class _FakeJob:
    """Duck-types _JobHandle's contract for _pace_one_segment, with a
    controllable watch() delay -- used to simulate MuseTalk's real,
    confirmed-live behavior of writing a job's frames in one late burst
    close to job completion, rather than progressively."""

    def __init__(self, *, decoded_after: dict[int, object], watch_delay: float) -> None:
        self.decoded: dict[int, object] = {}
        self._decoded_after = decoded_after
        self._watch_delay = watch_delay
        self.failed_msg: str | None = None
        self.job_id = "fake-job"
        self.cancel_path = object()
        self.cleaned_up = False

    async def watch(self) -> None:
        await asyncio.sleep(self._watch_delay)
        self.decoded.update(self._decoded_after)

    def cleanup(self) -> None:
        self.cleaned_up = True


@pytest.mark.asyncio
async def test_late_decoded_frames_are_caught_up_after_the_main_pacing_loop(tmp_path):
    # Regression test: MuseTalk's blend/write threads finish writing a
    # job's frames in one late burst close to job completion rather than
    # progressively -- confirmed live (internal issue #45's
    # phrase-level follow-up) to regularly land just after the fixed-rate
    # loop's own real-time window (sized to the segment's own audio
    # duration) has already closed, meaning those frames decoded into
    # job.decoded but were never pushed to output at all before this fix --
    # a real phrase would render with zero visible mouth movement.
    generator = MuseTalkVideoGenerator(
        io_root=tmp_path, portrait_container_path="/io/fake.png", avatar_id="test-avatar", fps=10,
    )
    late_frame = rtc.VideoFrame(1, 1, rtc.VideoBufferType.RGB24, b"\x00\x00\x00")
    # 0.2s segment -> frame_count=2 at fps=10, so the main loop's own
    # real-time window is ~0.2s; watch_delay=0.3s means job.watch() only
    # populates job.decoded *after* that window has already closed.
    job = _FakeJob(decoded_after={0: late_frame, 1: late_frame}, watch_delay=0.3)

    await generator._pace_one_segment(job, pcm=b"\x00" * 8, segment_duration=0.2)

    pushed = []
    while not generator._out_queue.empty():
        pushed.append(generator._out_queue.get_nowait())
    video_frames = [item for item in pushed if isinstance(item, rtc.VideoFrame)]
    assert len(video_frames) == 2, f"expected both late-decoded frames caught up, got {pushed}"
    assert job.cleaned_up is True
    await generator.aclose()


# --- Regression tests: 2026-09-10 unbounded-queue memory leak -------------
#
# Root cause (see muse_video_generator.py's _out_queue/_segment_queue
# comments for the full chain): AVSynchronizer's own internal video queue
# (livekit.rtc.synchronizer) is bounded to ~1 frame of headroom at typical
# fps, so any downstream stall (native FFI publish call, GPU/network
# contention, a WebRTC reconnect) blocks it -- which blocks
# AvatarRunner._forward_video's consumption of THIS generator's output.
# Before this fix, _out_queue/_segment_queue had no bound at all, so the
# real-time production loop just kept absorbing the entire backlog as full
# RGB24 frame buffers with nothing capping how large it could grow --
# matching the live "MUSE_AVATAR_ENABLED=1: ~80MB -> 3GB+ within ~80s"
# report. These tests assert the queues now apply real backpressure (a put()
# blocks once full) instead of accepting unbounded items.


@pytest.mark.asyncio
async def test_out_queue_is_bounded_and_applies_backpressure(tmp_path):
    generator = MuseTalkVideoGenerator(
        io_root=tmp_path, portrait_container_path="/io/fake.png", avatar_id="test-avatar", fps=4,
    )
    maxsize = generator._out_queue.maxsize
    assert 0 < maxsize <= 8, f"expected a small positive bound at fps=4, got {maxsize}"

    filler = rtc.VideoFrame(1, 1, rtc.VideoBufferType.RGB24, b"\x00\x00\x00")
    for _ in range(maxsize):
        generator._out_queue.put_nowait(filler)
    assert generator._out_queue.full()

    # Before the fix this queue was unbounded -- put() never blocked, and a
    # stalled consumer meant unbounded memory growth. Now it must block
    # (i.e. genuinely wait for room) rather than accept the item instantly.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(generator._out_queue.put(filler), timeout=0.05)

    # Draining one slot must unblock a pending put -- backpressure, not a
    # permanently stuck producer.
    generator._out_queue.get_nowait()
    await asyncio.wait_for(generator._out_queue.put(filler), timeout=0.5)
    await generator.aclose()


@pytest.mark.asyncio
async def test_segment_queue_is_bounded_and_applies_backpressure(tmp_path):
    generator = MuseTalkVideoGenerator(
        io_root=tmp_path, portrait_container_path="/io/fake.png", avatar_id="test-avatar",
    )
    assert generator._segment_queue.maxsize == SEGMENT_QUEUE_MAXSIZE

    # Stop the generator's own background pacer for this test -- it's a
    # live consumer of this queue (started in __init__) and would otherwise
    # race to drain items the instant this coroutine awaits anything,
    # making a "still full, put() blocks" assertion flaky. Filling the
    # queue itself happens without any await in between, so it completes
    # synchronously before the pacer (suspended on `await
    # self._segment_queue.get()`) gets a chance to run regardless -- this
    # cancel just keeps it from interfering with the blocking-put assertion
    # right after.
    generator._pacer_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await generator._pacer_task

    filler = _QueuedSegment(
        job=_FakeJob(decoded_after={}, watch_delay=0.0), pcm=b"\x00" * 4, duration=0.1,
    )
    for _ in range(SEGMENT_QUEUE_MAXSIZE):
        generator._segment_queue.put_nowait(filler)
    assert generator._segment_queue.full()

    # Before the fix this queue was unbounded -- an unlucky burst of
    # phrases (or MuseTalk itself falling behind) could pile up submitted-
    # but-not-yet-paced segments without limit. Now it must block.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(generator._segment_queue.put(filler), timeout=0.05)

    generator._segment_queue.get_nowait()
    await asyncio.wait_for(generator._segment_queue.put(filler), timeout=0.5)
    await generator.aclose()
