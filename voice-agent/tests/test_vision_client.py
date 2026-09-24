"""Unit tests for vision_client.py: the LiveKit agent's client for
musetalk-volta's `vision` sidecar (live perception of the human
participant's own camera, plus MiniCPM-V's text backbone as an
STT-artifact filter -- see vision_client.py's module docstring).

Uses the same tiny hand-written aiohttp.ClientSession fakes as
test_phrase_pipeline.py, for the same reason (no new dependency, the
ClientSession surface actually used here is small).
"""

from __future__ import annotations

import time

import aiohttp
import numpy as np
import pytest

from vision_client import (
    NEW_OBJECT_CONFIRM_S,
    FrameThrottle,
    ObjectPresenceTracker,
    SceneChangeLog,
    SceneNarrative,
    SceneState,
    VisionClient,
    WaveGestureDetector,
    frame_to_jpeg,
)


class _FakeResponse:
    def __init__(self, *, json_body=None, raise_exc=None):
        self._json_body = json_body
        self._raise_exc = raise_exc

    def raise_for_status(self) -> None:
        if self._raise_exc:
            raise self._raise_exc

    async def json(self) -> dict:
        return self._json_body

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.posted_urls: list[str] = []
        self.posted_kwargs: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        self.posted_urls.append(url)
        self.posted_kwargs.append(kwargs)
        return self._response


# -- SceneState -----------------------------------------------------------


def test_scene_state_default_is_stale():
    assert SceneState().is_stale(max_age_s=10.0)


def test_scene_state_fresh_is_not_stale():
    state = SceneState(caption="a person waves", updated_at=time.monotonic())
    assert not state.is_stale(max_age_s=10.0)


def test_scene_state_old_is_stale():
    state = SceneState(caption="a person waves", updated_at=time.monotonic() - 20.0)
    assert state.is_stale(max_age_s=10.0)


# -- VisionClient.perceive -------------------------------------------------


@pytest.mark.asyncio
async def test_perceive_builds_scene_state_from_response():
    response = _FakeResponse(
        json_body={
            "caption": "a person is dancing near a ship",
            "objects": [{"label": "person", "confidence": 0.9, "box": [0, 0, 1, 1]}],
            "face": {"leftEyeX": 0.4},
            "pose": None,
        }
    )
    session = _FakeSession(response)
    client = VisionClient("http://vision.local:18098", http=session)

    scene = await client.perceive(b"fake-jpeg-bytes")

    assert scene is not None
    assert scene.caption == "a person is dancing near a ship"
    assert scene.objects == [{"label": "person", "confidence": 0.9, "box": [0, 0, 1, 1]}]
    assert scene.face_present is True
    assert scene.pose_present is False
    assert scene.updated_at > 0
    assert session.posted_urls == ["http://vision.local:18098/perceive"]


@pytest.mark.asyncio
async def test_perceive_strips_trailing_slash_from_base_url():
    session = _FakeSession(_FakeResponse(json_body={"caption": None, "objects": [], "face": None, "pose": None}))
    client = VisionClient("http://vision.local:18098/", http=session)
    await client.perceive(b"x")
    assert session.posted_urls == ["http://vision.local:18098/perceive"]


@pytest.mark.asyncio
async def test_perceive_returns_none_on_failure_not_an_exception():
    session = _FakeSession(_FakeResponse(raise_exc=aiohttp.ClientError("vision sidecar down")))
    client = VisionClient("http://vision.local:18098", http=session)

    scene = await client.perceive(b"fake-jpeg-bytes")

    assert scene is None


@pytest.mark.asyncio
async def test_perceive_missing_optional_fields_default_safely():
    # A real /perceive response always includes all four keys, but this
    # must not crash if a future version of the sidecar omits one.
    session = _FakeSession(_FakeResponse(json_body={}))
    client = VisionClient("http://vision.local:18098", http=session)

    scene = await client.perceive(b"x")

    assert scene is not None
    assert scene.caption is None
    assert scene.objects == []
    assert scene.face_present is False
    assert scene.pose_present is False


# -- VisionClient.classify_transcript --------------------------------------


@pytest.mark.asyncio
async def test_classify_transcript_real_speech():
    session = _FakeSession(_FakeResponse(json_body={"is_real_speech": True}))
    client = VisionClient("http://vision.local:18098", http=session)

    assert await client.classify_transcript("What's the weather like today?") is True


@pytest.mark.asyncio
async def test_classify_transcript_artifact():
    session = _FakeSession(_FakeResponse(json_body={"is_real_speech": False}))
    client = VisionClient("http://vision.local:18098", http=session)

    assert await client.classify_transcript("mmm-hmm") is False


@pytest.mark.asyncio
async def test_classify_transcript_fails_open_on_error():
    # Must never silently eat real user speech just because the vision
    # sidecar is down/slow/still warming up -- see the module docstring.
    session = _FakeSession(_FakeResponse(raise_exc=aiohttp.ClientError("timeout")))
    client = VisionClient("http://vision.local:18098", http=session)

    assert await client.classify_transcript("anything") is True


@pytest.mark.asyncio
async def test_classify_transcript_fails_open_on_missing_field():
    session = _FakeSession(_FakeResponse(json_body={}))
    client = VisionClient("http://vision.local:18098", http=session)

    assert await client.classify_transcript("anything") is True


# -- VisionClient.observe_pose -----------------------------------------------


@pytest.mark.asyncio
async def test_observe_pose_returns_pose_metrics():
    session = _FakeSession(_FakeResponse(json_body={"pose": {"shoulderSpan": 0.3}}))
    client = VisionClient("http://vision.local:18098", http=session)

    pose = await client.observe_pose(b"fake-jpeg-bytes")

    assert pose == {"shoulderSpan": 0.3}
    assert session.posted_urls == ["http://vision.local:18098/gesture"]


@pytest.mark.asyncio
async def test_observe_pose_returns_none_when_no_pose_detected():
    session = _FakeSession(_FakeResponse(json_body={"pose": None}))
    client = VisionClient("http://vision.local:18098", http=session)

    assert await client.observe_pose(b"x") is None


@pytest.mark.asyncio
async def test_observe_pose_returns_none_on_failure():
    session = _FakeSession(_FakeResponse(raise_exc=aiohttp.ClientError("vision sidecar down")))
    client = VisionClient("http://vision.local:18098", http=session)

    assert await client.observe_pose(b"x") is None


# -- VisionClient.narrate ---------------------------------------------------


@pytest.mark.asyncio
async def test_narrate_posts_frame_and_log_and_returns_narrative():
    session = _FakeSession(_FakeResponse(json_body={"narrative": "The person is holding a mug and smiling."}))
    client = VisionClient("http://vision.local:18098", http=session)

    narrative = await client.narrate(b"fake-jpeg-bytes", "3s ago: a person waves", "person: present for the last 12s")

    assert narrative == "The person is holding a mug and smiling."
    assert session.posted_urls == ["http://vision.local:18098/narrate"]
    kwargs = session.posted_kwargs[0]
    # The frame goes as the raw JPEG body (matching /perceive and /query,
    # not a base64-in-JSON field) -- see the /narrate endpoint's Body(...)
    # parameter in vision/app.py.
    assert kwargs["data"] == b"fake-jpeg-bytes"
    assert kwargs["headers"] == {"Content-Type": "image/jpeg"}
    assert kwargs["params"] == {"raw_log": "3s ago: a person waves", "object_timeline": "person: present for the last 12s"}


@pytest.mark.asyncio
async def test_narrate_handles_missing_object_timeline():
    session = _FakeSession(_FakeResponse(json_body={"narrative": "Someone is at the desk."}))
    client = VisionClient("http://vision.local:18098", http=session)

    narrative = await client.narrate(b"x", "1s ago: someone sits down", None)

    assert narrative == "Someone is at the desk."
    assert session.posted_kwargs[0]["params"] == {"raw_log": "1s ago: someone sits down", "object_timeline": ""}


@pytest.mark.asyncio
async def test_narrate_returns_none_on_failure():
    session = _FakeSession(_FakeResponse(raise_exc=aiohttp.ClientError("lfm2vl-narrate down")))
    client = VisionClient("http://vision.local:18098", http=session)

    assert await client.narrate(b"x", "log", "timeline") is None


# -- SceneNarrative -----------------------------------------------------------


def test_scene_narrative_default_is_stale():
    assert SceneNarrative().is_stale(max_age_s=40.0)


def test_scene_narrative_fresh_is_not_stale():
    narrative = SceneNarrative(text="A person is at their desk.", updated_at=time.monotonic())
    assert not narrative.is_stale(max_age_s=40.0)


def test_scene_narrative_old_is_stale():
    narrative = SceneNarrative(text="A person is at their desk.", updated_at=time.monotonic() - 100.0)
    assert narrative.is_stale(max_age_s=40.0)


# -- ObjectPresenceTracker ----------------------------------------------------


def test_object_presence_starts_with_no_render():
    assert ObjectPresenceTracker().render_for_narration() is None


def test_object_presence_tracks_first_and_last_seen(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr("vision_client.time.monotonic", lambda: fake_now[0])
    tracker = ObjectPresenceTracker()

    tracker.observe([{"label": "person", "confidence": 0.9}])
    fake_now[0] += 12.0
    tracker.observe([{"label": "person", "confidence": 0.8}])

    rendered = tracker.render_for_narration()
    assert rendered == "person: present for the last 12s"


def test_object_presence_dedupes_repeated_labels_in_one_frame():
    tracker = ObjectPresenceTracker()
    tracker.observe([{"label": "person", "confidence": 0.9}, {"label": "person", "confidence": 0.5}])
    assert tracker.render_for_narration() == "person: present for the last 0s"


def test_object_presence_tracks_multiple_labels_independently(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr("vision_client.time.monotonic", lambda: fake_now[0])
    tracker = ObjectPresenceTracker()

    tracker.observe([{"label": "person", "confidence": 0.9}])
    fake_now[0] += 5.0
    tracker.observe([{"label": "person", "confidence": 0.9}, {"label": "bird", "confidence": 0.7}])

    rendered = tracker.render_for_narration()
    assert "person: present for the last 5s" in rendered
    assert "bird: present for the last 0s" in rendered


def test_object_presence_drops_label_after_grace_period_elapses(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr("vision_client.time.monotonic", lambda: fake_now[0])
    tracker = ObjectPresenceTracker(grace_s=5.0)

    tracker.observe([{"label": "bird", "confidence": 0.7}])
    fake_now[0] += 4.0
    tracker.observe([])  # bird missed this tick, but still within grace
    assert tracker.render_for_narration() == "bird: present for the last 4s"

    fake_now[0] += 6.0
    tracker.observe([])  # now well past the grace period
    assert tracker.render_for_narration() is None


def test_object_presence_ignores_entries_without_a_string_label():
    tracker = ObjectPresenceTracker()
    tracker.observe([{"confidence": 0.9}])  # malformed/missing label -- must not crash
    assert tracker.render_for_narration() is None


def test_object_presence_confirms_new_object_only_after_confirm_duration(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr("vision_client.time.monotonic", lambda: fake_now[0])
    tracker = ObjectPresenceTracker()

    newly = tracker.observe([{"label": "cup", "confidence": 0.8}])
    assert newly == []  # just arrived -- too recent to trust yet

    fake_now[0] += NEW_OBJECT_CONFIRM_S
    newly = tracker.observe([{"label": "cup", "confidence": 0.8}])
    assert newly == ["cup"]

    fake_now[0] += 1.0
    newly = tracker.observe([{"label": "cup", "confidence": 0.8}])
    assert newly == []  # already confirmed once -- doesn't refire every tick


def test_object_presence_brief_flicker_within_grace_does_not_reset_confirm_clock(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr("vision_client.time.monotonic", lambda: fake_now[0])
    tracker = ObjectPresenceTracker(grace_s=5.0)

    tracker.observe([{"label": "cup", "confidence": 0.8}])
    fake_now[0] += 0.5
    tracker.observe([])  # missed one tick, still within grace
    fake_now[0] += NEW_OBJECT_CONFIRM_S - 0.5
    newly = tracker.observe([{"label": "cup", "confidence": 0.8}])
    assert newly == ["cup"]  # confirm clock measured from first_seen_at, not reset by the gap


def test_object_presence_confirmed_flag_resets_if_object_leaves_and_returns(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr("vision_client.time.monotonic", lambda: fake_now[0])
    tracker = ObjectPresenceTracker(grace_s=1.0)

    tracker.observe([{"label": "cup", "confidence": 0.8}])
    fake_now[0] += NEW_OBJECT_CONFIRM_S
    assert tracker.observe([{"label": "cup", "confidence": 0.8}]) == ["cup"]

    fake_now[0] += 10.0  # well past grace_s -- entry pruned entirely
    tracker.observe([])
    assert tracker.observe([{"label": "cup", "confidence": 0.8}]) == []  # re-arrived -- fresh clock, too recent again
    fake_now[0] += NEW_OBJECT_CONFIRM_S
    newly = tracker.observe([{"label": "cup", "confidence": 0.8}])
    assert newly == ["cup"]  # a genuine new arrival after leaving -- fires again


# -- WaveGestureDetector -------------------------------------------------------


def _pose(*, shoulder_span=0.3, shoulder_y=0.5, left_x=0.5, left_y=0.2, left_vis=0.9, right_y=0.9, right_vis=0.9):
    return {
        "shoulderSpan": shoulder_span,
        "shoulderCenterY": shoulder_y,
        "leftWristX": left_x,
        "leftWristY": left_y,
        "leftWristVisibility": left_vis,
        "rightWristX": 0.5,
        "rightWristY": right_y,
        "rightWristVisibility": right_vis,
    }


def test_wave_detector_ignores_low_visibility_wrist(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr("vision_client.time.monotonic", lambda: fake_now[0])
    detector = WaveGestureDetector()
    for x in [0.3, 0.55, 0.3, 0.55, 0.3, 0.55, 0.3]:
        fake_now[0] += 0.1
        # Raised and oscillating in position, but MediaPipe isn't confident
        # about it (e.g. an occluded/off-frame wrist it's extrapolating) --
        # real values seen live: y=1.99, visibility=0.12.
        detector.observe(_pose(left_x=x, left_vis=0.12))
    assert not detector.detect_wave()


def test_wave_detector_no_pose_is_not_a_wave():
    detector = WaveGestureDetector()
    detector.observe(None)
    assert not detector.detect_wave()


def test_wave_detector_raised_but_still_is_not_a_wave(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr("vision_client.time.monotonic", lambda: fake_now[0])
    detector = WaveGestureDetector()
    for _ in range(8):
        fake_now[0] += 0.1
        detector.observe(_pose(left_x=0.5))  # raised, never moves side to side
    assert not detector.detect_wave()


def test_wave_detector_side_to_side_motion_is_a_wave(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr("vision_client.time.monotonic", lambda: fake_now[0])
    detector = WaveGestureDetector()
    for x in [0.3, 0.55, 0.3, 0.55, 0.3, 0.55, 0.3]:
        fake_now[0] += 0.1
        detector.observe(_pose(left_x=x))
    assert detector.detect_wave()


def test_wave_detector_jitter_below_threshold_does_not_count_as_motion(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr("vision_client.time.monotonic", lambda: fake_now[0])
    detector = WaveGestureDetector()
    # Movement well under WAVE_JITTER_FRACTION * shoulderSpan (0.03 * 0.3 = 0.009)
    for x in [0.500, 0.503, 0.499, 0.502, 0.498, 0.501, 0.499]:
        fake_now[0] += 0.1
        detector.observe(_pose(left_x=x))
    assert not detector.detect_wave()


def test_wave_detector_lowering_hand_resets_the_window(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr("vision_client.time.monotonic", lambda: fake_now[0])
    detector = WaveGestureDetector()
    for x in [0.3, 0.55, 0.3]:  # partway through a real wave
        fake_now[0] += 0.1
        detector.observe(_pose(left_x=x))
    fake_now[0] += 0.1
    detector.observe(_pose(left_y=0.6))  # hand lowered below shoulder -- not raised anymore
    assert not detector.detect_wave()
    # And the very next raise starts from an empty window, not a stitched-together one
    fake_now[0] += 0.1
    detector.observe(_pose(left_x=0.55))
    assert not detector.detect_wave()


def test_wave_detector_reset_clears_a_detected_wave(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr("vision_client.time.monotonic", lambda: fake_now[0])
    detector = WaveGestureDetector()
    for x in [0.3, 0.55, 0.3, 0.55, 0.3, 0.55, 0.3]:
        fake_now[0] += 0.1
        detector.observe(_pose(left_x=x))
    assert detector.detect_wave()

    detector.reset()
    assert not detector.detect_wave()


# -- SceneChangeLog.render_for_narration / is_empty --------------------------


def test_change_log_is_empty_when_nothing_observed():
    assert SceneChangeLog().is_empty()


def test_change_log_is_not_empty_after_an_observation():
    log = SceneChangeLog()
    log.observe("a person waves")
    assert not log.is_empty()


def test_change_log_render_for_narration_omits_camera_wrapper():
    log = SceneChangeLog()
    log.observe("a person waves")
    rendered = log.render_for_narration()
    assert rendered is not None
    assert "[Camera" not in rendered
    assert "a person waves" in rendered


def test_change_log_render_for_narration_none_when_empty():
    assert SceneChangeLog().render_for_narration() is None


# -- FrameThrottle ----------------------------------------------------------


def test_frame_throttle_accepts_first_frame():
    throttle = FrameThrottle(interval_s=2.0)
    assert throttle.should_accept() is True


def test_frame_throttle_rejects_within_window():
    throttle = FrameThrottle(interval_s=2.0)
    assert throttle.should_accept() is True
    assert throttle.should_accept() is False  # immediately after -- well inside the window


def test_frame_throttle_accepts_again_after_window(monkeypatch):
    fake_now = [100.0]
    monkeypatch.setattr("vision_client.time.monotonic", lambda: fake_now[0])

    throttle = FrameThrottle(interval_s=2.0)
    assert throttle.should_accept() is True

    fake_now[0] += 1.0
    assert throttle.should_accept() is False  # only 1s elapsed, window is 2s

    fake_now[0] += 1.5
    assert throttle.should_accept() is True  # now 2.5s elapsed


# -- frame_to_jpeg -----------------------------------------------------------


class _FakeVideoFrame:
    """Stands in for rtc.VideoFrame -- frame_to_jpeg only ever touches
    .data/.width/.height, so a real rtc.VideoFrame (which needs an actual
    LiveKit video buffer) isn't needed to exercise the real encoding path."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        # A simple horizontal gradient -- enough to confirm real pixel
        # data survives the RGB->BGR->JPEG round trip, not just that *some*
        # bytes come out.
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
        self.data = rgb.tobytes()


def test_frame_to_jpeg_produces_decodable_image():
    import cv2

    frame = _FakeVideoFrame(width=64, height=48)

    jpeg_bytes = frame_to_jpeg(frame)

    assert jpeg_bytes[:2] == b"\xff\xd8"  # JPEG magic bytes
    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (48, 64, 3)


def test_frame_to_jpeg_downscales_frames_above_the_max_edge():
    # Found live (2026-09-10): real webcam frames went out at native
    # camera resolution (confirmed 1280x720, unresized) straight to the
    # vision sidecar, costing far more tokens/latency than measured
    # against small test images -- see MAX_FRAME_EDGE_PX's comment.
    import cv2

    frame = _FakeVideoFrame(width=1280, height=720)

    jpeg_bytes = frame_to_jpeg(frame)

    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= 640
    # Aspect ratio preserved (16:9 in, 16:9 out).
    height, width = decoded.shape[:2]
    assert abs(width / height - 1280 / 720) < 0.01


def test_frame_to_jpeg_leaves_small_frames_unresized():
    import cv2

    frame = _FakeVideoFrame(width=320, height=180)

    jpeg_bytes = frame_to_jpeg(frame)

    decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (180, 320, 3)


# -- SceneChangeLog ---------------------------------------------------------


def test_change_log_starts_empty():
    assert SceneChangeLog().render() is None


def test_change_log_ignores_none_and_empty_caption():
    log = SceneChangeLog()
    log.observe(None)
    log.observe("")
    assert log.render() is None


def test_change_log_records_a_first_observation():
    log = SceneChangeLog()
    log.observe("a person is sitting at a desk")
    rendered = log.render()
    assert rendered is not None
    assert "a person is sitting at a desk" in rendered
    assert "0s ago" in rendered


def test_change_log_does_not_duplicate_consecutive_identical_captions():
    log = SceneChangeLog()
    log.observe("a person is sitting at a desk")
    log.observe("a person is sitting at a desk")
    log.observe("a person is sitting at a desk")
    rendered = log.render()
    assert rendered.count("a person is sitting at a desk") == 1


def test_change_log_records_each_distinct_change_in_order():
    log = SceneChangeLog()
    log.observe("a person is sitting at a desk")
    log.observe("a person is standing up")
    log.observe("a person is holding a red cup")
    rendered = log.render()
    # oldest first
    assert rendered.index("sitting at a desk") < rendered.index("standing up") < rendered.index("holding a red cup")


def test_change_log_reverts_to_repeating_an_earlier_caption_logs_a_new_entry():
    # Not deduped against anything but the immediately preceding entry --
    # a genuine "it changed back" is still a real change worth logging.
    log = SceneChangeLog()
    log.observe("a person is sitting at a desk")
    log.observe("a person is standing up")
    log.observe("a person is sitting at a desk")
    rendered = log.render()
    assert rendered.count("sitting at a desk") == 2


def test_change_log_prunes_beyond_max_entries():
    log = SceneChangeLog(max_entries=2)
    log.observe("state A")
    log.observe("state B")
    log.observe("state C")
    rendered = log.render()
    assert "state A" not in rendered
    assert "state B" in rendered
    assert "state C" in rendered


def test_change_log_prunes_beyond_max_age(monkeypatch):
    fake_now = [1000.0]
    monkeypatch.setattr("vision_client.time.monotonic", lambda: fake_now[0])

    log = SceneChangeLog(max_age_s=30.0)
    log.observe("old state")
    fake_now[0] += 40.0
    log.observe("new state")

    rendered = log.render()
    assert "old state" not in rendered
    assert "new state" in rendered
