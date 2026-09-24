"""Rendered-output verification for reaction -> MuseTalk job wiring
(internal issue #48).

Submits two MuseTalk jobs directly via the file-drop job protocol (same
avatar, same audio) -- one with an "amusement" reaction plan, one with
none -- and compares a frame from each via the already-running `feedback`
service's MediaPipe-backed `/observe` endpoint (musetalk-volta's own
landmark metrics API, no new dependency needed).

This exists to close a real verification gap: prior to this issue, a
reaction applied via the LiveKit voice agent's `/apply` endpoint was only
ever logged, never wired into a MuseTalk job payload at all -- confirmed
empirically, the avatar's visible expression never changed regardless of
what reaction was "applied", and the only way anyone would have noticed
was eyeballing a live screenshot. This test turns that into a fast (~15s),
repeatable, numeric assertion instead.

Requires the real musetalk-volta stack running (the resident container,
the `feedback` service, and a prepared ALP avatar bank) -- this is an
integration test, not a unit test, and is skipped if the stack isn't
reachable. Run manually with:

    docker compose ps                      # confirm musetalk-volta + feedback are up
    python -m pytest tests/test_reaction_render_verification.py -v

## Known gap (internal issue #49)

The measured effect here is real but small: raw source-bank frames for
this avatar show a ~0.030 smile-score difference between the quiet slot
and the amusement slot, but a full rendered job at strength=0.6 only
shows ~0.003 through the actual pipeline -- about 10x smaller than a
60%-strength blend of that raw delta should produce. The assertion below
is deliberately modest (amusement must simply score higher than neutral,
not by a specific large margin) so this test catches a *complete*
regression (reaction stops reaching the render at all, as it did before
this issue) without falsely failing on the *already-known*, separately
tracked "why is the effect this small" investigation.
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

IO_ROOT = Path(os.environ.get("MUSE_IO_ROOT", "./runtime"))
AVATAR_ID = "web-1e4285741243ef20d9f6-alp-semantic-bank-v9-kiss-settled-f24-3bba7329"
VIDEO_PATH = "/io/web_demo/motion/motion-1e4285741243ef20d9f6-semantic-bank-v9-kiss-f24-3bba7329.mp4"
BANK_START_FRAME = 8
FPS = 24
FEEDBACK_CONTAINER = "musetalk-volta-feedback-1"

AMUSEMENT_PLAN = {
    "reaction": "amusement",
    "strength": 0.6,
    "attack_ms": 200,
    "hold_ms": 800,
    "decay_ms": 700,
    "start_ratio": 0.0,
    "end_ratio": 0.18,
}


def _stack_is_up() -> bool:
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Health.Status}}", FEEDBACK_CONTAINER],
            capture_output=True, text=True, timeout=5,
        )
        return out.returncode == 0 and out.stdout.strip() == "healthy"
    except (OSError, subprocess.TimeoutExpired):
        return False


def _make_test_wav(path: Path, duration_s: float = 2.5) -> None:
    import math
    import struct
    import wave

    sr = 24000
    samples = [
        int(6000 * math.sin(2 * math.pi * 180 * i / sr) * (1 if (i // sr) % 2 == 0 else 0.3))
        for i in range(int(sr * duration_s))
    ]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _submit_job(reaction: dict | None, tag: str, wav_path: Path) -> tuple[str, Path]:
    segment_id = f"reactiontest-{tag}-{uuid.uuid4().hex[:8]}"
    job_id = f"live-{segment_id}"
    audio_host = IO_ROOT / "agent_video" / "audio" / f"{segment_id}.wav"
    stream_host = IO_ROOT / "agent_video" / "stream" / segment_id
    audio_host.parent.mkdir(parents=True, exist_ok=True)
    stream_host.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(wav_path, audio_host)

    payload = {
        "avatar_id": AVATAR_ID,
        "video": VIDEO_PATH,
        "audio": f"/io/agent_video/audio/{segment_id}.wav",
        "stream_dir": f"/io/agent_video/stream/{segment_id}",
        "bbox_shift": 0,
        "source_start_frame": 0,
        "bank_start_frame": BANK_START_FRAME,
        "fps": FPS,
        "stream_format": "jpg",
        "cancel_path": f"/io/muse_jobs/{job_id}.cancel",
    }
    if reaction:
        payload["reaction"] = reaction

    jobs_root = IO_ROOT / "muse_jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    (jobs_root / f"{job_id}.json").write_text(json.dumps(payload))
    return job_id, stream_host


def _wait_for_job(job_id: str, timeout_s: float = 60.0) -> None:
    jobs_root = IO_ROOT / "muse_jobs"
    done, err = jobs_root / f"{job_id}.done", jobs_root / f"{job_id}.err"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if err.is_file():
            raise RuntimeError(f"job {job_id} failed: {err.read_text()[-1000:]}")
        if done.is_file():
            return
        time.sleep(0.1)
    raise TimeoutError(f"job {job_id} timed out after {timeout_s}s")


def _pick_frame(stream_dir: Path, target_time_s: float, fps: int) -> Path:
    index = round(target_time_s * fps)
    candidates = sorted(stream_dir.glob("*.jpg"))
    assert candidates, f"no frames rendered in {stream_dir}"
    return min(candidates, key=lambda p: abs(int(p.stem) - index))


def _observe(frame_path: Path) -> dict:
    # The feedback service only `expose`s its port to the compose network,
    # not the host -- copy the frame in and run the HTTP call from inside
    # the container rather than adding a new host port mapping just for
    # this test.
    container_path = f"/tmp/{uuid.uuid4().hex}.jpg"
    subprocess.run(
        ["docker", "cp", str(frame_path), f"{FEEDBACK_CONTAINER}:{container_path}"],
        check=True, capture_output=True,
    )
    script = (
        "import urllib.request, json\n"
        f"with open({container_path!r}, 'rb') as f: data = f.read()\n"
        "req = urllib.request.Request('http://127.0.0.1:8095/observe', data=data, "
        "headers={'Content-Type': 'image/jpeg'}, method='POST')\n"
        "with urllib.request.urlopen(req, timeout=10) as resp: body = json.load(resp)\n"
        "print(json.dumps(body['metrics']))\n"
    )
    result = subprocess.run(
        ["docker", "exec", FEEDBACK_CONTAINER, "python3", "-c", script],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, f"feedback /observe failed: {result.stderr}"
    return json.loads(result.stdout.strip())


@pytest.mark.skipif(not _stack_is_up(), reason="musetalk-volta stack not running locally")
def test_amusement_reaction_measurably_smiles_more_than_no_reaction(tmp_path):
    wav_path = tmp_path / "test.wav"
    _make_test_wav(wav_path)

    amusement_job, amusement_dir = _submit_job(AMUSEMENT_PLAN, "amusement", wav_path)
    neutral_job, neutral_dir = _submit_job(None, "neutral", wav_path)
    _wait_for_job(amusement_job)
    _wait_for_job(neutral_job)

    # Hold window is attack(0.2s)..attack+hold(1.0s) at these plan values --
    # sample its middle.
    amusement_frame = _pick_frame(amusement_dir, target_time_s=0.6, fps=FPS)
    neutral_frame = _pick_frame(neutral_dir, target_time_s=0.6, fps=FPS)

    amusement_metrics = _observe(amusement_frame)
    neutral_metrics = _observe(neutral_frame)
    assert amusement_metrics is not None, "no face detected in amusement frame"
    assert neutral_metrics is not None, "no face detected in neutral frame"

    # Deliberately modest threshold -- see the "Known gap" note in this
    # file's module docstring (internal issue #49). This must catch a
    # *complete* regression (reaction never reaching the render at all,
    # the bug this test exists to prevent), not assert a specific large
    # effect size that isn't produced by the pipeline yet.
    delta = amusement_metrics["smile"] - neutral_metrics["smile"]
    assert delta > 0.001, (
        f"amusement reaction did not measurably increase smile score "
        f"(amusement={amusement_metrics['smile']!r}, neutral={neutral_metrics['smile']!r}, "
        f"delta={delta!r}); see internal issue #49 for the known "
        f"'effect is smaller than expected' investigation"
    )
