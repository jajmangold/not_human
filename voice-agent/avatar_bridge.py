"""Wires the MuseTalk VideoGenerator to a second room participant.

The main AgentSession publishes no audio of its own -- a second "avatar"
participant drives lip-sync and publishes the synchronized audio+video back
to the room. Both participants live in this same process/event loop (a
second rtc.Room connection with its own identity/token), avoiding a second
deployed process for a v1.

Earlier (pre internal issue #45 latency fix) this used
livekit.agents.voice.avatar's documented DataStreamAudioOutput/
DataStreamAudioReceiver pattern to carry TTS audio from the main session to
this avatar participant over the room's data channel. That's gone now:
agent.py's phrase pipeline (MuseTalkVoiceAgent._consume_phrases) calls
video_gen.push_audio()/set_next_reaction() directly, in-process, since both
participants already share this process -- one whole-turn round trip
through the framework's audio-output abstraction was itself part of what
made the old path feel slow, and per-phrase streaming needs finer-grained
control (one AudioSegmentEnd per phrase, immediate reactions) than that
abstraction exposes. AvatarRunner still requires *an* AudioReceiver to
satisfy its interface; _NullAudioReceiver below is a permanently-empty
stand-in since nothing sends it frames anymore.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator

from livekit import api, rtc
from livekit.agents.voice.avatar import AudioReceiver, AudioSegmentEnd, AvatarOptions, AvatarRunner
from PIL import Image

from muse_video_generator import MuseTalkVideoGenerator

logger = logging.getLogger("avatar-bridge")

AVATAR_IDENTITY = "musetalk-avatar"


class _NullAudioReceiver(AudioReceiver):
    """Satisfies AvatarRunner's AudioReceiver contract without ever
    producing a frame -- see module docstring for why nothing feeds this."""

    def notify_playback_finished(self, playback_position: float, interrupted: bool) -> None:
        return None

    def notify_playback_started(self) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame | AudioSegmentEnd]:
        return self

    async def __anext__(self) -> rtc.AudioFrame | AudioSegmentEnd:
        await asyncio.Event().wait()  # never resolves: this receiver has no source
        raise AssertionError("unreachable")


async def start_avatar_worker(
    *,
    livekit_url: str,
    api_key: str,
    api_secret: str,
    room_name: str,
    io_root: Path,
    portrait_container_path: str,
    portrait_host_path: Path,
    avatar_id: str,
    fps: int = 25,
    sample_rate: int = 24000,
    num_channels: int = 1,
    video_container_path: str | None = None,
    bank_start_frame: int = 0,
    source_start_frame: int = 0,
) -> tuple[rtc.Room, AvatarRunner, MuseTalkVideoGenerator]:
    """Join ``room_name`` as a second participant and start lip-sync video.

    Returns (room, runner, video_gen). The video generator is returned
    directly so a caller in the same process -- agent.py's phrase pipeline
    -- can call set_next_reaction()/push_audio() on it straight from the
    LLM-text side (see module docstring).
    """
    with Image.open(portrait_host_path) as image:
        width, height = image.size
        # PIL's "RGB" mode is already R,G,B byte order per pixel, matching
        # rtc.VideoBufferType.RGB24 directly -- no BGR swap needed (unlike
        # muse_video_generator's own cv2-decoded frames).
        initial_frame = rtc.VideoFrame(
            width, height, rtc.VideoBufferType.RGB24, image.convert("RGB").tobytes()
        )

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(AVATAR_IDENTITY)
        .with_grants(
            api.VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True)
        )
        .to_jwt()
    )

    avatar_room = rtc.Room()
    await avatar_room.connect(livekit_url, token)
    logger.info("avatar worker joined room %s as %s", room_name, AVATAR_IDENTITY)

    audio_recv = _NullAudioReceiver()
    video_gen = MuseTalkVideoGenerator(
        io_root=io_root,
        portrait_container_path=portrait_container_path,
        avatar_id=avatar_id,
        fps=fps,
        sample_rate=sample_rate,
        num_channels=num_channels,
        initial_frame=initial_frame,
        video_container_path=video_container_path,
        bank_start_frame=bank_start_frame,
        source_start_frame=source_start_frame,
    )
    runner = AvatarRunner(
        avatar_room,
        audio_recv=audio_recv,
        video_gen=video_gen,
        options=AvatarOptions(
            video_width=width,
            video_height=height,
            video_fps=fps,
            audio_sample_rate=sample_rate,
            audio_channels=num_channels,
        ),
    )
    await runner.start()
    logger.info("avatar runner started (%dx%d @ %dfps)", width, height, fps)
    return avatar_room, runner, video_gen
