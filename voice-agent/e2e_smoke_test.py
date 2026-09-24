"""Direct end-to-end verification of the live musetalk-voice-agent.service:
join the same LiveKit room a real browser would, publish real speech audio
on a synthetic mic track (no browser/mic needed), and confirm a reply audio
track comes back from the agent -- exercising the full
STT (crispasr-stt) -> LLM (gemma4-26b) -> TTS (Kokoro) loop for real."""
import asyncio
import os
import time
import wave

import numpy as np
from dotenv import load_dotenv
from livekit import api, rtc

load_dotenv(dotenv_path="../.env")

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880")
API_KEY = os.environ["LIVEKIT_API_KEY"]
API_SECRET = os.environ["LIVEKIT_API_SECRET"]
ROOM_NAME = os.environ.get("TEST_ROOM_NAME") or os.environ.get("LIVEKIT_ROOM_NAME", "musetalk-voice")
SPEECH_WAV = "corpus/spk260_2.wav"
SR = 16000


def load_pcm16(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == SR, f"expected {SR}Hz, got {w.getframerate()}"
        n = w.getnframes()
        raw = w.readframes(n)
    arr = np.frombuffer(raw, dtype=np.int16)
    if w.getnchannels() > 1:
        arr = arr.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
    return arr


async def main():
    token = (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity("e2e-smoke-test")
        .with_name("e2e-smoke-test")
        .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME))
        .to_jwt()
    )

    room = rtc.Room()
    received_frames: list[bytes] = []
    got_audio_back = asyncio.Event()

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        print(f"  [event] subscribed to track from {participant.identity}: {track.kind}")
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.create_task(_consume_audio(track))

    async def _consume_audio(track):
        stream = rtc.AudioStream(track)
        async for event in stream:
            received_frames.append(bytes(event.frame.data))
            got_audio_back.set()

    @room.on("participant_connected")
    def on_participant(p):
        print(f"  [event] participant connected: {p.identity}")

    print(f"connecting to {LIVEKIT_URL}, room={ROOM_NAME!r} ...")
    await room.connect(LIVEKIT_URL, token)
    print(f"connected. local identity={room.local_participant.identity}")
    print(f"remote participants already in room: {[p.identity for p in room.remote_participants.values()]}")

    # publish a synthetic "microphone" track
    source = rtc.AudioSource(SR, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    pub_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await room.local_participant.publish_track(track, pub_options)
    print("published synthetic mic track")

    # give the agent a moment to join/dispatch if it wasn't already in the room
    await asyncio.sleep(3)
    print(f"participants now: {[p.identity for p in room.remote_participants.values()]}")

    pcm = load_pcm16(SPEECH_WAV)
    duration_s = len(pcm) / SR
    print(f"feeding {duration_s:.2f}s of real speech from {SPEECH_WAV} ...")
    frame_ms = 20
    frame_samples = int(SR * frame_ms / 1000)
    t0 = time.perf_counter()
    sent = 0
    while sent < len(pcm):
        chunk = pcm[sent: sent + frame_samples]
        if len(chunk) < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - len(chunk)))
        frame = rtc.AudioFrame(
            data=chunk.tobytes(), sample_rate=SR, num_channels=1, samples_per_channel=frame_samples
        )
        await source.capture_frame(frame)
        sent += frame_samples
        target = t0 + sent / SR
        dt = target - time.perf_counter()
        if dt > 0:
            await asyncio.sleep(dt)
    print("finished feeding audio, sending trailing silence to let VAD detect end-of-turn ...")
    silence = np.zeros(frame_samples, dtype=np.int16)
    for _ in range(75):  # 1.5s of silence
        frame = rtc.AudioFrame(
            data=silence.tobytes(), sample_rate=SR, num_channels=1, samples_per_channel=frame_samples
        )
        await source.capture_frame(frame)
        await asyncio.sleep(frame_ms / 1000)

    print("waiting up to 20s for a reply audio track ...")
    try:
        await asyncio.wait_for(got_audio_back.wait(), timeout=20)
        print("REPLY AUDIO RECEIVED -- waiting a bit more to capture the full reply ...")
        await asyncio.sleep(6)
    except asyncio.TimeoutError:
        print("!! TIMEOUT -- no reply audio track received within 20s")

    total_bytes = sum(len(f) for f in received_frames)
    print(f"\ntotal reply audio received: {len(received_frames)} frames, {total_bytes} bytes "
          f"({total_bytes / 2 / SR:.2f}s at {SR}Hz mono s16)")

    if received_frames:
        out_path = "/tmp/e2e_reply_audio.wav"
        with wave.open(out_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(b"".join(received_frames))
        arr = np.frombuffer(b"".join(received_frames), dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0
        print(f"saved reply audio to {out_path}, RMS energy={rms:.4f} "
              f"({'looks like real signal, not silence' if rms > 0.001 else 'LOOKS LIKE SILENCE'})")

    await room.disconnect()
    return len(received_frames) > 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    print(f"\n=== RESULT: {'PASS -- got a reply' if ok else 'FAIL -- no reply audio'} ===")
