"""Tests whether the ~10.5s first-partial delay is a one-time per-session
GPU warm-up cost (first utterance on a fresh session pays it, subsequent
utterances on the SAME session don't) rather than something inherent to
every utterance."""
import asyncio
import json
import time

import numpy as np
import soundfile as sf
import websockets

WS_URL = "ws://127.0.0.1:18296"
CLIPS = ["corpus/spk1320_0.wav", "corpus/spk1320_1.wav", "corpus/spk1320_2.wav"]


async def feed_one(ws, wav_path, label):
    arr, sr = sf.read(wav_path, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    events = []
    t0 = time.perf_counter()

    async def sender():
        chunk_ms = 100
        chunk_samples = int(sr * chunk_ms / 1000)
        sent = 0
        while sent < len(arr):
            chunk = arr[sent: sent + chunk_samples]
            await ws.send(chunk.astype(np.float32).tobytes())
            sent += len(chunk)
            target = t0 + sent / sr
            dt = target - time.perf_counter()
            if dt > 0:
                await asyncio.sleep(dt)
        await ws.send("flush")

    async def receiver(stop_event):
        while not stop_event.is_set():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            recv_t = time.perf_counter() - t0
            try:
                ev = json.loads(msg)
            except json.JSONDecodeError:
                continue
            ev["_t"] = recv_t
            events.append(ev)
            if ev.get("final"):
                stop_event.set()

    stop_event = asyncio.Event()
    recv_task = asyncio.create_task(receiver(stop_event))
    await sender()
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=15)
    except asyncio.TimeoutError:
        pass
    stop_event.set()
    recv_task.cancel()

    partials = [e for e in events if not e.get("final")]
    finals = [e for e in events if e.get("final")]
    first_t = partials[0]["_t"] if partials else None
    print(f"  [{label}] first partial at t={first_t:.3f}s" if first_t else f"  [{label}] NO PARTIALS")
    if finals:
        print(f"  [{label}] final at t={finals[-1]['_t']:.3f}s: {finals[-1].get('text','')[:70]!r}")


async def main():
    async with websockets.connect(WS_URL, max_size=None) as ws:
        for i, clip in enumerate(CLIPS, 1):
            print(f"=== turn {i}: {clip} ===")
            await feed_one(ws, clip, f"turn{i}")


asyncio.run(main())
