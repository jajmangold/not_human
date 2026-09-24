"""Isolate the soak-test turn-3 hang: is it the specific clip, or state
accumulating on a persistent WS connection across turns?"""
import asyncio
import json
import sys
import time

import numpy as np
import soundfile as sf
import websockets

WS_URL = "ws://127.0.0.1:18294"
CLIP = "corpus/spk1320_2.wav"  # the clip that hung in the soak test


async def one_turn(ws, wav_path, label):
    arr, sr = sf.read(wav_path, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    chunk_ms = 100
    chunk_samples = int(sr * chunk_ms / 1000)
    t0 = time.perf_counter()
    sent = 0
    while sent < len(arr):
        chunk = arr[sent: sent + chunk_samples]
        await ws.send(chunk.astype(np.float32).tobytes())
        sent += len(chunk)
        target = t0 + sent / sr
        dt = target - time.perf_counter()
        if dt > 0:
            await asyncio.sleep(dt)
    print(f"  [{label}] fed {len(arr)/sr:.2f}s of audio, sending flush...")
    await ws.send("flush")
    t_flush = time.perf_counter()
    events_seen = []
    while time.perf_counter() - t_flush < 10:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            print(f"  [{label}] ...waiting ({time.perf_counter()-t_flush:.1f}s so far)")
            continue
        try:
            ev = json.loads(msg)
        except json.JSONDecodeError:
            print(f"  [{label}] non-JSON message: {msg!r}")
            continue
        events_seen.append(ev)
        if ev.get("final"):
            print(f"  [{label}] FINAL after {time.perf_counter()-t_flush:.2f}s: {ev.get('text')!r}")
            return True
    print(f"  [{label}] TIMEOUT -- {len(events_seen)} non-final events seen: "
          f"{[e.get('type') or e.get('text','')[:30] for e in events_seen]}")
    return False


async def test_persistent_connection():
    print("=== TEST A: persistent connection, turns 1/1.wav, 1.wav, 2.wav (soak-test order) ===")
    clips = ["corpus/spk1320_0.wav", "corpus/spk1320_1.wav", "corpus/spk1320_2.wav"]
    async with websockets.connect(WS_URL, max_size=None) as ws:
        for i, c in enumerate(clips, 1):
            ok = await one_turn(ws, c, f"persistent-turn{i}:{c}")
            if not ok:
                print(f"  !! failed at turn {i} on persistent connection")
                return


async def test_fresh_connection():
    print("\n=== TEST B: the hanging clip alone, on a totally FRESH connection ===")
    async with websockets.connect(WS_URL, max_size=None) as ws:
        ok = await one_turn(ws, CLIP, f"fresh:{CLIP}")
        print(f"  fresh-connection result: {'OK' if ok else 'HANG'}")


asyncio.run(test_persistent_connection())
asyncio.run(test_fresh_connection())
