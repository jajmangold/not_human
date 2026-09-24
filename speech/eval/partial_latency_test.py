"""Measures real partial-transcript latency from crispasr-stt's native WS
streaming interface, to answer the actual question nothuman#94 needs
answered: how fast do usable partials arrive relative to full LLM response
latency? (#94's acceptance criterion: "at least an order of magnitude
below LLM response latency".)

This is deliberately NOT wired into the agent -- pure plumbing/feasibility
measurement, separable from #94's reaction-classifier half (which depends
on AffectStateV1 / nothuman#90, not yet merged)."""
import asyncio
import json
import time

import numpy as np
import soundfile as sf
import websockets

WS_URL = "ws://127.0.0.1:18296"
CLIPS = [
    "corpus/spk1320_0.wav",   # 7.8s, multi-clause sentence
    "corpus/spk3575_0.wav",   # 12.7s, long sentence
    "scenario_d/interruption.wav",  # 12.7s
]


async def run_one(wav_path: str):
    arr, sr = sf.read(wav_path, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    duration_s = len(arr) / sr

    events = []
    async with websockets.connect(WS_URL, max_size=None) as ws:
        async def sender():
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
            await ws.send("flush")

        async def receiver(stop_event, t0_holder):
            while not stop_event.is_set():
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                recv_t = time.perf_counter()
                try:
                    ev = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                t0 = t0_holder.get("t0")
                ev["_recv_wall_s"] = (recv_t - t0) if t0 else None
                events.append(ev)
                if ev.get("final"):
                    stop_event.set()

        t0_holder = {}
        stop_event = asyncio.Event()
        recv_task = asyncio.create_task(receiver(stop_event, t0_holder))
        t0 = time.perf_counter()
        t0_holder["t0"] = t0
        send_task = asyncio.create_task(sender())
        await send_task
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=15)
        except asyncio.TimeoutError:
            pass
        stop_event.set()
        recv_task.cancel()

    print(f"\n=== {wav_path} ({duration_s:.2f}s audio, real-time-paced feed) ===")
    partials = [e for e in events if not e.get("final")]
    finals = [e for e in events if e.get("final")]
    if partials:
        first = partials[0]
        print(f"  FIRST PARTIAL at t={first['_recv_wall_s']:.3f}s into the utterance: {first.get('text','')[:60]!r}")
        gaps = [partials[i]["_recv_wall_s"] - partials[i-1]["_recv_wall_s"] for i in range(1, len(partials))]
        print(f"  {len(partials)} partials total, gaps between them: "
              f"min={min(gaps):.3f}s max={max(gaps):.3f}s mean={sum(gaps)/len(gaps):.3f}s" if gaps else f"  {len(partials)} partial(s) total")
    else:
        print("  !! NO PARTIALS RECEIVED -- only a final event, or nothing at all")
    if finals:
        print(f"  FINAL at t={finals[-1]['_recv_wall_s']:.3f}s (audio duration was {duration_s:.2f}s): {finals[-1].get('text','')!r}")
    return {"file": wav_path, "duration_s": duration_s, "n_partials": len(partials),
            "first_partial_s": partials[0]["_recv_wall_s"] if partials else None,
            "final_s": finals[-1]["_recv_wall_s"] if finals else None}


async def main():
    results = []
    for clip in CLIPS:
        results.append(await run_one(clip))
    return results


if __name__ == "__main__":
    asyncio.run(main())
