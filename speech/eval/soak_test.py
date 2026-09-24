"""Long-duration soak test: one persistent WS streaming connection,
repeatedly fed real audio in a loop with periodic flush boundaries
(simulating conversational turns), for an extended period. Watches for
memory growth (leak), latency drift, and crashes/disconnects -- none of
which the short adversarial-scenario tests earlier in this investigation
could catch, since they were all single short clips."""
import asyncio
import glob
import json
import subprocess
import time

import numpy as np
import soundfile as sf
import websockets

WS_URL = "ws://127.0.0.1:18291"
DURATION_S = float(__import__("os").environ.get("SOAK_DURATION_S", "1800"))  # default 30 min
SAMPLE_INTERVAL_S = 30

# loop through all available real clips (corpus + scenario wavs) for variety
CLIPS = sorted(glob.glob("corpus/spk*.wav")) + ["scenario_d/interruption.wav"]


def gpu_mem_mb():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "11"],
        capture_output=True, text=True,
    ).stdout.strip()
    return int(out)


def container_mem_mb():
    out = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", "crispasr-soak"],
        capture_output=True, text=True,
    ).stdout.strip()
    # format like "1.2GiB / 15.6GiB"
    used = out.split("/")[0].strip()
    if "GiB" in used:
        return float(used.replace("GiB", "")) * 1024
    if "MiB" in used:
        return float(used.replace("MiB", ""))
    return -1.0


async def one_turn(ws, wav_path, turn_id):
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
    await ws.send("flush")
    # wait for final
    final_text = None
    t_flush = time.perf_counter()
    while time.perf_counter() - t_flush < 15:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        try:
            ev = json.loads(msg)
        except json.JSONDecodeError:
            continue
        if ev.get("final"):
            final_text = ev.get("text")
            latency = time.perf_counter() - t_flush
            return latency, final_text
    return None, None  # timed out, no final


async def main():
    print(f"soak test starting: duration={DURATION_S:.0f}s, sample_interval={SAMPLE_INTERVAL_S}s")
    start = time.time()
    turn_id = 0
    clip_idx = 0
    log = []
    last_sample = 0
    n_timeouts = 0
    n_reconnects = 0
    ws = await websockets.connect(WS_URL, max_size=None)
    try:
        while time.time() - start < DURATION_S:
            wav_path = CLIPS[clip_idx % len(CLIPS)]
            clip_idx += 1
            turn_id += 1
            try:
                latency, text = await one_turn(ws, wav_path, turn_id)
            except (websockets.exceptions.ConnectionClosed, OSError) as e:
                latency, text = None, f"<connection error: {e}>"
            elapsed = time.time() - start
            ok = latency is not None
            print(f"  t={elapsed:6.0f}s  turn={turn_id:>4}  clip={wav_path:<30}  "
                  f"flush->final latency={latency if ok else 'TIMEOUT':<6}  "
                  f"text={(text or '')[:60]!r}")
            if elapsed - last_sample >= SAMPLE_INTERVAL_S or not ok:
                gpu_mb = gpu_mem_mb()
                cont_mb = container_mem_mb()
                log.append({"t": elapsed, "turn": turn_id, "gpu_mem_mb": gpu_mb,
                            "container_mem_mb": cont_mb, "latency_s": latency, "ok": ok})
                print(f"    [sample] gpu_mem={gpu_mb}MB  container_mem={cont_mb:.0f}MB")
                last_sample = elapsed
            if not ok:
                n_timeouts += 1
                print(f"    !! turn {turn_id} failed ({text}) -- reconnecting fresh (total failures: {n_timeouts})")
                try:
                    await ws.close()
                except Exception:
                    pass
                ws = await websockets.connect(WS_URL, max_size=None)
                n_reconnects += 1
    finally:
        await ws.close()

    log.append({"summary": True, "n_turns": turn_id, "n_timeouts": n_timeouts, "n_reconnects": n_reconnects})
    json.dump(log, open("soak_test_results.json", "w"), indent=2)
    print(f"\nfailures: {n_timeouts}/{turn_id} turns, {n_reconnects} reconnects")
    print(f"\n=== soak test done: {turn_id} turns over {time.time()-start:.0f}s ===")
    if log:
        mems = [s["gpu_mem_mb"] for s in log]
        lats = [s["latency_s"] for s in log if s["latency_s"]]
        print(f"GPU memory: min={min(mems)}MB max={max(mems)}MB delta={max(mems)-min(mems)}MB")
        if lats:
            print(f"flush->final latency: min={min(lats):.2f}s max={max(lats):.2f}s "
                  f"first={lats[0]:.2f}s last={lats[-1]:.2f}s")


asyncio.run(main())
