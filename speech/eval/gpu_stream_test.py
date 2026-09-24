import json
import os
import subprocess
import threading
import time

import numpy as np
import soundfile as sf

WAV = "scenario_d/interruption.wav"

LDLP = (
    "/scratch/hearing-stack-test-20260909/crispasr-build-cuda/src:"
    "/scratch/hearing-stack-test-20260909/crispasr-build-cuda/ggml/src:"
    "/scratch/hearing-stack-test-20260909/crispasr-build-cuda/ggml/src/ggml-cuda"
)
cmd = [
    "docker", "run", "--rm", "-i", "--gpus", "device=11",
    "-v", os.environ.get("EVAL_SCRATCH", ".") + ":/scratch",
    "-e", f"LD_LIBRARY_PATH={LDLP}",
    "yanwk/comfyui-boot:cu126-megapak",
    "/scratch/hearing-stack-test-20260909/crispasr-build-cuda/bin/crispasr",
    "-m", "/scratch/vision-stack-test-20260909/asr-models/ggml-base.en.bin",
    "--stream", "--stream-json",
    "--stream-step", "1000", "--stream-length", "10000",
    "--stream-final-on-silence-ms", "600",
]

arr, sr = sf.read(WAV, dtype="float32")
if arr.ndim > 1:
    arr = arr.mean(axis=1)
pcm = (arr * 32767.0).astype(np.int16).tobytes()
print(f"audio duration: {len(arr)/sr:.2f}s")

proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
events = []
t0_holder = {}


def reader():
    for line in proc.stdout:
        recv_t = time.time()
        line = line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev["_recv_wall_s"] = recv_t - t0_holder.get("t0", recv_t)
        events.append(ev)


rt = threading.Thread(target=reader)
rt.start()

chunk_ms = 100
chunk_bytes = int(sr * chunk_ms / 1000) * 2
t0 = time.time()
t0_holder["t0"] = t0
sent = 0
while sent < len(pcm):
    chunk = pcm[sent : sent + chunk_bytes]
    proc.stdin.write(chunk)
    proc.stdin.flush()
    sent += len(chunk)
    target_t = t0 + (sent / 2) / sr
    sleep_s = target_t - time.time()
    if sleep_s > 0:
        time.sleep(sleep_s)
proc.stdin.close()
feed_done_s = time.time() - t0
proc.wait(timeout=30)
rt.join(timeout=5)

print(f"audio fed in {feed_done_s:.2f}s (real-time pacing)")
for ev in events:
    print(f"  t={ev['_recv_wall_s']:6.2f}s  {ev.get('type'):8s} text={ev.get('text','')[:80]!r}")
finals = [e for e in events if e.get("type") == "final"]
if finals:
    print(f"\nfinal arrived at t={finals[-1]['_recv_wall_s']:.2f}s (audio duration was {len(arr)/sr:.2f}s)")
    print(f"FINAL TEXT: {finals[-1].get('text')!r}")
