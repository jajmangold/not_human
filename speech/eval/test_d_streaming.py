"""Scenario D: live/streaming mode, paced at real wall-clock rate (not
instant batch) -- the actual real-time-avatar code path. Tests:
  D1) does the gap+different-speaker content-drop bug from scenario C
      reproduce in --stream mode too?
  D2) genuine interruption/barge-in: speaker A cut off mid-sentence by
      speaker B overlapping -- does A finalize sanely, does B's partial
      appear promptly, what's the latency?
"""
import json
import os
import subprocess
import threading
import time

import numpy as np

from crisp_utils import BIN, MODEL, env, load_wav_mono16k, save_wav

os.makedirs("scenario_d", exist_ok=True)


def feed_realtime(wav_path: str, proc_stdin, chunk_ms=100):
    """Feed a WAV file's PCM into a subprocess's stdin at real wall-clock
    pace (chunk by chunk with real sleeps), simulating a live mic."""
    arr, sr = load_wav_mono16k(wav_path)
    pcm = (arr * 32767.0).astype(np.int16).tobytes()
    chunk_bytes = int(sr * chunk_ms / 1000) * 2  # int16 = 2 bytes/sample
    t0 = time.time()
    sent = 0
    while sent < len(pcm):
        chunk = pcm[sent : sent + chunk_bytes]
        proc_stdin.write(chunk)
        proc_stdin.flush()
        sent += len(chunk)
        target_t = t0 + (sent / 2) / sr
        sleep_s = target_t - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)
    proc_stdin.close()
    return t0


def run_stream_test(wav_path: str, label: str, extra_args=None, chunk_ms=100):
    extra_args = extra_args or []
    cmd = [
        BIN, "-m", MODEL, "--stream", "--stream-json",
        "--stream-step", "1000", "--stream-length", "10000",
        "--stream-final-on-silence-ms", "600",
    ] + extra_args
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env()
    )
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
    t0 = feed_realtime(wav_path, proc.stdin, chunk_ms=chunk_ms)
    t0_holder["t0"] = t0
    proc.wait(timeout=30)
    rt.join(timeout=5)
    stderr = proc.stderr.read().decode("utf-8", errors="replace")
    print(f"\n=== {label} ===")
    for ev in events:
        print(f"  t={ev['_recv_wall_s']:6.2f}s  {ev.get('type'):8s} "
              f"utt={ev.get('utterance_id')}  spk={ev.get('speaker','')}  "
              f"text={ev.get('text','')[:70]!r}")
    return events, stderr


# --- D1: does the gap+different-speaker drop bug reproduce in streaming? ---
events1, stderr1 = run_stream_test(
    "scenario_c/with_gap.wav", "D1: gap + different speaker, STREAMING mode"
)
finals1 = [e for e in events1 if e.get("type") == "final"]
print(f"\nD1 result: {len(finals1)} final utterance(s). "
      f"{'BUG REPRODUCED (2nd speaker missing)' if len(finals1) < 2 else 'both speakers present'}")

# --- D2: genuine interruption -- A talking, B barges in with real overlap ---
meta = json.load(open("corpus/meta.json"))
by_speaker = {}
for m in meta:
    by_speaker.setdefault(m["speaker"], []).append(m)
speakers = list(by_speaker.keys())
A, B = speakers[6], speakers[7]
a, sr = load_wav_mono16k(by_speaker[A][0]["file"])
b, _ = load_wav_mono16k(by_speaker[B][0]["file"])
# B barges in 1.5s before A would naturally finish -- true interruption
interrupt_at = max(0, len(a) - int(1.5 * sr))
total_len = max(len(a), interrupt_at + len(b))
mixed = np.zeros(total_len, dtype=np.float32)
mixed[: len(a)] += a
mixed[interrupt_at : interrupt_at + len(b)] += b
peak = np.max(np.abs(mixed))
if peak > 0.98:
    mixed = mixed * (0.98 / peak)
save_wav("scenario_d/interruption.wav", mixed, sr)
print(f"\nA={A} ({len(a)/sr:.1f}s): {by_speaker[A][0]['text'][:60]!r}")
print(f"B={B} interrupts at t={interrupt_at/sr:.1f}s: {by_speaker[B][0]['text'][:60]!r}")

events2, stderr2 = run_stream_test("scenario_d/interruption.wav", "D2: real interruption/barge-in, STREAMING mode")
json.dump(
    {"D1_events": events1, "D1_stderr_tail": stderr1[-1000:], "D2_events": events2, "D2_stderr_tail": stderr2[-1000:],
     "D2_interrupt_at_s": interrupt_at / sr},
    open("scenario_d/results.json", "w"), indent=2,
)
if not events2:
    print("\nD2: NO STREAMING EVENTS AT ALL -- stderr tail:")
    print(stderr2[-2000:])
