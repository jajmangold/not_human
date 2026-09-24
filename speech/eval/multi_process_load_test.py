"""Two independent server PROCESSES sharing one physical GPU, each hit
with a single request simultaneously -- tests whether multiple server
instances actually parallelize on the GPU's spare compute, or whether the
GPU itself is the bottleneck (in which case both would slow down
together)."""
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

WAV = "scenario_d/interruption.wav"
AUDIO_DURATION_S = 12.73
SERVERS = ["http://127.0.0.1:18280", "http://127.0.0.1:18281"]


def one_request(server):
    boundary = "----crispasrload"
    with open(WAV, "rb") as f:
        data = f.read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{server}/inference",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        resp.read()
    return time.perf_counter() - t0


print("=== solo baseline: each server alone, sequential ===")
solo = {}
for s in SERVERS:
    t = one_request(s)
    solo[s] = t
    print(f"  {s}: {t:.2f}s solo ({AUDIO_DURATION_S/t:.1f}x realtime)")

print("\n=== both servers hit SIMULTANEOUSLY, one request each ===")
with ThreadPoolExecutor(max_workers=2) as ex:
    t0 = time.perf_counter()
    futures = [ex.submit(one_request, s) for s in SERVERS]
    results = [f.result() for f in futures]
    wall = time.perf_counter() - t0

for s, t in zip(SERVERS, results):
    slowdown = t / solo[s]
    print(f"  {s}: {t:.2f}s concurrent ({AUDIO_DURATION_S/t:.1f}x realtime), "
          f"{slowdown:.2f}x slower than solo")
print(f"  wall time for both: {wall:.2f}s")
print(f"  combined aggregate throughput: {2*AUDIO_DURATION_S/wall:.1f}x realtime")
print(f"\n  verdict: {'GENUINE PARALLELISM -- both ran near solo speed' if max(results)/max(solo.values()) < 1.3 else 'GPU COMPUTE CONTENTION -- both slowed down together'}")
