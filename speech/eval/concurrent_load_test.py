"""Empirically test whether the persistent server's /inference endpoint
serializes concurrent requests (single model_mutex) or genuinely
parallelizes them, and measure the real latency-vs-concurrency curve."""
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SERVER = "http://127.0.0.1:18280"
WAV = "scenario_d/interruption.wav"  # 12.73s of audio
AUDIO_DURATION_S = 12.73


def one_request():
    boundary = "----crispasrload"
    with open(WAV, "rb") as f:
        data = f.read()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{SERVER}/inference",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp.read()
        return time.perf_counter() - t0, None
    except Exception as e:
        return time.perf_counter() - t0, str(e)


def run_concurrency_level(n):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda _: one_request(), range(n)))
    wall_s = time.perf_counter() - t0
    latencies = [r[0] for r in results]
    errors = [r[1] for r in results if r[1]]
    mean_lat = statistics.mean(latencies)
    max_lat = max(latencies)
    # if requests were serialized, wall time for n concurrent requests
    # should be roughly n * (single-request time); if truly parallel,
    # wall time should stay close to a single request's time.
    print(f"\nn={n:>2} concurrent requests:")
    print(f"  wall time for batch: {wall_s:.2f}s")
    print(f"  per-request latency: mean={mean_lat:.2f}s  max={max_lat:.2f}s  min={min(latencies):.2f}s")
    print(f"  errors: {len(errors)}/{n}" + (f"  ({errors[0]})" if errors else ""))
    print(f"  effective aggregate throughput: {n * AUDIO_DURATION_S / wall_s:.1f}x realtime "
          f"({'scaling with concurrency -- real parallelism' if wall_s < 1.5 * (latencies[0] if n==1 else mean_lat) else ''})")
    return wall_s, mean_lat, max_lat, len(errors)


if __name__ == "__main__":
    print("=== baseline: single request ===")
    base_lat, base_err = one_request()
    print(f"single request latency: {base_lat:.2f}s ({AUDIO_DURATION_S/base_lat:.1f}x realtime), error={base_err}")

    for n in [2, 4, 8]:
        run_concurrency_level(n)
