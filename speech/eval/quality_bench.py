import os
"""Compare WER and batch throughput across model sizes against the
persistent server, to find the best-quality model that still stays
comfortably faster than real-time. Reuses the same corpus/refs as the
original scenario-A WER baseline for an apples-to-apples comparison."""
import json
import time
import urllib.request

from crisp_utils import wer

SERVER = "http://127.0.0.1:18080"
CORPUS_META = "corpus/meta.json"
# Model paths as seen by the *server* process (its own filesystem, e.g. a container mount).
MODEL_DIR = os.environ.get("CRISPASR_MODEL_DIR", "/models")
THROUGHPUT_WAV = "scenario_d/interruption.wav"
THROUGHPUT_DURATION_S = 12.73  # measured earlier from soundfile

MODELS = {
    "base.en": MODEL_DIR + "/ggml-base.en.bin",
    "small.en": MODEL_DIR + "/ggml-small.en.bin",
    "medium.en": MODEL_DIR + "/ggml-medium.en.bin",
    "large-v3-turbo": MODEL_DIR + "/ggml-large-v3-turbo.bin",
    "large-v3": MODEL_DIR + "/ggml-large-v3.bin",
}


def load_model(path):
    boundary = "----crispasrload"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{path}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(
        f"{SERVER}/load",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        text = resp.read().decode(errors="replace")
    return text, time.time() - t0


def transcribe(wav_path):
    boundary = "----crispasrbench"
    with open(wav_path, "rb") as f:
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
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def main():
    meta = json.load(open(CORPUS_META))
    results = {}
    for name, path in MODELS.items():
        print(f"\n=== {name} ===")
        _, load_s = load_model(path)
        print(f"  loaded in {load_s:.2f}s")

        # WER over the full corpus
        wers = []
        for clip in meta:
            r = transcribe(clip["file"])
            hyp = " ".join(seg.get("text", "") for seg in r.get("segments", []))
            w = wer(clip["text"], hyp)
            wers.append(w)
        mean_wer = sum(wers) / len(wers)
        print(f"  mean WER over {len(wers)} clips: {mean_wer:.4f}")

        # batch throughput on the interruption clip, 3 runs
        times = []
        for i in range(3):
            t0 = time.perf_counter()
            r = transcribe(THROUGHPUT_WAV)
            dt = time.perf_counter() - t0
            times.append(dt)
        best = min(times)
        rtf = THROUGHPUT_DURATION_S / best
        print(f"  throughput runs: {[f'{t:.2f}s' for t in times]}  -> best={best:.2f}s ({rtf:.1f}x realtime)")

        results[name] = {
            "load_s": load_s,
            "mean_wer": mean_wer,
            "wers": wers,
            "throughput_runs_s": times,
            "best_throughput_s": best,
            "realtime_factor": rtf,
        }

    json.dump(results, open("quality_bench_results.json", "w"), indent=2)
    print("\n\n=== SUMMARY ===")
    print(f"{'model':<16}{'mean WER':>10}{'best batch s':>14}{'realtime x':>12}")
    for name, r in results.items():
        print(f"{name:<16}{r['mean_wer']:>10.4f}{r['best_throughput_s']:>14.2f}{r['realtime_factor']:>11.1f}x")


if __name__ == "__main__":
    main()
