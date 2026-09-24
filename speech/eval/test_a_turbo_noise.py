"""Re-run Scenario A's noise/SNR adversarial curve against large-v3-turbo
(the quality-bench winner) via the persistent server, to confirm the WER
win holds under noise and isn't just a clean-corpus artifact."""
import json
import os

import numpy as np

from crisp_utils import load_wav_mono16k, mix_at_snr, save_wav, wer
from quality_bench import load_model, transcribe

os.makedirs("scenario_a_turbo", exist_ok=True)
meta = json.load(open("corpus/meta.json"))
noise_meta = json.load(open("noise/meta.json"))

print("loading large-v3-turbo on server...")
_, load_s = load_model(os.environ.get("CRISPASR_MODEL_DIR", "/models") + "/ggml-large-v3-turbo.bin")
print(f"  loaded in {load_s:.2f}s")

by_speaker = {}
for m in meta:
    by_speaker.setdefault(m["speaker"], m)
test_utts = list(by_speaker.values())[:3]

results = {"clean_baseline": [], "snr_curve": []}

print("=== clean baseline WER (8 speakers x first clip each) ===")
seen = set()
for m in meta:
    if m["speaker"] in seen:
        continue
    seen.add(m["speaker"])
    r = transcribe(m["file"])
    hyp = " ".join(seg.get("text", "") for seg in r.get("segments", []))
    w = wer(m["text"], hyp)
    results["clean_baseline"].append({"speaker": m["speaker"], "ref": m["text"], "hyp": hyp, "wer": w})
    print(f"  spk={m['speaker']:>6}  wer={w:.3f}  ref={m['text'][:50]!r}")

babble_srcs = [m["file"] for m in meta if m not in test_utts][:6]
babble_chunks = []
for f in babble_srcs:
    a, _ = load_wav_mono16k(f)
    babble_chunks.append(a)
maxlen = max(len(c) for c in babble_chunks)
babble = np.zeros(maxlen, dtype=np.float32)
for c in babble_chunks:
    offset = np.random.randint(0, maxlen - len(c) + 1) if maxlen > len(c) else 0
    babble[offset : offset + len(c)] += c
all_noise_types = {m["category"]: m["file"] for m in noise_meta}
noise_types = {
    "babble_6talker": None,
    "vacuum_cleaner": all_noise_types["vacuum_cleaner"],
    "engine": all_noise_types["engine"],
    "siren": all_noise_types["siren"],
}

print("\n=== SNR degradation curve (3 utterances x 4 noise types x 4 SNR levels) ===")
snr_levels = [20, 10, 0, -5]
for m in test_utts:
    speech, sr = load_wav_mono16k(m["file"])
    for noise_name, noise_file in noise_types.items():
        if noise_name == "babble_6talker":
            noise, _ = babble, 16000
        else:
            noise, _ = load_wav_mono16k(noise_file)
        for snr in snr_levels:
            mixed = mix_at_snr(speech, noise, snr)
            out_wav = f"scenario_a_turbo/{m['speaker']}_{noise_name}_{snr}dB.wav"
            save_wav(out_wav, mixed)
            r = transcribe(out_wav)
            hyp = " ".join(seg.get("text", "") for seg in r.get("segments", []))
            w = wer(m["text"], hyp)
            results["snr_curve"].append(
                {"speaker": m["speaker"], "noise": noise_name, "snr_db": snr, "wer": w, "hyp": hyp[:120]}
            )
            print(f"  spk={m['speaker']:>6}  noise={noise_name:>16}  snr={snr:>4}dB  wer={w:.3f}")

json.dump(results, open("scenario_a_turbo/results.json", "w"), indent=2)

clean_wers = [r["wer"] for r in results["clean_baseline"]]
print(f"\nCLEAN baseline mean WER: {np.mean(clean_wers):.3f}  (base.en was 0.211)")
for noise_name in noise_types:
    for snr in snr_levels:
        ws = [r["wer"] for r in results["snr_curve"] if r["noise"] == noise_name and r["snr_db"] == snr]
        print(f"  {noise_name:>16} @ {snr:>4}dB: mean WER {np.mean(ws):.3f}")
