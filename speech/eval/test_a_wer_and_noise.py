"""Scenario A: clean-baseline WER + controlled-SNR adversarial noise curve.
Real LibriSpeech speech, real ESC-50 noise (+ synthetic babble), exact known
ground truth, exact controlled SNR -- not eyeballed."""
import glob
import json
import os

import numpy as np

from crisp_utils import full_text, load_wav_mono16k, mix_at_snr, run_batch, save_wav, wer

os.makedirs("scenario_a", exist_ok=True)
meta = json.load(open("corpus/meta.json"))
noise_meta = json.load(open("noise/meta.json"))

# pick 3 clean utterances from 3 different speakers for the noise curve
by_speaker = {}
for m in meta:
    by_speaker.setdefault(m["speaker"], m)
test_utts = list(by_speaker.values())[:3]

results = {"clean_baseline": [], "snr_curve": []}

print("=== A1: clean baseline WER (8 speakers x first clip each) ===")
seen = set()
for m in meta:
    if m["speaker"] in seen:
        continue
    seen.add(m["speaker"])
    r = run_batch(m["file"])
    hyp = full_text(r)
    w = wer(m["text"], hyp)
    results["clean_baseline"].append({"speaker": m["speaker"], "ref": m["text"], "hyp": hyp, "wer": w})
    print(f"  spk={m['speaker']:>6}  wer={w:.3f}  ref={m['text'][:50]!r}")

# build a synthetic babble noise (multi-talker chatter -- the closest proxy
# for "TV / crowd in the background") from OTHER speakers' clips, since
# ESC-50 has no speech-babble category
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
    "babble_6talker": None,  # special-cased below -- the real "TV/crowd" proxy
    "vacuum_cleaner": all_noise_types["vacuum_cleaner"],
    "engine": all_noise_types["engine"],
    "siren": all_noise_types["siren"],
}

print("\n=== A2: SNR degradation curve (3 utterances x 4 noise types x 4 SNR levels) ===")
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
            out_wav = f"scenario_a/{m['speaker']}_{noise_name}_{snr}dB.wav"
            save_wav(out_wav, mixed)
            r = run_batch(out_wav)
            hyp = full_text(r)
            w = wer(m["text"], hyp)
            results["snr_curve"].append(
                {"speaker": m["speaker"], "noise": noise_name, "snr_db": snr, "wer": w, "hyp": hyp[:120]}
            )
            print(f"  spk={m['speaker']:>6}  noise={noise_name:>16}  snr={snr:>4}dB  wer={w:.3f}")

json.dump(results, open("scenario_a/results.json", "w"), indent=2)

clean_wers = [r["wer"] for r in results["clean_baseline"]]
print(f"\nCLEAN baseline mean WER: {np.mean(clean_wers):.3f}")
for noise_name in noise_types:
    for snr in snr_levels:
        ws = [r["wer"] for r in results["snr_curve"] if r["noise"] == noise_name and r["snr_db"] == snr]
        print(f"  {noise_name:>16} @ {snr:>4}dB: mean WER {np.mean(ws):.3f}")
