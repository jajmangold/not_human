"""Scenario B: multi-speaker sequential diarization -- the 'person tracking'
question. Does the same real speaker, appearing in non-contiguous turns
separated by other speakers, get the SAME predicted cluster ID both times?
Real ground truth: we control the exact turn order."""
import json

import numpy as np

from crisp_utils import load_wav_mono16k, run_batch, save_wav

meta = json.load(open("corpus/meta.json"))
by_speaker = {}
for m in meta:
    by_speaker.setdefault(m["speaker"], []).append(m)

# script: A, B, C, A(again), D, B(again) -- non-contiguous repeats, the real
# test of "is this the same person as before"
speakers = list(by_speaker.keys())[:4]
A, B, C, D = speakers
script = [(A, 0), (B, 0), (C, 0), (A, 1), (D, 0), (B, 1)]

segments = []  # (speaker, start_s, end_s, text)
audio_chunks = []
t = 0.0
sr = 16000
gap = 0.5
for spk, clip_idx in script:
    clip = by_speaker[spk][clip_idx]
    arr, _ = load_wav_mono16k(clip["file"])
    start = t
    end = t + len(arr) / sr
    segments.append({"speaker": spk, "start": start, "end": end, "text": clip["text"]})
    audio_chunks.append(arr)
    audio_chunks.append(np.zeros(int(gap * sr), dtype=np.float32))
    t = end + gap

full = np.concatenate(audio_chunks)
save_wav("scenario_b/multispeaker_sequential.wav", full, sr)
json.dump(segments, open("scenario_b/ground_truth.json", "w"), indent=2)
print("script:", [s for s, _ in script], "-> speakers", {A: "A", B: "B", C: "C", D: "D"})
print(f"total duration: {len(full)/sr:.1f}s")

r = run_batch("scenario_b/multispeaker_sequential.wav", extra_args=["--diarize-speakers"], timeout=180)
pred_segments = r.get("transcription", [])

print("\n=== predicted segments ===")
for seg in pred_segments:
    print(f"  [{seg['offsets']['from']/1000:6.2f}-{seg['offsets']['to']/1000:6.2f}s] "
          f"speaker={seg.get('speaker','?'):>3}  text={seg['text'][:60]!r}")

# score: for each ground-truth segment, find the predicted segment(s)
# overlapping its midpoint, take majority predicted-cluster vote
print("\n=== ground truth vs predicted-cluster mapping ===")
gt_to_pred = {}
for gseg in segments:
    mid = (gseg["start"] + gseg["end"]) / 2
    match = None
    for pseg in pred_segments:
        if pseg["offsets"]["from"] / 1000 <= mid <= pseg["offsets"]["to"] / 1000:
            match = pseg.get("speaker", "?")
            break
    gt_to_pred.setdefault(gseg["speaker"], []).append(match)
    print(f"  real={gseg['speaker']:>6}  turn@{mid:5.1f}s  -> predicted cluster {match}")

print("\n=== consistency check (same real speaker -> same predicted cluster every time?) ===")
all_consistent = True
for spk, preds in gt_to_pred.items():
    consistent = len(set(preds)) == 1
    all_consistent &= consistent
    print(f"  real speaker {spk}: predicted clusters across their turns = {preds}  "
          f"{'CONSISTENT' if consistent else 'INCONSISTENT (FAIL)'}")

# also check no two DIFFERENT real speakers collapsed onto the same cluster
pred_sets = {spk: set(preds) for spk, preds in gt_to_pred.items()}
collisions = []
spks = list(pred_sets.keys())
for i in range(len(spks)):
    for j in range(i + 1, len(spks)):
        if pred_sets[spks[i]] & pred_sets[spks[j]]:
            collisions.append((spks[i], spks[j], pred_sets[spks[i]] & pred_sets[spks[j]]))
if collisions:
    print(f"\nCROSS-SPEAKER CLUSTER COLLISIONS (FAIL): {collisions}")
else:
    print("\nNo cross-speaker cluster collisions.")

print(f"\nOVERALL: {'PASS' if all_consistent and not collisions else 'FAIL'}")
