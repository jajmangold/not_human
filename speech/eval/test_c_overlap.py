"""Scenario C: overlapping speech stress test -- two speakers talking at the
same time, at increasing overlap fractions. The hardest realistic case for
both ASR and diarization; measures where it actually breaks, not whether."""
import json

import numpy as np

from crisp_utils import full_text, load_wav_mono16k, run_batch, save_wav, wer

meta = json.load(open("corpus/meta.json"))
by_speaker = {}
for m in meta:
    by_speaker.setdefault(m["speaker"], []).append(m)
speakers = list(by_speaker.keys())
A, B = speakers[4], speakers[5]  # fresh pair, not reused from scenario B
clip_a, clip_b = by_speaker[A][0], by_speaker[B][0]
sr = 16000
a, _ = load_wav_mono16k(clip_a["file"])
b, _ = load_wav_mono16k(clip_b["file"])

print(f"speaker A={A} ({len(a)/sr:.1f}s): {clip_a['text'][:60]!r}")
print(f"speaker B={B} ({len(b)/sr:.1f}s): {clip_b['text'][:60]!r}")

results = []
for overlap_frac in (0.0, 0.10, 0.30, 0.50, 0.80, 1.0):
    overlap_samples = int(min(len(a), len(b)) * overlap_frac)
    total_len = len(a) + len(b) - overlap_samples
    mixed = np.zeros(total_len, dtype=np.float32)
    mixed[: len(a)] += a
    b_start = len(a) - overlap_samples
    mixed[b_start : b_start + len(b)] += b
    peak = np.max(np.abs(mixed))
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    out = f"scenario_c/overlap_{int(overlap_frac*100)}pct.wav"
    save_wav(out, mixed, sr)

    r = run_batch(out, extra_args=["--diarize-speakers"], timeout=120)
    segs = r.get("transcription", [])
    hyp_full = full_text(r)
    wer_a = wer(clip_a["text"], hyp_full)
    wer_b = wer(clip_b["text"], hyp_full)
    n_speakers_found = len({s.get("speaker") for s in segs})
    results.append(
        {
            "overlap_pct": int(overlap_frac * 100),
            "n_segments": len(segs),
            "n_speakers_found": n_speakers_found,
            "wer_vs_A": wer_a,
            "wer_vs_B": wer_b,
            "hyp": hyp_full[:200],
        }
    )
    print(
        f"\noverlap={int(overlap_frac*100):3d}%  segs={len(segs)}  speakers_found={n_speakers_found}  "
        f"wer_vs_A={wer_a:.2f}  wer_vs_B={wer_b:.2f}"
    )
    for s in segs:
        print(f"    [{s['offsets']['from']/1000:5.2f}-{s['offsets']['to']/1000:5.2f}s] "
              f"spk={s.get('speaker','?')}  {s['text'][:70]!r}")

json.dump(results, open("scenario_c/results.json", "w"), indent=2)
print("\n=== summary: overlap% -> min(WER vs either speaker), speakers detected ===")
for r in results:
    best_wer = min(r["wer_vs_A"], r["wer_vs_B"])
    print(f"  {r['overlap_pct']:3d}%  best_single_speaker_wer={best_wer:.2f}  speakers_found={r['n_speakers_found']}")
