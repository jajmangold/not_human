"""Crosstalk fix attempt: run a pretrained speech-separation model
(SepFormer, WSJ0-2mix) upstream of CrispASR on the same overlap_*.wav
clips from Scenario C, split into two separated streams, transcribe each
separately, and compare against the original single-stream (no
separation) transcription -- same ground truth, same overlap percentages,
directly comparable to the original disqualifying finding."""
import json
import os

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.separation import SepformerSeparation

import quality_bench
quality_bench.SERVER = "http://127.0.0.1:18292"  # dedicated GPU-14 server
from quality_bench import transcribe
from crisp_utils import wer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

meta = json.load(open("corpus/meta.json"))
by_speaker = {}
for m in meta:
    by_speaker.setdefault(m["speaker"], []).append(m)
speakers = list(by_speaker.keys())
A, B = speakers[4], speakers[5]  # identical pair to test_c_overlap.py
clip_a, clip_b = by_speaker[A][0], by_speaker[B][0]
print(f"speaker A={A}: {clip_a['text'][:60]!r}")
print(f"speaker B={B}: {clip_b['text'][:60]!r}")

os.makedirs("scenario_c_separated", exist_ok=True)

print("\nloading SepFormer (WSJ0-2mix)...")
separator = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="sepformer-wsj02mix-cache",
    run_opts={"device": DEVICE},
)
print(f"loaded on {DEVICE}")

results = []
for overlap_pct in (0, 10, 30, 50, 80, 100):
    wav_path = f"scenario_c/overlap_{overlap_pct}pct.wav"
    if not os.path.exists(wav_path):
        print(f"  !! missing {wav_path}, skipping")
        continue

    # --- baseline: original single-stream transcription, no separation ---
    r = transcribe(wav_path)
    hyp_orig = " ".join(seg.get("text", "") for seg in r.get("segments", []))
    wer_orig_a = wer(clip_a["text"], hyp_orig)
    wer_orig_b = wer(clip_b["text"], hyp_orig)

    # --- separation ---
    est_sources = separator.separate_file(path=wav_path)  # (time, n_src)
    est_sources = est_sources.detach().cpu().numpy()
    sep_texts = []
    sep_wers_best = []
    for src_idx in range(est_sources.shape[-1]):
        track = est_sources[..., src_idx].squeeze()
        track = track / (np.max(np.abs(track)) + 1e-9) * 0.95
        out_path = f"scenario_c_separated/overlap_{overlap_pct}pct_src{src_idx}.wav"
        sf.write(out_path, track.astype(np.float32), 8000)  # sepformer-wsj02mix is 8kHz
        # upsample to 16kHz for crispasr (it was trained/tested at 16kHz throughout this investigation)
        import scipy.signal as sps
        track16k = sps.resample(track, int(len(track) * 16000 / 8000))
        out_path16k = f"scenario_c_separated/overlap_{overlap_pct}pct_src{src_idx}_16k.wav"
        sf.write(out_path16k, track16k.astype(np.float32), 16000)
        r_sep = transcribe(out_path16k)
        hyp_sep = " ".join(seg.get("text", "") for seg in r_sep.get("segments", []))
        w_a = wer(clip_a["text"], hyp_sep)
        w_b = wer(clip_b["text"], hyp_sep)
        sep_texts.append(hyp_sep)
        sep_wers_best.append(min(w_a, w_b))
        print(f"  [sep src {src_idx}] wer_vs_A={w_a:.2f} wer_vs_B={w_b:.2f}  text={hyp_sep[:80]!r}")

    # best pairing: does one separated source better match A and the other B?
    # (permutation-invariant -- SepFormer doesn't guarantee src0=A)
    results.append({
        "overlap_pct": overlap_pct,
        "orig_wer_vs_A": wer_orig_a,
        "orig_wer_vs_B": wer_orig_b,
        "orig_best_wer": min(wer_orig_a, wer_orig_b),
        "orig_hyp": hyp_orig[:150],
        "sep_texts": [t[:150] for t in sep_texts],
        "sep_best_wers": sep_wers_best,
        "sep_mean_best_wer": float(np.mean(sep_wers_best)) if sep_wers_best else None,
    })
    print(f"overlap={overlap_pct:3d}%  ORIGINAL best_wer={min(wer_orig_a, wer_orig_b):.2f}  "
          f"SEPARATED mean_best_wer={np.mean(sep_wers_best):.2f}\n")

json.dump(results, open("scenario_c_separated/results.json", "w"), indent=2)

print("\n=== SUMMARY: original (no separation) vs. SepFormer-separated ===")
print(f"{'overlap%':>9}{'orig best WER':>16}{'separated mean best WER':>26}")
for r in results:
    print(f"{r['overlap_pct']:>9}{r['orig_best_wer']:>16.2f}{r['sep_mean_best_wer']:>26.2f}")
