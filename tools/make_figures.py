"""Regenerate figures/ from the recorded result files. Needs matplotlib only."""
import csv, json, statistics as st
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EVAL, OUT = ROOT / "speech/eval", ROOT / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
C = {"base": "#8a8f98", "turbo": "#d9480f"}

# 1. noise robustness: WER vs SNR, per noise type, two model runs
def curve(path):
    agg = defaultdict(list)
    for r in json.load(open(path))["snr_curve"]:
        agg[(r["noise"], r["snr_db"])].append(r["wer"])
    return {k: st.mean(v) for k, v in agg.items()}
a, b = curve(EVAL / "results/scenario_a/results.json"), curve(EVAL / "results/scenario_a_turbo/results.json")
noises = sorted({k[0] for k in a}); snrs = [20, 10, 0, -5]
fig, ax = plt.subplots(1, len(noises), figsize=(9.5, 2.6), sharey=True)
for x, n in zip(ax, noises):
    x.plot(snrs, [a[(n, s)] for s in snrs], "o-", c=C["base"], label="baseline run")
    x.plot(snrs, [b[(n, s)] for s in snrs], "o-", c=C["turbo"], label="large-v3-turbo")
    x.axhline(1, c="k", lw=.6, ls=":"); x.set_title(n.replace("_", " "), fontsize=9)
    x.set_xticks(snrs); x.invert_xaxis(); x.set_xlabel("SNR (dB)")
ax[0].set_ylabel("mean WER (8 speakers)"); ax[0].legend(frameon=False, fontsize=8)
fig.suptitle("Competing speech breaks ASR; machine noise barely does (WER > 1 = hallucination)", fontsize=10, y=1.04)
fig.savefig(OUT / "asr_noise_curve.png", bbox_inches="tight"); plt.close(fig)

# 2. model bench: accuracy vs speed
q = json.load(open(EVAL / "quality_bench_results.json"))
fig, ax = plt.subplots(figsize=(5.2, 3.3))
for name, v in q.items():
    ax.scatter(v["realtime_factor"], v["mean_wer"], s=50, c=C["turbo"] if "turbo" in name else C["base"], zorder=3)
    ax.annotate(name, (v["realtime_factor"], v["mean_wer"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
ax.set_xlabel("batch throughput (x realtime, higher = faster)"); ax.set_ylabel("mean WER, 24 clean clips")
ax.set_title("large-v3-turbo beats large-v3 on both axes here", fontsize=9)
fig.savefig(OUT / "asr_model_tradeoff.png", bbox_inches="tight"); plt.close(fig)

# 3. soak: 307 turns on one persistent WebSocket
rows = [r for r in json.load(open(EVAL / "soak_test_results.json")) if "latency_s" in r]
summ = [r for r in json.load(open(EVAL / "soak_test_results.json")) if r.get("summary")][0]
t = [r["t"] / 60 for r in rows]
fig, ax = plt.subplots(2, 1, figsize=(6.2, 3.8), sharex=True)
ax[0].plot(t, [r["latency_s"] for r in rows], c=C["turbo"], lw=1.2); ax[0].set_ylabel("turn latency (s)")
ax[1].plot(t, [r["gpu_mem_mb"] / 1024 for r in rows], c=C["base"], lw=1.2); ax[1].set_ylabel("GPU mem (GiB)")
ax[1].set_xlabel("minutes")
ax[0].set_title(f"Soak: {summ['n_turns']} turns, {summ['n_timeouts']} timeouts, {summ['n_reconnects']} reconnects", fontsize=9)
fig.savefig(OUT / "asr_soak.png", bbox_inches="tight"); plt.close(fig)

# 4. perception latency vs the 40 ms (25 fps) frame budget
rows = list(csv.DictReader(open(ROOT / "data/vision_latency.csv")))
names = [r["layer"] for r in rows][::-1]
mid = [(float(r["mean_ms_low"]) + float(r["mean_ms_high"])) / 2 for r in rows][::-1]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.barh(names, mid, color=[C["turbo"] if m <= 40 else C["base"] for m in mid], zorder=3)
ax.axvline(40, c="k", ls="--", lw=1); ax.text(44, len(names) - 0.55, "25 fps budget (40 ms)", fontsize=8, va="bottom")
ax.set_xscale("log"); ax.set_xlabel("mean single-frame latency (ms, log)")
ax.set_title("Only face landmarks clearly fit a per-frame budget; VLMs are ~1 Hz signals", fontsize=9)
fig.savefig(OUT / "perception_latency.png", bbox_inches="tight"); plt.close(fig)
# 5. TTS -> STT closed loop: spectrograms, transcript, and the phantom trailing word
import wave
import numpy as np

def read_wav(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768.0, w.getframerate()

rt = json.load(open(ROOT / "media/audio/roundtrip.json"))
fig, axes = plt.subplots(len(rt["clips"]), 1, figsize=(7.2, 1.9 * len(rt["clips"])))
for ax, c in zip(axes, rt["clips"]):
    y, sr = read_wav(ROOT / "media/audio" / c["file"])
    ax.specgram(y, NFFT=1024, Fs=sr, noverlap=768, cmap="magma", vmin=-120, vmax=-40)
    ax.set_ylim(0, 8000); ax.set_yticks([0, 4000, 8000]); ax.set_yticklabels(["0", "4k", "8k"], fontsize=7)
    ax.grid(False); ax.set_ylabel(c["voice"], fontsize=8)
    ref = c["reference"].lower().replace(",", "").replace(".", "").split()
    hyp = c["hypothesis_normalized"].replace(",", "").replace(".", "").split()
    extra = hyp[-1] if hyp and hyp[-1].lower() not in ref else None
    ax.set_title(f'WER {c["wer"]:.3f}' + (f'   extra word never spoken, after the last word: "{extra}"' if extra else "   no errors"),
                 fontsize=8, loc="left", color=C["turbo"] if extra else "black")
axes[-1].set_xlabel("seconds")
fig.suptitle("Preset-voice TTS read back by the STT stack (mean WER %.3f): every error is one word the speaker never said" % rt["mean_wer"], fontsize=8.5, y=1.0)
fig.tight_layout(); fig.savefig(OUT / "tts_roundtrip.png", bbox_inches="tight"); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
