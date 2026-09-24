"""Room-acoustics realism check: every noise-robustness test so far used
additive noise mixing (real noise, real SNR, but still a direct-path clean
speech signal). Nothing tested what a real microphone actually captures --
speech convolved with a room's real reverberation. Real RIRs from MIT's
recorded reverberation survey (davidscripka/MIT_environmental_impulse_responses),
9 rooms spanning small/dry (bedroom) to large/reverberant (gym, stairwell,
parking lot, outdoor amphitheater)."""
import glob
import json

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from crisp_utils import wer
import quality_bench
quality_bench.SERVER = "http://127.0.0.1:18292"  # dedicated GPU-14 instance, avoids soak-test contention on GPU 11
from quality_bench import transcribe

CORPUS_META = "corpus/meta.json"
RIR_DIR = "rir"


def load_mono(path):
    arr, sr = sf.read(path, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return arr, sr


def convolve_rir(speech, rir):
    wet = fftconvolve(speech, rir)[: len(speech) + len(rir) - 1]
    # normalize to match the dry signal's peak so the model isn't just
    # reacting to loudness changes -- isolates the effect of reverberation
    # itself, not level
    peak_dry = np.max(np.abs(speech)) + 1e-9
    peak_wet = np.max(np.abs(wet)) + 1e-9
    wet = wet * (peak_dry / peak_wet) * 0.95
    return wet.astype(np.float32)


def main():
    meta = json.load(open(CORPUS_META))
    # 3 utterances, 3 different speakers, for the room sweep
    by_speaker = {}
    for m in meta:
        by_speaker.setdefault(m["speaker"], m)
    test_utts = list(by_speaker.values())[:3]

    rir_files = sorted(glob.glob(f"{RIR_DIR}/*.wav"))
    print(f"testing {len(test_utts)} utterances x {len(rir_files)} real rooms")

    results = {"dry_baseline": [], "room_sweep": []}

    print("\n=== dry (anechoic) baseline ===")
    for m in test_utts:
        r = transcribe(m["file"])
        hyp = " ".join(seg.get("text", "") for seg in r.get("segments", []))
        w = wer(m["text"], hyp)
        results["dry_baseline"].append({"speaker": m["speaker"], "wer": w, "hyp": hyp})
        print(f"  spk={m['speaker']:>6}  wer={w:.3f}")

    print("\n=== real-room reverberation sweep ===")
    import os
    os.makedirs("scenario_room", exist_ok=True)
    for m in test_utts:
        speech, sr = load_mono(m["file"])
        for rir_path in rir_files:
            rir, rir_sr = load_mono(rir_path)
            assert rir_sr == sr, f"sr mismatch: {rir_sr} vs {sr}"
            wet = convolve_rir(speech, rir)
            room_name = rir_path.split("/")[-1].replace(".wav", "")
            out_wav = f"scenario_room/{m['speaker']}_{room_name}.wav"
            sf.write(out_wav, wet, sr, subtype="PCM_16")
            r = transcribe(out_wav)
            hyp = " ".join(seg.get("text", "") for seg in r.get("segments", []))
            w = wer(m["text"], hyp)
            results["room_sweep"].append({"speaker": m["speaker"], "room": room_name, "wer": w, "hyp": hyp[:150]})
            print(f"  spk={m['speaker']:>6}  room={room_name:<45}  wer={w:.3f}")

    json.dump(results, open("scenario_room/results.json", "w"), indent=2)

    dry_wers = [r["wer"] for r in results["dry_baseline"]]
    print(f"\nDRY baseline mean WER: {np.mean(dry_wers):.3f}")
    for rir_path in rir_files:
        room_name = rir_path.split("/")[-1].replace(".wav", "")
        ws = [r["wer"] for r in results["room_sweep"] if r["room"] == room_name]
        print(f"  {room_name:<45}: mean WER {np.mean(ws):.3f}")


if __name__ == "__main__":
    main()
