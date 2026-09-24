"""Shared utilities for the CrispASR adversarial test harness."""
import json
import os
import re
import subprocess

import numpy as np
import soundfile as sf

BIN = os.environ.get("CRISPASR_BIN", "crispasr")
# Configuration (all optional):
#   CRISPASR_BIN    path to the crispasr binary                     (default: `crispasr` on PATH)
#   CRISPASR_MODEL  ggml model used by the CLI-mode scripts         (default: models/ggml-base.en.bin)
#   CRISPASR_GPU    PCI-bus-order GPU index to run on               (default: 0)
#   NVIDIA_LIBS     extra colon-separated library dirs for CUDA     (default: none)
MODEL = os.path.abspath(os.environ.get("CRISPASR_MODEL", "models/ggml-base.en.bin"))
NVLIBS = os.environ.get("NVIDIA_LIBS", "")


def env():
    e = os.environ.copy()
    e["LD_LIBRARY_PATH"] = NVLIBS + ":" + e.get("LD_LIBRARY_PATH", "")
    e["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    e["CUDA_VISIBLE_DEVICES"] = os.environ.get("CRISPASR_GPU", "0")
    return e


def normalize_text(s: str) -> str:
    s = s.upper()
    s = re.sub(r"[^A-Z0-9' ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def wer(ref: str, hyp: str) -> float:
    import jiwer

    r, h = normalize_text(ref), normalize_text(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return jiwer.wer(r, h)


def run_batch(wav_path: str, extra_args: list[str] | None = None, timeout=120) -> dict:
    """Run crispasr in file mode, return parsed -ojf JSON output."""
    extra_args = extra_args or []
    json_path = wav_path + ".json"
    if os.path.exists(json_path):
        os.remove(json_path)
    cmd = [BIN, "-m", MODEL, "-f", wav_path, "-ojf"] + extra_args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env())
    if not os.path.exists(json_path):
        return {"error": proc.stderr[-3000:], "returncode": proc.returncode, "stdout": proc.stdout[-1000:]}
    d = json.load(open(json_path))
    d["_stderr_tail"] = proc.stderr[-500:]
    return d


def full_text(result: dict) -> str:
    return " ".join(seg.get("text", "") for seg in result.get("transcription", []))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Tile/crop noise to speech length, scale to hit target SNR (dB), mix."""
    if len(noise) < len(speech):
        reps = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[: len(speech)]
    speech_rms = np.sqrt(np.mean(speech.astype(np.float64) ** 2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise.astype(np.float64) ** 2) + 1e-12)
    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    noise_scaled = noise * (target_noise_rms / (noise_rms + 1e-12))
    mixed = speech.astype(np.float64) + noise_scaled
    peak = np.max(np.abs(mixed))
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    return mixed.astype(np.float32)


def load_wav_mono16k(path: str) -> tuple[np.ndarray, int]:
    arr, sr = sf.read(path, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample

        n = int(len(arr) * 16000 / sr)
        arr = resample(arr, n).astype(np.float32)
        sr = 16000
    return arr, sr


def save_wav(path: str, arr: np.ndarray, sr: int = 16000):
    sf.write(path, arr, sr, subtype="PCM_16")
