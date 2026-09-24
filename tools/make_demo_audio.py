"""Synthesize the demo lines with Kokoro *preset* voices only, and write a manifest.

Provenance is by construction: fixed text, named preset voice, recorded model hash. No reference
recordings, no cloning, no real-person audio. Needs `pip install kokoro-onnx soundfile` and the
Kokoro-82M v1.0 ONNX model + voices.npz (see NOTICE for licenses).

  KOKORO_DIR=/path/to/kokoro-82m-v1.0-onnx python tools/make_demo_audio.py
"""
import hashlib, json, os
from pathlib import Path
import soundfile as sf
from kokoro_onnx import Kokoro

ROOT = Path(__file__).resolve().parent.parent
KDIR = Path(os.environ["KOKORO_DIR"])
MODEL, VOICES = KDIR / "onnx/model.onnx", KDIR / "voices.npz"
OUT = ROOT / "media/audio"; OUT.mkdir(parents=True, exist_ok=True)

LINES = [
    ("af_heart", "en-us", "noise-vs-voices",
     "Six talkers babbling at zero decibels push the recognizer past one hundred percent word error rate. "
     "A siren at minus five decibels barely moves it."),
    ("am_michael", "en-us", "brow-compensation",
     "The brow compensation made things worse. Leakage went from zero point one one seven to zero point eight five eight, "
     "and it took a measurement to notice."),
    ("bf_emma", "en-gb", "launch-bound",
     "Ninety four percent of the object detector's latency was not the graphics card computing. "
     "It was the graphics card being asked to launch five hundred and thirty kernels for every frame."),
    ("bm_george", "en-gb", "open-bug",
     "One bug is still open. The browser never rendered the avatar's video track, "
     "and we wrote down the next thing to try."),
]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
tts = Kokoro(str(MODEL), str(VOICES))
manifest = {"model": "Kokoro-82M v1.0 ONNX (fp32)", "model_sha256": sha(MODEL), "voices_sha256": sha(VOICES),
            "voice_type": "preset (no cloning, no reference audio)", "speed": 1.0, "clips": []}
for voice, lang, slug, text in LINES:
    samples, sr = tts.create(text, voice=voice, speed=1.0, lang=lang)
    name = f"{slug}__{voice}.wav"
    sf.write(OUT / name, samples, sr, subtype="PCM_16")
    manifest["clips"].append({"file": name, "voice": voice, "lang": lang, "text": text,
                              "seconds": round(len(samples) / sr, 2), "sample_rate": sr})
    print(name, manifest["clips"][-1]["seconds"], "s")
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
