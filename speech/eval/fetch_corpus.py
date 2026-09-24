"""Pull a diverse multi-speaker corpus (real ground-truth transcripts + speaker
IDs) from LibriSpeech test-clean, for building controlled adversarial audio
test scenarios against CrispASR."""
import io
import json
import os

import soundfile as sf
from datasets import Audio, load_dataset

OUT = "corpus"
os.makedirs(OUT, exist_ok=True)

ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

by_speaker: dict[str, list] = {}
meta = []

for ex in ds:
    spk = str(ex["speaker_id"])
    arr, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]))
    dur = len(arr) / sr
    # want short-ish, clean utterances (3-8s) for easy splicing
    if not (3.0 <= dur <= 8.0):
        continue
    by_speaker.setdefault(spk, [])
    if len(by_speaker[spk]) >= 3:
        continue
    idx = len(by_speaker[spk])
    fname = f"{OUT}/spk{spk}_{idx}.wav"
    sf.write(fname, arr, sr, subtype="PCM_16")
    by_speaker[spk].append(fname)
    meta.append({"speaker": spk, "file": fname, "text": ex["text"], "duration_s": dur, "sr": sr})
    if len({s for s in by_speaker if len(by_speaker[s]) >= 3}) >= 8:
        break

json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print(f"speakers: {len(by_speaker)}")
for spk, files in by_speaker.items():
    print(f"  {spk}: {len(files)} clips")
print("total clips:", len(meta))
