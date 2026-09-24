"""Pull real labeled noise clips (ESC-50, crowd/TV/appliance categories most
relevant to the 'background noise, TV' scenario from musetalk-volta#27) for
controlled-SNR adversarial mixing."""
import io
import json
import os

import soundfile as sf
from datasets import Audio, load_dataset

OUT = "noise"
os.makedirs(OUT, exist_ok=True)

ds = load_dataset("ashraq/esc50", split="train")
ds = ds.cast_column("audio", Audio(decode=False))
wanted = {"crowd", "clapping", "footsteps", "vacuum_cleaner", "washing_machine", "engine", "car_horn", "siren"}
meta = []
seen_cats = set()
for ex in ds:
    cat = ex["category"]
    if cat not in wanted:
        continue
    if cat in seen_cats:
        continue
    arr, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]))
    fname = f"{OUT}/{cat}.wav"
    sf.write(fname, arr, sr, subtype="PCM_16")
    meta.append({"category": cat, "file": fname, "sr": sr, "duration_s": len(arr) / sr})
    seen_cats.add(cat)
    if seen_cats >= wanted:
        break

json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print("noise categories fetched:", sorted(seen_cats))
