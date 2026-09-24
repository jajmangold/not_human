"""TTS -> STT closed loop: transcribe media/audio/*.wav with CrispASR and score against the
source text (same normalization and WER as speech/eval). Writes media/audio/roundtrip.json.

  CRISPASR_BIN=... CRISPASR_MODEL=.../ggml-large-v3-turbo.bin CRISPASR_GPU=9 python tools/roundtrip_wer.py
"""
import json, re, sys
from pathlib import Path

from num2words import num2words

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "speech/eval"))
import crisp_utils as cu  # noqa: E402

def spell_numbers(s: str) -> str:
    """Uniform normalizer applied to the STT output only: digits -> words, % -> percent.
    (The references are written the way they are spoken.) No other edits: extra words stay in the score."""
    s = re.sub(r"(-?\d+(?:\.\d+)?)\s*%", lambda m: m.group(1) + " percent", s)
    def num(m):
        x = m.group(0)
        return num2words(float(x) if "." in x else int(x)).replace("-", " ").replace(",", "")
    return re.sub(r"\d+(?:\.\d+)?", num, s)


AUDIO = ROOT / "media/audio"
manifest = json.loads((AUDIO / "manifest.json").read_text())
rows = []
for clip in manifest["clips"]:
    r = cu.run_batch(str(AUDIO / clip["file"]))
    if "transcription" not in r:  # fail loudly: an empty hypothesis must never become "WER 1.0"
        raise SystemExit(f"transcription failed for {clip['file']}: {str(r.get('error'))[:600]}")
    hyp = cu.full_text(r).strip() if "transcription" in r else ""
    rows.append({"file": clip["file"], "voice": clip["voice"], "wer": round(cu.wer(clip["text"], spell_numbers(hyp)), 4),
                 "reference": clip["text"], "hypothesis": hyp, "hypothesis_normalized": spell_numbers(hyp), "error": r.get("error")})
    print(f'{clip["voice"]:11s} WER {rows[-1]["wer"]:.3f}  {hyp[:70]}')
mean = sum(x["wer"] for x in rows) / len(rows)
out = {"stt_model": Path(cu.MODEL).name, "mean_wer": round(mean, 4), "clips": rows}
(AUDIO / "roundtrip.json").write_text(json.dumps(out, indent=2))
print("mean WER", out["mean_wer"])
