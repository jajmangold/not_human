"""Generate index.html (the GitHub Pages player) from the media manifests.

GitHub's Markdown strips <video>/<audio>, so playable media lives on this page; the .md files embed
GIFs/figures inline and link here for sound. Run: python tools/make_media_page.py
"""
import html, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
audio = json.loads((ROOT / "media/audio/manifest.json").read_text())
rt = {c["file"]: c for c in json.loads((ROOT / "media/audio/roundtrip.json").read_text())["clips"]}
expr = json.loads((ROOT / "media/video/expressions/manifest.json").read_text())
e = html.escape

speech = "\n".join(
    f'''<figure><figcaption><b>{e(c["voice"])}</b> · {c["seconds"]} s · STT WER {rt[c["file"]]["wer"]:.3f}</figcaption>
<audio controls preload="none" src="media/audio/{e(c["file"])}"></audio>
<p>{e(c["text"])}</p><p class="hyp">heard: {e(rt[c["file"]]["hypothesis"])}</p></figure>''' for c in audio["clips"])

by = {}
for c in expr["clips"]:
    by.setdefault(c["portrait"], []).append(c)
gallery = ""
for p, cs in sorted(by.items()):
    gallery += f"<h3 id='{e(p)}'>{e(p)}</h3><div class='grid'>" + "".join(
        f'''<figure><video controls loop muted playsinline preload="metadata" src="media/video/expressions/{e(c["file"])}"></video>
<figcaption>{e(c["expression"])} <small>({e(c["bank_profile"])}, {e(c["fps"])} fps)</small></figcaption></figure>''' for c in cs) + "</div>"

page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>not_human — media</title>
<style>
:root{{--bg:#fff;--fg:#1f2328;--mut:#59636e;--line:#d1d9e0;--acc:#d9480f}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0d1117;--fg:#e6edf3;--mut:#9198a1;--line:#30363d;--acc:#ff8a4c}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:920px;margin:0 auto;padding:24px 16px 64px}}
h1{{margin:.2em 0}} h2{{margin-top:2em;border-bottom:1px solid var(--line);padding-bottom:.2em}}
a{{color:var(--acc)}} .mut,figcaption small,.hyp{{color:var(--mut)}} .hyp{{font-size:.9em;margin-top:-.6em}}
figure{{margin:1em 0}} audio{{width:100%}} video{{width:100%;border-radius:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}} .grid figure{{margin:0}}
img{{max-width:100%}}
</style></head><body><main>
<h1>not_human — media</h1>
<p class="mut">GitHub's README can't play audio or video, so it lives here. Back to the
<a href="https://github.com/jajmangold/not_human">repository</a>. Everything below is generated from the manifests in <code>media/</code>.</p>

<h2 id="speech">Speech: preset-voice TTS, read back by the STT stack</h2>
<p>Kokoro <b>preset</b> voices only (no cloning), then transcribed by the same STT service.
Mean WER {json.loads((ROOT / "media/audio/roundtrip.json").read_text())["mean_wer"]:.3f}; every error is one extra word the
speaker never said at the very end. n = 4 clips: an illustration, not a benchmark.</p>
{speech}
<p><a href="figures/tts_roundtrip.png"><img alt="spectrograms with transcripts" src="figures/tts_roundtrip.png"></a></p>

<h2 id="expressions">Expression bank renders</h2>
<p>Silent, deliberately subtle (reaction strength is capped). The three portraits are
<b>AI-generated according to the repository owner</b>; the repo holds no generation record for them.</p>
{gallery}

<h2 id="a2v">Audio-to-video sample (with sound)</h2>
<p>An LTX-2.3 output: an AI-generated dinner scene with a synthetic voice. An illustration of the
approach in <a href="https://github.com/jajmangold/not_human/blob/main/docs/lab-notebook/04-audio-to-video-ltx.md">notebook 04</a>,
<b>not</b> evidence that its lip sync worked.</p>
<video controls playsinline preload="metadata" src="media/video/ltx-a2v-dinner-scene.mp4" style="max-width:512px"></video>
<p class="mut">LTX-2 Community License applies to the model; see NOTICE.</p>
</main></body></html>
"""
(ROOT / "index.html").write_text(page)
(ROOT / ".nojekyll").write_text("")
print("wrote index.html", len(page), "bytes")
