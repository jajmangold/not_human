# Known issues, gaps and corrections

Everything here is something we know is wrong, missing or unverified. If a claim elsewhere in the
repo conflicts with this page, this page wins.

## What has and has not been verified in *this* checkout

| check | result |
|---|---|
| `spec/` unit tests | 190 passed |
| `voice-agent/` unit tests (deps from `uv.lock`) | 147 passed |
| `stack/` unit tests | 111 passed, 22 skipped, 0 failed. The skips need live GPU services or fixtures |
| `tools/make_figures.py` | regenerates all four figures from recorded data |
| CrispASR patch | `git apply --check` succeeds against the pinned upstream commit |
| **Docker image builds** | **not run here.** CrispASR needs ~25 min and a CUDA host; lfm2vl builds llama.cpp from a fork |
| Live stack (`docker compose up`) | **not run here.** Assumes a multi-GPU host, see below |

The numbers in the lab notebook were measured on the original hardware at the time and come from
recorded result files or issue threads. They were not re-measured for this release.

## Broken or inconsistent

- **Kokoro "voice lab" route.** `web/app.py` calls `/voice-lab` on the Kokoro service. The last
  playground commit moved `docker-compose.yml` to the upstream Kokoro-FastAPI image, which has no
  such route; the custom `kokoro/app.py` that does is no longer what compose runs. The test that
  asserts the old wiring (`test_voice_lab_interpolates_and_persists_style_vectors`) is skipped with
  this explanation. Named voice-blend support was in a further branch that conflicted with the
  playground work and was **not merged**.
- **Avatar video in the LiveKit agent is off by default** (`MUSE_AVATAR_ENABLED=0`; upstream code
  defaulted to `1`). Two reasons: an unbounded-queue memory leak whose fix has not been confirmed
  in a live session, and a separate unsolved bug where the browser never rendered the avatar's
  tracks ([06](lab-notebook/06-voice-loop-and-bridge.md)).
- **Rendered reaction strength arrives ~10x weaker than commanded** (tracked upstream as #49).
  A render-verification test that catches it is checked in.
- **Streamed avatar frames arrive in a late burst** (3.84 s of empty polls, then 74 frames in
  0.85 s; suspected GIL contention; upstream #47).
- **25 fps native lip sync is not reached** on the long benchmark: 15.4 fps steady, 22.2 fps on
  primed 4-frame live batches ([05](lab-notebook/05-dispatch-bound.md)).

## Documentation errors we found and fixed

- Comments in `docker-compose.yml` and `crispasr-stt/Dockerfile` credited **0.211 WER to
  "small.en / CPU faster-whisper."** The recorded harness shows 0.211 for **base.en** on the same
  GPU server and 0.182 for small.en; the CPU faster-whisper setup was never measured in that
  harness. Comments corrected.
- The original notes label the baseline noise sweep both "small" and "base.en." The 8-clip
  baseline result file does not record which model produced it. Figures label it "baseline run."
- `spec` ADR-0001 and a pinned constant called the sm_70 target "Turing." sm_70 is Volta.
  Constant, test and ADR corrected (the ADR carries a correction note).
- The session notes state the authority-resolution rule as a formula
  (`resolved = commanded + …`). The code implements deterministic writer arbitration instead.
  `spec/README.md` describes the code.
- An earlier draft credited a cuDNN workaround with costing tensor-core throughput. The hardware
  has no working tensor cores ([05](lab-notebook/05-dispatch-bound.md)).

## Findings we retracted

1. "Finalization inconsistency" in the streaming server: a test-script truncation artifact.
2. "Named-speaker identification never fires": a grep for `speaker_db` against a log line spelled
   `speaker-db`.
3. "1.02x realtime": measured audio feed pacing, not compute.
4. "Headroom for concurrent sessions": wrong; requests serialize on one mutex
   ([01](lab-notebook/01-hearing-stack.md)).
5. "Crosstalk is an unfixable architectural limit": it is fixable with separation, with caveats.

## Harness portability

`speech/eval/` scripts read `CRISPASR_BIN`, `CRISPASR_MODEL`, `CRISPASR_GPU` and
`CRISPASR_MODEL_DIR`. They were made configurable during release prep: an earlier commit still
hardcoded the original scratch paths and GPU index. `gpu_stream_test.py` is Docker-specific and
kept as historical evidence only. The CLI binary needs CUDA runtime libraries on the host; the
round trip in `media/` was run inside a CrispASR container image for that reason.

## Not done

- **CrispASR patches are not upstreamed** and have no native regression tests.
- **The crosstalk separation gate is documented but not implemented**; separation is wired into
  nothing.
- **Speech-separation** is fixed at two sources.
- **Only three portrait identities** were used for controllability measurement; interactions
  between simultaneously-active controls are recorded but not analysed.
- **OCR correctness was never verified**; only latency was measured.
- **The `spec/` adapters and compositor were never built**; the running stack does not use the
  contracts.
- **Hardware assumptions.** Compose defaults to GPU index 0 for every service; the original
  deployment spread services over ~10 specific cards. On a single GPU you will hit the
  serialization and VRAM limits described in the notebook. All target architectures are sm_70.

## Licensing (see NOTICE, checked against upstream on 2026-09-24)

- **Two services pull in `ultralytics` (AGPL-3.0):** `stack/vision` and `stack/advanced_live_portrait`.
  Kept and labelled, not swapped ([vision notice](../stack/vision/AGPL_NOTICE.md)).
- **The ALP path is research-use-only:** the node has no license, and it loads InsightFace's
  `buffalo_l` models (non-commercial research only). See
  [its notice](../stack/advanced_live_portrait/LICENSE_NOTICE.md). Unresolved until the authors
  are asked.
- **Only the `crispasr-stt` and `lfm2vl` images are publishable**; the ALP and vision images are not.
- LFM2.5-VL uses the non-OSS `lfm1.0` license. ESC-50 is non-commercial. The MIT impulse-response
  dataset states no license.
- IDOL / SMPL-X are not included and are license-gated.
- The copyright line in `LICENSE` names the GitHub account (`jajmangold`). Replace it with a legal
  name if you want one before publication.

## Media

**Included:** [`media/audio/`](../media/audio/): four Kokoro preset-voice lines and their STT
round-trip scores. Provenance by construction (scripted text, named preset voice, hashed model).
The earlier playground/voice-lab WAVs were *not* used: that path mixes preset, clone and convert
modes with no per-file record, so a clone of someone's recording could be among them.

**Not included: face videos.** Candidates were reviewed frame by frame and held back:

| candidate | reason held back |
|---|---|
| expression-bank clips (10 portrait identities) | 6 of 10 are real people or stock photos, including personal family photos; the remaining portraits have no generation record |
| `cartoon_talk` | heavily degraded, blurry render; not representative of anything that worked |
| `idol_final_keep` (and the `idol_*` series) | needs the license-gated SMPL-X body model, which is not included |
| `greenman_talk15_tight`, `farmer_song_horse`, `pastor_altar_final` | photoreal faces whose source images could not be traced |

A face clip can go in as soon as its source portrait is provably synthetic (generation prompt or
job record kept alongside). Attestation from the owner is enough; the records on disk are not.

## Deliberately excluded

- Any media derived from real people, including personal photographs used during development.
- A face-swap dataset built from real people's photos.
- The JobHUD interview companion (separate project).
- Fleet automation for the original agent runners.
