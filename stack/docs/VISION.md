# Live perception for the LiveKit voice agent

Background: the 2026-09-09 nothuman vision-stack spike
(docs/lab-notebook/03-perception-latency.md)
benchmarked candidate real-time perception layers against recorded clips
(no camera exists on that host). The `vision` (and, as of 2026-09-10,
`lfm2vl`) services are where those findings became a real, running
pipeline for internal issue #45's LiveKit agent -- specifically for
seeing the *human participant's own camera*, not the avatar's driving
frames that `feedback`/`upper-body` already analyze for MuseTalk
rendering.

## Architecture (updated 2026-09-10)

Three small vision/language models, split by what each actually measured
well at -- not one model doing everything:

| Model | Service | Used for | Measured |
|---|---|---|---|
| LiquidAI/LFM2.5-VL-450M | `lfm2vl` | ambient caption, on-demand targeted visual question | ~150-185ms/request, correct on every caption/count/description test tried |
| LiquidAI/LFM2.5-VL-3B | `lfm2vl-narrate` | background scene-narrative consolidation (~20s cadence) | same latency class as the 450M model for short answers (~162-183ms); a detailed 4-sentence description measured ~1.4s, meaningfully more accurate/detailed than the 450M model attempted |
| MiniCPM-V-4.6 | `vision` (in-process) | STT-artifact classification (text only) | ~500ms/request; LFM2.5-VL tested unreliable here (see below) |

`vision` also owns YOLO26n (object detection, in-process) and proxies to
the already-running `feedback`/`upper-body` MediaPipe services for
face/pose, unchanged from the original design.

**`POST /perceive`** (on `vision`): one JPEG frame in, aggregated result
out -- `objects` (YOLO26n), `face`/`pose` (proxied), `caption` (proxied to
`lfm2vl`). `objects` is also surfaced directly to gemma4-26b (see "Trust
calibration" below), not just consumed internally.

**`POST /query`** (on `vision`, proxied to `lfm2vl`): `{question}` +
one JPEG frame -> a targeted answer to that specific question, run fresh
against the current frame. Exists because a generic rolling caption
cannot answer something specific ("how many fingers am I holding up") --
found live asking exactly that and getting "there's no camera" instead of
an attempt, because nothing told the model finger-counting was the actual
question.

**`POST /narrate`** (on `vision`, proxied to `lfm2vl-narrate`): current
JPEG frame + `raw_log` (agent.py's rolling changelog of distinct ambient
captions) + `object_timeline` (YOLO per-label presence over time) -> one
consolidated narrative sentence. Added 2026-09-10 per "is there anything
we can do with yolo+the running log to boost temporal coherence... maybe
something running in the background cleaning up the storyline" -- the raw
per-tick changelog reads as a list of disconnected snapshots, not a
storyline, and the model driving the actual conversation (also a small,
weak model) doesn't reliably synthesize one from it turn to turn. A
background task on a slow (~20s) cadence does that consolidation once,
so the conversational model gets a coherent sentence instead. Uses the
LARGER 3B model, not the 450M one every other endpoint here uses --
deliberately, so it can look at the actual current frame and verify/
correct against it rather than only trusting whatever the smaller,
faster, less accurate model already wrote into the log. Not on any
conversational hot path -- `agent.py`'s `_run_scene_narrator` background
task calls this, never gating a turn -- so it can afford the extra
latency of a bigger model for meaningfully better output. Falls back to
the raw changelog in `agent.py`'s `_scene_note()` whenever the narrative
hasn't been refreshed recently (narrator hasn't run yet, or
`lfm2vl-narrate` is down).

**`POST /classify_transcript`** (on `vision`, MiniCPM-V in-process):
`{text}` -> `{is_real_speech}`. Used by `agent.py`'s
`on_user_turn_completed` as an STT-artifact filter before transcripts
reach gemma4-26b -- log-only as of 2026-09-10, see "Known limitations"
below.

## Trust calibration: gemma4-26b must not treat the 450M model as ground truth

Found live (2026-09-10): shown a toy parrot wearing a tiny pirate hat,
`lfm2vl` correctly identified "parrot" but wrongly attributed the hat to
the person holding it, not the parrot -- and gemma4-26b repeated that
wrong detail as fact instead of treating it as one small model's possibly-
wrong read. `agent.py`'s `VISION_CAPABILITY_CLAUSE` now explicitly
distinguishes "the camera feed is real" (always true) from "every detail
the small vision model claims is true" (not true -- it can confidently
state wrong specifics), and instructs the model to cross-check
specific/detailed claims against YOLO's object list. YOLO's detections
(previously computed by `/perceive` but never surfaced to gemma at all)
are now rendered via `agent.py`'s `_render_yolo_objects()` and appended
to both the ambient and targeted notes as `"YOLO detected: ..."` -- a
second, independent signal the model can weigh a specific claim against,
rather than just repeating whatever the 450M model said.

## Why LFM2.5-VL-450M replaced MiniCPM-V for the image-based calls

Real numbers, measured 2026-09-10 on this fleet's hardware, chasing "can
the finger-counting feature get under 1 second":

| Path | Single-request latency | Notes |
|---|---:|---|
| MiniCPM-V-4.6, `transformers` | ~939ms-3.1s | the original implementation |
| MiniCPM-V-4.6, GGUF/llama-server (Vulkan) | ~1.0s | doesn't batch under concurrent load (spike finding) |
| LFM2.5-VL-450M-Extract, `transformers` | ~976ms | wrong on free-text description (trained for structured extraction, not captioning) |
| LFM2.5-VL-450M (base), `transformers` | ~939ms | correct captions, but still not under 1s |
| LFM2.5-VL-450M (base), GGUF/llama.cpp (mainline) | crash / ~0.2-0.5 tok/s | see below -- not a config mistake, a real upstream gap |
| **LFM2.5-VL-450M (base), GGUF/llama.cpp (fixed fork)** | **~150-185ms** | **what's deployed** |

The GGUF/llama.cpp path for LFM2.5-VL only works with an **unmerged**
upstream PR applied -- see `../lfm2vl/vendor/README.md` for the full
story (mainline segfaults on two different release builds; the official
`server-cuda` Docker image loads the model but processes prompts at a
pathological ~0.2-0.5 tokens/second without the fix). This is a real,
confirmed gap in llama.cpp's LFM2 support, not a local misconfiguration --
verified against llama.cpp's own supported-multimodal-models
documentation (LFM2/LFM2.5-VL isn't listed) and the actual open PR
(ggml-org/llama.cpp#25524, opened 2026-07-10, still unmerged as of this
writing).

**Why MiniCPM-V stayed for `classify_transcript`**: LFM2.5-VL-450M was
also tested against the exact proven classification prompt used there,
and answered `ARTIFACT` for every test transcript tried, including
obviously real text ("What time is it?"). Not a like-for-like
replacement for that specific task -- kept MiniCPM-V there rather than
regress a working feature chasing a speed win that didn't apply.

## Setting it up

Three model sources this stack needs that nothing else in this repo
already provides:

1. **YOLO26n weights** (`yolo26n.pt`): download via `ultralytics` once
   (`python3 -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"`) and
   place at `${YOLO_MODEL_DIR:-./models/yolo26n}/yolo26n.pt`.
   No SHA256 pin yet, unlike every other model in this compose file --
   worth adding before this goes further than a first deploy.
2. **MiniCPM-V-4.6** (`openbmb/MiniCPM-V-4.6`, `transformers` checkpoint,
   Apache-2.0): pulled automatically by `transformers` on first container
   start into the `VISION_HF_CACHE_DIR` volume. Budget time for that on
   the very first `docker compose up vision`.
3. **LFM2.5-VL-450M** (`LiquidAI/LFM2.5-VL-450M-GGUF`, LFM1.0 license):
   two files, place both at
   `${LFM2VL_MODEL_DIR:-./models/lfm2vl}/`:
   - `LFM2.5-VL-450M-Q8_0.gguf` (main model, 379,219,104 bytes,
     sha256 `263aca93039e22140d55e046831c700c796affa8143d7638581c488a30c712bc`)
   - `mmproj-LFM2.5-VL-450m-F32.gguf` (vision projector, 376,952,256 bytes,
     sha256 `418afaca450bb30715c38c394a059ab2b1d46c1678c4c43f48aa9569fa00a5b5`)
4. **LFM2.5-VL-3B** (`LiquidAI/LFM2.5-VL-3B-GGUF`, LFM1.0 license, for
   `lfm2vl-narrate` -- same image/binary as `lfm2vl`, just bigger weights):
   two files, place both at
   `${LFM2VL_NARRATE_MODEL_DIR:-./models/lfm2vl-3b}/`:
   - `LFM2.5-VL-3B-Q4_K_M.gguf` (main model, 1,674,455,072 bytes,
     sha256 `2436cf4bbac9a16e5dfc7799a140e583c2bc15958f4b24a391f8a8b10ecb888`)
   - `mmproj-LFM2.5-VL-3B-Q8_0.gguf` (vision projector, 583,109,984 bytes,
     sha256 `ecbbe7097f696dba67172738d79c9f01132cdb6c0b457606315e268df3d67e6`)
   Uses the same fixed llama.cpp fork as the 450M model -- confirmed
   working with zero code changes, just different mounted weight files
   via `docker-compose.yml`'s `command:` override.

GPU: `lfm2vl` / `vision` / `lfm2vl-narrate` default to `${LFM2VL_GPU_ID:-6}`
/ `${VISION_GPU_ID:-11}` / `${LFM2VL_NARRATE_GPU_ID:-8}` respectively --
`lfm2vl-narrate` deliberately NOT co-located with `lfm2vl` (GPU 6) even
though combined footprint would likely fit -- narration isn't on the
conversational hot path, but it's real GPU compute that would otherwise
compete with `lfm2vl`'s ambient-caption calls, which ARE latency-sensitive
(see `DEFAULT_FRAME_INTERVAL_S`'s history in `agent/vision_client.py`).
Combined footprint (YOLO26n + MiniCPM-V + LFM2.5-VL-450M) measured
comfortably under 16GB in this session's testing (peak ~6.3GB when
co-located on one card during verification), so sharing one card between
`lfm2vl`/`vision` is a reasonable override if a dedicated second card
isn't available -- just keep `lfm2vl-narrate` on its own card. Re-check
idleness before first deploy (`nvidia-smi --query-compute-apps`) -- fleet
usage shifts over time.

## Known limitations, carried over from the spike

- **No camera anywhere in this fleet to test against live** -- every
  number in the table above comes from replaying recorded clips or
  single extracted frames, not a real video call, EXCEPT the
  finger-counting feature itself, which has been exercised against a
  real webcam live (the original motivating bug report and its fix).
- **MiniCPM-V's batching requires uniform frame shape** -- mixing
  differently-sized images in one batch crashes `vit_merger`'s reshape
  (found running this work's own smoke test). No longer relevant to the
  image-based calls (moved to LFM2.5-VL, which doesn't do client-side
  batching -- `llama-server`'s own slot system handles concurrency), but
  still true of MiniCPM-V's remaining `classify_transcript` role if that
  code is ever extended to accept images again.
- **The STT-artifact classifier is not reliable enough for enforcement**:
  a longer live conversation (2026-09-10) found a 67% false-positive rate
  on the turns it actually flagged for dropping, including clearly real,
  substantive content ("Talk like a futuristic cyborg robot from now
  on.", "No toothbrush, just the pliers."). `VISION_STT_FILTER_ENFORCE`
  defaults to log-only (`0`) as a result -- verdicts are still logged for
  future prompt tuning, but never act on a turn. Given a bare transcript
  with no conversation context, it does correctly flag context-free
  fragments/noise ("mmm-hmm", a cough), but calling standalone "Thank
  you."/"bye" real speech is reasonable in isolation and those specific
  hallucination shapes are already handled upstream by `agent.py`'s VAD
  tuning anyway. See `minicpmv_runtime.py`'s `CLASSIFY_PROMPT_TEMPLATE`
  comment for the v1/v2 prompt history. Don't flip enforcement back on
  without a fresh before/after test against a live session.
- **Real webcam frames must be resized before encoding, not just the
  small test images latency was originally tuned against**: found live
  (2026-09-10) after pushing ambient cadence to 500ms -- real frames were
  going out at native camera resolution (confirmed 1280x720, unresized),
  costing ~788-806 tokens/request and 370-450ms against `lfm2vl` instead
  of the ~150-185ms measured against small (384x216) test images, since
  `lfm2vl`'s SigLIP2-NaFlex vision encoder scales token count with input
  resolution. Fixed with a bounded-longest-edge resize
  (`MAX_FRAME_EDGE_PX = 640` in `agent/vision_client.py`'s
  `frame_to_jpeg()`) before JPEG encoding -- verified against the real
  frame that caught the bug: `/query` latency 578ms -> 274ms (~2.1x
  faster), identical caption/answer content before and after. Ambient
  cadence itself was reverted from 500ms back to 1s pending a fresh live
  re-test now that frames are properly bounded -- the resize fix wasn't
  assumed to make 500ms safe again without re-measuring.
- **`lfm2vl`'s vendored llama.cpp build depends on an unmerged upstream
  PR** -- see `../lfm2vl/vendor/README.md` for exactly what to do once
  ggml-org/llama.cpp#25524 merges (re-vendor from a mainline release
  instead of the fork). `lfm2vl-narrate` shares the same vendored build
  (same image, different mounted weights), so the same note applies to
  both.
