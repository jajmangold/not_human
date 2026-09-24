# MuseTalk Volta resident

## Canonical control stream

The renderer-independent face representation lives under `avatar/`. It stores
MediaPipe's 52 ARKit-compatible blendshape scores, rigid head pose, confidence,
and timestamps in `controls.npz` with a versioned `metadata.json` sidecar.
Extract a still or video with:

```bash
PYTHONPATH=. python scripts/extract_face_controls.py \
  INPUT --model /path/to/face_landmarker.task --output controls --derived
```

See [docs/AVATAR-CONTROLS.md](docs/AVATAR-CONTROLS.md) for units and replay
conventions, and [docs/CONTROLLABILITY.md](docs/CONTROLLABILITY.md) for
measuring how ALP's sliders actually move that space and where they leak into
each other.

This repository is the focused Docker Compose restore of the Volta MuseTalk 1.5
resident service. It uses the archived `musetalk:v100` image and mounts the
archived speed patches; model weights and generated media stay outside Git.

## Bring it up

The Volta host must have the archived image loaded from
`musetalk_v100.tar` and the MuseTalk 1.5 model cache
under the archive drive. The first setup is:

```bash
cp .env.example .env
python3 scripts/preflight.py
docker load -i musetalk_v100.tar
docker compose up -d musetalk kokoro affect feedback upper-body web
docker compose logs -f --tail=50 musetalk kokoro affect feedback upper-body web
```

Wait for `MUSE_SERVER_READY` (also written to `/io/MUSE_SERVER_READY`). The
default GPU is Volta device 4; change
`MUSE_TALK_GPU_ID` in `.env` if that card is leased. `scripts/musetalk.sh up`
combines the preflight and start steps.

The face-detector checkpoint is cached separately at the path in
`MUSE_TALK_TORCH_CACHE`; preflight intentionally fails if it is absent instead
of allowing the container to download it during startup.

## Submit a job

Place a portrait image/video and WAV file below the configured I/O directory,
then submit container paths. The resident watches `/io/muse_jobs` and writes
`.done` or `.err` markers next to each job.

```bash
mkdir -p "$MUSE_TALK_IO_DIR/input" "$MUSE_TALK_IO_DIR/outputs"
python3 scripts/submit_job.py \
  --video /io/input/portrait.png \
  --audio /io/input/voice.wav \
  --output /io/outputs/portrait_talk.mp4 \
  --bbox-shift 0
```

The service uses the archived Volta optimizations: TAESD decode, optional
TAESD encode, no ping-pong frame preparation for equal-length jobs, threaded
mask preparation/blending, a fixed-shape CUDA graph for UNet plus TAESD, and a
responsive resident job-file loop. Set
`MUSE_USE_TAESD=0` if the cache does not contain the archived TAESD weights.
Set `MUSE_USE_CUDA_GRAPH=0` to roll back graph capture; it is enabled by default.

## Warm resident path

Avatar preparation is cached by portrait file and bounding-box settings. The
first job for a portrait pays face detection, parsing, and latent encoding; later
jobs reuse those materials and do not repeat that work. Keep `MUSE_CACHE_AVATARS=1`
(the default) for low-latency utterances. You can give a portrait an explicit
stable cache key with `--avatar-id` when submitting jobs:

```bash
python3 scripts/submit_job.py \
  --avatar-id horse-girl \
  --video /io/input/horse-girl.png \
  --audio /io/input/line.wav \
  --output /io/outputs/line.mp4
```

`MUSE_BATCH_SIZE` controls the warm UNet/VAE batch size. Tune it on the target
card; larger batches can improve throughput but increase first-frame latency and
VRAM use. This is the resident/prepared-avatar path described by upstream
MuseTalk; the file-job protocol still renders a complete MP4 rather than a live
socket stream.

Prepared avatar material is persisted in `MUSE_TALK_RESULTS_DIR`, so a resident
restart does not force face detection and VAE encoding again.

For frame-at-a-time delivery, enqueue short WAV chunks with
`scripts/submit_stream_chunk.py`. The resident writes frames into the requested
`/io` directory as soon as blending completes and creates a `.done` marker when
the chunk is finished:

```bash
python3 scripts/submit_stream_chunk.py \
  --avatar-id horse-girl \
  --video /io/input/horse-girl.png \
  --audio /io/input/chunks/0001.wav \
  --stream-dir /io/streams/horse-girl/0001 \
  --batch-size 4 \
  --stream-format jpg \
  --chunk-id horse-girl-0001
```

Set `MUSE_AUDIO_WARMUP=1` (default) so the first Whisper feature-extractor call is
paid during resident startup instead of on the first utterance. The chunk mode
is low-latency file delivery, not a WebSocket; a caller can watch the frame
directory and feed the PNGs to its own display/encoder.
For lower first-frame latency, use a smaller stream microbatch (the web demo
uses `--batch-size 4`); larger batches maximize
throughput but delay the first emitted frame.

## Talking-portrait web demo

The Compose stack also includes a browser demo at
`http://<host-or-tailscale-address>:${MUSE_DEMO_PORT:-8092}`. The default
`MUSE_DEMO_BIND=0.0.0.0` makes it reachable from the network; set it to
`127.0.0.1` when local-only access is preferred. Upload a PNG/JPEG/WebP portrait,
ask a question, and select a Kokoro voice. The only browser workflow is live:
the service streams an answer from `qwen27b` at
`http://localhost:8000/v1`, splits it into short spoken phrases, and
pipelines Kokoro synthesis with MuseTalk animation.

Portrait selection is the start of the session; no question is required yet.
The server immediately prepares the moving source and the browser plays one
random readiness gesture (`wink`, `nod`, `smile`, `surprise-wink`, or `kiss`).
It then uses only the retained expression bank for quiet eye motion and reactions
while waiting and between spoken phrases. The question controls remain disabled
until that live idle loop is ready. The WebSocket remains open after each answer,
so more questions reuse the already-warm portrait without another upload or ready
gesture. Ready frames come from the same cached ALP
generation as the MuseTalk source, so this does not add a second GPU pass.

Before MuseTalk prepares the avatar, the optional resident
AdvancedLivePortrait service generates a deterministic 28-frame source clip. The
source bank is encoded with lossless RGB H.264 (`libx264rgb`, CRF 0) and a PNG
archive; this avoids chroma ringing or JPEG edges becoming part of later eye,
blink, and mouth composites. The encoder revision is included in the cache key,
so older lossy motion banks are never reused.
The first eight frames carry the visible ready gesture. The retained 20 frames
form a random-access bank with quiet eye motion, a blink, and bounded two-level
amusement, surprise, skepticism, concern, agreement, disagreement, interest,
thinking, plus real left/right yaw and up/down pitch controls. While waiting,
MuseTalk expands only the quiet slots and
one blink into a four-second idle source; the browser treats the open-eye
frames as a monotonic rest axis and injects fast blinks from a bounded
log-normal renewal schedule with a hard refractory period. During
speech, MuseTalk continuously interpolates the bank on a phrase-local
attack/hold/decay trajectory, then remains the final audio-driven mouth stage. This
ordering matters: applying LivePortrait after MuseTalk would repaint the mouth
and weaken lip synchronization. The loop is cached by portrait bytes, settings,
ready gesture, and pinned upstream revision. MuseTalk deliberately caches only
the expression-bank portion of the clip, and every speaking job uses that exact moving
source path; it never falls back to reopening the still upload.
Set `ADVANCED_LIVE_PORTRAIT_ENABLED=0` to use the original still-portrait path.
The web service returns HTTP 503 from `/api/healthz` until every required
resident and sidecar is healthy, rate-limits session creation and feedback
uploads, and expires abandoned sessions after `MUSE_DEMO_SESSION_TTL` seconds.
The limiter is intentionally in-process because the Compose web service runs a
single Uvicorn worker; a multi-worker deployment should move that state to a
shared store.
For a deployment outside the private tailnet, set `MUSE_DEMO_AUTH_TOKEN` in the
workspace environment; clients must send it as a bearer token.

A CPU-only `affect` service uses the already-local, checksum-pinned
`sentence-transformers/all-MiniLM-L6-v2` ONNX model as a semantic sampler. It
samples once per completed Qwen phrase, not once per token. Lightweight lexical
cues disambiguate cases such as "no, that is incorrect" where a generic sentence
embedding is not a trained emotion classifier. Kokoro WAV energy, onset, and
pause ratio are sampled separately and fused into a maximum 0.72 expression
strength of 0.68. Repeated labels are attenuated, and every trajectory has bounded
attack, hold, decay, and cooldown metadata. The browser exposes the same
trajectory against `AudioContext.currentTime`; Web Audio remains the master
clock, optional webcam MediaPipe owns gross pose, AdvancedLivePortrait owns the upper
face, and MuseTalk owns the mouth. No Whisper pass is needed for the avatar's own
speech because the streamed Qwen text is already available.

The adapter builds from
[PowerHouseMan/ComfyUI-AdvancedLivePortrait](https://github.com/PowerHouseMan/ComfyUI-AdvancedLivePortrait)
revision `3bba732915e22f18af0d221b9c5c282990181f1b`; upstream code is
not vendored here. The upstream revision references a `LICENSE` in
`pyproject.toml` but does not contain that file, so redistribution terms are
not asserted by this repository. Put its five Kijai LivePortrait safetensors
under `$ADVANCED_LIVE_PORTRAIT_MODEL_DIR/liveportrait/` and
`face_yolov8n.pt` under `$ADVANCED_LIVE_PORTRAIT_MODEL_DIR/ultralytics/`.
Weights and generated motion clips remain external to Git.
The deployed Kijai model revision is
`59f30f36d7b791929c25437df7461d5b0e0010b1`; SHA-256 values are:
`38bef5de50a92bf1fc66e8c511051a19dfacdf80c37f8713425ec15dc9ca7d34`
(appearance),
`3568cd410e29d046771acb55ecfdfe4c7c197d345bd8b7f95942ef63130b6c9e`
(motion),
`f7b7834bd6039b4088f72e5161e60ad366f68a3763df8a3eac0bc0f9d46fdbbf`
(warping),
`ca04fbec765745e9eae836d2d7522c274647b277ce5f25104fa1705b75222212`
(SPADE), and
`60725cf3523ae413880da28ed583c9e84c5c25695a0cef1b77210ea31cc424ea`
(stitching). The face detector is Bingsu/adetailer revision
`53cc19de382014514d9d4038601d261a7faa9b7b`,
`70b640f8f60b1cf0dcc72f30caf3da9495eb2fb6509da48c53374ad6806e6a9c`.
The service performs a synthetic 28-frame CUDA warmup before becoming healthy
and holds about 1.94 GiB VRAM. The semantic-bank-v4 warmup measured 16.56
seconds. A cold portrait reached ready in 21.64 seconds, including 16-frame
MuseTalk preprocessing and the 64-frame silent prime. The stage is
therefore resident and cache-oriented, not
claimed as native per-frame real-time inference on the flashed CMP card.
The active `semantic-bank-v13-blink-brow-fix` profile keeps the same model and
memory shape, gives the mouth one stable mask during silence, and reserves a
fully closed-eye source for an independent blink lane. Relative to v11, its
gaze/head anchors are sign-coherent, `pupil_y` is neutral because the
calibrated actuator couples into eyelids and brows on this identity, and ALP
roll is neutral so the browser upper-body controller is the sole roll owner.
Relative to v12, the closed-eye slot uses full closure (`blink=-20`, not a
partial `-14`) with no eyebrow counter-term: measured against 3 identities
(`runtime/evidence/issue-30/blink-brow-fix/`), every negative eyebrow value
made the brow-down leak larger, monotonically, rather than canceling it --
full closure alone measured lowest (worst-case brow delta 0.117 vs 0.858 for
the previous `-14`/`-32` pairing). Runtime events use only
the measured live-safe ranges (pupil-x +/-5, yaw/pitch +/-4/+/-3, blink -14 to
+6, eyebrow -8..+5, smile <=0.7), follow selected gaze changes after the eyes
acquire their target, and complete a smooth return inside the phrase instead of
looping as sway. The rotated RGB frame, landmarks, and latent all condition
MuseTalk, which still repaints the mouth last. RGB pose interpolation uses
backward optical-flow warping rather than dissolving two displaced faces. Gaze
is then limited to a soft orbital mask whose lower edge is hard-zero before the
mouth; blink keeps its separate upper-face mask.

Kokoro uses the full FP32 `onnx/model.onnx` export and the aggregated
`voices.npz` archive. Keep those external to Git under `KOKORO_MODEL_DIR`; the
assets are from [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX).
This is intentional on the dual E5-2650L v3 host: it has AVX2 but no native
FP16 arithmetic, so the smaller q8f16 export took about 1.9 seconds per second
of audio. FP32 with a bounded 12-thread ONNX Runtime pool measured about 0.37
seconds per second of audio. `KOKORO_INTRA_OP_THREADS` and
`KOKORO_INTER_OP_THREADS` expose those measured defaults.
The browser receives WAV and JPEG payloads directly on one binary WebSocket; it
does not fetch a file for every frame. A Web Audio clock controls frame
presentation. Kokoro audio plays directly through Web Audio at its native rate;
it is never routed through a video element or time-stretched to follow late
frames. The visible canvas is the canonical display surface. MuseTalk generates
mouth updates at 16 fps by default; speech frames use webcam-style
sample-and-hold on that clock and crossfade the final MuseTalk frame into a
slowly varying closed-mouth rest state instead of snapping across phrase gaps.
A bounded log-normal renewal
process schedules independent fast blinks, while a critically damped controller
moves among quiet eye/brow/rest-mouth targets without repeating a fixed loop.
Google MediaPipe Face Landmarker 1.0.1 runs in a browser worker for webcam
pose only. The checksum-pinned native MediaPipe 0.10.35 `feedback` service
observes generated canvas JPEGs at a backpressured 12.5 Hz and returns eye,
brow, mouth, smile/smirk, and pose metrics. It never receives camera frames.
Those measurements apply slow bounded blink-depth/duration and motion-amplitude
corrections; Web Audio remains the direct and monotonic lip-frame clock. Webcam
mocap is requested automatically after portrait upload; its button remains as
a manual enable/disable fallback. It adds smoothed, bounded head translation,
roll, scale, and pose shear to the entire composed portrait at display cadence.
Camera frames are processed locally and never sent
over the WebSocket; webcam mouth movement is deliberately ignored so it cannot
fight MuseTalk lip sync. Camera access requires HTTPS or localhost.

An optional CPU `upper-body` sidecar fits a MediaPipe pose rig once when the
portrait is uploaded, then returns audio-clocked phrase plans for breathing,
sternum, shoulder, torso, neck, and bounded gross-head motion. The browser
applies those plans with a small piecewise-affine mesh before its existing
whole-frame pose transform. Respiration uses variable inhale, exhale, and rest
states; there is no periodic sway or bob oscillator. Generated-canvas pose
observations close the loop at the same backpressured feedback cadence used for
the face. MuseTalk remains authoritative for the mouth and AdvancedLivePortrait
for upper-face residuals. Set `UPPER_BODY_ENABLED=0` to bypass preparation,
planning, feedback, and rendering without disabling the talking portrait.
Provenance, archived IDOL measurements, GUAVA qualification status, ownership,
and rollback details are in [`docs/UPPER-BODY.md`](docs/UPPER-BODY.md).

The Face Landmarker task file is external to Git at
`$MEDIAPIPE_FACE_LANDMARKER_DIR/face_landmarker.task`. The pinned Google
float16/1 artifact is 3,758,596 bytes with SHA-256
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
The Docker build pins the Apache-2.0 `@mediapipe/tasks-vision` package to 1.0.1
through `package-lock.json`; its JavaScript and WASM are served by this app
rather than a runtime CDN.
The four-frame microbatch is deliberately latency-oriented. A silent four-frame inference
microbatch also warms the avatar while Qwen and Kokoro are working, so model
warmup does not sit in front of the first visible frame.

On the flashed CMP 100-210 card, a primed four-frame live batch currently
generates at about 19-22 fps, with headroom over the demo's 16 fps mouth-update
clock. `MUSE_JOB_POLL_INTERVAL=0.01` prevents the resident from inserting the
old one-second pause between phrase jobs. Native 25 fps mouth inference still
requires moving the convolution-heavy UNet from FP16 HMMA to W8A8 DP4A; that
kernel work is tracked as the W8A8 DP4A conv work (see docs/lab-notebook/05-volta-hardware.md).

With the semantic source interpolator enabled, the final two-turn browser gate
measured 18.7-20.6 MuseTalk fps with 60 ms blend-drain backlog against the 16
fps actuator clock. MiniLM classification measured 59-64 ms per phrase on CPU.

Portrait preparation starts as soon as the upload completes and overlaps Qwen
generation. Qwen token ingestion, Kokoro synthesis, and MuseTalk rendering use
bounded queues, so a slow stage applies backpressure rather than building an
ever-older slideshow. Kokoro still requires a complete short phrase; it is not
a sample-level streaming TTS engine.

## Provenance

The mounted files come from the `vrm_automation` commit `16075bf`
(`muse_patch: MuseTalk speed optimizations (~3x) + launch script`). The model
UNet checksum used by the dinner qualification is
`7ebf6c98c181e20838e4c0054e96e944ac60d5d692cc01db42839fe11b787007`.
# Kokoro Playground

Open `/playground` in the same deployment as `/voice`. The lab keeps the live
portrait conversation route separate and provides three deliberately comparable
paths: native Kokoro preset/mixed voices, KokoClone text-to-clone, and Kanade
audio-to-audio re-voicing. Each take exposes normalized text, duration, model,
and settings and can be saved into a browser-local take deck for A/B listening.

Gemma 4 26B is an optional speech director. It rewrites text into clean spoken
copy and returns a visible delivery note; the note is never sent to Kokoro as
if it were supported emotion markup. Kokoro's practical controls are wording,
punctuation, voice choice/mixing, and speed. The `kokoro_hack` PSO style-vector
experiment is intentionally not silently enabled: it requires a separate
PyTorch/Kokoro checkpoint and emotion encoder, so it should be added only as a
measured optional backend with neutral-vs-steered evidence.

The Voice Lab exposes KokoVoiceLab-style vector exploration: choose a source
and target preset, use `-1` / `+1` to reach either endpoint, `0` for the
midpoint, and `±2` for restrained extrapolation. Named vectors beginning with
`custom_` persist under `KOKORO_CUSTOM_VOICE_DIR` and appear in the voice list
after the service reloads.

KokoClone is built from the pinned upstream revision in `kokoclone/Dockerfile`.
Its Kanade and auxiliary Hugging Face assets are downloaded lazily into the
external `KOKOCLONE_MODEL_DIR` / cache paths on the first clone request. If
those assets or the service are unavailable, native Kokoro and LiveKit remain
healthy and the UI reports cloning as optional/warming.

Useful references: [KokoClone](https://github.com/Ashish-Patnaik/kokoclone),
[kokoro_hack](https://github.com/eryawww/kokoro_hack), and the existing
[Kokoro-ONNX model](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX).
