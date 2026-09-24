# Upper-body live portrait stage

This stage adds restrained respiration and conversational torso/shoulder motion
without changing the mouth renderer. It is enabled by default in Compose and can
be bypassed with `UPPER_BODY_ENABLED=0`.

## Runtime contract

One component owns each degree of freedom:

| Region | Authority |
|---|---|
| Mouth interior, lip contour, speech jaw | MuseTalk |
| Blink, eyes, brows, cheeks, facial residuals, sparse semantic yaw/pitch | AdvancedLivePortrait |
| Torso, shoulders, sternum, neck, subtle roll/tilt | upper-body controller; browser webcam pose overrides gross head pose when present |
| Timing | Web Audio monotonic clock |

On upload, the CPU sidecar fits MediaPipe shoulder, sternum, neck, and hip
anchors. The browser deforms a 5-by-6 piecewise-affine mesh at display cadence,
then applies the existing whole-frame head transform. Since the mouth and upper
face are already in the source bitmap, this single coherent warp does not repaint
or independently translate either region. Each destination triangle clip
overlaps its neighbors by 0.85 pixels, and an opaque copy of the source frame
underlays the mesh. This prevents independently antialiased triangle edges from
exposing dark grid seams without changing the affine mapping.

Respiration is a variable-duration inhale/exhale/rest state machine. A critically
damped second-order controller approaches each respiratory and phrase-level pose
target. Speech smoothly turns an in-progress inhale toward exhalation rather than
resetting phase. There is deliberately no periodic sway, bob, or random walk.
The existing generated-canvas feedback loop observes face and pose concurrently
at 12.5 Hz with backpressure; a failed optional pose observation does not discard
the face observation or stop the session.

The ALP random-access bank also carries real Euler yaw and pitch endpoints.
Selected gaze shifts can recruit one slower head-follow event, and semantic nods
can recruit one down-and-return pitch event. Those events transform the face
landmarks and latent before MuseTalk repaints the mouth, while backward optical
flow carries the final RGB through one spatially coherent pose. The client has
no whole-canvas head-follow fallback: ALP owns rendered face yaw/pitch and the
fitted browser upper-body controller owns roll, neck, torso, and breathing. This
prevents two controllers from fighting over the same degree of freedom.

## Assets and provenance

| Asset | Revision / checksum | Terms and handling |
|---|---|---|
| MediaPipe Pose Landmarker Lite float16/1 | SHA-256 `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` | Google MediaPipe model, external read-only mount; Apache-2.0 project |
| Archived IDOL source | `yiyuzhuang/IDOL` at `9fd9296c28e8f8f9ed5f5c594f3df1574b8ec82d`, with recorded local studio patches | The pinned tree contains no root license file; do not redistribute it or its checkpoint from this repository |
| Archived IDOL checkpoint | `<archive>/studio/splat/IDOL/work_dirs/ckpt/model.ckpt`, 8,832,048,103 bytes | External archive only; no weights in Git |
| Archived IDOL environment | `<archive>/idol_volta.tar`; archive SHA-256 `cd44a8aea2e5fb9129e4bd181faec4f561795cc8efbf1b934d6add1fd3916a69`; image ID `sha256:e2e301975b129e6fc9b74e23d641ff5119c2b6cdf492bcb86320cd10012012f5` | PyTorch 2.3.1, CUDA 11.8, sm_70 extensions; external archive only |
| GUAVA source | `Pixel-Talk/GUAVA` at `356067ad0e68521277f5b3a174944b8ea9a92874` | Apache-2.0 source; not vendored |

The inventory wiki was checked before model retrieval. Targeted, shallow searches
of the archived studio and the known the archive drive model, habitat, motion, and model
cache roots found no raw `SMPLX_NEUTRAL_2020.npz` or FLAME 2020 model. GUAVA
requires those separately licensed assets, so its weights/runtime have not been
promoted or represented as locally qualified. This is an artifact-rights gate,
not a CUDA limitation: upstream states compute capability 7.0 and at least 6 GB
VRAM, which includes these cards.

## Recovered systems and benchmark decision

The archived studio already had useful SMPL-X work:

- `greenman/build_motion.py` maps spine, neck, head, and collar joints and adds
  breathing, speech neck motion, torso follow, and weight shift. Its joint
  ownership is retained; its open-loop sine timing is not.
- `splat_charfull.py` contains K-nearest-triangle deformation and true linear
  blend skinning.
- IDOL's resident server caches the model and one encoded avatar. Archived job
  receipts cover 8 to 1,160 frames. Representative 374-frame runs took 52.1,
  55.7, 55.8, 73.7, and 127.1 seconds: 7.18, 6.71, 6.70, 5.07, and 2.94 fps.
  The quickest archived result remains far below a webcam-rate display loop.
- Known archived failure modes were underside-neck gaps, identity drift at wider
  pose, composite zoom drift, and mouth damage from a second face-swap/restoration
  pass. Those are why the live stage keeps MuseTalk's composited frame intact.

| Candidate | Avatar init | Render rate | VRAM | Identity / stability | Decision |
|---|---:|---:|---:|---|---|
| Archived IDOL resident | Model load plus per-identity encode; cached thereafter | 2.94-7.18 fps in archived receipts | Single 16 GB card | Proven full upper body, but archived neck/face/composite defects | Recovered as comparison and offline proof, not live default |
| GUAVA upstream | Claims sub-second avatar construction | Claims real time | Upstream says >=6 GB | Promising explicit expressive upper body; local identity/stability unmeasured without licensed SMPL-X/FLAME inputs | Gate pending assets; no invented result |
| Current fitted mesh | One CPU pose fit on upload | 58.23 display fps; 0.459 ms mean warp cost | No added GPU residency | Preserves the exact MuseTalk/ALP bitmap; bounded deformation and measured pose feedback | Live default |

The current stage is intentionally a fast deformation/controller layer, not a
claim that GUAVA or IDOL has been reimplemented. A future native GUAVA sidecar may
replace the renderer behind the same ownership and clock contract after the
licensed inputs are supplied and an sm_70 A/B clears identity, neck stability,
VRAM, cold-init, and sustained-frame-rate gates.

### Deterministic browser calibration

Append `?motion-calibration=1` to the demo URL to expose
`window.setPortraitMotionCalibration(state)` for automated browser QA. Supported
states are `neutral`, `inhale`, `phrase-left`, `nod`, and `live`. The calibration
freezes all other motion sources, reports fitted-mesh displacement in pixels via
`window.portraitTelemetry.motionCalibration.geometry`, and restores the complete
pre-calibration state when returned to `live`. The API is absent without the
query flag and does not add a user-facing animation control.

The collapsed **Motion values and feedback** panel in the demo is the native
debug surface for tuning this pipeline. After a run, **Copy values** returns one
JSON snapshot containing the ALP-style expression vocabulary (pitch, yaw, roll,
blink, eyebrow, wink, pupil, AAA/EEE/WOO, and smile), the actual MiniLM control
values and reaction timing, scheduled gaze/blink/head events, upper-body pose and
respiration, plus the latest MediaPipe face and shoulder summaries. It does not
copy or persist camera frames. The same static control schema is available at
`/api/motion/controls`, and `window.copyPortraitMotionValues()` returns the exact
JSON string for automated browser capture.

For direct expression tuning, open `/motion-editor`. It is a small native editor,
not a copy of the upstream Gradio UI: upload one image, move the ALP controls,
and the browser debounces changes into `POST /api/motion/edit`. The resident GPU
sidecar renders one PNG per latest slider state at `/edit`; the browser never
queues an unbounded slider storm, and the same JSON control values can be copied
back into a tuning request. This endpoint is an image calibration tool only; it
does not replace the live MuseTalk mouth stream.

The fixed 416-by-512 portrait calibration found the original full-inhale mesh
displacement was only 0.794 px. The tuned profile measures 1.408 px for inhale,
3.917 px for the bounded phrase-left endpoint, and 1.970 px for the nod endpoint;
all existing pose clamps remain unchanged. Pair-specific RTX0 Qwen review found
all three motions visible, subtle rather than exaggerated, identity-stable, and
free of grid/distortion artifacts. Captures are outside Git under
`web_demo/motion_qa/calibration-v2/{baseline,tuned}` in the durable Compose IO
root.

## Live qualification

Evidence is stored outside Git under
`runtime/evidence/issue-23/live-upper-body-v1/` in the durable MuseTalk runtime.
The known-good portrait produced 184/184 idle face and body observations, then
61/61 observations across a three-sentence streamed turn. The session returned
to `Alive and listening` with no browser errors. Idle shoulder-span standard
deviation was 0.006586 and shoulder-center-Y standard deviation was 0.004096.
Six 100-frame mesh samples averaged 0.459 ms per frame; headless Chrome sustained
58.23 requestAnimationFrame fps. The resident RTX0 Qwen visual gate reported no
mesh seam, neck/shoulder discontinuity, duplicate mouth, or layout regression in
idle and speaking captures.

A follow-up seam regression used a high-contrast 512-by-512 synthetic warp. The
original exact-edge clips produced 7,122 partially transparent pixels with
minimum alpha 156; overlapped clips plus the opaque underlay produced zero
partial or transparent pixels and minimum alpha 255. A fresh live portrait then
produced 102/102 face and body observations with no browser errors. RTX0 Qwen
reported no horizontal, vertical, or diagonal mesh lines, mouth duplication, or
neck/shoulder tearing in `runtime/evidence/issue-23/grid-seam-fix/idle-after.png`.

The archived IDOL image import check confirms PyTorch 2.3.1/CUDA 11.8 and working
`pytorch3d`, `simple_knn`, and Gaussian-rasterizer imports. A fresh replay from
that image, the read-only source/checkpoint archive, and `static_talk.npy`
produced eight 512-by-512 frames on GPU 14. Its measured render loop was 1.50
seconds, or 5.33 fps; the complete post-model-load portrait pass was 21.46
seconds. RTX0 Qwen visual QA found a coherent clothed upper body with no visible
face, neck, shoulder, torso, duplicate-mouth, or gross-rendering defect in the
sampled frame. The replay also exposed two unpinned secondary downloads (VGG16
and rembg U2Net) into the disposable container, so the baseline is not yet an
offline-hermetic service.

A rebuild from its
historical untracked Dockerfile is not reproducible as written: unpinned
`bitsandbytes` currently replaces the base with PyTorch 2.14/CUDA 13, and native
extension failures are hidden by pipelines ending in `tail`. Use the preserved
image for recovery until a separately governed pinned rebuild removes that
mutable dependency and enables `pipefail`.

## Rollback

Set `UPPER_BODY_ENABLED=0` and recreate only `web`. The sidecar may remain healthy
in the Compose project but the web app will skip prepare, plan, observation, and
mesh rendering. This leaves Qwen, Kokoro, MuseTalk, AdvancedLivePortrait, facial
feedback, and the durable avatar/runtime cache unchanged.
