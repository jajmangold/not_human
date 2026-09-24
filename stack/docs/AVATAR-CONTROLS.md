# Canonical avatar controls

The avatar stack exchanges renderer-independent face controls through the
`musetalk-avatar-controls/v1` format. `controls.npz` contains one row per
timestamp and `metadata.json` binds the column order and units.

The 52 blendshape columns use MediaPipe Face Landmarker's ARKit-compatible
names and float32 scores in `[0, 1]`. `head_rotation` stores XYZ Euler angles in
radians. `head_translation` and the optional `head_transform` preserve the raw
MediaPipe facial transformation matrix. Timestamps are float64 seconds from
the first frame and must be strictly increasing.

**The pinned MediaPipe model's real output is not BLENDSHAPE_NAMES verbatim.**
Verified empirically against `face_landmarker.task` (mediapipe 0.10.35): the
model returns 52 categories, but category 0 is `_neutral` (a synthetic
"how neutral does this face look" score, not an ARKit shape) and `tongueOut`
is never emitted at all — there's no tongue-visibility signal in a monocular
RGB face crop. `NEUTRAL_CATEGORY_NAME` names the former; it's captured
separately as the optional `neutral_score` array, not mixed into the 52-column
canonical array. `FaceLandmarkerExtractor.UNSUPPORTED_BLENDSHAPES` names the
latter — currently `{"tongueOut"}` — and every extraction/calibration script
records it as `unmeasured_blendshapes` in `metadata.json`. Treat a `0.0` in an
unmeasured column as "never measured," not "measured closed"; the extractor
raises rather than silently drops any category it doesn't recognize, so a
future model swap that changes this set is caught immediately.

**Confidence is detection validity, not a graded score.** MediaPipe Face
Landmarker's IMAGE-mode Python API exposes no continuous per-detection
confidence — every landmark's `presence`/`visibility` is `None` for this task
(checked empirically) — so `confidence` from this extractor is only ever 1.0
(a face was found) or 0.0 (it wasn't). `metadata.json` records this as
`confidence_semantics: "binary_detection_validity"`. A different extractor
with a real graded confidence should say so explicitly in its own metadata
rather than let this default be assumed.

**Camera axes and units** (Google's MediaPipe Face Geometry module —
[3D Face Transform](https://developers.googleblog.com/mediapipe-3d-face-transform/)):
`head_translation`/`head_transform` live in a right-handed, orthonormal metric
coordinate space with a virtual perspective camera at the origin, looking down
`-Z` (so a face in front of the camera has negative Z); the canonical face
model's metric unit is the centimeter. This is the module's own documented
convention, not independently re-measured here — and its *absolute* magnitude
reflects the module's built-in default virtual-camera field of view, not a
calibrated measurement of true physical camera distance, so don't treat a
translation value as ground-truth depth without separate calibration. Relative
motion between frames of the same portrait/session is far more trustworthy
than any single frame's absolute translation.

Extract a still or video with:

```bash
PYTHONPATH=. python scripts/extract_face_controls.py \
  /path/to/portrait-or-video.png \
  --model /path/to/face_landmarker.task \
  --output controls \
  --derived
```

The extractor keeps raw measurements. Velocity and acceleration are optional
finite differences stored as separate arrays; filtering belongs in a later
replay/controller stage so it cannot overwrite the measurement baseline.

To measure the current ALP slider implementation against this canonical space,
run the calibration helper against the resident ALP endpoint:

```bash
PYTHONPATH=. python scripts/calibrate_liveportrait.py \
  /path/to/portrait.png --model /path/to/face_landmarker.task \
  --alp-url http://127.0.0.1:8093 --output /runtime/alp-calibration \
  --points 5 --random-samples 16 --keep-frames
```

The output contains the measured `controls.npz`, `metadata.json`, and
`samples.jsonl`, which pairs every requested ALP vector with the extracted
blendshapes, pose, raw head transform, and neutral score. Keep the output
under runtime storage.

**Calibration samples are not a temporal sequence.** Each row is an
independently requested, arbitrary slider setting — consecutive rows can
belong to unrelated sweeps — so the runner writes `temporal_sequence: false`
and never sets `include_derived`. Computing velocity/acceleration across
these samples' synthetic ordinal timestamps would manufacture fictitious
motion at every sweep/control boundary rather than measure anything real.
Only genuine video extraction (`extract_face_controls.py` on a video input)
has real inter-frame spacing and may enable `--derived`.

AdvancedLivePortrait slider values never appear in this format. Its adapter will
consume the canonical packet and report its own calibration and controllability
metrics. MuseTalk remains an audio-conditioned mouth renderer until a later
closed-loop experiment proves a unified renderer improves lip sync.

No code from QtMeshEditor (#869/#909) has been ported into this stack; if a
future slice proposes reusing it, review its license and reproduce parity
independently first rather than accepting its issue text as proof.
