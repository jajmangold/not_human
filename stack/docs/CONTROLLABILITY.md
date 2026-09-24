# ALP controllability and cross-coupling measurement (#30)

`scripts/measure_controllability.py` measures how AdvancedLivePortrait's 12
sliders actually move the canonical control space from `docs/AVATAR-CONTROLS.
md`, on real identities, using the resident ALP endpoint and the canonical
`FaceLandmarkerExtractor` (#28) -- not a derived summary metric.

## What it samples, per identity

- **Repeated neutral baseline** (`--neutral-repeats`, default 6): all twelve
  controls at zero, rendered more than once. This measures render/detection
  noise. Measured empirically: the resident ALP + MediaPipe pipeline is
  **exactly deterministic** for identical (portrait, controls) input --
  repeat renders are bit-for-bit identical, so real repeat std is `0.0`, not
  just small. Because of that, `avatar.controls.controllability.
  region_leakage` combines the (here, zero) statistical noise floor with a
  fixed `absolute_floor` (default `0.02` on the `[0, 1]` blendshape scale) so
  a purely multiplicative noise threshold doesn't call every nonzero delta
  "leakage."
- **Named six-point sweep per control** (`avatar.controls.calibration.
  named_grid`): min, q1, default, mid, q3, max -- one control at a time, the
  other eleven held at their own default (0.0), matching the vocabulary of
  the prior manual slider-matrix QA run.
- **Bounded combined controls** (`--combined-samples`, default 24): 2-3
  controls simultaneously active, each drawn from the middle half of its own
  range (not its extremes). These are held out from the per-control
  leakage/envelope analysis below -- they exist for #31's adapter evaluation,
  not to be mixed into the single-axis measurement.

Every attempted render is recorded; a render or detection failure is written
to `failures.json`, never silently dropped.

## What the report says

For each identity: a noise floor, and per control: mouth-region leakage
(`eye_to_mouth_leakage`, only for eye/brow-region controls: `pupil_x`,
`pupil_y`, `blink`, `wink`, `eyebrow`) and brow-region leakage
(`blink_to_brow_leakage`, for `blink` specifically) beyond the noise floor,
plus a per-identity `safe_envelope_by_tolerance`: the widest span around the
control's default where cross-region coupling stays under a given magnitude,
at three tiers -- `negligible` (<=0.05 on the `[0, 1]` blendshape scale),
`mild` (<=0.15), `severe` (<=0.30).

Each control's own intended blendshape target is excluded from its own
cross-coupling severity (`avatar.controls.controllability.
CROSS_COUPLING_REGIONS`) -- e.g. `eyebrow` moving brow channels is its job,
not leakage; `smile`/`aaa`/`eee`/`woo` moving mouth channels is their job.
This mapping is our own judgment call from the sliders' documented semantics,
not independently verified ground truth; each control's raw
`mouth_leakage_by_point`/`brow_leakage_by_point` (with `max_abs_delta` per
point, home-region inclusion notwithstanding) are still reported in full, so a
different judgment call can be re-applied to the same data without
rerendering. `top_response_channels_at_strongest_point` is the home-region-
agnostic empirical counterpart: whichever canonical channels actually moved
most, no assumption about which one "should" respond.

**Why tiers, not one binary envelope:** an earlier zero-tolerance ("no
channel moves at all") version of this envelope collapsed to `[0.0, 0.0]` for
every one of the 12 controls, including ones with no plausible mechanism for
strong coupling (e.g. `rotate_yaw`) -- because *some* nonzero cross-region
delta shows up by the innermost sampled point (q1/q3) for essentially every
control, at magnitudes far below what a person would call a real effect. A
single global zero-tolerance bar carries no differentiating information once
every control fails it identically; the three named magnitude tiers do.

**The envelope is still only as precise as the six sampled points.** A tier
can only resolve to the nearest sampled point on each side, not a
continuously interpolated boundary; a control whose `mild` envelope is
`[0.0, 0.0]` needs finer sampling between `default` and `q1`/`q3` to find its
true boundary, not a claim that no nonzero value clears that tier. Finer
sampling for controls that hit this floor is a natural, explicitly-scoped
follow-up, not done in this slice (the issue's own guidance is to start with
hundreds of samples, not preemptively densify).

`cross_identity_safe_envelope` is, per tolerance tier, the intersection
(tightest, most conservative) of every identity's own envelope for a control
-- the range every measured identity agrees clears that tier.

## Held-out data for #31

Every identity's `controls.npz`/`samples.jsonl` records the bounded combined-
control samples (`control: "combined"`) alongside the single-axis sweep, but
this report's per-control analysis never reads them -- they exist for #31's
adapter evaluation. When #31 trains a canonical-controls-to-ALP adapter, treat
the combined-control rows across all identities, and at least one identity's
single-axis rows entirely, as held-out evaluation data rather than folding
them into training.

## Running it

ALP is not published to the host; run inside `musetalk-feedback` attached to
the compose network:

```bash
docker run --rm --network musetalk-volta_default \
  -v ./models/mediapipe-face-landmarker:/models:ro \
  -v /path/to/three/portraits:/img:ro \
  -v "$PWD":/workspace:ro -w /workspace -e PYTHONPATH=/workspace \
  -v /path/to/output:/out \
  musetalk-feedback:local python3 scripts/measure_controllability.py \
    /img/portrait-a.png /img/portrait-b.png /img/portrait-c.png \
    --model /models/face_landmarker.task \
    --alp-url http://advanced-live-portrait:8093 \
    --output /out
```

`--only <control>` (repeatable) restricts the sweep to specific controls for
a fast smoke run. `--qwen` additionally runs one Qwen visual review per
control per identity against `LLM_BASE_URL`/`LLM_MODEL`/`LLM_API_KEY` --
best-effort and supplementary; the numeric report above is the required
deliverable and does not depend on it. Keep `--output` under runtime storage,
not this repo.
