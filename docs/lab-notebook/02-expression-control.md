# 02 · Expression control: measuring what the face actually does

*The renderer accepts sliders. The question is whether moving one slider moves only the thing
you meant.*

Code: [`stack/advanced_live_portrait/`](../../stack/advanced_live_portrait/) (renderer),
[`stack/avatar/controls/`](../../stack/avatar/controls/) (control vocabulary + measurement),
[`stack/feedback/`](../../stack/feedback/) (MediaPipe observer),
[`stack/scripts/measure_controllability.py`](../../stack/scripts/measure_controllability.py)

## Architecture in one paragraph

Three things want to move the same face. **MuseTalk** owns the mouth (audio-driven lip sync).
**AdvancedLivePortrait (ALP)** owns lids, brows, horizontal gaze and sparse head events, via a
bank of pre-rendered expression slots. A **browser upper-body layer** owns roll, translation,
zoom and breathing. A small text classifier (MiniLM) *biases event planning* but does not drive
individual frames. A MediaPipe observer (478 landmarks) reads the *generated* frames back and
reports eye, brow, mouth and pose metrics at ~12.5 Hz (measured feedback latency:
mean 81.7 ms, range 63–107 ms, n=10).

Getting these to agree on who owns which degree of freedom was most of the work.

## The review that found three sign and ownership bugs

A read-only review of the running system (by a model, checked against source) found, before any
new motion was added:

- **Gaze/head sign conflict.** In the calibrated bank, `pupil_x` and `rotate_yaw` moved in
  *opposite* directions. Every "look left" also turned the head right.
- **`pupil_y` is not vertical gaze on this portrait.** It strongly changes lid aperture, brows,
  pitch and smile. It was removed from gaze anchors and a de-rolled, eye-corner-normalized gaze
  metric replaced it.
- **Roll had three owners** (ALP slots, browser autonomic pose, upper-body pose). Now one.

Verdict at the time: *no-go for adding new motion; go for a coherence slice.* The slice
(sign coherence, bank hygiene, de-rolled metrics, single roll owner) shipped with 76 passing tests
and a structural startup QA (face found, both-eye closure, gaze separation, mouth stability, yaw
and pitch metric separation all passed). A model-based visual pass was more conservative and
marked the tiny static head deltas `warn`. We kept the warning.

## Measurement first

Then the interesting part: an analysis core that, for each of 12 controls, renders a grid of
values through the resident renderer, reads the landmarks back, and reports

- a **noise floor** (the pipeline turned out to be *bit-deterministic*: std exactly 0.0 across
  repeated identical renders, so the floor had to be absolute, not multiplicative),
- **per-region leakage** (does the eye control move the mouth? does blink move the brows?),
- a **tiered safe envelope** (negligible / mild / severe leakage).

Two bugs in the analyzer itself, both caught before the reported run:

1. It counted a control's *intended* region as leakage, so every control looked terrible.
2. With a zero-tolerance bar and (1), all 12 envelopes collapsed to an identical `[0.0, 0.0]`.
   Fixed with home-region exclusion and three tiers instead of one binary bar.

Run: 3 identities, **306 samples, 0 failures, ~3 minutes**. Head pitch/yaw/roll came out the
narrowest envelopes (pitch stays `[0, 0]` even at the `severe` tier). `aaa`/`eee` mouth
shapes showed essentially no brow leakage across their whole range.

## The "compensation" that made it worse

Blink was known to raise the eyebrows, which reads as *startled*, not human. The deployed fix
was a brow counter-term (`blink=-14, eyebrow=-32`), documented as cancelling the lift.
Nobody had verified that. With the new measurement:

| configuration | worst-case brow leakage (max abs delta) |
|---|---:|
| no blink compensation, blink depth -14 | 0.070 – 0.250 |
| **deployed "compensation"** (brow -32) | **0.858** |
| full closure (blink -20), no brow term | **0.117** |

Sweeping the brow term from 0 to -40 at fixed blink depth made leakage larger *monotonically*:
the counter-term compounded the lift instead of cancelling it. Sweeping blink depth
(-14 … -20) against brow (0 / +3 / +5) put the minimum at full closure with no brow term
across the whole grid: a **~7.3x improvement**, and inside the "mild" tier. It was deployed live
and re-measured against the public endpoint: worst case 0.117, matching the prediction.

**What we did not measure:** how it reads perceptually. The fix is driven by the numeric
leakage metric, not a fresh human or model visual pass.

## Smaller things that cost days

- **Alpha-blending adjacent frames ghosted the mouth.** Sample-and-hold fixed it.
- **Browser-only head motion is only translate/skew/roll.** Convincing yaw and pitch needed real
  ALP anchors. Achieved separation: yaw 0.0254, pitch 0.0329 against mouth delta 0.0098.
- **Feedback metrics were contaminated by roll** until landmarks were de-rolled and iris height
  was normalized against the eye-corner line.

## Open

- **Reaction blend strength arrives ~10x weaker in the rendered frame** than commanded
  (issue #49). Open. The render-verification test that catches it is checked in.
- **Streamed frames arrive in a late burst** (3.84 s of empty polls, then 74 frames in 0.85 s;
  suspected GIL contention). Open.
- Combined-control samples (2–3 controls active at once) are recorded but were deliberately held
  out of the first report. Interactions between controls are not characterised.
- Only three portrait identities were measured.
