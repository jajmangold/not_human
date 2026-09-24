# Automatic motion feedback

Every visual soak should retain the ordered output frames, `face_bbox.json`, the
controller events that produced them, and the audio clock.  The first deterministic
gate is:

```bash
python scripts/analyze_motion_feedback.py evidence/blink-run/frames --fps 16 --check
```

It measures the observed upper-face trajectory and rejects a blink when lower-face
motion is more than 18% of upper-face motion.  It also reports closure/reopening
timing and trajectory jerk.  This catches ownership and compositing regressions
without depending on a face detector.

The complementary feedback lane can run OpenFace over the same frame sequence and retain
its per-frame Action Units and pose next to controller telemetry.  Target AU45 for
blinks; AU01/02/04 for brows; AU12 for smile; and AU25/26 for mouth/jaw leakage.
Evaluate distributions and trajectories, not single magic values:

- closure is faster than reopening, with a brief rather than crushed peak;
- bilateral eye closure is normally coupled, while winks are explicit sparse events;
- inter-blink intervals use a refractory bounded renewal process, not a metronome;
- idle AU25/26 motion stays small, smooth, and independent of AU45;
- pose and expression velocity, acceleration, and jerk remain bounded;
- intended attack/hold/decay events align with observed AU onset/peak/offset.

Calibrate the target bands from consented conversational recordings and a licensed
spontaneous-expression corpus such as BP4D+, stratified by pose, speech state, and
expression.  Keep a periodic visual verdict because automatic AU detectors can
confuse neighboring actions.  A useful optimization record is one row per event:

`controller target -> rendered AU/landmarks -> timing/leak/jerk metrics -> visual verdict`

## Live controller

The Compose stack now runs a persistent native MediaPipe observer over the
generated canvas JPEG at 12.5 Hz. It records normalized left/right eye aperture,
iris gaze and vergence, bilateral eye/brow asymmetry, brow height, smile, smirk,
mouth aperture, face center, scale, yaw, pitch, and roll. Those observations
close three deliberately slow, bounded loops:

- completed idle blinks adjust duration and closure depth within narrow limits;
- breathing gain is adjusted only when shoulder response has enough visibility
  and correlation to be identifiable. Face-pose variance remains telemetry; it
  is not used as a noise-chasing gain loop.

The rolling telemetry is inspectable at `window.portraitTelemetry`, including
near-neutral occupancy and gaze/head and bilateral-brow correlations. Calibration is
session-local: it sends only the generated portrait to the same-origin backend;
it never sends camera frames or persists a person's biometric measurements.

## Startup calibration

Compose also starts a non-blocking `startup-qa` one-shot after MuseTalk,
AdvancedLivePortrait, and the MediaPipe face observer are healthy. It uses the
frequently exercised `portrait-d08aa48606ebf5e43a2a.png` from the shared runtime
as its fixed calibration subject. The live web service does not depend on this
job and remains available if the portrait or visual judge is unavailable.

The job renders neutral, left/right gaze, blink, amusement, surprise,
skepticism, concern, agreement, disagreement, interest, thinking, and isolated
left/right yaw plus up/down pitch with the
same silent audio and cached avatar. It retains raw ALP anchors, final MuseTalk
frames, neutral/control pairs, a labeled contact sheet, MediaPipe measurements,
lower-mouth pixel leakage for both gaze directions, normalized pose-mouth
stability, dedicated high-resolution yaw/pitch sheets, and the exact Qwen response
under `/io/web_demo/startup_qa/<UTC timestamp>/`. `latest.json` points to the
newest report. Override the subject or judge with `STARTUP_QA_PORTRAIT`,
`STARTUP_QA_VISION_BASE_URL`, and `STARTUP_QA_VISION_MODEL`.
The report separates structural failure from a visual `warn`: a Qwen aesthetic
finding is retained and made visible but never prevents the live service from
starting.

Automatic movement uses one persistent **semi-Markov** controller, not independent
blink, gaze, and pose timers. States carry explicit dwell distributions and choose
a permitted next state. Gaze events are placed first; blink scheduling suppresses
the roughly 480 ms pre-gaze interval and can probabilistically recruit a blink in
the slower head-follow portion. A three-factor correlated latent maps orienting,
engagement, and asymmetry into bounded pose residuals. This is where Markov
structure is useful: choosing behavior and dwell. It must not synthesize each
channel or frame independently.

MiniLM emits a low-amplitude, phrase-rate context bias (smile, smirk, brow, gaze,
nod, tilt, lean, and recoil). Its visible strength and residual mixtures are
deliberately attenuated so it cannot fire a posed expression directly. MuseTalk
remains authoritative over the mouth while
the strongest semantic primitive plus at most two attenuated residual primitives
select AdvancedLivePortrait expression sources.

Head motion has explicit ownership. AdvancedLivePortrait holds plus/minus four
degree yaw and plus/minus three degree pitch endpoints; sparse runtime events use
only a fraction of them, after eye acquisition, and return fully inside the
phrase. The posed face landmarks and latent reach MuseTalk so the speech mouth
stays registered to the rotated face. Backward optical flow carries the RGB
through the same geometry instead of cross-dissolving two displaced faces.
The browser no longer has a whole-canvas head-follow fallback: ALP owns rendered
face yaw/pitch through sparse Euler anchors, and the fitted upper-body layer is
the sole browser owner for neck, shoulders, torso, breathing, and roll/tilt.
Webcam mocap remains the gross tracked-pose override. All channels have hard
limits; there is no periodic semantic head sway.

Gaze is a fixation process, not continuous noise. Three mouth-stable ALP pupil-x
anchors represent left, camera contact, and right. The calibrated pupil-y
actuator is deliberately neutral in the live bank because it changes eyelid
aperture and brows on this identity rather than isolating vertical gaze. At idle, the browser
holds camera contact for irregular multi-second intervals, makes a fast
100--180 ms saccade into a short gaze aversion, then returns to camera. During
speech, a phrase-spanning scheduler produces the same move/hold/return events on
the audio clock and shares its event clock with blinks; it never resets at phrase boundaries or alternates sources on
a metronome. Pupil motion is composited after MuseTalk through a dedicated soft
orbital mask which reaches zero at half the detected face height. The lower
face is therefore copied exactly from MuseTalk. Gaze controls also never drive
browser whole-canvas translation or shear, so attention changes cannot move or
take ownership of the lips.

The compositor uses the same lower-face ownership rule for blinks: its vertical
feather reaches zero at 58% of the detected face height, leaving a hard mouth
boundary below that line. Completed stream jobs perform a final two-pass frame
drain after MuseTalk writes its `.done` marker. If a client disconnects, the web
service writes per-job cancellation markers; the resident renderer checks them
between batches and bounded session cleanup removes only transient audio, job,
and stream files. Shared portrait and motion-bank caches are retained.

## Distribution analysis

MediaPipe observation JSONL can be converted into an identity/session-specific
reference report without inventing universal expression amplitudes:

```bash
python scripts/analyze_landmark_series.py observations.jsonl --output motion-profile.json
```

Each line contains a timestamp and either the metric object itself or
`{"time": ..., "state": ..., "metrics": {...}}`. The report contains robust
amplitude percentiles, velocity/acceleration/jerk RMS, near-neutral occupancy,
state counts, and the strongest cross-channel correlations. Run it separately
for speaking and listening recordings before promoting a target profile.

References:

- OpenFace Action Units: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units
- BP4D-Spontaneous: https://doi.org/10.1016/j.imavis.2014.06.008
- High-speed blink kinematics: https://pmc.ncbi.nlm.nih.gov/articles/PMC4043155/
- Automated AU measurement limitations: https://pmc.ncbi.nlm.nih.gov/articles/PMC8235167/
- Naturalistic facial temporal states: https://elifesciences.org/articles/79581
- Conversational temporal order: https://www.nature.com/articles/s41598-025-30403-9
- Spontaneous versus deliberate motion: https://pmc.ncbi.nlm.nih.gov/articles/PMC2843933/
- Blink/head coupling (preprint): https://pmc.ncbi.nlm.nih.gov/articles/PMC13345164/
