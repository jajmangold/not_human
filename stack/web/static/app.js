const form = document.querySelector('#live-form');
const portraitInput = document.querySelector('#portrait');
const promptInput = document.querySelector('#prompt');
const voiceInput = document.querySelector('#voice');
const speedInput = document.querySelector('#speed');
const goLive = document.querySelector('#go-live');
const formStatus = document.querySelector('#form-status');
const streamStatus = document.querySelector('#stream-status');
const affectStatus = document.querySelector('#affect-status');
const answer = document.querySelector('#answer');
const motionValues = document.querySelector('#motion-values');
const copyMotionValues = document.querySelector('#copy-motion-values');
const downloadMotionValues = document.querySelector('#download-motion-values');
const motionValuesStatus = document.querySelector('#motion-values-status');
const canvas = document.querySelector('#live-canvas');
const mocapCamera = document.querySelector('#mocap-camera');
const mocapToggle = document.querySelector('#mocap-toggle');
const mocapStatus = document.querySelector('#mocap-status');
const context = canvas.getContext('2d', { alpha: false });
const blinkLayer = document.createElement('canvas');
const blinkContext = blinkLayer.getContext('2d');
const upperBodyLayer = document.createElement('canvas');
const upperBodyContext = upperBodyLayer.getContext('2d');
const feedbackFaceLayer = document.createElement('canvas');
const feedbackFaceContext = feedbackFaceLayer.getContext('2d', { alpha: false });
const HEADER_BYTES = 9, AUDIO_PACKET = 1, FRAME_PACKET = 2, READY_FRAME_PACKET = 3, IDLE_FRAME_PACKET = 4;
const chunks = new Map();
const readyFrames = new Map(), idleFrames = new Map();

let audioContext, socket, basePortrait, activeChunk = null;
let nextPlaybackSeq = 0, animationRunning = false, playbackGeneration = 0, serverComplete = false;
let portraitReady = false, readyGesture = '', expectedReadyFrames = 0, readyStartedAt = null, readyPlayed = false, idleFps = 8;
let quietIdleIndices = [], blinkIdleIndices = [], nextIdleBlinkAt = Infinity, idleBlinkStartedAt = null;
let lastIdleBlinkAt = -Infinity;
let idleGestureStart = Infinity;
let blinkFaceBox = null;
let idleBlinkShape = null, mouthSettleFrame = null, mouthSettleStartedAt = null;
let blinkFeedbackUntil = -Infinity;
let blinkFeedbackId = 0, activeBlinkFeedbackId = 0;
let mocapWorkerPromise, mocapStream, mocapReady = false, mocapEnabled = false, mocapStarting = false;
let mocapNeutral = null, lastWebcamFrame = 0, lastOutputFrame = 0, lastPhaseUpdate = 0;
let avPhaseSeconds = 0, audioSamples = [], mouthSamples = [];
let outputFaceTracked = false, phaseScore = null, mocapDelegates = '';
let outputFeedbackFailures = 0, outputFeedbackRetryAt = -Infinity;
let webcamFaceTracked = false;
let lastAffectUpdate = 0;
let lastMotionValuesUpdate = 0, latestMotionPlan = null, motionControlSchema = null;
const mocapBusy = { webcam: false, output: false, blink: false };
const mocapWorkers = { webcam: null, output: null };
const objectUrlForImage = Symbol('objectUrlForImage');
const motion = {
  current: { x: 0, y: 0, rotation: 0, zoom: 0, shearX: 0, shearY: 0, blend: 0 },
  target: { x: 0, y: 0, rotation: 0, zoom: 0, shearX: 0, shearY: 0, blend: 0 },
};
const autonomic = {
  lastAt: null, rest: 0.15, velocity: 0, target: 0.15, nextTargetAt: 0, state: 'attentive',
  pose: { x: 0, y: 0, rotation: 0 }, poseVelocity: { x: 0, y: 0, rotation: 0 },
  poseTarget: { x: 0, y: 0, rotation: 0 },
};
const semanticMotion = {
  lastAt: null,
  current: { x: 0, y: 0, rotation: 0, zoom: 0, shearX: 0, shearY: 0 },
  velocity: { x: 0, y: 0, rotation: 0, zoom: 0, shearX: 0, shearY: 0 },
  target: { x: 0, y: 0, rotation: 0, zoom: 0, shearX: 0, shearY: 0 },
};
const attention = {
  lastAt: null, current: 0.5, velocity: 0, target: 0.5,
  headCurrent: 0.5, headVelocity: 0,
  fixation: 'camera', nextAt: 0, saccades: 0,
};
const expressionFeedback = {
  neutralEyeOpen: null, blinkSamples: [], durationScale: 1, peakScale: 1,
  lastMetrics: null, completedBlinks: 0, poseSamples: [],
  observations: 0, missedObservations: 0, history: [], motionSummary: null,
};
const upperBody = {
  rig: null, state: 'inhale', stateStartedAt: 0, stateDuration: 1.6,
  breath: 0, breathVelocity: 0, breathTarget: 0, amplitude: 0.6,
  feedbackGain: 1, feedbackAdjustments: 0,
  // The browser is the sole roll owner. Idle/autonomic roll is routed through
  // this target and blended with the phrase-level upper-body roll below.
  idleHeadRollTarget: 0,
  pose: { sternumY: 0, shoulderLift: 0, torsoYaw: 0, headPitch: 0, headRoll: 0 },
  poseVelocity: { sternumY: 0, shoulderLift: 0, torsoYaw: 0, headPitch: 0, headRoll: 0 },
  lastAt: null, observations: 0, missedObservations: 0, history: [], summary: null,
};
const motionCalibration = {
  allowed: new URLSearchParams(window.location.search).get('motion-calibration') === '1',
  active: false, state: 'live', geometry: null, saved: null,
};
const behaviorStates = {
  attentive: { rest: [0.10, 0.30], dwell: [2.0, 4.8], pose: [0.0015, 0.0010, 0.0020], next: ['observe', 'settle', 'thinking'] },
  observe: { rest: [0.30, 0.52], dwell: [1.4, 3.2], pose: [0.0025, 0.0012, 0.0030], next: ['attentive', 'settle'] },
  thinking: { rest: [0.48, 0.72], dwell: [1.2, 2.8], pose: [0.0020, 0.0015, 0.0035], next: ['settle', 'attentive'] },
  settle: { rest: [0.18, 0.38], dwell: [0.9, 1.8], pose: [0.0008, 0.0007, 0.0012], next: ['attentive', 'observe'] },
};
window.portraitTelemetry = { expressionFeedback, upperBody, autonomic, semanticMotion, attention, motionCalibration };

const ALP_EDITOR_DEFAULTS = {
  rotate_pitch: 0, rotate_yaw: 0, rotate_roll: 0, blink: 0, eyebrow: 0,
  wink: 0, pupil_x: 0, pupil_y: 0, aaa: 0, eee: 0, woo: 0, smile: 0,
};

function motionSnapshot() {
  const plan = latestMotionPlan || {};
  const reaction = plan.reaction || {};
  return {
    captured_at: new Date().toISOString(),
    renderer: {
      profile: motionControlSchema?.profile || 'semantic-bank-v13-blink-brow-fix',
      ready_gesture: readyGesture || null,
      alp_frames: expectedReadyFrames ? expectedReadyFrames + 20 : 28,
      alp_editor_schema: motionControlSchema?.controls || null,
      ownership: motionControlSchema?.ownership || null,
      alp_editor_defaults: ALP_EDITOR_DEFAULTS,
    },
    last_phrase: {
      sequence: plan.seq ?? null,
      text: plan.text || null,
      reaction: reaction.reaction || 'neutral',
      strength: reaction.strength ?? 0,
      minilm_controls: reaction.controls || {},
      primitive_mix: reaction.primitive_mix || [],
      timing: {
        attack_ms: reaction.attack_ms ?? null,
        hold_ms: reaction.hold_ms ?? null,
        decay_ms: reaction.decay_ms ?? null,
        clock: reaction.clock || 'audio',
      },
      prosody: plan.prosody || null,
      blinks: reaction.blinks || [],
      gaze_events: reaction.gaze_events || [],
      head_events: reaction.head_events || [],
      upper_body: reaction.upper_body || null,
    },
    current_actuators: {
      canvas: { ...motion.current },
      autonomic_pose: { ...autonomic.pose },
      semantic_motion: { ...semanticMotion.current },
      attention: { fixation: attention.fixation, value: attention.current, head: attention.headCurrent },
      upper_body: { breath: upperBody.breath, pose: { ...upperBody.pose }, state: upperBody.state },
    },
    feedback: {
      face_last_metrics: expressionFeedback.lastMetrics,
      face_summary: expressionFeedback.motionSummary,
      completed_blinks: expressionFeedback.completedBlinks,
      blink_peak_scale: expressionFeedback.peakScale,
      upper_body_summary: upperBody.summary,
      face_observations: expressionFeedback.observations,
      upper_body_observations: upperBody.observations,
    },
  };
}

function refreshMotionValues() {
  if (!motionValues) return;
  motionValues.textContent = JSON.stringify(motionSnapshot(), null, 2);
}

window.copyPortraitMotionValues = async () => {
  const payload = JSON.stringify(motionSnapshot(), null, 2);
  try {
    await navigator.clipboard.writeText(payload);
    if (motionValuesStatus) motionValuesStatus.textContent = 'Copied.';
  } catch (_) {
    if (motionValuesStatus) motionValuesStatus.textContent = 'Clipboard unavailable; select the JSON above.';
  }
  return payload;
};

copyMotionValues?.addEventListener('click', () => window.copyPortraitMotionValues());
downloadMotionValues?.addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(motionSnapshot(), null, 2)], { type: 'application/json' });
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'portrait-motion-values.json';
  link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  if (motionValuesStatus) motionValuesStatus.textContent = 'Downloaded.';
});

fetch('/api/motion/controls').then((response) => response.ok ? response.json() : null)
  .then((schema) => { motionControlSchema = schema; refreshMotionValues(); })
  .catch(() => { /* The panel remains useful with its local defaults. */ });

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

function normalRandom() {
  const u = Math.max(Number.EPSILON, Math.random());
  const v = Math.max(Number.EPSILON, Math.random());
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function nextBlinkInterval() {
  return clamp(Math.exp(1.48 + 0.28 * normalRandom()), 2.6, 8.2);
}

function enterBehaviorState(now, name) {
  const state = behaviorStates[name] || behaviorStates.attentive;
  autonomic.state = name in behaviorStates ? name : 'attentive';
  autonomic.target = state.rest[0] + Math.random() * (state.rest[1] - state.rest[0]);
  const prior = activeChunk ? chunkFor(activeChunk.seq).reaction?.motion_prior : null;
  const latent = prior?.latent || {
    orient: clamp(normalRandom() * 0.28, -1, 1),
    engagement: clamp(normalRandom() * 0.20, -1, 1),
    asymmetry: clamp(normalRandom() * 0.16, -1, 1),
  };
  // Feedback currently closes only identifiable blink/phase/breath loops. Do
  // not pretend a face-pose variance loop is active; semantic attenuation is
  // supplied by the server plan and remains stable across the phrase.
  const gain = prior?.gain ?? 0.46;
  // One low-dimensional cause now moves translation and roll together.  This
  // preserves human-like covariance instead of sampling three unrelated axes.
  const idleRoll = clamp((latent.orient * 0.0026 + latent.asymmetry * 0.0007) * gain, -0.0022, 0.0022);
  if (upperBody.rig) upperBody.idleHeadRollTarget = idleRoll;
  autonomic.poseTarget = {
    x: clamp((latent.orient * 0.0017 + latent.asymmetry * 0.00035) * gain, -0.0014, 0.0014),
    y: clamp((latent.engagement * 0.0011 - latent.orient * 0.00025) * gain, -0.0009, 0.0009),
    // Keep the legacy controller numerically stable for no-rig fallback, but
    // do not render it when the fitted upper-body controller is active.
    rotation: upperBody.rig ? 0 : idleRoll,
  };
  autonomic.nextTargetAt = now + state.dwell[0] + Math.random() * (state.dwell[1] - state.dwell[0]);
}

function updateAutonomic(now) {
  const dt = autonomic.lastAt === null ? 0 : clamp(now - autonomic.lastAt, 0, 0.05);
  autonomic.lastAt = now;
  if (now >= autonomic.nextTargetAt) {
    const state = behaviorStates[autonomic.state] || behaviorStates.attentive;
    let choices = state.next;
    const controls = activeChunk ? chunkFor(activeChunk.seq).reaction?.controls : null;
    if (controls?.gaze_focus > 0.08) choices = ['attentive', 'observe', ...choices];
    if (Math.abs(controls?.gaze_x || 0) > 0.06) choices = ['thinking', ...choices];
    enterBehaviorState(now, choices[Math.floor(Math.random() * choices.length)]);
  }
  // Critically damped second-order response: correlated life, no random walk
  // drift and no snapped preset changes.
  const omega = 1.7;
  const acceleration = omega * omega * (autonomic.target - autonomic.rest) - 2 * omega * autonomic.velocity;
  autonomic.velocity += acceleration * dt;
  autonomic.rest = clamp(autonomic.rest + autonomic.velocity * dt, 0, 1);
  for (const key of Object.keys(autonomic.pose)) {
    const poseAcceleration = 1.3 * 1.3 * (autonomic.poseTarget[key] - autonomic.pose[key])
      - 2 * 1.3 * autonomic.poseVelocity[key];
    autonomic.poseVelocity[key] += poseAcceleration * dt;
    autonomic.pose[key] += autonomic.poseVelocity[key] * dt;
  }
}

function chooseAttentionFixation(now) {
  if (attention.fixation !== 'camera') {
    // Eye contact resumes after a short aversion; do not wander continuously.
    attention.fixation = 'camera';
    attention.target = 0.47 + Math.random() * 0.06;
    attention.nextAt = now + 2.0 + Math.random() * 3.4;
    attention.saccades += 1;
    return;
  }
  if (Math.random() < 0.72) {
    attention.fixation = Math.random() < 0.5 ? 'left-up' : 'right-down';
    attention.target = attention.fixation === 'left-up'
      ? 0.12 + Math.random() * 0.15 : 0.73 + Math.random() * 0.15;
    attention.nextAt = now + 0.42 + Math.random() * 0.78;
    // Some gaze shifts recruit a blink while the slower head movement catches
    // up.  Keep it probabilistic and respect the blink refractory interval.
    if (!activeChunk && now - lastIdleBlinkAt > 2.4 && Math.random() < 0.38) {
      nextIdleBlinkAt = Math.min(nextIdleBlinkAt, now + 0.11 + Math.random() * 0.10);
    }
  } else {
    // Tiny refixation around the camera breaks a perfectly frozen stare.
    attention.target = 0.45 + Math.random() * 0.10;
    attention.nextAt = now + 1.2 + Math.random() * 2.2;
  }
  attention.saccades += 1;
}

function updateAttention(now) {
  const dt = attention.lastAt === null ? 0 : clamp(now - attention.lastAt, 0, 0.05);
  attention.lastAt = now;
  if (activeChunk) {
    const chunk = chunkFor(activeChunk.seq);
    const elapsedMs = Math.max(0, now - activeChunk.startedAt) * 1000;
    let scheduledTarget = 0.5;
    for (const event of chunk.reaction?.gaze_events || []) {
      const start = Number(event.at_ms || 0) + Number(event.move_ms || 90) + 35;
      const move = Number(event.head_follow_ms || 230);
      const hold = Math.max(0, Number(event.hold_ms || 500) - move * 0.45);
      const returning = Number(event.return_ms || 180) + 90;
      const local = elapsedMs - start;
      if (local < 0 || local >= move + hold + returning) continue;
      let amount;
      if (local < move) amount = smoothstep(local / move);
      else if (local < move + hold) amount = 1;
      else amount = 1 - smoothstep((local - move - hold) / returning);
      const direction = Number(event.direction || -1) < 0 ? -1 : 1;
      scheduledTarget = 0.5 + direction * Number(event.amplitude || 1) * amount * 0.22;
    }
    attention.target = scheduledTarget;
  } else if (now >= attention.nextAt) {
    chooseAttentionFixation(now);
  }
  // Eyes saccade in roughly 100-180 ms, much faster than the head follows.
  const omega = 24;
  const acceleration = omega * omega * (attention.target - attention.current)
    - 2 * omega * attention.velocity;
  attention.velocity += acceleration * dt;
  attention.current = clamp(attention.current + attention.velocity * dt, 0, 1);
  // The eyes acquire the fixation first; the head follows with lower gain and
  // a much slower response, then counter-rotates as eye contact resumes.
  const headOmega = 4.2;
  const headAcceleration = headOmega * headOmega * (attention.target - attention.headCurrent)
    - 2 * headOmega * attention.headVelocity;
  attention.headVelocity += headAcceleration * dt;
  attention.headCurrent = clamp(attention.headCurrent + attention.headVelocity * dt, 0, 1);
}

function semanticMotionTarget(now) {
  const zero = { x: 0, y: 0, rotation: 0, zoom: 0, shearX: 0, shearY: 0 };
  if (!activeChunk) return zero;
  const chunk = chunkFor(activeChunk.seq), plan = chunk.reaction;
  const elapsed = now - activeChunk.startedAt;
  const duration = chunk.duration || activeChunk.source.buffer.duration;
  const strength = clamp(Number(plan?.strength || 0), 0, 0.68);
  if (!plan?.controls || strength < 0.01 || elapsed < 0) return zero;
  // Controls already carry MiniLM confidence. Divide the phrase envelope by
  // its strength so it shapes time without applying confidence twice.
  const activity = clamp(reactionEnvelope(plan, elapsed, duration) / Math.max(0.01, strength), 0, 1);
  const controls = plan.controls;
  const lean = clamp(Number(controls.lean || 0), 0, 0.45);
  const recoil = clamp(Number(controls.recoil || 0), 0, 0.45);
  // Gross pitch/roll (including nod and tilt) belongs exclusively to the
  // fitted upper-body controller. ALP owns gaze inside an eye-only mask; a
  // canvas gaze transform would incorrectly drag the mouth with the eyes.
  return {
    x: 0,
    y: clamp((lean * 0.015 - recoil * 0.012) * activity, -0.0055, 0.0055),
    rotation: 0,
    zoom: clamp((lean * 0.050 - recoil * 0.060) * activity, -0.005, 0.005),
    shearX: 0,
    shearY: clamp(-recoil * 0.012 * activity, -0.006, 0.006),
  };
}

function updateSemanticMotion(now) {
  const dt = semanticMotion.lastAt === null ? 0 : clamp(now - semanticMotion.lastAt, 0, 0.05);
  semanticMotion.lastAt = now;
  semanticMotion.target = semanticMotionTarget(now);
  // Faster than the idle posture controller, but still critically damped.
  // This preserves a single smooth webcam-style trajectory across phrases.
  const omega = 5.2;
  for (const key of Object.keys(semanticMotion.current)) {
    const acceleration = omega * omega * (semanticMotion.target[key] - semanticMotion.current[key])
      - 2 * omega * semanticMotion.velocity[key];
    semanticMotion.velocity[key] += acceleration * dt;
    semanticMotion.current[key] += semanticMotion.velocity[key] * dt;
  }
}

function rangeSample(range, fallback) {
  if (!Array.isArray(range) || range.length !== 2) return fallback;
  return Number(range[0]) + Math.random() * (Number(range[1]) - Number(range[0]));
}

function enterRespiratoryState(now, state) {
  const profile = upperBody.rig?.rest_profile || {};
  const respiration = activeChunk ? chunkFor(activeChunk.seq).reaction?.upper_body?.respiration : null;
  const cycle = respiration?.rate_bpm ? 60 / clamp(Number(respiration.rate_bpm), 11, 17.5) : null;
  const inhaleRatio = clamp(Number(respiration?.inhale_ratio || 0.37), 0.33, 0.42);
  upperBody.state = state; upperBody.stateStartedAt = now;
  if (state === 'inhale') {
    upperBody.stateDuration = cycle ? cycle * inhaleRatio : rangeSample(profile.inhale_seconds, 1.6);
    upperBody.breathTarget = 1;
    upperBody.amplitude = respiration?.gain
      ? clamp(Number(respiration.gain), 0.48, 0.78) : rangeSample(profile.amplitude, 0.62);
  } else if (state === 'exhale') {
    upperBody.stateDuration = cycle ? cycle * (1 - inhaleRatio) * 0.94 : rangeSample(profile.exhale_seconds, 2.6);
    upperBody.breathTarget = 0;
  } else {
    upperBody.stateDuration = rangeSample(profile.rest_seconds, 0.18);
    upperBody.breathTarget = 0;
  }
}

function setUpperBodyRig(rig, now) {
  upperBody.rig = rig?.enabled ? rig : null;
  upperBody.idleHeadRollTarget = 0;
  // Once the fitted browser owner is available, discard any pre-rig roll so
  // the canvas cannot carry a second residual into the same frame.
  autonomic.pose.rotation = 0; autonomic.poseVelocity.rotation = 0; autonomic.poseTarget.rotation = 0;
  upperBody.lastAt = now; upperBody.breath = 0; upperBody.breathVelocity = 0;
  enterRespiratoryState(now, 'inhale');
}

function beginSpeechRespiration(now) {
  if (!upperBody.rig) return;
  const respiration = activeChunk ? chunkFor(activeChunk.seq).reaction?.upper_body?.respiration : null;
  if (respiration?.speech_phase !== 'exhale') return;
  // People speak on an exhalation. Preserve an exhale already in progress so
  // adjacent streamed phrases do not visibly restart the breathing clock.
  if (upperBody.state !== 'exhale') enterRespiratoryState(now, 'exhale');
}

function updateUpperBody(now) {
  if (!upperBody.rig) return;
  const dt = upperBody.lastAt === null ? 0 : clamp(now - upperBody.lastAt, 0, 0.05);
  upperBody.lastAt = now;
  if (now - upperBody.stateStartedAt >= upperBody.stateDuration) {
    const speechPhase = activeChunk ? chunkFor(activeChunk.seq).reaction?.upper_body?.respiration?.speech_phase : null;
    if (speechPhase === 'exhale') {
      // Hold the low-pressure end of the exhale until a streamed phrase gap;
      // never visibly inhale in the middle of voiced audio.
      upperBody.state = 'exhale'; upperBody.stateStartedAt = now;
      upperBody.stateDuration = 0.4; upperBody.breathTarget = 0;
    } else {
      enterRespiratoryState(now, upperBody.state === 'inhale' ? 'exhale' : (upperBody.state === 'exhale' ? 'rest' : 'inhale'));
    }
  }
  const breathOmega = upperBody.state === 'inhale' ? 2.25 : 1.45;
  const breathAcceleration = breathOmega * breathOmega * (upperBody.breathTarget - upperBody.breath)
    - 2 * breathOmega * upperBody.breathVelocity;
  upperBody.breathVelocity += breathAcceleration * dt;
  upperBody.breath = clamp(upperBody.breath + upperBody.breathVelocity * dt, 0, 1);

  const plan = activeChunk ? chunkFor(activeChunk.seq).reaction?.upper_body : null;
  const elapsed = activeChunk ? now - activeChunk.startedAt : 0;
  const duration = activeChunk ? (chunkFor(activeChunk.seq).duration || activeChunk.source.buffer.duration) : 1;
  const activity = activeChunk ? clamp(reactionEnvelope(chunkFor(activeChunk.seq).reaction, elapsed, duration) / 0.45, 0, 1) : 0;
  const target = {
    sternumY: Number(plan?.pose?.sternum_y || 0) * activity,
    shoulderLift: Number(plan?.pose?.shoulder_lift || 0) * activity,
    torsoYaw: Number(plan?.pose?.torso_yaw || 0) * activity,
    headPitch: Number(plan?.pose?.head_pitch || 0) * activity,
    // Idle and phrase roll share one critically-damped browser actuator.
    headRoll: clamp(upperBody.idleHeadRollTarget + Number(plan?.pose?.head_roll || 0) * activity, -0.0045, 0.0045),
  };
  const nodStart = 0.08, nodDuration = clamp(duration * 0.42, 0.34, 0.62);
  const nodPhase = clamp((elapsed - nodStart) / nodDuration, 0, 1);
  const nodPulse = elapsed >= nodStart && elapsed <= nodStart + nodDuration
    ? Math.sin(Math.PI * nodPhase) ** 2 : 0;
  target.headPitch += Number(plan?.pose?.nod_impulse || 0) * nodPulse;
  const omega = Number(plan?.spring?.omega || 2.4);
  for (const key of Object.keys(upperBody.pose)) {
    const acceleration = omega * omega * (target[key] - upperBody.pose[key])
      - 2 * omega * upperBody.poseVelocity[key];
    upperBody.poseVelocity[key] += acceleration * dt;
    upperBody.pose[key] += upperBody.poseVelocity[key] * dt;
  }
}

function closeImage(image) {
  if (image?.[objectUrlForImage]) {
    URL.revokeObjectURL(image[objectUrlForImage]);
    image.removeAttribute?.('src');
    delete image[objectUrlForImage];
  }
  image?.close?.();
}

function imageDimensions(image) {
  return {
    width: image.width || image.naturalWidth,
    height: image.height || image.naturalHeight,
  };
}

async function decodeImage(source) {
  if (typeof window.createImageBitmap === 'function') {
    try { return await window.createImageBitmap(source); } catch (_) { /* WebKit fallback below */ }
  }
  const url = URL.createObjectURL(source);
  try {
    const image = new Image();
    image.decoding = 'async';
    if (typeof image.decode === 'function') {
      image.src = url;
      await image.decode();
    } else await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error('The browser could not decode this image.'));
      image.src = url;
    });
    image[objectUrlForImage] = url;
    return image;
  } catch (error) {
    URL.revokeObjectURL(url);
    throw error;
  }
}

function newAudioContext() {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) throw new Error('This browser does not support live audio playback.');
  try { return new AudioContextClass({ latencyHint: 'interactive' }); }
  catch (_) { return new AudioContextClass(); }
}

function chunkFor(seq) {
  if (!chunks.has(seq)) chunks.set(seq, { frames: new Map(), fps: 8, audio: null, reaction: null, prosody: null });
  return chunks.get(seq);
}

function smoothstep(value) {
  const bounded = clamp(value, 0, 1); return bounded * bounded * (3 - 2 * bounded);
}

function reactionEnvelope(plan, elapsed, duration) {
  if (!plan || plan.reaction === 'neutral' || elapsed < 0) return 0;
  let attack = Math.max(0.08, Number(plan.attack_ms || 200) / 1000);
  let hold = Math.max(0, Number(plan.hold_ms || 320) / 1000);
  let decay = Math.max(0.12, Number(plan.decay_ms || 700) / 1000);
  const available = Math.max(0.2, duration - 0.08), total = attack + hold + decay;
  if (total > available) { const scale = available / total; attack *= scale; hold *= scale; decay *= scale; }
  const startRatio = clamp(Number(plan.start_ratio ?? 0), 0, 0.32);
  const endRatio = clamp(Number(plan.end_ratio ?? 0.18), 0, 0.32);
  let envelope = 0;
  if (elapsed < attack) envelope = startRatio + (1 - startRatio) * smoothstep(elapsed / attack);
  else if (elapsed < attack + hold) envelope = 1;
  else if (elapsed < attack + hold + decay) envelope = 1 - (1 - endRatio) * smoothstep((elapsed - attack - hold) / decay);
  else envelope = endRatio;
  return envelope * clamp(Number(plan.strength || 0), 0, 0.68);
}

function idleBlinkAmount(elapsed, shape) {
  if (!shape || elapsed < 0) return 0;
  if (elapsed < shape.close) return smoothstep(elapsed / shape.close);
  if (elapsed < shape.close + shape.hold) return 1;
  if (elapsed < shape.close + shape.hold + shape.open) {
    return 1 - smoothstep((elapsed - shape.close - shape.hold) / shape.open);
  }
  return 0;
}

function refreshIdleIndices() {
  const indices = [...idleFrames.keys()].sort((a, b) => a - b);
  quietIdleIndices = indices.filter((index) => index < idleGestureStart);
  blinkIdleIndices = indices.filter((index) => index >= idleGestureStart);
}

function updateAffectController(now) {
  if (!activeChunk || now - lastAffectUpdate < 0.1) return;
  lastAffectUpdate = now;
  const chunk = chunkFor(activeChunk.seq);
  const elapsed = now - activeChunk.startedAt;
  const value = reactionEnvelope(chunk.reaction, elapsed, chunk.duration || activeChunk.source.buffer.duration);
  const label = chunk.reaction?.reaction || 'neutral';
  affectStatus.textContent = `${label} ${(value * 100).toFixed(0)}% · audio clock · semantics/phrase · prosody 50 Hz · face ${chunk.fps} fps`;
}

function mocapLabel(extra = '') {
  const pose = !mocapEnabled ? 'Webcam pose off' : (webcamFaceTracked ? 'Webcam pose tracking' : 'Webcam waiting for face');
  const tracking = outputFaceTracked ? 'output face tracked' : 'render feedback off';
  const sync = `phase ${avPhaseSeconds >= 0 ? '+' : ''}${Math.round(avPhaseSeconds * 1000)} ms`;
  const confidence = phaseScore === null ? '' : ` · lock ${phaseScore.toFixed(2)}`;
  mocapStatus.textContent = `${pose} · ${tracking} · ${sync}${confidence}${extra ? ` · ${extra}` : ''}`;
}

function startMocapWorker(role) {
  return new Promise((resolve, reject) => {
    const worker = new Worker('/static/mocap-worker.js?v=feedback3', { type: 'module' });
    mocapWorkers[role] = worker;
    worker.onmessage = ({ data }) => {
      if (data.type === 'ready') {
        resolve(data.delegate); return;
      }
      if (data.type === 'error') {
        mocapStatus.textContent = `MediaPipe unavailable: ${data.detail}`; reject(new Error(data.detail)); return;
      }
      if (data.type === 'frame-error') {
        mocapBusy[data.stream] = false;
        if (data.stream === 'output') {
          outputFeedbackFailures += 1; outputFaceTracked = false;
          outputFeedbackRetryAt = (audioContext?.currentTime || 0)
            + Math.min(30, 0.5 * (2 ** Math.min(6, outputFeedbackFailures - 1)));
          mocapLabel('output feedback tracker retry scheduled');
        } else {
          // A failed camera texture can otherwise be posted again every 50 ms.
          // Stop only the optional webcam lane; rendered-output feedback remains live.
          mocapEnabled = false; webcamFaceTracked = false;
          for (const track of mocapStream?.getTracks?.() || []) track.stop();
          mocapStream = null; mocapCamera.srcObject = null;
          mocapToggle.textContent = 'Enable webcam mocap';
          mocapToggle.setAttribute('aria-pressed', 'false');
          mocapLabel('webcam tracker stopped after a local frame error');
        }
        return;
      }
      if (data.type !== 'result') return;
      mocapBusy[data.stream] = false;
      if (data.stream === 'webcam') updateWebcamMotion(data.metrics);
      else updateOutputFeedback(data);
    };
    worker.onerror = (event) => {
      mocapStatus.textContent = `MediaPipe worker failed: ${event.message}`;
      reject(new Error(event.message));
    };
    worker.postMessage({ type: 'init', role });
  });
}

function ensureMocapWorker() {
  if (mocapWorkerPromise) return mocapWorkerPromise;
  // Render feedback runs in the same Compose stack because the browser WASM
  // graph is unreliable on WebKit and this Chromium build. Camera frames are
  // never sent; the optional webcam graph remains a lazy local worker.
  mocapWorkerPromise = Promise.resolve().then(() => {
    mocapReady = true; mocapDelegates = 'output:backend-CPU'; mocapLabel(`MediaPipe ${mocapDelegates}`);
    outputFeedbackFailures = 0; outputFeedbackRetryAt = -Infinity;
  });
  return mocapWorkerPromise;
}

function updateWebcamMotion(metrics) {
  if (!mocapEnabled || !metrics) return;
  if (!webcamFaceTracked) { webcamFaceTracked = true; mocapLabel(); }
  if (!mocapNeutral) { mocapNeutral = metrics; return; }
  const scale = metrics.eyeDistance / mocapNeutral.eyeDistance;
  motion.target = {
    x: clamp(-(metrics.centerX - mocapNeutral.centerX) * 1.15, -0.035, 0.035),
    y: clamp((metrics.centerY - mocapNeutral.centerY) * 0.8, -0.025, 0.025),
    rotation: clamp(-(metrics.roll - mocapNeutral.roll), -0.09, 0.09),
    zoom: clamp(scale - 1, -0.035, 0.035),
    shearX: clamp(-(metrics.yaw - mocapNeutral.yaw) * 0.12, -0.045, 0.045),
    shearY: clamp((metrics.pitch - mocapNeutral.pitch) * 0.08, -0.035, 0.035),
    blend: 1,
  };
}

function interpolateAudio(time) {
  if (audioSamples.length < 2 || time < audioSamples[0].time || time > audioSamples.at(-1).time) return null;
  for (let index = audioSamples.length - 1; index > 0; index -= 1) {
    const upper = audioSamples[index], lower = audioSamples[index - 1];
    if (lower.time <= time) {
      const mix = (time - lower.time) / Math.max(0.0001, upper.time - lower.time);
      return lower.value + (upper.value - lower.value) * mix;
    }
  }
  return null;
}

function correlation(pairs) {
  if (pairs.length < 8) return 0;
  const meanA = pairs.reduce((sum, pair) => sum + pair[0], 0) / pairs.length;
  const meanB = pairs.reduce((sum, pair) => sum + pair[1], 0) / pairs.length;
  let numerator = 0, squareA = 0, squareB = 0;
  for (const [a, b] of pairs) {
    const da = a - meanA, db = b - meanB;
    numerator += da * db; squareA += da * da; squareB += db * db;
  }
  return numerator / Math.max(0.000001, Math.sqrt(squareA * squareB));
}

function updatePhaseController(now) {
  if (now - lastPhaseUpdate < 0.5 || mouthSamples.length < 10 || audioSamples.length < 20) return;
  lastPhaseUpdate = now;
  let bestLag = 0, bestScore = -1;
  for (let lag = -0.12; lag <= 0.1201; lag += 0.02) {
    const pairs = mouthSamples.map((sample) => [sample.value, interpolateAudio(sample.time + lag)])
      .filter((pair) => pair[1] !== null);
    const score = correlation(pairs);
    if (score > bestScore) { bestScore = score; bestLag = lag; }
  }
  if (bestScore >= 0.32 && Math.abs(bestLag) >= 0.015) {
    const correction = clamp(-bestLag * 0.2, -0.005, 0.005);
    avPhaseSeconds = clamp(avPhaseSeconds + correction, -0.04, 0.04);
  }
  phaseScore = bestScore; mocapLabel();
}

function updateOutputFeedback(data) {
  outputFeedbackFailures = 0; outputFeedbackRetryAt = -Infinity;
  expressionFeedback.observations += 1;
  if (!data.bodySkipped) upperBody.observations += 1;
  if (!data.bodySkipped && !data.bodyMetrics) upperBody.missedObservations += 1;
  else if (!data.bodySkipped) {
    upperBody.history.push({ time: data.sampleTime, metrics: data.bodyMetrics, commandedBreath: upperBody.breath, speaking: Boolean(activeChunk) });
    if (upperBody.history.length > 300) upperBody.history.splice(0, upperBody.history.length - 300);
    if (upperBody.history.length >= 40 && upperBody.observations % 20 === 0) {
      const usable = upperBody.history.slice(-120).filter(
        (sample) => Number(sample.metrics.shoulderVisibility ?? sample.metrics.visibility ?? 0) >= 0.35
      );
      const spans = usable.map((sample) => Number(sample.metrics.shoulderSpan || 0));
      const centers = usable.map((sample) => Number(sample.metrics.shoulderCenterY || 0));
      const breaths = usable.map((sample) => Number(sample.commandedBreath || 0));
      const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
      const sd = (values) => { const center = mean(values); return Math.sqrt(mean(values.map((value) => (value - center) ** 2))); };
      if (usable.length >= 48) {
        const breathMean = mean(breaths), spanMean = mean(spans);
        const variance = mean(breaths.map((value) => (value - breathMean) ** 2));
        const covariance = mean(breaths.map((value, index) => (value - breathMean) * (spans[index] - spanMean)));
        const normalizedResponse = Math.abs(covariance / Math.max(0.000001, variance)) / Math.max(0.001, spanMean);
        const breathCorrelation = correlation(usable.map((sample, index) => [breaths[index], spans[index]]));
        // Adjust only when the observed signal is credible and no webcam pose
        // contaminates it. The narrow gain bound prevents noise-chasing.
        if (!webcamFaceTracked && sd(breaths) >= 0.12 && Math.abs(breathCorrelation) >= 0.30) {
          const correction = normalizedResponse < 0.0025 ? 1.01 : (normalizedResponse > 0.012 ? 0.99 : 1);
          if (correction !== 1) upperBody.feedbackAdjustments += 1;
          upperBody.feedbackGain = clamp(upperBody.feedbackGain * correction, 0.88, 1.08);
        }
        upperBody.summary = {
          samples: usable.length,
          shoulderSpanStd: sd(spans), shoulderCenterYStd: sd(centers),
          breathShoulderCorrelation: breathCorrelation,
          normalizedBreathResponse: normalizedResponse,
          feedbackGain: upperBody.feedbackGain,
          feedbackAdjustments: upperBody.feedbackAdjustments,
        };
      }
    }
  }
  if (!data.metrics) { expressionFeedback.missedObservations += 1; return; }
  if (!outputFaceTracked) { outputFaceTracked = true; mocapLabel(); }
  expressionFeedback.lastMetrics = data.metrics;
  expressionFeedback.history.push({ time: data.sampleTime, metrics: data.metrics, speaking: Boolean(activeChunk) });
  if (expressionFeedback.history.length > 300) {
    expressionFeedback.history.splice(0, expressionFeedback.history.length - 300);
  }
  if (expressionFeedback.history.length >= 40 && expressionFeedback.history.length % 20 === 0) {
    const names = ['gazeX', 'gazeY', 'yaw', 'pitch', 'roll', 'leftBrow', 'rightBrow', 'smile', 'smirk'];
    const values = Object.fromEntries(names.map((name) => [
      name, expressionFeedback.history.map((sample) => Number(sample.metrics[name] || 0)),
    ]));
    const robust = Object.fromEntries(names.map((name) => {
      const sorted = [...values[name]].sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)];
      const iqr = Math.max(0.001, sorted[Math.floor(sorted.length * 0.75)] - sorted[Math.floor(sorted.length * 0.25)]);
      return [name, { median, iqr }];
    }));
    const deviations = expressionFeedback.history.map((_, index) => (
      Math.max(...names.map((name) => Math.abs(values[name][index] - robust[name].median) / robust[name].iqr))
    ));
    expressionFeedback.motionSummary = {
      samples: expressionFeedback.history.length,
      nearNeutralFraction: deviations.filter((value) => value <= 0.75).length / deviations.length,
      gazeHeadCorrelation: correlation(expressionFeedback.history.map((_, index) => [values.gazeX[index], values.yaw[index]])),
      browCorrelation: correlation(expressionFeedback.history.map((_, index) => [values.leftBrow[index], values.rightBrow[index]])),
    };
  }
  const eyeOpen = (data.metrics.leftEyeOpen + data.metrics.rightEyeOpen) / 2;
  if (!activeChunk) {
    if (data.blinkId > 0 && data.blinkId === activeBlinkFeedbackId) {
      expressionFeedback.blinkSamples.push({ time: data.sampleTime, eyeOpen });
    }
    if (!data.blinkId) {
      expressionFeedback.neutralEyeOpen = expressionFeedback.neutralEyeOpen === null
        ? eyeOpen : expressionFeedback.neutralEyeOpen * 0.96 + eyeOpen * 0.04;
      expressionFeedback.poseSamples.push([
        data.metrics.centerX, data.metrics.centerY, data.metrics.roll,
        data.metrics.leftBrow, data.metrics.rightBrow, data.metrics.smile, data.metrics.smirk,
      ]);
      // The observer sees webcam and renderer motion combined, so its center
      // variance is telemetry—not a valid actuator error signal. Blink and AV
      // phase retain their independently identifiable closed loops.
      if (expressionFeedback.poseSamples.length >= 120) expressionFeedback.poseSamples.splice(0, 60);
    }
  }
  if (activeChunk && data.seq === activeChunk.seq && data.sampleTime >= 0) {
    mouthSamples.push({ time: data.sampleTime, value: data.metrics.mouth });
    mouthSamples = mouthSamples.filter((sample) => data.sampleTime - sample.time <= 1.8);
    updatePhaseController(data.sampleTime);
  }
}

function finishBlinkFeedback(blinkId) {
  if (blinkId !== activeBlinkFeedbackId) return;
  const samples = expressionFeedback.blinkSamples;
  const neutral = expressionFeedback.neutralEyeOpen;
  if (neutral && samples.length >= 2) {
    const active = samples.filter((sample) => sample.eyeOpen < neutral * 0.85);
    const peakRatio = Math.min(...samples.map((sample) => sample.eyeOpen)) / neutral;
    if (active.length >= 1) {
      // Even the cropped face-only observer cannot estimate a ~200 ms event's
      // duration without aliasing. Keep biologically bounded timing open-loop;
      // close depth is identifiable from a single sub-threshold observation.
      const amplitudeCorrection = peakRatio < 0.08 ? 0.97 : (peakRatio > 0.22 ? 1.025 : 1);
      expressionFeedback.peakScale = clamp(expressionFeedback.peakScale * amplitudeCorrection, 0.82, 1.08);
      expressionFeedback.completedBlinks += 1;
    }
  }
  expressionFeedback.blinkSamples = [];
  activeBlinkFeedbackId = 0;
}

async function postMocapFrame(source, stream, timestampMs, sampleTime = 0, seq = -1, blinkId = 0) {
  if (!mocapReady || mocapBusy[stream]) return;
  mocapBusy[stream] = true;
  try {
    if (stream === 'output' && source === canvas) {
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.76));
      if (!blob) throw new Error('could not encode feedback frame');
      const response = await fetch('/api/feedback', {
        method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob,
      });
      if (!response.ok) throw new Error(`feedback observer failed (${response.status})`);
      const result = await response.json();
      mocapBusy.output = false;
      outputFeedbackFailures = 0; outputFeedbackRetryAt = -Infinity;
      updateOutputFeedback({ metrics: result.metrics, bodyMetrics: result.body_metrics,
        bodySkipped: Boolean(result.body_skipped), sampleTime, seq, blinkId });
      return;
    }
    if (typeof window.createImageBitmap !== 'function') { mocapBusy[stream] = false; return; }
    const bitmap = await window.createImageBitmap(source);
    mocapWorkers[stream].postMessage({ type: 'frame', bitmap, timestampMs, sampleTime, seq }, [bitmap]);
  } catch (_) {
    mocapBusy[stream] = false;
    if (stream === 'output') {
      outputFeedbackFailures += 1; outputFaceTracked = false;
      outputFeedbackRetryAt = (audioContext?.currentTime || 0)
        + Math.min(30, 0.5 * (2 ** Math.min(6, outputFeedbackFailures - 1)));
      mocapLabel(`output feedback retry ${Math.round(outputFeedbackRetryAt - (audioContext?.currentTime || 0))}s`);
    }
  }
}

async function postBlinkFeedbackFrame(now) {
  if (!blinkFaceBox || mocapBusy.blink || activeBlinkFeedbackId <= 0) return;
  mocapBusy.blink = true;
  const blinkId = activeBlinkFeedbackId;
  try {
    // This lane is independent of the continuously busy full-frame observer,
    // so the first actually rendered closed-eye frame is sampled immediately.
    feedbackFaceLayer.width = 256; feedbackFaceLayer.height = 256;
    const [x1, y1, x2, y2] = blinkFaceBox;
    const faceWidth = Math.max(1, x2 - x1), faceHeight = Math.max(1, y2 - y1);
    const side = Math.min(Math.max(faceWidth, faceHeight) * 1.65, Math.min(canvas.width, canvas.height));
    const sx = clamp((x1 + x2 - side) / 2, 0, Math.max(0, canvas.width - side));
    const sy = clamp((y1 + y2 - side) / 2, 0, Math.max(0, canvas.height - side));
    feedbackFaceContext.drawImage(canvas, sx, sy, side, side, 0, 0, 256, 256);
    const blob = await new Promise((resolve) => feedbackFaceLayer.toBlob(resolve, 'image/jpeg', 0.76));
    if (!blob) throw new Error('could not encode blink feedback frame');
    const response = await fetch('/api/feedback?body=false', {
      method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blob,
    });
    if (!response.ok) throw new Error(`blink observer failed (${response.status})`);
    const result = await response.json();
    updateOutputFeedback({ metrics: result.metrics, bodyMetrics: null, bodySkipped: true,
      sampleTime: now, seq: -1, blinkId });
  } catch (_) { /* A missed blink sample must not disable continuous feedback. */ }
  finally { mocapBusy.blink = false; }
}

function sampleAudio(now) {
  if (!activeChunk?.analyser || now < activeChunk.startedAt) return;
  const waveform = activeChunk.waveform;
  activeChunk.analyser.getFloatTimeDomainData(waveform);
  const rms = Math.sqrt(waveform.reduce((sum, value) => sum + value * value, 0) / waveform.length);
  const time = now - activeChunk.startedAt;
  audioSamples.push({ time, value: Math.log1p(rms * 40) });
  audioSamples = audioSamples.filter((sample) => time - sample.time <= 2.0);
}

function tickMocap(now) {
  if (!motionCalibration.active) {
    updateAutonomic(now);
    updateAttention(now);
    updateSemanticMotion(now);
    updateUpperBody(now);
  }
  for (const key of Object.keys(motion.current)) {
    motion.current[key] += (motion.target[key] - motion.current[key]) * 0.16;
  }
  const timestampMs = performance.now();
  if (mocapEnabled && mocapCamera.readyState >= 2 && now - lastWebcamFrame >= 0.05) {
    lastWebcamFrame = now; postMocapFrame(mocapCamera, 'webcam', timestampMs);
  }
  if (!motionCalibration.active && activeBlinkFeedbackId === 0 && now >= outputFeedbackRetryAt && canvas.width && now - lastOutputFrame >= 0.08) {
    lastOutputFrame = now;
    postMocapFrame(
      canvas, 'output', timestampMs,
      activeChunk ? now - activeChunk.startedAt : now,
      activeChunk ? activeChunk.seq : -1,
      now <= blinkFeedbackUntil ? activeBlinkFeedbackId : 0,
    );
  }
  if (now - lastMotionValuesUpdate >= 0.25) {
    lastMotionValuesUpdate = now;
    refreshMotionValues();
  }
}

async function disableMocap() {
  for (const track of mocapStream?.getTracks() || []) track.stop();
  mocapStream = null; mocapCamera.srcObject = null; mocapEnabled = false; mocapNeutral = null; webcamFaceTracked = false;
  motion.target = { x: 0, y: 0, rotation: 0, zoom: 0, shearX: 0, shearY: 0, blend: 0 };
  mocapToggle.textContent = 'Enable webcam mocap'; mocapToggle.setAttribute('aria-pressed', 'false'); mocapLabel();
}

async function enableMocap() {
  if (mocapEnabled || mocapStarting) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    throw new Error('Webcam mocap needs HTTPS (or localhost). Output lip-sync feedback is still active.');
  }
  mocapStarting = true; mocapToggle.disabled = true;
  try {
    await ensureMocapWorker();
    mocapStream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 24, max: 30 } }, audio: false });
    if (!mocapWorkers.webcam) {
      const delegate = await startMocapWorker('webcam');
      mocapDelegates = `${mocapDelegates}/webcam:${delegate}`;
    }
    mocapCamera.srcObject = mocapStream; await mocapCamera.play();
    mocapEnabled = true; mocapNeutral = null; webcamFaceTracked = false; motion.target.blend = 1;
    mocapToggle.textContent = 'Disable webcam mocap'; mocapToggle.setAttribute('aria-pressed', 'true'); mocapLabel('move naturally to drive the portrait');
  } finally {
    mocapStarting = false; mocapToggle.disabled = false;
  }
}

mocapToggle.addEventListener('click', async () => {
  mocapToggle.disabled = true;
  try {
    if (mocapEnabled) await disableMocap(); else await enableMocap();
  } catch (error) { await disableMocap(); mocapStatus.textContent = error.message; }
  finally { mocapToggle.disabled = false; }
});

async function makeWebcam(file) {
  playbackGeneration += 1;
  if (socket) socket.close();
  if (activeChunk?.source) { activeChunk.source.onended = null; try { activeChunk.source.stop(); } catch (_) { /* stopped */ } }
  if (audioContext) await audioContext.close();
  closeImage(basePortrait);
  for (const chunk of chunks.values()) for (const bitmap of chunk.frames.values()) closeImage(bitmap);
  for (const bitmap of readyFrames.values()) closeImage(bitmap);
  for (const bitmap of idleFrames.values()) closeImage(bitmap);
  chunks.clear(); readyFrames.clear(); idleFrames.clear(); activeChunk = null; nextPlaybackSeq = 0; animationRunning = false; serverComplete = false;
  latestMotionPlan = null; refreshMotionValues();
  portraitReady = false; expectedReadyFrames = 0; readyStartedAt = null; readyPlayed = false;
  quietIdleIndices = []; blinkIdleIndices = []; idleGestureStart = Infinity; blinkFaceBox = null;
  nextIdleBlinkAt = Infinity; idleBlinkStartedAt = null;
  lastIdleBlinkAt = -Infinity;
  idleBlinkShape = null; mouthSettleFrame = null; mouthSettleStartedAt = null;
  blinkFeedbackUntil = -Infinity;
  activeBlinkFeedbackId = 0;
  autonomic.lastAt = null; autonomic.rest = 0.15; autonomic.velocity = 0;
  autonomic.target = 0.15; autonomic.nextTargetAt = 0; autonomic.state = 'attentive';
  autonomic.pose = { x: 0, y: 0, rotation: 0 }; autonomic.poseVelocity = { x: 0, y: 0, rotation: 0 };
  autonomic.poseTarget = { x: 0, y: 0, rotation: 0 };
  semanticMotion.lastAt = null;
  semanticMotion.current = { x: 0, y: 0, rotation: 0, zoom: 0, shearX: 0, shearY: 0 };
  semanticMotion.velocity = { x: 0, y: 0, rotation: 0, zoom: 0, shearX: 0, shearY: 0 };
  semanticMotion.target = { x: 0, y: 0, rotation: 0, zoom: 0, shearX: 0, shearY: 0 };
  attention.lastAt = null; attention.current = 0.5; attention.velocity = 0;
  attention.headCurrent = 0.5; attention.headVelocity = 0;
  attention.target = 0.5; attention.fixation = 'camera'; attention.nextAt = 0; attention.saccades = 0;
  expressionFeedback.neutralEyeOpen = null; expressionFeedback.blinkSamples = [];
  expressionFeedback.durationScale = 1; expressionFeedback.peakScale = 1; expressionFeedback.completedBlinks = 0;
  expressionFeedback.poseSamples = [];
  expressionFeedback.observations = 0; expressionFeedback.missedObservations = 0;
  expressionFeedback.history = []; expressionFeedback.motionSummary = null;
  upperBody.rig = null; upperBody.lastAt = null; upperBody.observations = 0; upperBody.missedObservations = 0;
  upperBody.history = []; upperBody.summary = null; upperBody.feedbackGain = 1; upperBody.feedbackAdjustments = 0;
  upperBody.idleHeadRollTarget = 0;
  upperBody.pose = { sternumY: 0, shoulderLift: 0, torsoYaw: 0, headPitch: 0, headRoll: 0 };
  upperBody.poseVelocity = { sternumY: 0, shoulderLift: 0, torsoYaw: 0, headPitch: 0, headRoll: 0 };
  audioSamples = []; mouthSamples = []; avPhaseSeconds = 0; outputFaceTracked = false; phaseScore = null;
  outputFeedbackFailures = 0; outputFeedbackRetryAt = -Infinity;
  lastAffectUpdate = 0; affectStatus.textContent = 'Reaction controller waiting for speech.';
  audioContext = newAudioContext(); await audioContext.resume();
  basePortrait = await decodeImage(file);
  const dimensions = imageDimensions(basePortrait);
  canvas.width = dimensions.width; canvas.height = dimensions.height;
  context.drawImage(basePortrait, 0, 0);
}

function canvasSnapshot() {
  const snapshot = document.createElement('canvas');
  snapshot.width = canvas.width; snapshot.height = canvas.height;
  snapshot.getContext('2d', { alpha: false }).drawImage(canvas, 0, 0);
  return snapshot;
}

function affineForTriangle(source, target) {
  const [s0, s1, s2] = source, [d0, d1, d2] = target;
  const denominator = s0.x * (s1.y - s2.y) + s1.x * (s2.y - s0.y) + s2.x * (s0.y - s1.y);
  if (Math.abs(denominator) < 0.001) return null;
  return {
    a: (d0.x * (s1.y - s2.y) + d1.x * (s2.y - s0.y) + d2.x * (s0.y - s1.y)) / denominator,
    c: (d0.x * (s2.x - s1.x) + d1.x * (s0.x - s2.x) + d2.x * (s1.x - s0.x)) / denominator,
    e: (d0.x * (s1.x * s2.y - s2.x * s1.y) + d1.x * (s2.x * s0.y - s0.x * s2.y) + d2.x * (s0.x * s1.y - s1.x * s0.y)) / denominator,
    b: (d0.y * (s1.y - s2.y) + d1.y * (s2.y - s0.y) + d2.y * (s0.y - s1.y)) / denominator,
    d: (d0.y * (s2.x - s1.x) + d1.y * (s0.x - s2.x) + d2.y * (s1.x - s0.x)) / denominator,
    f: (d0.y * (s1.x * s2.y - s2.x * s1.y) + d1.y * (s2.x * s0.y - s0.x * s2.y) + d2.y * (s0.x * s1.y - s1.x * s0.y)) / denominator,
  };
}

function drawMeshTriangle(image, source, target) {
  const transform = affineForTriangle(source, target);
  if (!transform) return;
  // Canvas clips antialias each triangle independently. Expand only the clip
  // (not the affine mapping) so neighboring triangles overlap by a subpixel
  // instead of leaving translucent seams that become dark on the output canvas.
  const centroid = target.reduce((point, vertex) => ({
    x: point.x + vertex.x / target.length,
    y: point.y + vertex.y / target.length,
  }), { x: 0, y: 0 });
  const clipTarget = target.map((vertex) => {
    const dx = vertex.x - centroid.x, dy = vertex.y - centroid.y;
    const distance = Math.max(1, Math.hypot(dx, dy));
    const scale = (distance + 0.85) / distance;
    return { x: centroid.x + dx * scale, y: centroid.y + dy * scale };
  });
  upperBodyContext.save();
  upperBodyContext.beginPath(); upperBodyContext.moveTo(clipTarget[0].x, clipTarget[0].y);
  upperBodyContext.lineTo(clipTarget[1].x, clipTarget[1].y); upperBodyContext.lineTo(clipTarget[2].x, clipTarget[2].y);
  upperBodyContext.closePath(); upperBodyContext.clip();
  upperBodyContext.transform(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f);
  upperBodyContext.drawImage(image, 0, 0, canvas.width, canvas.height);
  upperBodyContext.restore();
}

function upperBodyGeometry(rig) {
  if (!rig || !canvas.width || !canvas.height) return null;
  const center = Number(rig.neck?.x || 0.5);
  const shoulderXs = [Number(rig.leftShoulder?.x || 0.30), Number(rig.rightShoulder?.x || 0.70)].sort((a, b) => a - b);
  const [left, right] = shoulderXs;
  const columns = [0, clamp(left, 0.08, center - 0.04), center, clamp(right, center + 0.04, 0.92), 1];
  const shoulderY = (Number(rig.leftShoulder?.y || 0.64) + Number(rig.rightShoulder?.y || 0.64)) / 2;
  const rows = [0, clamp(Number(rig.neck?.y || shoulderY - 0.06), 0.20, shoulderY), shoulderY,
    clamp(Number(rig.sternum?.y || shoulderY + 0.10), shoulderY + 0.03, 0.88),
    clamp(Number(rig.hipCenter?.y || 0.94), shoulderY + 0.12, 0.98), 1];
  const breath = upperBody.breath * upperBody.amplitude;
  // At the normalized 416x512 feed, the previous peak respiration was below
  // one pixel and disappeared into raster antialiasing. These remain well
  // below 1% of frame size, but put a full inhale above the perceptual floor.
  const expand = Number(rig.shoulderSpan || 0.4) * canvas.width * (0.0074 * breath * upperBody.feedbackGain);
  const lift = canvas.height * (0.0030 * breath * upperBody.feedbackGain + upperBody.pose.shoulderLift);
  const sternum = canvas.height * (0.0018 * breath * upperBody.feedbackGain + upperBody.pose.sternumY);
  const yaw = canvas.width * upperBody.pose.torsoYaw;
  const rowExpand = [0, 0, 0.58, 1.0, 0.35, 0];
  const rowLift = [0, 0, 1.0, 0.58, 0.16, 0];
  const rowYaw = [0, 0, 0.10, 0.55, 0.80, 0];
  const source = rows.map((y) => columns.map((x) => ({ x: x * canvas.width, y: y * canvas.height })));
  const target = rows.map((y, rowIndex) => columns.map((x) => {
    const centerWeight = 1 - Math.min(1, Math.abs(x - center) / Math.max(center, 1 - center));
    const direction = x < center ? -1 : (x > center ? 1 : 0);
    return {
      x: x * canvas.width + direction * expand * rowExpand[rowIndex] * centerWeight + yaw * rowYaw[rowIndex] * centerWeight,
      y: y * canvas.height - lift * rowLift[rowIndex] * centerWeight - sternum * rowExpand[rowIndex] * centerWeight,
    };
  }));
  const displacement = source.flatMap((row, rowIndex) => row.map((point, columnIndex) => (
    Math.hypot(target[rowIndex][columnIndex].x - point.x, target[rowIndex][columnIndex].y - point.y)
  )));
  return { source, target, maxDisplacementPx: Math.max(...displacement), expandPx: expand, liftPx: lift, sternumPx: sternum, yawPx: yaw };
}

function upperBodyWarp(image) {
  const rig = upperBody.rig;
  if (!rig || !canvas.width || !canvas.height) return image;
  if (upperBodyLayer.width !== canvas.width || upperBodyLayer.height !== canvas.height) {
    upperBodyLayer.width = canvas.width; upperBodyLayer.height = canvas.height;
  }
  upperBodyContext.setTransform(1, 0, 0, 1, 0, 0);
  upperBodyContext.clearRect(0, 0, canvas.width, canvas.height);
  // Keep the layer opaque even if a browser rasterizer leaves fractional
  // coverage at an outer boundary. Deformation is sub-percent, so this underlay
  // is invisible except where it replaces what would otherwise be a black seam.
  upperBodyContext.drawImage(image, 0, 0, canvas.width, canvas.height);
  const geometry = upperBodyGeometry(rig);
  const { source, target } = geometry;
  motionCalibration.geometry = geometry;
  for (let row = 0; row < source.length - 1; row += 1) for (let column = 0; column < source[row].length - 1; column += 1) {
    drawMeshTriangle(image, [source[row][column], source[row + 1][column], source[row + 1][column + 1]], [target[row][column], target[row + 1][column], target[row + 1][column + 1]]);
    drawMeshTriangle(image, [source[row][column], source[row + 1][column + 1], source[row][column + 1]], [target[row][column], target[row + 1][column + 1], target[row][column + 1]]);
  }
  return upperBodyLayer;
}

function setMotionCalibration(state = 'neutral') {
  if (!motionCalibration.allowed) throw new Error('motion calibration requires ?motion-calibration=1');
  const states = {
    neutral: { breath: 0, amplitude: 0.68, pose: { sternumY: 0, shoulderLift: 0, torsoYaw: 0, headPitch: 0, headRoll: 0 } },
    inhale: { breath: 1, amplitude: 0.68, pose: { sternumY: 0, shoulderLift: 0, torsoYaw: 0, headPitch: 0, headRoll: 0 } },
    'phrase-left': { breath: 0.35, amplitude: 0.68, pose: { sternumY: 0.004, shoulderLift: 0.004, torsoYaw: -0.006, headPitch: 0, headRoll: -0.007 } },
    nod: { breath: 0.25, amplitude: 0.68, pose: { sternumY: 0.002, shoulderLift: 0.002, torsoYaw: 0, headPitch: 0.008, headRoll: 0 } },
  };
  if (state === 'live') {
    if (motionCalibration.saved) {
      const saved = motionCalibration.saved;
      upperBody.breath = saved.breath; upperBody.breathVelocity = saved.breathVelocity; upperBody.amplitude = saved.amplitude;
      upperBody.idleHeadRollTarget = saved.idleHeadRollTarget;
      Object.assign(upperBody.pose, saved.upperBodyPose); Object.assign(upperBody.poseVelocity, saved.upperBodyPoseVelocity);
      Object.assign(motion.current, saved.motionCurrent); Object.assign(motion.target, saved.motionTarget);
      Object.assign(autonomic.pose, saved.autonomicPose); Object.assign(autonomic.poseVelocity, saved.autonomicVelocity);
      Object.assign(autonomic.poseTarget, saved.autonomicTarget);
      Object.assign(semanticMotion.current, saved.semanticCurrent); Object.assign(semanticMotion.velocity, saved.semanticVelocity);
      Object.assign(semanticMotion.target, saved.semanticTarget);
      Object.assign(attention, saved.attention);
      const now = audioContext?.currentTime || 0;
      upperBody.lastAt = now; upperBody.stateStartedAt = now;
      autonomic.lastAt = now; semanticMotion.lastAt = now; attention.lastAt = now;
    }
    motionCalibration.active = false; motionCalibration.state = state; motionCalibration.saved = null;
    ensureAnimation();
    return null;
  }
  if (!(state in states)) throw new Error(`unknown motion calibration state: ${state}`);
  if (!motionCalibration.active) {
    motionCalibration.saved = {
      breath: upperBody.breath, breathVelocity: upperBody.breathVelocity, amplitude: upperBody.amplitude,
      idleHeadRollTarget: upperBody.idleHeadRollTarget,
      upperBodyPose: { ...upperBody.pose }, upperBodyPoseVelocity: { ...upperBody.poseVelocity },
      motionCurrent: { ...motion.current }, motionTarget: { ...motion.target },
      autonomicPose: { ...autonomic.pose }, autonomicVelocity: { ...autonomic.poseVelocity }, autonomicTarget: { ...autonomic.poseTarget },
      semanticCurrent: { ...semanticMotion.current }, semanticVelocity: { ...semanticMotion.velocity }, semanticTarget: { ...semanticMotion.target },
      attention: {
        current: attention.current, velocity: attention.velocity, target: attention.target,
        headCurrent: attention.headCurrent, headVelocity: attention.headVelocity,
        fixation: attention.fixation, nextAt: attention.nextAt, saccades: attention.saccades,
      },
    };
  }
  motionCalibration.active = true; motionCalibration.state = state;
  const selected = states[state];
  upperBody.breath = selected.breath; upperBody.breathVelocity = 0; upperBody.amplitude = selected.amplitude;
  upperBody.idleHeadRollTarget = 0;
  Object.assign(upperBody.pose, selected.pose);
  Object.keys(upperBody.poseVelocity).forEach((key) => { upperBody.poseVelocity[key] = 0; });
  Object.keys(motion.current).forEach((key) => { motion.current[key] = 0; motion.target[key] = 0; });
  Object.keys(autonomic.pose).forEach((key) => { autonomic.pose[key] = 0; autonomic.poseVelocity[key] = 0; autonomic.poseTarget[key] = 0; });
  Object.keys(semanticMotion.current).forEach((key) => { semanticMotion.current[key] = 0; semanticMotion.velocity[key] = 0; semanticMotion.target[key] = 0; });
  attention.current = 0.5; attention.target = 0.5; attention.velocity = 0;
  attention.headCurrent = 0.5; attention.headVelocity = 0;
  motionCalibration.geometry = upperBodyGeometry(upperBody.rig);
  ensureAnimation();
  return motionCalibration.geometry;
}

if (motionCalibration.allowed) window.setPortraitMotionCalibration = setMotionCalibration;

function drawTransformed(bitmap, alpha = 1) {
  const value = motion.current, blend = value.blend;
  // Webcam owns gross pose when present. Semantic motion remains visible as a
  // small residual and cannot fight a tracked person's head movement.
  const semanticGain = webcamFaceTracked ? 0.35 : 1;
  const semantic = semanticMotion.current;
  // ALP owns rendered face yaw/pitch through its sparse Euler bank. Attention
  // still schedules eye fixations, but its old whole-canvas head fallback is
  // intentionally removed: it fought the ALP direction and dragged the mouth.
  const attentionHead = 0;
  const browserRoll = upperBody.rig ? upperBody.pose.headRoll * semanticGain : autonomic.pose.rotation;
  const cover = 1.012 + 0.11 * blend;
  context.save();
  context.translate(
    canvas.width * (0.5 + value.x * blend + autonomic.pose.x + semantic.x * semanticGain + attentionHead * 0.0011),
    canvas.height * (0.5 + value.y * blend + autonomic.pose.y + semantic.y * semanticGain),
  );
  context.rotate(value.rotation * blend + semantic.rotation * semanticGain
    + browserRoll + attentionHead * 0.0018);
  context.transform(
    1, value.shearY * blend + (semantic.shearY + upperBody.pose.headPitch) * semanticGain,
    value.shearX * blend + semantic.shearX * semanticGain, 1, 0, 0,
  );
  const scale = cover * (1 + value.zoom * blend + semantic.zoom * semanticGain);
  context.scale(scale, scale);
  context.globalAlpha = alpha;
  const rendered = upperBodyWarp(bitmap);
  context.drawImage(rendered, -canvas.width / 2, -canvas.height / 2, canvas.width, canvas.height);
  context.restore();
}

function drawBlinkOverlay(bitmap, alpha) {
  if (!bitmap || !blinkFaceBox || alpha <= 0) return false;
  if (blinkLayer.width !== canvas.width || blinkLayer.height !== canvas.height) {
    blinkLayer.width = canvas.width; blinkLayer.height = canvas.height;
  }
  const [x1, y1, x2, y2] = blinkFaceBox;
  const faceWidth = Math.max(1, x2 - x1), faceHeight = Math.max(1, y2 - y1);
  const left = Math.max(0, x1 - faceWidth * 0.12);
  const right = Math.min(canvas.width, x2 + faceWidth * 0.12);
  const top = Math.max(0, y1 - faceHeight * 0.12);
  const bottom = Math.min(canvas.height, y1 + faceHeight * 0.58);
  const fadeStart = Math.max(top, y1 + faceHeight * 0.43);

  blinkContext.clearRect(0, 0, blinkLayer.width, blinkLayer.height);
  blinkContext.globalCompositeOperation = 'source-over';
  blinkContext.drawImage(bitmap, 0, 0, blinkLayer.width, blinkLayer.height);
  blinkContext.globalCompositeOperation = 'destination-in';
  const vertical = blinkContext.createLinearGradient(0, 0, 0, blinkLayer.height);
  const topFull = Math.min(fadeStart, Math.max(top, y1 - faceHeight * 0.02));
  vertical.addColorStop(0, 'rgba(0,0,0,0)');
  vertical.addColorStop(clamp(top / blinkLayer.height, 0, 1), 'rgba(0,0,0,0)');
  vertical.addColorStop(clamp(topFull / blinkLayer.height, 0, 1), 'rgba(0,0,0,1)');
  vertical.addColorStop(clamp(fadeStart / blinkLayer.height, 0, 1), 'rgba(0,0,0,1)');
  vertical.addColorStop(clamp(bottom / blinkLayer.height, 0, 1), 'rgba(0,0,0,0)');
  vertical.addColorStop(1, 'rgba(0,0,0,0)');
  blinkContext.fillStyle = vertical;
  blinkContext.fillRect(0, 0, blinkLayer.width, blinkLayer.height);
  const horizontal = blinkContext.createLinearGradient(0, 0, blinkLayer.width, 0);
  const innerLeft = Math.min(right, left + faceWidth * 0.10);
  const innerRight = Math.max(left, right - faceWidth * 0.10);
  horizontal.addColorStop(0, 'rgba(0,0,0,0)');
  horizontal.addColorStop(clamp(left / blinkLayer.width, 0, 1), 'rgba(0,0,0,0)');
  horizontal.addColorStop(clamp(innerLeft / blinkLayer.width, 0, 1), 'rgba(0,0,0,1)');
  horizontal.addColorStop(clamp(innerRight / blinkLayer.width, 0, 1), 'rgba(0,0,0,1)');
  horizontal.addColorStop(clamp(right / blinkLayer.width, 0, 1), 'rgba(0,0,0,0)');
  horizontal.addColorStop(1, 'rgba(0,0,0,0)');
  blinkContext.fillStyle = horizontal;
  blinkContext.fillRect(0, 0, blinkLayer.width, blinkLayer.height);
  blinkContext.globalCompositeOperation = 'source-over';
  drawTransformed(blinkLayer, alpha);
  return true;
}

function drawPair(lower, upper, mix, lowerIsSnapshot = false) {
  if (canvas.width !== lower.width || canvas.height !== lower.height) { canvas.width = lower.width; canvas.height = lower.height; }
  context.globalAlpha = 1; context.fillStyle = '#080a0d'; context.fillRect(0, 0, canvas.width, canvas.height);
  if (lowerIsSnapshot) context.drawImage(lower, 0, 0, canvas.width, canvas.height); else drawTransformed(lower);
  if (upper && mix > 0) drawTransformed(upper, mix);
  context.globalAlpha = 1;
}

function drawFrameMap(frames, position, loop, selectedIndices = null) {
  const indices = selectedIndices || [...frames.keys()].sort((a, b) => a - b);
  if (!indices.length) return false;
  const start = indices[0], span = indices.at(-1) - start + 1;
  const localPosition = loop ? position % span : Math.min(position, span - 1);
  const framePosition = start + localPosition;
  let lowerIndex = indices[0];
  for (const index of indices) if (index <= framePosition) lowerIndex = index;
  const upperIndex = indices.find((index) => index > lowerIndex) ?? (loop ? indices[0] : undefined);
  const lower = frames.get(lowerIndex), upper = frames.get(upperIndex);
  let mix = 0;
  if (upper) {
    const distance = upperIndex > lowerIndex ? upperIndex - lowerIndex : span - (lowerIndex - start) + (upperIndex - start);
    mix = clamp((framePosition - lowerIndex + (framePosition < lowerIndex ? span : 0)) / distance, 0, 1);
  }
  drawPair(lower, upper, mix); return true;
}

function drawWaitingFrame(now) {
  if (!readyPlayed && readyStartedAt !== null && readyFrames.size) {
    const position = (now - readyStartedAt) * idleFps;
    if (position < expectedReadyFrames) return drawFrameMap(readyFrames, position, false);
    readyPlayed = true;
    formStatus.textContent = portraitReady ? `${readyGesture.replace('-', ' ')} complete. Ready when you are.` : 'Settling into quiet micro-movements…';
  }
  const quietPosition = attention.current * Math.max(0, quietIdleIndices.length - 1);
  if (quietIdleIndices.length && blinkIdleIndices.length && blinkFaceBox && idleBlinkStartedAt === null && now >= nextIdleBlinkAt) {
    idleBlinkStartedAt = now;
    lastIdleBlinkAt = now;
    idleBlinkShape = {
      close: (0.055 + Math.random() * 0.023) * expressionFeedback.durationScale,
      hold: (0.010 + Math.random() * 0.014) * expressionFeedback.durationScale,
      open: (0.115 + Math.random() * 0.040) * expressionFeedback.durationScale,
    };
    blinkFeedbackUntil = now + idleBlinkShape.close + idleBlinkShape.hold + idleBlinkShape.open + 0.22;
    activeBlinkFeedbackId = ++blinkFeedbackId;
    if (expressionFeedback.lastMetrics) {
      expressionFeedback.blinkSamples = [{
        time: now,
        eyeOpen: (expressionFeedback.lastMetrics.leftEyeOpen + expressionFeedback.lastMetrics.rightEyeOpen) / 2,
      }];
    }
  }
  if (idleBlinkStartedAt !== null) {
    const elapsed = now - idleBlinkStartedAt;
    const amount = idleBlinkAmount(elapsed, idleBlinkShape);
    if (elapsed < idleBlinkShape.close + idleBlinkShape.hold + idleBlinkShape.open) {
      const blinkIndex = blinkIdleIndices.reduce((best, index) => (
        Math.abs(index - (blinkIdleIndices[0] + 2)) < Math.abs(best - (blinkIdleIndices[0] + 2)) ? index : best
      ), blinkIdleIndices[0]);
      drawFrameMap(idleFrames, quietPosition, false, quietIdleIndices);
      drawBlinkOverlay(idleFrames.get(blinkIndex), clamp(amount * expressionFeedback.peakScale, 0, 1));
      postBlinkFeedbackFrame(now);
      return true;
    }
    // The backend observer result can arrive just after the visual trajectory
    // ends. Let that final reopening sample land before closing the event.
    const completedBlinkId = activeBlinkFeedbackId;
    setTimeout(() => {
      finishBlinkFeedback(completedBlinkId);
      if (completedBlinkId === activeBlinkFeedbackId || activeBlinkFeedbackId === 0) blinkFeedbackUntil = -Infinity;
    }, 220);
    idleBlinkStartedAt = null;
    idleBlinkShape = null;
    nextIdleBlinkAt = now + nextBlinkInterval();
  }
  const drewIdle = drawFrameMap(idleFrames, quietPosition, false, quietIdleIndices.length ? quietIdleIndices : null);
  if (drewIdle && mouthSettleFrame && mouthSettleStartedAt !== null) {
    const mix = smoothstep((now - mouthSettleStartedAt) / 0.24);
    if (mix < 1) {
      context.save(); context.globalAlpha = 1 - mix;
      context.drawImage(mouthSettleFrame, 0, 0, canvas.width, canvas.height);
      context.restore();
    } else { mouthSettleFrame = null; mouthSettleStartedAt = null; }
  }
  return drewIdle;
}

function drawCurrentFrame(generation) {
  if (generation !== playbackGeneration) return;
  const now = audioContext.currentTime;
  if (!activeChunk) {
    tickMocap(now);
    if (drawWaitingFrame(now) || !portraitReady) requestAnimationFrame(() => drawCurrentFrame(generation));
    else animationRunning = false;
    return;
  }
  const chunk = chunkFor(activeChunk.seq);
  sampleAudio(now); tickMocap(now); updateAffectController(now);
  const requestedPosition = Math.max(0, now - activeChunk.startedAt + avPhaseSeconds) * chunk.fps;
  const position = Math.max(activeChunk.lastPosition, requestedPosition);
  activeChunk.lastPosition = position;
  const indices = [...chunk.frames.keys()].sort((a, b) => a - b);
  let lowerIndex = indices[0];
  for (const index of indices) if (index <= position) lowerIndex = index;
  const lower = chunk.frames.get(lowerIndex);
  if (lower) {
    if (now < activeChunk.startedAt && activeChunk.carryFrame) {
      // A webcam holds its previous complete frame during pre-roll. Dissolving
      // two full face frames produces two simultaneous mouth shapes and makes
      // phrase boundaries look soft or doubled.
      drawPair(activeChunk.carryFrame, null, 0, true);
    } else {
      // Present generated speech frames with sample-and-hold semantics, just as
      // a video/webcam element does. Full-frame alpha interpolation ghosts lip
      // and eyelid edges whenever adjacent source frames differ.
      drawPair(lower, null, 0);
    }
    for (const [index, oldBitmap] of chunk.frames) if (index < lowerIndex) { closeImage(oldBitmap); chunk.frames.delete(index); }
  }
  requestAnimationFrame(() => drawCurrentFrame(generation));
}

function ensureAnimation() {
  if (animationRunning) return;
  const generation = playbackGeneration;
  animationRunning = true; requestAnimationFrame(() => drawCurrentFrame(generation));
}

function finishTurnWhenPlaybackEnds() {
  if (!serverComplete || activeChunk || chunks.size) return;
  streamStatus.textContent = 'Alive and listening.';
  affectStatus.textContent = 'Neutral settle · reaction cooldown.';
  formStatus.textContent = 'Response complete; ask another question whenever you like.';
  setAskEnabled(true); promptInput.select();
}

function tryStartPlayback() {
  if (activeChunk) return;
  const chunk = chunks.get(nextPlaybackSeq);
  if (!chunk || !chunk.audio || chunk.frames.size === 0) return;
  const source = audioContext.createBufferSource(); source.buffer = chunk.audio;
  source.playbackRate.value = 1;
  const analyser = audioContext.createAnalyser(); analyser.fftSize = 512; analyser.smoothingTimeConstant = 0.25;
  source.connect(analyser); analyser.connect(audioContext.destination);
  const seq = nextPlaybackSeq, generation = playbackGeneration;
  const startedAt = audioContext.currentTime + 0.06;
  audioSamples = []; mouthSamples = []; lastPhaseUpdate = 0;
  activeChunk = { seq, source, analyser, waveform: new Float32Array(analyser.fftSize), startedAt, carryFrame: canvasSnapshot(), lastPosition: 0 };
  beginSpeechRespiration(startedAt);
  source.onended = () => {
    if (generation !== playbackGeneration) return;
    mouthSettleFrame = canvasSnapshot(); mouthSettleStartedAt = audioContext.currentTime;
    for (const bitmap of chunk.frames.values()) closeImage(bitmap);
    chunk.frames.clear(); chunks.delete(seq); nextPlaybackSeq = seq + 1; activeChunk = null; tryStartPlayback();
    if (!activeChunk) {
      attention.fixation = 'camera'; attention.target = 0.5;
      attention.nextAt = audioContext.currentTime + 1.4 + Math.random() * 2.2;
    }
    if (!activeChunk) affectStatus.textContent = 'Neutral settle · reaction cooldown.';
    finishTurnWhenPlaybackEnds();
  };
  source.start(startedAt); streamStatus.textContent = `Speaking phrase ${seq + 1} live…`; ensureAnimation();
}

async function receiveBinary(buffer) {
  const view = new DataView(buffer); if (view.byteLength < HEADER_BYTES) return;
  const kind = view.getUint8(0), seq = view.getUint32(1), index = view.getUint32(5), payload = buffer.slice(HEADER_BYTES);
  if (kind === READY_FRAME_PACKET || kind === IDLE_FRAME_PACKET) {
    const frames = kind === READY_FRAME_PACKET ? readyFrames : idleFrames;
    frames.set(index, await decodeImage(new Blob([payload], { type: 'image/jpeg' })));
    if (kind === IDLE_FRAME_PACKET) refreshIdleIndices();
    if (kind === READY_FRAME_PACKET && frames.size === expectedReadyFrames && readyStartedAt === null) {
      readyStartedAt = audioContext.currentTime;
    }
    ensureAnimation(); return;
  }
  const chunk = chunkFor(seq);
  if (kind === AUDIO_PACKET) chunk.audio = await audioContext.decodeAudioData(payload.slice(0));
  else if (kind === FRAME_PACKET) chunk.frames.set(index, await decodeImage(new Blob([payload], { type: 'image/jpeg' })));
  tryStartPlayback();
}

async function loadVoices() {
  try {
    const response = await fetch('/api/voices'), data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Voice service unavailable');
    voiceInput.replaceChildren(...data.voices.map((name) => new Option(name, name)));
    voiceInput.value = data.voices.includes('af_bella') ? 'af_bella' : data.voices[0];
  } catch (error) { formStatus.textContent = error.message; }
}

function setAskEnabled(enabled) {
  promptInput.disabled = !enabled; voiceInput.disabled = !enabled; speedInput.disabled = !enabled; goLive.disabled = !enabled;
}

async function preparePortrait(portrait) {
  setAskEnabled(false); answer.textContent = '';
  formStatus.textContent = 'Processing the portrait…'; streamStatus.textContent = 'Building its ready gesture and steady idle loop…';
  try {
    await makeWebcam(portrait);
    if (window.isSecureContext && navigator.mediaDevices?.getUserMedia && !mocapEnabled) {
      enableMocap().catch(async (error) => {
        await disableMocap();
        mocapStatus.textContent = `Automatic webcam mocap unavailable: ${error.message}`;
      });
    }
    ensureAnimation();
    const body = new FormData(); body.append('portrait', portrait);
    const response = await fetch('/api/live/start', { method: 'POST', body }), session = await response.json();
    if (!response.ok) throw new Error(session.detail || 'Could not start live session');
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${protocol}//${location.host}${session.websocket}`); socket.binaryType = 'arraybuffer';
    const generation = playbackGeneration;
    socket.onopen = () => { if (generation === playbackGeneration) formStatus.textContent = 'Teaching the portrait how to move…'; };
    socket.onmessage = async (message) => {
      if (generation !== playbackGeneration) return;
      try {
        if (message.data instanceof ArrayBuffer) { await receiveBinary(message.data); return; }
        const data = JSON.parse(message.data);
        if (data.type === 'assistant_start') formStatus.textContent = `${data.model} is answering…`;
        else if (data.type === 'assistant_delta') answer.textContent += data.text;
        else if (data.type === 'motion_preparing') streamStatus.textContent = 'Creating a one-time ready gesture and micro-expressions…';
        else if (data.type === 'motion_ready') streamStatus.textContent = `${data.gesture.replace('-', ' ')} ready; preparing the steady idle face…`;
        else if (data.type === 'ready_gesture') { readyGesture = data.gesture; idleFps = data.fps; expectedReadyFrames = data.frames; readyStartedAt = null; readyPlayed = false; }
        else if (data.type === 'ready_gesture_loaded') streamStatus.textContent = `Coming alive with a ${data.gesture.replace('-', ' ')}…`;
        else if (data.type === 'idle_start') idleFps = data.fps;
        else if (data.type === 'portrait_ready') {
          portraitReady = true; serverComplete = false; streamStatus.textContent = 'Alive and ready.';
          idleGestureStart = data.idle_gesture_start;
          blinkFaceBox = Array.isArray(data.face_bbox) && data.face_bbox.length === 4 ? data.face_bbox.map(Number) : null;
          setUpperBodyRig(data.upper_body_rig, audioContext.currentTime);
          refreshIdleIndices();
          nextIdleBlinkAt = audioContext.currentTime + 2.6 + Math.random() * 3.8;
          idleBlinkStartedAt = null; idleBlinkShape = null;
          attention.current = 0.5; attention.target = 0.5; attention.velocity = 0;
          attention.headCurrent = 0.5; attention.headVelocity = 0;
          attention.fixation = 'camera'; attention.nextAt = audioContext.currentTime + 2.0 + Math.random() * 2.6;
          formStatus.textContent = `Ready gesture: ${data.gesture.replace('-', ' ')}. Ask anything.`;
          promptInput.placeholder = 'Ask anything…'; setAskEnabled(true); promptInput.focus(); ensureAnimation();
        }
        else if (data.type === 'media_start') {
          const chunk = chunkFor(data.seq); chunk.fps = data.fps; chunk.duration = data.duration; chunk.reaction = data.reaction; chunk.prosody = data.prosody;
          latestMotionPlan = { text: data.text, reaction: data.reaction, prosody: data.prosody, seq: data.seq };
          refreshMotionValues();
        }
        else if (data.type === 'render_start') streamStatus.textContent = `Animating phrase ${data.seq + 1}…`;
        else if (data.type === 'complete') { serverComplete = true; formStatus.textContent = 'Response rendered; finishing live speech…'; finishTurnWhenPlaybackEnds(); }
        else if (data.type === 'error') { formStatus.textContent = data.detail; goLive.disabled = false; socket.close(); }
      } catch (error) { formStatus.textContent = error.message; goLive.disabled = false; socket.close(); }
    };
    socket.onerror = () => { if (generation === playbackGeneration) formStatus.textContent = 'Live connection failed.'; };
    socket.onclose = () => {
      if (generation !== playbackGeneration) return;
      if (formStatus.textContent.endsWith('…')) formStatus.textContent = 'Live connection closed.';
      setAskEnabled(false);
    };
  } catch (error) { formStatus.textContent = error.message; streamStatus.textContent = 'Could not prepare the live portrait.'; setAskEnabled(false); }
}

portraitInput.addEventListener('change', () => {
  const portrait = portraitInput.files[0]; if (portrait) preparePortrait(portrait);
});

form.addEventListener('submit', (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!portraitReady || !prompt || socket?.readyState !== WebSocket.OPEN) return;
  serverComplete = false; goLive.disabled = true; answer.textContent = ''; formStatus.textContent = 'Qwen is starting its answer…';
  streamStatus.textContent = 'Listening for the first spoken phrase…';
  socket.send(JSON.stringify({ prompt, voice: voiceInput.value, speed: Number(speedInput.value) }));
});

loadVoices();
ensureMocapWorker().catch(() => { /* status is set by the worker */ });
