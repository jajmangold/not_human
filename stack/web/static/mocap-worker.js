import { FaceLandmarker, FilesetResolver } from '/static/vendor/mediapipe/vision_bundle.mjs';

let landmarker;
let role;
let inputCanvas;
let inputContext;

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function meanPoint(points, indices) {
  return {
    x: indices.reduce((sum, index) => sum + points[index].x, 0) / indices.length,
    y: indices.reduce((sum, index) => sum + points[index].y, 0) / indices.length,
  };
}

function normalizedIris(points, irisIndices, cornerIndices, eyeDistance) {
  const iris = meanPoint(points, irisIndices);
  const first = points[cornerIndices[0]], second = points[cornerIndices[1]];
  const corners = [first.x, second.x].sort((a, b) => a - b);
  const span = Math.max(0.001, corners[1] - corners[0]);
  const cornerDx = second.x - first.x;
  const t = clamp((iris.x - first.x) / (Math.abs(cornerDx) >= 0.001 ? cornerDx : 0.001), 0, 1);
  const cornerLineY = first.y + t * (second.y - first.y);
  return {
    x: clamp(2 * ((iris.x - corners[0]) / span - 0.5), -1, 1),
    y: clamp(4 * ((iris.y - cornerLineY) / Math.max(0.001, eyeDistance)), -1, 1),
  };
}

function derollLandmarks(points, center, roll) {
  const cosine = Math.cos(-roll), sine = Math.sin(-roll);
  return points.map((point) => {
    const dx = point.x - center.x, dy = point.y - center.y;
    return {
      x: cosine * dx - sine * dy + center.x,
      y: sine * dx + cosine * dy + center.y,
    };
  });
}

function metrics(result) {
  const rawPoints = result.faceLandmarks?.[0];
  if (!rawPoints || rawPoints.length < 478) return null;
  const rawLeftEye = rawPoints[33], rawRightEye = rawPoints[263];
  const dx = rawRightEye.x - rawLeftEye.x, dy = rawRightEye.y - rawLeftEye.y;
  const eyeDistance = Math.max(0.001, Math.hypot(dx, dy));
  const eyeMidX = (rawLeftEye.x + rawRightEye.x) / 2;
  const eyeMidY = (rawLeftEye.y + rawRightEye.y) / 2;
  const roll = Math.atan2(dy, dx);
  // Keep the webcam worker's measurements aligned with the backend observer:
  // de-roll shape metrics and normalize gazeY to the eye-corner line, not the
  // aperture-sensitive lid gap.
  const points = derollLandmarks(rawPoints, { x: eyeMidX, y: eyeMidY }, roll);
  const leftEye = points[33], rightEye = points[263], nose = points[1];
  const leftUpper = points[159], leftLower = points[145];
  const rightUpper = points[386], rightLower = points[374];
  const leftBrow = points[105], rightBrow = points[334];
  const upperLip = points[13], lowerLip = points[14];
  const leftMouth = points[61], rightMouth = points[291];
  const leftEyeOpen = clamp(Math.hypot(leftLower.x - leftUpper.x, leftLower.y - leftUpper.y) / eyeDistance, 0, 0.5);
  const rightEyeOpen = clamp(Math.hypot(rightLower.x - rightUpper.x, rightLower.y - rightUpper.y) / eyeDistance, 0, 0.5);
  const leftBrowValue = clamp((leftUpper.y - leftBrow.y) / eyeDistance, -0.2, 0.8);
  const rightBrowValue = clamp((rightUpper.y - rightBrow.y) / eyeDistance, -0.2, 0.8);
  const leftGaze = normalizedIris(points, [468, 469, 470, 471, 472], [33, 133], eyeDistance);
  const rightGaze = normalizedIris(points, [473, 474, 475, 476, 477], [362, 263], eyeDistance);
  return {
    centerX: eyeMidX,
    centerY: eyeMidY,
    eyeDistance,
    roll,
    yaw: clamp((nose.x - eyeMidX) / eyeDistance, -0.8, 0.8),
    pitch: clamp((nose.y - eyeMidY) / eyeDistance, 0.1, 1.8),
    mouth: clamp(Math.hypot(lowerLip.x - upperLip.x, lowerLip.y - upperLip.y) / eyeDistance, 0, 0.8),
    leftEyeOpen,
    rightEyeOpen,
    eyeAsymmetry: clamp(leftEyeOpen - rightEyeOpen, -0.25, 0.25),
    gazeX: clamp((leftGaze.x + rightGaze.x) / 2, -1, 1),
    gazeY: clamp((leftGaze.y + rightGaze.y) / 2, -1, 1),
    gazeVergence: clamp(leftGaze.x - rightGaze.x, -1, 1),
    leftBrow: leftBrowValue,
    rightBrow: rightBrowValue,
    browAsymmetry: clamp(leftBrowValue - rightBrowValue, -0.5, 0.5),
    smile: clamp(((upperLip.y + lowerLip.y) / 2 - (leftMouth.y + rightMouth.y) / 2) / eyeDistance, -0.4, 0.4),
    smirk: clamp((rightMouth.y - leftMouth.y) / eyeDistance, -0.4, 0.4),
  };
}

async function createLandmarker(fileset, model, delegate) {
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetBuffer: new Uint8Array(model.slice(0)), delegate },
    // Explicit worker-owned canvas avoids the internal DOM/WebKit fallback and
    // gives the graph one stable GL token for its lifetime.
    canvas: new OffscreenCanvas(1, 1),
    // IMAGE mode avoids the task runtime's broken cross-frame GL texture-token
    // reuse on this Chromium build. The controller already rate-limits frames
    // and performs its own temporal filtering.
    runningMode: 'IMAGE',
    numFaces: 1,
    minFaceDetectionConfidence: 0.45,
    minFacePresenceConfidence: 0.45,
    minTrackingConfidence: 0.45,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  });
}

async function initialize(workerRole) {
  role = workerRole;
  // The module-worker flag selects vision_wasm_module_internal.js. Without it,
  // the classic loader keeps ModuleFactory in the wrong global scope.
  const fileset = await FilesetResolver.forVisionTasks('/static/vendor/mediapipe/wasm', true);
  const response = await fetch('/api/mocap/model');
  if (!response.ok) throw new Error(`Face Landmarker model failed (${response.status})`);
  const model = await response.arrayBuffer();
  // XNNPACK is stable for both small feedback samples and webcam frames.
  // WebGL can initialize successfully and then abort on repeated canvas
  // ImageBitmaps, producing a frame-error storm instead of falling back.
  const delegate = 'CPU';
  landmarker = await createLandmarker(fileset, model, delegate);
  postMessage({ type: 'ready', role, delegate });
}

self.onmessage = async ({ data }) => {
  if (data.type === 'init') {
    try { await initialize(data.role); } catch (error) { postMessage({ type: 'error', role: data.role, detail: error.message }); }
    return;
  }
  if (data.type === 'close') {
    landmarker?.close(); close(); return;
  }
  if (data.type !== 'frame') return;
  try {
    let input = data.bitmap;
    if (data.pixels) {
      if (!inputCanvas || inputCanvas.width !== data.width || inputCanvas.height !== data.height) {
        inputCanvas = new OffscreenCanvas(data.width, data.height);
        inputContext = inputCanvas.getContext('2d', { alpha: false });
      }
      inputContext.putImageData(
        new ImageData(new Uint8ClampedArray(data.pixels), data.width, data.height), 0, 0,
      );
      input = inputCanvas;
    }
    const result = landmarker.detect(input);
    data.bitmap?.close();
    postMessage({ type: 'result', stream: role, seq: data.seq, sampleTime: data.sampleTime, metrics: metrics(result) });
  } catch (error) {
    data.bitmap?.close();
    postMessage({ type: 'frame-error', stream: role, detail: error.message });
  }
};
