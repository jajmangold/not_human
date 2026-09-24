const portraitInput = document.querySelector('#editor-portrait');
const sourceImage = document.querySelector('#editor-source');
const outputImage = document.querySelector('#editor-output');
const status = document.querySelector('#editor-status');
const sliders = document.querySelector('#editor-sliders');
const resetButton = document.querySelector('#editor-reset');
const copyButton = document.querySelector('#editor-copy');
const valuesStatus = document.querySelector('#editor-values-status');
const valuesOutput = document.querySelector('#editor-values');

const FALLBACK_CONTROLS = {
  rotate_pitch: { label: 'Rotate Pitch', min: -20, max: 20, step: 0.5, default: 0 },
  rotate_yaw: { label: 'Rotate Yaw', min: -20, max: 20, step: 0.5, default: 0 },
  rotate_roll: { label: 'Rotate Roll', min: -20, max: 20, step: 0.5, default: 0 },
  blink: { label: 'Blink', min: -20, max: 20, step: 0.5, default: 0 },
  eyebrow: { label: 'Eyebrow', min: -40, max: 20, step: 0.5, default: 0 },
  wink: { label: 'Wink', min: 0, max: 25, step: 0.5, default: 0 },
  pupil_x: { label: 'Pupil X', min: -20, max: 20, step: 0.5, default: 0 },
  pupil_y: { label: 'Pupil Y', min: -20, max: 20, step: 0.5, default: 0 },
  aaa: { label: 'AAA', min: -30, max: 120, step: 1, default: 0 },
  eee: { label: 'EEE', min: -20, max: 20, step: 0.2, default: 0 },
  woo: { label: 'WOO', min: -20, max: 20, step: 0.2, default: 0 },
  smile: { label: 'Smile', min: -2, max: 2, step: 0.01, default: 0 },
};

let controls = FALLBACK_CONTROLS;
let state = Object.fromEntries(Object.entries(controls).map(([name, spec]) => [name, Number(spec.default)]));
let sourceFile = null;
let sourceUrl = null;
let outputUrl = null;
let queued = false;
let rendering = false;
let requestNumber = 0;

function prettyNumber(value) {
  const number = Number(value);
  return Number.isInteger(number) ? String(number) : number.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
}

function snapshot() {
  return {
    endpoint: '/api/motion/edit',
    source: sourceFile?.name || null,
    controls: Object.fromEntries(Object.entries(state).map(([name, value]) => [name, Number(value)])),
    control_ranges: controls,
    updated_at: new Date().toISOString(),
  };
}

function refreshValues() {
  valuesOutput.textContent = JSON.stringify(snapshot(), null, 2);
}

function scheduleRender() {
  queued = true;
  refreshValues();
  if (rendering || !sourceFile) return;
  window.setTimeout(renderLatest, 90);
}

async function renderLatest() {
  if (rendering || !queued || !sourceFile) return;
  queued = false; rendering = true;
  const currentRequest = ++requestNumber;
  const form = new FormData(); form.append('portrait', sourceFile, sourceFile.name);
  for (const [name, value] of Object.entries(state)) form.append(name, String(value));
  status.textContent = 'Rendering…';
  try {
    const response = await fetch('/api/motion/edit', { method: 'POST', body: form });
    if (!response.ok) {
      let detail = `Render failed (${response.status})`;
      try { detail = (await response.json()).detail || detail; } catch (_) { /* plain error */ }
      throw new Error(detail);
    }
    const blob = await response.blob();
    if (currentRequest !== requestNumber) return;
    if (outputUrl) URL.revokeObjectURL(outputUrl);
    outputUrl = URL.createObjectURL(blob); outputImage.src = outputUrl;
    status.textContent = 'Live expression ready.';
  } catch (error) {
    if (currentRequest === requestNumber) status.textContent = error.message;
  } finally {
    rendering = false;
    if (queued) window.setTimeout(renderLatest, 90);
  }
}

function renderSliders() {
  sliders.replaceChildren();
  for (const [name, spec] of Object.entries(controls)) {
    const label = document.createElement('label');
    const title = document.createElement('span'); title.textContent = spec.label;
    const row = document.createElement('span'); row.className = 'slider-row';
    const input = document.createElement('input'); input.type = 'range'; input.min = spec.min; input.max = spec.max; input.step = spec.step; input.value = state[name]; input.dataset.control = name;
    const output = document.createElement('output'); output.textContent = prettyNumber(state[name]); output.dataset.valueFor = name;
    input.addEventListener('input', () => {
      state[name] = Number(input.value); output.textContent = prettyNumber(state[name]); scheduleRender();
    });
    row.append(input, output); label.append(title, row); sliders.append(label);
  }
}

function reset() {
  state = Object.fromEntries(Object.entries(controls).map(([name, spec]) => [name, Number(spec.default)]));
  renderSliders(); scheduleRender();
}

portraitInput.addEventListener('change', () => {
  sourceFile = portraitInput.files?.[0] || null;
  if (sourceUrl) URL.revokeObjectURL(sourceUrl);
  sourceUrl = sourceFile ? URL.createObjectURL(sourceFile) : null;
  sourceImage.src = sourceUrl || ''; outputImage.removeAttribute('src');
  status.textContent = sourceFile ? 'Rendering neutral frame…' : 'Choose a portrait to begin.';
  scheduleRender();
});

resetButton.addEventListener('click', reset);
copyButton.addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(JSON.stringify(snapshot(), null, 2)); valuesStatus.textContent = 'Copied.'; }
  catch (_) { valuesStatus.textContent = 'Clipboard unavailable; select the JSON above.'; }
});

fetch('/api/motion/controls').then((response) => response.ok ? response.json() : null).then((schema) => {
  if (schema?.controls) controls = schema.controls;
  state = Object.fromEntries(Object.entries(controls).map(([name, spec]) => [name, Number(spec.default)]));
  renderSliders(); refreshValues();
}).catch(() => { renderSliders(); refreshValues(); });

window.portraitMotionEditor = { getValues: snapshot, render: scheduleRender, reset };
