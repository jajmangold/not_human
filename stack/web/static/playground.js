const $ = (selector) => document.querySelector(selector);
const form = $('#playground-form'), mode = $('#mode'), voice = $('#voice'), lang = $('#lang'), speed = $('#speed');
const referenceWrap = $('#reference-wrap'), sourceWrap = $('#source-wrap'), reference = $('#reference'), source = $('#source');
const audio = $('#audio'), save = $('#save'), takes = $('#takes');
const labSource = $('#lab-source'), labTarget = $('#lab-target'), labFactor = $('#lab-factor');
let latest = null, saved = [];

function updateMode() {
  const cloning = mode.value !== 'preset';
  referenceWrap.hidden = !cloning; sourceWrap.hidden = mode.value !== 'convert';
  reference.required = cloning; source.required = mode.value === 'convert';
}
speed.addEventListener('input', () => $('#speed-value').textContent = `${Number(speed.value).toFixed(2)}×`);
mode.addEventListener('change', updateMode); updateMode();

async function boot() {
  try {
    const capabilities = await fetch('/api/playground/capabilities').then((response) => response.json());
    const voicePayload = await fetch('/api/voices').then((response) => response.json()).catch(() => ({}));
    const voices = voicePayload.voices || ['af_bella', 'af_heart', 'af_sarah', 'am_adam', 'am_michael', 'bf_emma', 'bf_isabella', 'bm_daniel', 'bm_george'];
    voice.replaceChildren(...voices.map((name) => new Option(name, name)));
    const labVoices = voices.filter((name) => !name.startsWith('custom_'));
    labSource.replaceChildren(...labVoices.map((name) => new Option(name, name)));
    labTarget.replaceChildren(...labVoices.map((name) => new Option(name, name)));
    voice.insertAdjacentHTML('afterbegin', '<option value="af_heart,af_bella">af_heart + af_bella (blend)</option>');
    voice.value = 'af_heart'; labSource.value = 'af_heart'; labTarget.value = 'am_eric';
    document.querySelector('.voice-lab-card').hidden = !capabilities.voice_lab;
    $('#capability').textContent = `Kokoro ${capabilities.kokoro ? 'ready' : 'offline'} · KokoClone ${capabilities.kokoclone ? 'ready' : 'optional / warming'} · Gemma ${capabilities.gemma_models?.length ? 'available' : 'unavailable'}`;
  } catch (error) { $('#capability').textContent = 'Service check failed; direct generation may still recover.'; }
}

function updateLabFactor() {
  const value = Number(labFactor.value); const label = value === 0 ? 'midpoint' : value < -1 ? 'beyond source' : value > 1 ? 'beyond target' : value < 0 ? 'toward source' : 'toward target';
  $('#lab-factor-value').textContent = `${value.toFixed(2)} · ${label}`;
}
labFactor.addEventListener('input', updateLabFactor); updateLabFactor();

$('#lab-generate').addEventListener('click', async () => {
  const button = $('#lab-generate'); button.disabled = true; $('#lab-status').textContent = 'Rendering style-vector preview…';
  const body = new FormData(); body.append('text', $('#text').value); body.append('source', labSource.value); body.append('target', labTarget.value); body.append('factor', labFactor.value); body.append('name', $('#lab-name').value); body.append('speed', speed.value); body.append('lang', lang.value);
  try { const response = await fetch('/api/playground/voice-lab', {method: 'POST', body}); const result = await response.json(); if (!response.ok) throw new Error(result.detail || 'voice lab failed'); $('#lab-audio').hidden = false; $('#lab-audio').src = result.audio_url; $('#lab-audio').load(); $('#lab-status').textContent = result.saved ? `Saved ${result.voice}. It is now available in the voice menu after refresh.` : 'Ephemeral preview ready. Save it if you want to reuse it.'; }
  catch (error) { $('#lab-status').textContent = error.message; } finally { button.disabled = false; }
});

function addTake(result) {
  const card = document.createElement('article'); card.className = 'take';
  card.innerHTML = `<div><strong>${result.mode}</strong><span>${result.voice} · ${Number(result.duration).toFixed(2)}s</span></div><audio controls src="${result.audio_url}"></audio><small>${result.plan.delivery}</small>`;
  takes.querySelector('.status')?.remove(); takes.prepend(card);
}

form.addEventListener('submit', async (event) => {
  event.preventDefault(); const button = $('#generate'); button.disabled = true; $('#form-status').textContent = 'Generating…';
  const body = new FormData(); body.append('text', $('#text').value); body.append('mode', mode.value); body.append('voice', voice.value); body.append('speed', speed.value); body.append('lang', lang.value); body.append('style', $('#style').value); body.append('director', $('#director').checked ? 'true' : 'false'); body.append('director_model', 'gemma4-26b');
  if (reference.files[0]) body.append('reference', reference.files[0]); if (source.files[0]) body.append('source', source.files[0]);
  try {
    const response = await fetch('/api/playground/generate', {method: 'POST', body}); const result = await response.json(); if (!response.ok) throw new Error(result.detail || 'generation failed');
    latest = result; audio.src = result.audio_url; audio.load(); $('#result-title').textContent = `${result.mode} · ${result.voice}`; $('#delivery').textContent = `${result.plan.delivery} · ${result.plan.model}`; $('#normalized').textContent = result.normalized_text; $('#metadata').textContent = JSON.stringify(result, null, 2); save.disabled = false; $('#form-status').textContent = 'Ready. Listen, save it, then change one thing and try again.';
  } catch (error) { $('#form-status').textContent = error.message; } finally { button.disabled = false; }
});
save.addEventListener('click', () => { if (latest) { addTake(latest); saved.push(latest); save.disabled = true; } });
$('#clear').addEventListener('click', () => { saved = []; takes.innerHTML = '<p class="status">Saved takes will appear here for A/B listening.</p>'; });
boot();
