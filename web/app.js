/* Wiring. Fetches health, drives the capture, posts the analysis.
 *
 * The one behaviour worth stating: the image is posted to 127.0.0.1 and
 * nowhere else. There is no telemetry call in this file and no third-party
 * origin anywhere in the page, which is why there is no build step and no CDN
 * font. A page that fetches a stylesheet from someone else's server has
 * already told them a face-analysis tool was opened.
 */

import { Capture } from '/static/capture.js';
import { renderReport, renderCatalogue } from '/static/report.js';

const $ = (sel) => document.querySelector(sel);

const els = {
  video: $('#video'),
  overlay: $('#overlay'),
  still: $('#still'),
  viewport: $('#viewport'),
  placeholder: $('#placeholder'),
  guidance: $('#guidance'),
  start: $('#start'),
  shutter: $('#shutter'),
  retake: $('#retake'),
  file: $('#file'),
  profile: $('#profile'),
  analyse: $('#analyse'),
  showcat: $('#showcat'),
  status: $('#status'),
  results: $('#results'),
  readout: $('#readout'),
  fields: $('#fields'),
};

let profileBlob = null;

const capture = new Capture({
  video: els.video,
  overlay: els.overlay,
  still: els.still,
  viewport: els.viewport,
  placeholder: els.placeholder,
  onMetrics: paintMetrics,
});

function field(root, name) {
  return root.querySelector(`[data-field="${name}"]`);
}

function paintMetrics(m) {
  if (!m || m.luma === undefined) return;
  const exposure = field(els.guidance, 'exposure');
  exposure.textContent = `${m.exposure} (${m.luma.toFixed(0)})`;
  exposure.className = m.exposure === 'ok' ? '' : 'warn';

  const focus = field(els.guidance, 'focus');
  focus.textContent = `${m.focusVerdict} (${m.focus.toFixed(0)})`;
  focus.className = m.focusVerdict === 'ok' ? '' : 'warn';

  const tilt = field(els.guidance, 'tilt');
  if (m.tiltDeg === null || m.tiltDeg === undefined) {
    tilt.textContent = 'no sensor';
    tilt.className = '';
  } else {
    tilt.textContent = `${m.tiltDeg.toFixed(1)}°`;
    tilt.className = Math.abs(m.tiltDeg) < 2 ? '' : 'warn';
  }
}

function say(text, working = false) {
  els.status.hidden = false;
  els.status.textContent = text;
  els.status.className = working ? 'status working' : 'status';
}

function clearStatus() {
  els.status.hidden = true;
  els.status.textContent = '';
}

async function health() {
  try {
    const res = await fetch('/health');
    const h = await res.json();
    field(els.readout, 'device').textContent = h.device;
    field(els.readout, 'pipeline').textContent = h.pipeline_available ? 'ready' : 'absent';
    field(els.readout, 'tier').textContent = h.license_tier;
    field(els.readout, 'store').textContent = h.storing_uploads ? 'stored' : 'in memory';
    field(els.readout, 'n').textContent = `${h.n_measurements} measurements`;
    if (!h.pipeline_available) {
      say(
        'This build has no analysis pipeline installed, so Analyse will return 503. ' +
          'The catalogue below is live.',
      );
    }
  } catch (err) {
    say(`could not reach the local server: ${err.message}`);
  }
}

async function showCatalogue() {
  const res = await fetch('/catalogue');
  renderCatalogue(els.results, await res.json());
}

function ready() {
  els.analyse.disabled = !capture.blob;
  els.shutter.disabled = !capture.live;
  els.retake.disabled = !capture.blob;
}

els.start.addEventListener('click', async () => {
  try {
    clearStatus();
    await capture.start();
    els.start.textContent = 'Restart camera';
  } catch (err) {
    say(`camera unavailable: ${err.message}. Use a file instead.`);
  }
  ready();
});

els.shutter.addEventListener('click', async () => {
  try {
    await capture.capture();
    clearStatus();
  } catch (err) {
    say(err.message);
  }
  ready();
});

els.retake.addEventListener('click', async () => {
  try {
    await capture.retake();
  } catch (err) {
    say(err.message);
  }
  ready();
});

els.file.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  await capture.adopt(file);
  clearStatus();
  ready();
});

els.profile.addEventListener('change', (e) => {
  profileBlob = e.target.files[0] || null;
  if (profileBlob) say(`profile view attached: ${profileBlob.name}`, true);
});

els.showcat.addEventListener('click', () => showCatalogue());

els.analyse.addEventListener('click', async () => {
  if (!capture.blob) return;
  const form = new FormData();
  form.append('frontal', capture.blob, 'frontal.jpg');
  if (profileBlob) form.append('profile', profileBlob, 'profile.jpg');
  for (const [name, value] of new FormData(els.fields).entries()) {
    if (value !== '') form.append(name, value);
  }

  els.analyse.disabled = true;
  say('measuring, and propagating landmark covariance through every formula…', true);
  try {
    const res = await fetch('/analyze', { method: 'POST', body: form });
    const payload = await res.json();
    if (res.ok) {
      clearStatus();
      renderReport(els.results, payload);
      els.results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      const reasons = (payload.reasons || []).map((r) => `\n  ${r}`).join('');
      say(`${res.status} ${payload.status || ''}: ${payload.detail || 'no detail'}${reasons}`);
    }
  } catch (err) {
    say(`request failed: ${err.message}`);
  }
  els.analyse.disabled = false;
});

health();
showCatalogue();
ready();
