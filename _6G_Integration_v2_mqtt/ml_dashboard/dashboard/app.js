// ml_dashboard frontend — talks only to its own backend (same origin).

const API = {
  status:    '/api/status',
  versions:  '/api/versions',
  retrain:   '/api/retrain',
  promote:   '/api/promote',
  switch:    '/api/switch',
};

let _retrainPollHandle = null;

// ---------------------------------------------------------------------------
// Status panel
// ---------------------------------------------------------------------------
async function refreshStatus() {
  const panel = document.getElementById('status-panel');
  panel.innerHTML = '<div class="loading">Loading...</div>';
  try {
    const r = await fetch(API.status);
    const data = await r.json();

    const inf  = data.inference_server || {};
    const reg  = data.registry || {};
    const cells = [];

    // Inference server status
    if (inf.error) {
      cells.push(cell('Inference server', `unreachable: ${inf.error}`, 'error'));
    } else {
      cells.push(cell('Currently loaded', inf.loaded_as || inf.model_version || '—',
                      inf.loaded_as && inf.loaded_as.startsWith('mlflow:') ? 'ok' : 'warn'));
      cells.push(cell('Model version (base)', inf.model_version || '—'));
      cells.push(cell('Uses barometer', String(inf.uses_barometer ?? '—')));
      cells.push(cell('Uptime',
                      inf.uptime_seconds != null ? Math.round(inf.uptime_seconds) + 's' : '—'));
    }

    // Registry status
    if (reg.error) {
      cells.push(cell('Production alias', `not set (${reg.error})`, 'warn'));
    } else if (reg.version != null) {
      cells.push(cell('Production version', `v${reg.version}`, 'ok'));
    }

    // Drift between alias and loaded
    if (data.alias_matches_loaded === false) {
      cells.push(cell(
        'Sync',
        'Production alias has moved but inference server still on old version — click hot-swap',
        'warn'
      ));
    } else if (data.alias_matches_loaded === true) {
      cells.push(cell('Sync', 'Server is on the Production version', 'ok'));
    }

    panel.innerHTML = cells.join('');
  } catch (e) {
    panel.innerHTML = `<div class="status-cell"><div class="status-value error">Failed: ${e.message}</div></div>`;
  }
}

function cell(label, value, cls) {
  return `<div class="status-cell">
    <div class="status-label">${escapeHtml(label)}</div>
    <div class="status-value ${cls || ''}">${escapeHtml(String(value))}</div>
  </div>`;
}

// ---------------------------------------------------------------------------
// Registered versions
// ---------------------------------------------------------------------------
async function loadVersions() {
  const tbody = document.getElementById('versions-tbody');
  tbody.innerHTML = '<tr><td colspan="4" class="loading">Loading...</td></tr>';
  try {
    const r = await fetch(API.versions);
    const data = await r.json();
    if (!data.versions || data.versions.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="loading">No registered versions yet — run a retrain with --register</td></tr>';
      return;
    }
    tbody.innerHTML = data.versions.map(v => `
      <tr>
        <td><strong>v${v.version}</strong></td>
        <td>${formatTime(v.creation_time)}</td>
        <td>${(v.aliases || []).map(a =>
          `<span class="alias-tag ${a.toLowerCase()}">${escapeHtml(a)}</span>`).join('') || '—'}</td>
        <td>
          <button class="btn-mini" onclick="promote(${v.version}, 'Production')">Set Production</button>
          <button class="btn-mini" onclick="promote(${v.version}, 'Staging')">Set Staging</button>
        </td>
      </tr>
    `).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="4" class="status-line error">Failed: ${e.message}</td></tr>`;
  }
}

async function promote(version, alias) {
  if (alias === 'Production' &&
      !confirm(`Move the Production alias to v${version}?\n\n` +
               `This will route /model/switch to v${version} on the next hot-swap.`)) {
    return;
  }
  try {
    const r = await fetch(API.promote, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ version, alias }),
    });
    if (!r.ok) {
      const t = await r.text();
      alert(`Promote failed: ${t}`);
      return;
    }
    await loadVersions();
    await refreshStatus();
  } catch (e) {
    alert(`Promote failed: ${e.message}`);
  }
}

// ---------------------------------------------------------------------------
// Retrain — POST + poll for output
// ---------------------------------------------------------------------------
async function startRetrain() {
  const btn        = document.getElementById('retrain-btn');
  const statusEl   = document.getElementById('retrain-status');
  const logEl      = document.getElementById('retrain-log');
  const modelVer   = document.getElementById('retrain-version').value.trim() || 'v0';
  const dataset    = document.getElementById('retrain-dataset').value;
  const doRegister = document.getElementById('retrain-register').checked;

  btn.disabled = true;
  statusEl.textContent = 'Starting...';
  statusEl.className = 'status-line';
  logEl.textContent = '';

  try {
    const r = await fetch(API.retrain, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_version: modelVer, dataset, do_register: doRegister }),
    });
    if (!r.ok) {
      statusEl.textContent = `Failed to start: ${await r.text()}`;
      statusEl.className = 'status-line error';
      btn.disabled = false;
      return;
    }
    const { job_id } = await r.json();
    pollRetrain(job_id, btn, statusEl, logEl);
  } catch (e) {
    statusEl.textContent = `Failed to start: ${e.message}`;
    statusEl.className = 'status-line error';
    btn.disabled = false;
  }
}

function pollRetrain(jobId, btn, statusEl, logEl) {
  if (_retrainPollHandle) clearInterval(_retrainPollHandle);

  const tick = async () => {
    try {
      const r = await fetch(`${API.retrain}/${jobId}`);
      if (!r.ok) {
        statusEl.textContent = `Poll failed: ${await r.text()}`;
        statusEl.className = 'status-line error';
        return;
      }
      const job = await r.json();
      logEl.textContent = (job.output || []).join('\n');
      logEl.scrollTop = logEl.scrollHeight;
      statusEl.textContent = `${job.status}  •  ${job.duration}s  •  ${jobId}`;

      if (job.status === 'ok') {
        statusEl.className = 'status-line ok';
        clearInterval(_retrainPollHandle);
        btn.disabled = false;
        loadVersions();
        refreshStatus();
      } else if (job.status === 'failed') {
        statusEl.className = 'status-line error';
        clearInterval(_retrainPollHandle);
        btn.disabled = false;
      } else {
        statusEl.className = 'status-line';
      }
    } catch (e) {
      statusEl.textContent = `Poll error: ${e.message}`;
      statusEl.className = 'status-line error';
    }
  };
  tick();
  _retrainPollHandle = setInterval(tick, 1500);
}

// ---------------------------------------------------------------------------
// Hot-swap
// ---------------------------------------------------------------------------
async function hotswap(alias) {
  if (!confirm(`Hot-swap the live inference server to alias "${alias}"?\n\n` +
               `This affects real prediction traffic.`)) {
    return;
  }
  await doSwitch({ mlflow_alias: alias });
}

async function hotswapVersion() {
  if (!confirm('Roll back to the file-based v0 model?')) return;
  await doSwitch({ version: 'v0' });
}

async function doSwitch(body) {
  const statusEl = document.getElementById('switch-status');
  statusEl.textContent = 'Switching...';
  statusEl.className = 'status-line';
  try {
    const r = await fetch(API.switch, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const text = await r.text();
    if (!r.ok) {
      statusEl.textContent = `Failed: ${text}`;
      statusEl.className = 'status-line error';
      return;
    }
    statusEl.textContent = `OK: ${text}`;
    statusEl.className = 'status-line ok';
    setTimeout(refreshStatus, 500);
  } catch (e) {
    statusEl.textContent = `Failed: ${e.message}`;
    statusEl.className = 'status-line error';
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function formatTime(ms) {
  if (!ms) return '—';
  try { return new Date(ms).toLocaleString(); } catch { return String(ms); }
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
refreshStatus();
loadVersions();
setInterval(refreshStatus, 10000);
