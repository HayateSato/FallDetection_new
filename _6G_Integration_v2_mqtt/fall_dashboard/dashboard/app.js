// Caregiver dashboard — 6G/Charite

const API = {
  patients: '/api/patients',
  falls:    '/api/falls',
  stream:   '/api/stream',
};

let inDetailView = false;
let eventSource = null;

// observation_id -> patient_id for in-flight possible-fall badges
const pendingAlerts = new Map();

// ---------------------------------------------------------------------------
// Patients tab
// ---------------------------------------------------------------------------
async function loadPatients() {
  const list = document.getElementById('patient-list');
  list.innerHTML = '<p class="loading">Loading patients...</p>';
  try {
    const resp = await fetch(API.patients);
    const data = await resp.json();
    if (!data.patients || data.patients.length === 0) {
      list.innerHTML = '<p class="empty">No patients registered yet. Set PATIENT_IDS in .env.</p>';
      document.getElementById('stat-patients').textContent = '0';
      return;
    }
    list.innerHTML = data.patients.map((p) => `
      <div class="patient-card ${p.fall_count > 0 ? 'has-falls' : ''}" data-patient-id="${escapeHtml(p.patient_id)}" onclick="openPatient('${escapeHtml(p.patient_id)}')">
        <div class="patient-name">${escapeHtml(p.patient_id)}</div>
        <div class="patient-meta">
          <span class="badge">${p.fall_count} falls</span>
          ${p.session_active ? '<span class="badge badge-active">Active</span>' : ''}
          ${p.mac_id ? `<span class="badge">${escapeHtml(p.mac_id)}</span>` : ''}
        </div>
        <div class="patient-card-hint">Tap to view history</div>
      </div>
    `).join('');
    document.getElementById('stat-patients').textContent = data.patients.length;
    updateFallsToday(data.patients);
    reapplyPendingBadges();
  } catch (e) {
    list.innerHTML = `<p class="error">Failed to load patients: ${e.message}</p>`;
  }
}

function updateFallsToday(patients) {
  const total = patients.reduce((sum, p) => sum + (p.fall_count || 0), 0);
  document.getElementById('stat-falls-today').textContent = total;
}

// ---------------------------------------------------------------------------
// Patient detail view
// ---------------------------------------------------------------------------
async function openPatient(patientId) {
  inDetailView = true;
  document.getElementById('tab-patients').classList.add('hidden');
  document.getElementById('tab-patient-detail').classList.remove('hidden');
  document.getElementById('detail-patient-name').textContent = patientId;
  document.getElementById('detail-tbody').innerHTML = '<tr><td colspan="4" class="loading">Loading...</td></tr>';
  document.getElementById('detail-falls-24h').textContent = '—';
  document.getElementById('detail-confirmed').textContent = '—';
  document.getElementById('detail-help').textContent = '—';

  try {
    const params = new URLSearchParams({ patient_id: patientId, only_falls: 'false', hours: '24', limit: '500' });
    const resp = await fetch(`${API.falls}?${params}`);
    const data = await resp.json();
    const rows = data.falls || [];

    const total      = rows.length;
    const confirmed  = rows.filter((r) => r.patient_confirmed === 1).length;
    const helpNeeded = rows.filter((r) => r.needs_help === true).length;

    document.getElementById('detail-falls-24h').textContent = total;
    document.getElementById('detail-confirmed').textContent = confirmed;
    document.getElementById('detail-help').textContent = helpNeeded;

    if (rows.length === 0) {
      document.getElementById('detail-tbody').innerHTML =
        '<tr><td colspan="4" class="empty">No fall events in the last 24 hours.</td></tr>';
      return;
    }

    document.getElementById('detail-tbody').innerHTML = rows.map((r) => `
      <tr>
        <td>${formatTime(r.detection_time)}</td>
        <td>${formatConfirmed(r.patient_confirmed)}</td>
        <td>${r.needs_help === true ? '<span class="tag tag-yes">Yes</span>' : '<span class="tag tag-no">No</span>'}</td>
        <td>${r.confidence != null ? (r.confidence * 100).toFixed(0) + '%' : '—'}</td>
      </tr>
    `).join('');
  } catch (e) {
    document.getElementById('detail-tbody').innerHTML =
      `<tr><td colspan="4" class="error">Failed to load: ${e.message}</td></tr>`;
  }
}

function backToPatients() {
  inDetailView = false;
  document.getElementById('tab-patient-detail').classList.add('hidden');
  document.getElementById('tab-patients').classList.remove('hidden');
  loadPatients();
}

// ---------------------------------------------------------------------------
// Live stream (SSE)
// ---------------------------------------------------------------------------
function connectStream() {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(API.stream);

  eventSource.addEventListener('connected', () => setStreamStatus(true));

  eventSource.onmessage = (msg) => {
    setStreamStatus(true);
    try {
      const event = JSON.parse(msg.data);
      handleFallEvent(event);
    } catch (e) {
      console.warn('Bad SSE message:', msg.data);
    }
  };

  eventSource.onerror = () => setStreamStatus(false);
}

function setStreamStatus(connected) {
  const el = document.getElementById('stat-stream');
  el.textContent = connected ? 'Connected' : 'Disconnected';
  el.className = 'stat-value ' + (connected ? 'status-connected' : 'status-disconnected');
}

function handleFallEvent(event) {
  console.log('Fall event:', event);

  const status = event.status;  // 'pending' or 'confirmed'
  const obsId  = event.observation_id;

  if (status === 'pending') {
    if (obsId) pendingAlerts.set(obsId, event.patient_id);
    markPatientPending(event.patient_id, obsId);
    return;
  }

  // Confirmed: clear the pending badge for this observation
  if (obsId && pendingAlerts.has(obsId)) {
    clearPatientPending(pendingAlerts.get(obsId), obsId);
    pendingAlerts.delete(obsId);
  }

  // Show red alert banner when caregiver needs to act:
  //   -1 = no response from patient -> treat as serious
  //    1 + needs_help = confirmed fall, rescue needed
  const confirmed = event.patient_confirmed;  // int: 1, 0, -1
  const needsHelp = event.needs_help;
  const shouldAlert = (confirmed === -1 || (confirmed === 1 && needsHelp === true));
  if (!shouldAlert) return;

  const banner = document.getElementById('alert-banner');
  const text   = document.getElementById('alert-text');
  const label  = event.mac_id || event.patient_id || 'unknown';
  const reason = confirmed === -1 ? 'no response from patient' : 'patient confirmed, needs help';
  text.textContent = `FALL ALERT — ${label} (${reason}, confidence ${event.confidence ?? '?'})`;
  banner.classList.remove('hidden');

  if (!inDetailView) loadPatients();
}

function markPatientPending(patientId, obsId) {
  const card = document.querySelector(`.patient-card[data-patient-id="${CSS.escape(patientId)}"]`);
  if (!card) return;
  card.classList.add('is-pending');
  const meta = card.querySelector('.patient-meta');
  if (!meta) return;
  // Only add one badge per observation_id
  const existing = obsId ? meta.querySelector(`.badge-pending[data-obs-id="${CSS.escape(obsId)}"]`) : null;
  if (existing) return;
  const badge = document.createElement('span');
  badge.className = 'badge badge-pending';
  badge.dataset.obsId = obsId || '';
  badge.textContent = 'Possible fall';
  meta.appendChild(badge);
}

function clearPatientPending(patientId, obsId) {
  const card = document.querySelector(`.patient-card[data-patient-id="${CSS.escape(patientId)}"]`);
  if (!card) return;
  const selector = obsId
    ? `.badge-pending[data-obs-id="${CSS.escape(obsId)}"]`
    : '.badge-pending';
  const badge = card.querySelector(selector);
  if (badge) badge.remove();
  if (!card.querySelector('.badge-pending')) card.classList.remove('is-pending');
}

function reapplyPendingBadges() {
  pendingAlerts.forEach((patientId, obsId) => markPatientPending(patientId, obsId));
}

function dismissAlert() {
  document.getElementById('alert-banner').classList.add('hidden');
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function formatTime(iso) {
  if (!iso) return '—';
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}

function formatConfirmed(c) {
  if (c === 1)  return '<span class="tag tag-yes">Confirmed</span>';
  if (c === 0)  return '<span class="tag tag-no">Not a fall</span>';
  if (c === -1) return '<span class="tag tag-pending">No response</span>';
  return '<span class="tag tag-pending">—</span>';
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
loadPatients();
connectStream();
setInterval(() => {
  if (!inDetailView) loadPatients();
}, 15000);
