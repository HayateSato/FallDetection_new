// Caregiver dashboard — 6G/Charite

const API = {
  patients: '/api/patients',
  falls:    '/api/falls',
  stream:   '/api/stream',
};

let inDetailView = false;
let eventSource = null;

// observation_id -> patient_id for in-flight possible-fall notices
const pendingAlerts = new Map();

// patient_ids that have received a confirmed fall + help-needed alert
const confirmedPatients = new Set();

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
      list.innerHTML = '<p class="empty">No patients registered yet. Click "+ Add Patient" to get started.</p>';
      document.getElementById('stat-patients').textContent = '0';
      return;
    }
    list.innerHTML = data.patients.map((p) => `
      <div class="patient-card" data-patient-id="${escapeHtml(p.patient_id)}" onclick="openPatient('${escapeHtml(p.patient_id)}')">
        <button class="btn-delete-patient" onclick="deletePatient(event, '${escapeHtml(p.patient_id)}')" title="Remove patient">&#x2715;</button>
        <div class="patient-name">${escapeHtml(p.patient_id)}</div>
        ${p.name ? `<div class="patient-display-name">${escapeHtml(p.name)}</div>` : ''}
        <div class="patient-meta">
          <span class="badge">${p.fall_count} falls</span>
          ${p.mac_id ? `<span class="badge">${escapeHtml(p.mac_id)}</span>` : ''}
        </div>
        <div class="patient-card-hint">Tap to view history</div>
      </div>
    `).join('');
    document.getElementById('stat-patients').textContent = data.patients.length;
    updateFallsToday(data.patients);
    reapplyPendingNotices();
    reapplyConfirmedNotices();
  } catch (e) {
    list.innerHTML = `<p class="error">Failed to load patients: ${e.message}</p>`;
  }
}

function updateFallsToday(patients) {
  const total = patients.reduce((sum, p) => sum + (p.fall_count_today || 0), 0);
  document.getElementById('stat-falls-today').textContent = total;
}

// ---------------------------------------------------------------------------
// Patient detail view
// ---------------------------------------------------------------------------
async function openPatient(patientId) {
  // Opening the card is the caregiver's acknowledgement — clear the pending notice
  for (const [obsId, pid] of pendingAlerts) {
    if (pid === patientId) pendingAlerts.delete(obsId);
  }
  clearPatientPending(patientId);

  inDetailView = true;
  document.getElementById('tab-patients').classList.add('hidden');
  document.getElementById('tab-patient-detail').classList.remove('hidden');
  document.getElementById('detail-patient-name').textContent = patientId;
  document.getElementById('detail-tbody').innerHTML = '<tr><td colspan="4" class="loading">Loading...</td></tr>';
  document.getElementById('detail-falls-24h').textContent = '—';
  document.getElementById('detail-confirmed').textContent = '—';
  document.getElementById('detail-help').textContent = '—';

  try {
    const params = new URLSearchParams({ patient_id: patientId, only_falls: 'true', hours: '24', limit: '500' });
    const resp = await fetch(`${API.falls}?${params}`);
    const data = await resp.json();
    const rows = data.falls || [];

    const total      = rows.length;
    const confirmed  = rows.filter((r) => r.patient_confirmed === 1 || r.patient_confirmed === 'yes').length;
    const helpNeeded = rows.filter((r) => r.needs_help === true || r.needs_help === 1).length;

    document.getElementById('detail-falls-24h').textContent = total;
    document.getElementById('detail-confirmed').textContent = confirmed;
    document.getElementById('detail-help').textContent = helpNeeded;

    if (rows.length === 0) {
      document.getElementById('detail-tbody').innerHTML =
        '<tr><td colspan="3" class="empty">No fall events in the last 24 hours.</td></tr>';
      return;
    }

    document.getElementById('detail-tbody').innerHTML = rows.map((r) => `
      <tr>
        <td>${formatTime(r.detection_time)}</td>
        <td>${formatConfirmed(r.patient_confirmed)}</td>
        <td>${(r.needs_help === true || r.needs_help === 1) ? '<span class="tag tag-yes">Yes</span>' : '<span class="tag tag-no">No</span>'}</td>
      </tr>
    `).join('');
  } catch (e) {
    document.getElementById('detail-tbody').innerHTML =
      `<tr><td colspan="3" class="error">Failed to load: ${e.message}</td></tr>`;
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
    markPatientPending(event.patient_id);
    return;
  }

  // Confirmed event: escalate card if patient asked for help.
  if (event.needs_help === true) {
    markPatientConfirmed(event.patient_id);
  }
  if (!inDetailView) loadPatients();
}

// ---------------------------------------------------------------------------
// Pending-notice helpers
// ---------------------------------------------------------------------------

function markPatientPending(patientId) {
  const card = document.querySelector(`.patient-card[data-patient-id="${CSS.escape(patientId)}"]`);
  if (!card) return;
  // Only inject one notice even if multiple possible-fall events arrive
  if (!card.querySelector('.pending-notice')) {
    const notice = document.createElement('div');
    notice.className = 'pending-notice';
    notice.textContent = 'Possible fall, wait for confirmation';
    card.insertBefore(notice, card.firstChild);
  }
  card.classList.add('is-pending');
}

function markPatientConfirmed(patientId) {
  // Remove from pending — confirmed supersedes possible-fall state
  for (const [obsId, pid] of pendingAlerts) {
    if (pid === patientId) pendingAlerts.delete(obsId);
  }
  confirmedPatients.add(patientId);

  const card = document.querySelector(`.patient-card[data-patient-id="${CSS.escape(patientId)}"]`);
  if (!card) return;

  // Update banner text (reuse existing notice element if present)
  let notice = card.querySelector('.pending-notice');
  if (notice) {
    notice.textContent = 'Fall is confirmed';
    notice.classList.add('confirmed-notice');
  } else {
    notice = document.createElement('div');
    notice.className = 'pending-notice confirmed-notice';
    notice.textContent = 'Fall is confirmed';
    card.insertBefore(notice, card.firstChild);
  }

  // Add the big help message below the patient name if not already present
  if (!card.querySelector('.help-overlay')) {
    const overlay = document.createElement('div');
    overlay.className = 'help-overlay';
    overlay.textContent = 'Help is requested';
    card.appendChild(overlay);
  }

  card.classList.add('is-pending', 'is-confirmed');
}

function clearPatientPending(patientId) {
  confirmedPatients.delete(patientId);
  const card = document.querySelector(`.patient-card[data-patient-id="${CSS.escape(patientId)}"]`);
  if (!card) return;
  const notice = card.querySelector('.pending-notice');
  if (notice) notice.remove();
  const overlay = card.querySelector('.help-overlay');
  if (overlay) overlay.remove();
  card.classList.remove('is-pending', 'is-confirmed');
}

function reapplyPendingNotices() {
  const seen = new Set();
  pendingAlerts.forEach((patientId) => {
    if (!seen.has(patientId)) {
      seen.add(patientId);
      markPatientPending(patientId);
    }
  });
}

function reapplyConfirmedNotices() {
  confirmedPatients.forEach((patientId) => markPatientConfirmed(patientId));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function formatTime(iso) {
  if (!iso) return '—';
  try { return new Date(iso).toLocaleString(); } catch { return iso; }
}

// patient_confirmed int encoding (matches InfluxDB and SSE events):
//   1  = patient said yes, it was a fall  ("yes")
//   0  = patient denied, false positive   ("no")
//  -1  = no response within timeout       ("not_answered")
function formatConfirmed(c) {
  if (c === 1 || c === 'yes')           return '<span class="tag tag-yes">Confirmed</span>';
  if (c === 0 || c === 'no')            return '<span class="tag tag-no">Not a fall</span>';
  if (c === -1 || c === 'not_answered') return '<span class="tag tag-pending">No response</span>';
  return '<span class="tag tag-pending">—</span>';
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

// ---------------------------------------------------------------------------
// Add Patient modal
// ---------------------------------------------------------------------------

function showAddPatientModal() {
  document.getElementById('modal-patient-id').value = '';
  document.getElementById('modal-patient-name').value = '';
  document.getElementById('modal-patient-mac').value = '';
  document.getElementById('modal-error').classList.add('hidden');
  document.getElementById('modal-overlay').classList.remove('hidden');
  document.getElementById('modal-patient-id').focus();
}

function hideAddPatientModal() {
  document.getElementById('modal-overlay').classList.add('hidden');
}

function handleModalOverlayClick(event) {
  if (event.target === document.getElementById('modal-overlay')) hideAddPatientModal();
}

async function submitAddPatient() {
  const patientId = document.getElementById('modal-patient-id').value.trim();
  const name      = document.getElementById('modal-patient-name').value.trim();
  const macId     = document.getElementById('modal-patient-mac').value.trim();
  const errorEl   = document.getElementById('modal-error');
  const submitBtn = document.getElementById('modal-submit');

  if (!patientId) {
    errorEl.textContent = 'Patient ID is required.';
    errorEl.classList.remove('hidden');
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = 'Adding...';
  errorEl.classList.add('hidden');

  try {
    const resp = await fetch('/api/patients', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient_id: patientId, name: name || null, mac_id: macId || null }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    hideAddPatientModal();
    loadPatients();
  } catch (e) {
    errorEl.textContent = `Failed: ${e.message}`;
    errorEl.classList.remove('hidden');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Add Patient';
  }
}

// ---------------------------------------------------------------------------
// Delete patient
// ---------------------------------------------------------------------------

async function deletePatient(event, patientId) {
  event.stopPropagation();
  if (!confirm(`Remove "${patientId}" from the dashboard?\n\nFall history in InfluxDB is not affected.`)) return;
  try {
    const resp = await fetch(`/api/patients/${encodeURIComponent(patientId)}`, { method: 'DELETE' });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    loadPatients();
  } catch (e) {
    alert(`Failed to remove patient: ${e.message}`);
  }
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
loadPatients();
connectStream();
