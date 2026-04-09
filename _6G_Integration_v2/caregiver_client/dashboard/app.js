// Caregiver dashboard — 6G/Charite minimal client
// All API calls are same-origin, no auth.

const API = {
  patients: '/api/patients',
  falls:    '/api/falls',
  stream:   '/api/stream',
  confirm:  (id) => `/api/falls/${id}/confirm`,
};

let currentTab = 'patients';
let eventSource = null;

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab').forEach((b) => b.classList.remove('active'));
  document.querySelector(`.tab[onclick="switchTab('${tab}')"]`).classList.add('active');
  document.getElementById('tab-patients').classList.toggle('hidden', tab !== 'patients');
  document.getElementById('tab-history').classList.toggle('hidden', tab !== 'history');
  if (tab === 'history') loadHistory();
  if (tab === 'patients') loadPatients();
}

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
      <div class="patient-card ${p.fall_count > 0 ? 'has-falls' : ''}">
        <div class="patient-name">${escapeHtml(p.mac_id || p.patient_id)}</div>
        <div class="patient-meta">
          <span class="badge">${p.fall_count} falls</span>
          ${p.session_active ? '<span class="badge badge-active">Active</span>' : ''}
        </div>
      </div>
    `).join('');
    document.getElementById('stat-patients').textContent = data.patients.length;
    updateFallsToday(data.patients);
  } catch (e) {
    list.innerHTML = `<p class="error">Failed to load patients: ${e.message}</p>`;
  }
}

function updateFallsToday(patients) {
  const total = patients.reduce((sum, p) => sum + (p.fall_count || 0), 0);
  document.getElementById('stat-falls-today').textContent = total;
}

// ---------------------------------------------------------------------------
// Fall history tab
// ---------------------------------------------------------------------------
async function loadHistory() {
  const tbody  = document.getElementById('history-tbody');
  const filter = document.getElementById('filter-patient').value.trim();
  const conf   = document.getElementById('filter-confirmed').value;

  tbody.innerHTML = '<tr><td colspan="5" class="loading">Loading...</td></tr>';

  const params = new URLSearchParams({ only_falls: 'true', limit: '500' });
  if (filter) params.append('patient_id', filter);

  try {
    const resp = await fetch(`${API.falls}?${params}`);
    const data = await resp.json();
    let rows = data.falls || [];
    if (conf) rows = rows.filter((r) => r.patient_confirmed === conf);

    if (rows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" class="empty">No fall events.</td></tr>';
      return;
    }

    tbody.innerHTML = rows.map((r) => `
      <tr class="row-fall">
        <td>${formatTime(r.detection_time)}</td>
        <td>${escapeHtml(r.mac_id || r.patient_id)}</td>
        <td>${formatConfirmed(r.patient_confirmed)}</td>
        <td>
          <button class="btn-mini" onclick="confirmFall(${r.id}, 'yes')">Yes</button>
          <button class="btn-mini" onclick="confirmFall(${r.id}, 'no')">No</button>
        </td>
      </tr>
    `).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="5" class="error">Failed to load: ${e.message}</td></tr>`;
  }
}

async function confirmFall(id, confirmed) {
  try {
    await fetch(`${API.confirm(id)}?confirmed=${confirmed}`, { method: 'POST' });
    loadHistory();
  } catch (e) {
    alert('Could not save confirmation: ' + e.message);
  }
}

function resetFilters() {
  document.getElementById('filter-patient').value = '';
  document.getElementById('filter-confirmed').value = '';
  loadHistory();
}

// ---------------------------------------------------------------------------
// Live stream (SSE)
// ---------------------------------------------------------------------------
function connectStream() {
  if (eventSource) eventSource.close();
  eventSource = new EventSource(API.stream);

  eventSource.addEventListener('connected', () => {
    setStreamStatus(true);
  });

  eventSource.onmessage = (msg) => {
    setStreamStatus(true);
    try {
      const event = JSON.parse(msg.data);
      handleFallEvent(event);
    } catch (e) {
      console.warn('Bad SSE message:', msg.data);
    }
  };

  eventSource.onerror = () => {
    setStreamStatus(false);
    // EventSource auto-reconnects; no manual retry needed.
  };
}

function setStreamStatus(connected) {
  const el = document.getElementById('stat-stream');
  el.textContent = connected ? 'Connected' : 'Disconnected';
  el.className = 'stat-value ' + (connected ? 'status-connected' : 'status-disconnected');
}

function handleFallEvent(event) {
  console.log('Fall event:', event);
  const banner = document.getElementById('alert-banner');
  const text   = document.getElementById('alert-text');
  text.textContent = `FALL DETECTED — ${event.patient_id || 'unknown patient'} (confidence ${event.confidence ?? '?'})`;
  banner.classList.remove('hidden');
  // Refresh visible data
  if (currentTab === 'patients') loadPatients();
  if (currentTab === 'history')  loadHistory();
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
  if (c === 'yes')          return '<span class="tag tag-yes">Yes</span>';
  if (c === 'no')           return '<span class="tag tag-no">No</span>';
  return '<span class="tag tag-pending">Not answered</span>';
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
setInterval(() => { if (currentTab === 'patients') loadPatients(); }, 15000);
