/**
 * Care-Giver Dashboard — app.js
 *
 * Features:
 *  - Login with JWT auth
 *  - Two tabs: Patient list / Fall history
 *  - Live fall alert banner via SSE (Redis fall_events channel)
 *  - Patient cards with click-through to per-patient fall table (includes feedback)
 *  - Fall history tab: time range, patient, user_fall, need_help filters + pagination
 */

const API_BASE = "/api/caregiver";   // nginx routes this to caregiver_api:8002
const PAGE_SIZE = 50;

let token = localStorage.getItem("cg_token") || null;
let sseConnection = null;
let historyOffset = 0;
let historyTotal = 0;
let lastFilters = {};

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

window.addEventListener("DOMContentLoaded", () => {
  if (token) showDashboard();
});

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

async function login() {
  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value;
  const errEl = document.getElementById("login-error");
  errEl.textContent = "";

  try {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || "Login failed");
    const data = await res.json();
    token = data.access_token;
    localStorage.setItem("cg_token", token);
    showDashboard();
  } catch (e) {
    errEl.textContent = e.message;
  }
}

function logout() {
  token = null;
  localStorage.removeItem("cg_token");
  if (sseConnection) { sseConnection.close(); sseConnection = null; }
  document.getElementById("dashboard").classList.add("hidden");
  document.getElementById("login-screen").classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

function showDashboard() {
  document.getElementById("login-screen").classList.add("hidden");
  document.getElementById("dashboard").classList.remove("hidden");
  loadPatients();
  loadStats();
  loadHistory();
  connectFallStream();
  setInterval(loadStats, 60_000);
  setInterval(loadPatients, 30_000);
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

function switchTab(tab) {
  document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(t => t.classList.add("hidden"));
  document.getElementById(`tab-${tab}`).classList.remove("hidden");
  event.currentTarget.classList.add("active");
}

// ---------------------------------------------------------------------------
// Patient List
// ---------------------------------------------------------------------------

async function loadPatients() {
  const res = await apiFetch("/patients");
  if (!res) return;
  const { patients } = await res.json();
  renderPatientList(patients);
}

function renderPatientList(patients) {
  const container = document.getElementById("patient-list");
  if (!patients.length) {
    container.innerHTML = "<p class='loading'>No patient sessions recorded yet.</p>";
    return;
  }
  container.innerHTML = patients.map(p => {
    const hasFall = p.fall_count > 0;
    const lastSeen = p.last_seen ? new Date(p.last_seen).toLocaleString() : "—";
    const confidence = p.last_confidence != null ? (p.last_confidence * 100).toFixed(1) + "%" : "—";
    return `
      <div class="patient-card ${hasFall ? "has-fall" : ""}" onclick="openDetail('${escHtml(p.participant_name)}')">
        <div class="name">${escHtml(p.participant_name)}</div>
        <div class="meta">Gender: ${p.gender || "—"} &nbsp;|&nbsp; Last confidence: ${confidence}</div>
        <div class="meta">Last seen: ${lastSeen}</div>
        <span class="badge ${hasFall ? "badge-red" : "badge-green"}">${p.fall_count} fall${p.fall_count !== 1 ? "s" : ""}</span>
        <span class="badge badge-grey">${p.active ? "Active" : "Ended"}</span>
      </div>
    `;
  }).join("");
}

// ---------------------------------------------------------------------------
// Patient Detail
// ---------------------------------------------------------------------------

async function openDetail(name) {
  document.getElementById("patient-list").classList.add("hidden");
  const detail = document.getElementById("patient-detail");
  detail.classList.remove("hidden");
  document.getElementById("detail-name").textContent = name;
  document.getElementById("fall-tbody").innerHTML =
    "<tr><td colspan='6' class='loading'>Loading...</td></tr>";
  document.getElementById("detail-empty").classList.add("hidden");

  const res = await apiFetch(`/patients/${encodeURIComponent(name)}/falls?limit=100`);
  if (!res) return;
  const { falls } = await res.json();

  const tbody = document.getElementById("fall-tbody");
  if (!falls.length) {
    tbody.innerHTML = "";
    document.getElementById("detail-empty").classList.remove("hidden");
    return;
  }
  tbody.innerHTML = falls.map(f => `
    <tr>
      <td>${new Date(f.timestamp).toLocaleString()}</td>
      <td>${(f.confidence * 100).toFixed(1)}%</td>
      <td>${f.model_version || "—"}</td>
      <td>${f.latency_ms != null ? f.latency_ms + " ms" : "—"}</td>
      <td>${feedbackLabel(f.user_fall)}</td>
      <td>${feedbackLabel(f.need_help)}</td>
    </tr>
  `).join("");
}

function closeDetail() {
  document.getElementById("patient-detail").classList.add("hidden");
  document.getElementById("patient-list").classList.remove("hidden");
}

// ---------------------------------------------------------------------------
// Fall History Tab
// ---------------------------------------------------------------------------

function buildHistoryParams(offset = 0) {
  const patient   = document.getElementById("filter-patient").value.trim();
  const fromInput = document.getElementById("filter-from").value;
  const toInput   = document.getElementById("filter-to").value;
  const userFall  = document.getElementById("filter-user-fall").value;
  const needHelp  = document.getElementById("filter-need-help").value;

  const params = new URLSearchParams({ limit: PAGE_SIZE, offset });
  if (patient)   params.set("patient", patient);
  if (fromInput) params.set("from_ts", new Date(fromInput).toISOString());
  if (toInput)   params.set("to_ts",   new Date(toInput).toISOString());
  if (userFall)  params.set("user_fall", userFall);
  if (needHelp)  params.set("need_help", needHelp);
  return params;
}

async function loadHistory(offset = 0) {
  historyOffset = offset;
  const params = buildHistoryParams(offset);
  const res = await apiFetch(`/falls/history?${params}`);
  if (!res) return;

  const data = await res.json();
  historyTotal = data.total;
  renderHistoryTable(data.falls);
  updatePagination();
}

function applyFilters() {
  loadHistory(0);
}

function resetFilters() {
  document.getElementById("filter-patient").value = "";
  document.getElementById("filter-from").value = "";
  document.getElementById("filter-to").value = "";
  document.getElementById("filter-user-fall").value = "";
  document.getElementById("filter-need-help").value = "";
  loadHistory(0);
}

function historyPage(dir) {
  const newOffset = historyOffset + dir * PAGE_SIZE;
  if (newOffset < 0 || newOffset >= historyTotal) return;
  loadHistory(newOffset);
}

function renderHistoryTable(falls) {
  const tbody = document.getElementById("history-tbody");
  document.getElementById("history-count").textContent =
    `${historyTotal} event${historyTotal !== 1 ? "s" : ""}`;

  if (!falls.length) {
    tbody.innerHTML = "<tr><td colspan='7' class='loading'>No falls match the current filters.</td></tr>";
    return;
  }
  tbody.innerHTML = falls.map(f => `
    <tr>
      <td>${new Date(f.timestamp).toLocaleString()}</td>
      <td>${escHtml(f.participant || "—")}</td>
      <td>${f.confidence != null ? (f.confidence * 100).toFixed(1) + "%" : "—"}</td>
      <td>${f.model_version || "—"}</td>
      <td>${f.inference_mode || "—"}</td>
      <td>${feedbackLabel(f.user_fall)}</td>
      <td>${feedbackLabel(f.need_help)}</td>
    </tr>
  `).join("");
}

function updatePagination() {
  const page = Math.floor(historyOffset / PAGE_SIZE) + 1;
  const totalPages = Math.ceil(historyTotal / PAGE_SIZE) || 1;
  document.getElementById("page-info").textContent = `Page ${page} of ${totalPages}`;
  document.getElementById("btn-prev").disabled = historyOffset <= 0;
  document.getElementById("btn-next").disabled = historyOffset + PAGE_SIZE >= historyTotal;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

async function loadStats() {
  const res = await apiFetch("/stats/summary");
  if (!res) return;
  const d = await res.json();
  document.getElementById("stat-falls").textContent = d.falls_today ?? "—";
  document.getElementById("stat-confirmed").textContent = d.confirmed_falls_today ?? "—";
  document.getElementById("stat-help").textContent = d.help_requested_today ?? "—";
  const conf = d.avg_confidence_today != null ? (d.avg_confidence_today * 100).toFixed(1) + "%" : "—";
  document.getElementById("stat-confidence").textContent = conf;
}

// ---------------------------------------------------------------------------
// Live Fall SSE Stream
// ---------------------------------------------------------------------------

function connectFallStream() {
  if (sseConnection) sseConnection.close();
  sseConnection = new EventSource(`${API_BASE}/patients/stream`);

  sseConnection.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.fall_detected) {
      showFallAlert(data);
      loadPatients();
      loadStats();
    }
  };

  sseConnection.onerror = () => {
    console.warn("SSE connection error, auto-reconnecting...");
  };
}

function showFallAlert(event) {
  const banner = document.getElementById("alert-banner");
  const conf = event.confidence != null ? ` (${(event.confidence * 100).toFixed(0)}% confidence)` : "";
  const ts = event.timestamp ? new Date(event.timestamp).toLocaleTimeString() : "—";
  document.getElementById("alert-text").textContent =
    `FALL DETECTED — ${event.patient_id}${conf} at ${ts}`;
  banner.classList.remove("hidden");
  setTimeout(dismissAlert, 30_000);
}

function dismissAlert() {
  document.getElementById("alert-banner").classList.add("hidden");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function apiFetch(path) {
  if (!token) return null;
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (res.status === 401) { logout(); return null; }
  return res;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// Maps user_fall / need_help integer codes to a labelled HTML span.
function feedbackLabel(val) {
  switch (Number(val)) {
    case 1:  return '<span class="fb-yes">Yes</span>';
    case 2:  return '<span class="fb-no">No</span>';
    case 3:  return '<span class="fb-timeout">No answer</span>';
    default: return '<span class="fb-pending">Pending</span>';
  }
}
