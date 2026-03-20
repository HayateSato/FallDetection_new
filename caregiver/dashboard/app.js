/**
 * Care-Giver Dashboard — app.js
 *
 * Features:
 *  - Login with JWT auth
 *  - Patient list with live fall status badges
 *  - Full-screen alert banner via SSE when a fall occurs
 *  - Patient detail view with fall history table
 *  - Stats summary header (updated every 60s)
 */

const API_BASE = "/api/caregiver";   // nginx routes this to caregiver_api:8002
let token = localStorage.getItem("cg_token") || null;
let sseConnection = null;

// ---- Bootstrap ----

window.addEventListener("DOMContentLoaded", () => {
  if (token) {
    showDashboard();
  }
});

// ---- Auth ----

async function login() {
  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value;
  const errEl = document.getElementById("login-error");
  errEl.hidden = true;

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
    errEl.hidden = false;
  }
}

function logout() {
  token = null;
  localStorage.removeItem("cg_token");
  if (sseConnection) { sseConnection.close(); sseConnection = null; }
  document.getElementById("dashboard").hidden = true;
  document.getElementById("login-screen").hidden = false;
}

// ---- Dashboard ----

function showDashboard() {
  document.getElementById("login-screen").hidden = true;
  document.getElementById("dashboard").hidden = false;
  loadPatients();
  loadStats();
  connectFallStream();
  setInterval(loadStats, 60_000);
  setInterval(loadPatients, 30_000);
}

// ---- Patient List ----

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
    const activeLabel = p.active ? "Active" : "Ended";
    const lastSeen = p.last_seen ? new Date(p.last_seen).toLocaleString() : "—";
    const confidence = p.last_confidence != null ? (p.last_confidence * 100).toFixed(1) + "%" : "—";
    return `
      <div class="patient-card ${hasFall ? "has-fall" : ""}" onclick="openDetail('${p.participant_name}')">
        <div class="name">${p.participant_name}</div>
        <div class="meta">Gender: ${p.gender || "—"} &nbsp;|&nbsp; Last confidence: ${confidence}</div>
        <div class="meta">Last seen: ${lastSeen}</div>
        <span class="badge ${hasFall ? "badge-red" : "badge-green"}">${p.fall_count} fall${p.fall_count !== 1 ? "s" : ""}</span>
        <span class="badge badge-grey">${activeLabel}</span>
      </div>
    `;
  }).join("");
}

// ---- Patient Detail ----

async function openDetail(name) {
  document.getElementById("patient-list").hidden = true;
  const detail = document.getElementById("patient-detail");
  detail.hidden = false;
  document.getElementById("detail-name").textContent = name;
  document.getElementById("fall-tbody").innerHTML = "<tr><td colspan='4' class='loading'>Loading...</td></tr>";

  const res = await apiFetch(`/patients/${encodeURIComponent(name)}/falls?limit=100`);
  if (!res) return;
  const { falls } = await res.json();

  const tbody = document.getElementById("fall-tbody");
  const empty = document.getElementById("detail-empty");
  if (!falls.length) {
    tbody.innerHTML = "";
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  tbody.innerHTML = falls.map(f => `
    <tr>
      <td>${new Date(f.timestamp).toLocaleString()}</td>
      <td>${(f.confidence * 100).toFixed(1)}%</td>
      <td>${f.model_version || "—"}</td>
      <td>${f.latency_ms != null ? f.latency_ms + " ms" : "—"}</td>
    </tr>
  `).join("");
}

function closeDetail() {
  document.getElementById("patient-detail").hidden = true;
  document.getElementById("patient-list").hidden = false;
}

// ---- Stats ----

async function loadStats() {
  const res = await apiFetch("/stats/summary");
  if (!res) return;
  const data = await res.json();
  document.getElementById("stat-falls").textContent = `${data.falls_today} falls today`;
  document.getElementById("stat-predictions").textContent = `${data.predictions_today} predictions today`;
  const conf = data.avg_confidence_today != null ? (data.avg_confidence_today * 100).toFixed(1) + "%" : "—";
  document.getElementById("stat-confidence").textContent = `avg confidence: ${conf}`;
}

// ---- Live Fall SSE Stream ----

function connectFallStream() {
  if (sseConnection) sseConnection.close();

  // Note: EventSource doesn't support custom headers, so we use query param for auth
  // In production, use a separate auth endpoint to get a short-lived SSE token
  sseConnection = new EventSource(`${API_BASE}/patients/stream`);

  sseConnection.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.fall_detected) {
      showFallAlert(data);
      loadPatients();  // refresh patient list immediately
    }
  };

  sseConnection.onerror = () => {
    // EventSource auto-reconnects — no manual retry needed
    console.warn("SSE connection error, auto-reconnecting...");
  };
}

function showFallAlert(event) {
  const banner = document.getElementById("alert-banner");
  const conf = event.confidence != null ? ` (${(event.confidence * 100).toFixed(0)}% confidence)` : "";
  document.getElementById("alert-text").textContent =
    `FALL DETECTED — ${event.patient_id}${conf} at ${new Date(event.timestamp).toLocaleTimeString()}`;
  banner.hidden = false;
  // Auto-dismiss after 30 seconds
  setTimeout(dismissAlert, 30_000);
}

function dismissAlert() {
  document.getElementById("alert-banner").hidden = true;
}

// ---- Helpers ----

async function apiFetch(path) {
  if (!token) return null;
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (res.status === 401) {
    logout();
    return null;
  }
  return res;
}
