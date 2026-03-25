/**
 * Operator Dashboard — app.js
 *
 * Panels:
 *   1. Model info + hot-switch via POST /api/ml/model/switch
 *   2. Live server health via GET /api/ml/health
 *   3. Grafana links (opens in new tab)
 *   4. Recent inference log via GET /api/caregiver/... (reads Postgres)
 */

const ML_API = "/api/ml";
const CG_API = "/api/caregiver";

let API_KEY = localStorage.getItem("op_api_key") || "";

function saveApiKey() {
  const input = document.getElementById("api-key-input");
  const status = document.getElementById("api-key-status");
  API_KEY = input.value.trim();
  if (!API_KEY) { status.textContent = "Key is empty."; return; }
  localStorage.setItem("op_api_key", API_KEY);
  status.textContent = "Saved — connecting…";
  loadModelInfo();
  loadModelList();
  loadServerHealth();
  loadRecentLog();
}

// ---- Boot ----

window.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("api-key-input");
  if (API_KEY) {
    input.value = API_KEY;
    document.getElementById("api-key-status").textContent = "Key loaded from storage.";
  }
  loadModelInfo();
  loadModelList();
  loadServerHealth();
  loadRecentLog();
  setInterval(loadServerHealth, 15_000);
  setInterval(loadRecentLog, 30_000);
});

// ---- Model Info ----

async function loadModelInfo() {
  const data = await mlFetch("/model/info");
  if (!data) return;
  document.getElementById("active-model").textContent   = data.version?.toUpperCase() || "—";
  document.getElementById("model-features").textContent = data.num_features ?? "—";
  document.getElementById("model-baro").textContent     = data.uses_barometer ? "Yes" : "No";
}

async function loadModelList() {
  const data = await mlFetch("/model/list");
  if (!data) return;
  const select = document.getElementById("model-select");
  select.innerHTML = "";
  for (const [version, desc] of Object.entries(data.available_versions || {})) {
    const opt = document.createElement("option");
    opt.value = version;
    opt.textContent = `${version.toUpperCase()} — ${desc}`;
    select.appendChild(opt);
  }
}

async function switchModel() {
  const version = document.getElementById("model-select").value;
  const statusEl = document.getElementById("switch-status");
  if (!version) { statusEl.textContent = "Select a model first."; return; }
  if (!API_KEY)  { statusEl.textContent = "Enter API key and click Connect first."; return; }

  statusEl.textContent = "Switching…";
  const res = await mlFetchRaw("/model/switch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ version }),
  });
  if (!res) { statusEl.textContent = "No response — is ml_server running?"; return; }
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    statusEl.textContent = `Failed (${res.status}): ${err.detail || "unknown error"}`;
    return;
  }
  const data = await res.json();
  statusEl.textContent = `Switched to ${data.new_version?.toUpperCase()} (${data.num_features} features)`;
  loadModelInfo();
}

// ---- Server Health ----

async function loadServerHealth() {
  const serverStatusEl = document.getElementById("server-status");
  const data = await mlFetch("/health");
  if (!data) {
    serverStatusEl.textContent = "OFFLINE";
    serverStatusEl.className   = "badge badge-red";
    document.getElementById("metric-status").textContent  = "Offline";
    document.getElementById("metric-uptime").textContent  = "—";
    document.getElementById("metric-version").textContent = "—";
    return;
  }
  serverStatusEl.textContent = data.status === "ok" ? "ONLINE" : "ERROR";
  serverStatusEl.className   = data.status === "ok" ? "badge badge-green" : "badge badge-red";
  document.getElementById("uptime").textContent = `Uptime: ${formatUptime(data.uptime_seconds)}`;
  document.getElementById("metric-status").textContent  = data.status?.toUpperCase();
  document.getElementById("metric-uptime").textContent  = formatUptime(data.uptime_seconds);
  document.getElementById("metric-version").textContent = data.model_version?.toUpperCase() || "—";
}

function formatUptime(seconds) {
  if (seconds == null) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`;
}

// ---- Processing Config (placeholder — backend wiring pending) ----

function applyConfig() {
  const window_s = document.getElementById("cfg-window").value;
  const source   = document.querySelector('input[name="data-source"]:checked').value;
  const status   = document.getElementById("config-status");
  if (source === "csv") {
    status.textContent = "CSV/MinIO data source not yet wired. Change noted.";
    return;
  }
  // TODO: POST to ml_server /config endpoint once implemented
  status.textContent = `Config noted — window: ${window_s}s, source: ${source}. (Backend wiring pending)`;
}

// ---- Recent Inference Log (reads /inferences from ml_server → Postgres) ----

async function loadRecentLog() {
  const tbody = document.getElementById("log-tbody");
  try {
    const res = await fetch(`${ML_API}/inferences?limit=20`);
    if (!res.ok) {
      tbody.innerHTML = `<tr><td colspan="7" class="loading">ML server unavailable (${res.status}). Is it running?</td></tr>`;
      return;
    }
    const data = await res.json();
    const rows = data.inferences || [];
    if (rows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="loading">No inferences yet. Send a prediction to see data here.</td></tr>`;
      return;
    }
    tbody.innerHTML = rows.map(r => `
      <tr>
        <td>${new Date(r.timestamp).toLocaleTimeString()}</td>
        <td>${r.participant || "—"}</td>
        <td class="${r.fall_detected ? "fall-yes" : "fall-no"}">${r.fall_detected ? "FALL" : "No"}</td>
        <td>${r.confidence != null ? (r.confidence * 100).toFixed(1) + "%" : "—"}</td>
        <td>${r.model_version?.toUpperCase() || "—"}</td>
        <td>${r.latency_ms != null ? r.latency_ms + "ms" : "—"}</td>
        <td>${r.window_size != null ? r.window_size + " smp" : "—"}</td>
      </tr>
    `).join("");
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="7" class="loading">Cannot reach caregiver API. Is Docker running?</td></tr>`;
  }
}

// ---- Helpers ----

/** Returns raw Response so callers can inspect status codes. */
async function mlFetchRaw(path, options = {}) {
  try {
    return await fetch(`${ML_API}${path}`, {
      ...options,
      headers: { "X-API-Key": API_KEY || "", ...(options.headers || {}) },
    });
  } catch (e) {
    return null;
  }
}

async function mlFetch(path, options = {}) {
  const keyStatus = document.getElementById("api-key-status");
  try {
    const res = await fetch(`${ML_API}${path}`, {
      ...options,
      headers: {
        "X-API-Key": API_KEY || "",
        ...(options.headers || {}),
      },
    });
    if (res.status === 403) {
      keyStatus.textContent = "API key rejected (403). Check key and click Connect.";
      return null;
    }
    if (res.status === 401) {
      keyStatus.textContent = "API key missing (401). Enter key and click Connect.";
      return null;
    }
    if (!res.ok) return null;
    if (API_KEY) keyStatus.textContent = "Connected.";
    return await res.json();
  } catch (e) {
    keyStatus.textContent = "Cannot reach ml_server. Is it running?";
    return null;
  }
}
