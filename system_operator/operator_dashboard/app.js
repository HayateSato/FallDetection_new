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
  loadConfig();
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

// ---- Processing Config ----

/** Populate all config dropdowns from the server's current state on page load. */
async function loadConfig() {
  const data = await mlFetch("/config");
  if (!data) return;
  _applyConfigToUI(data);
}

function _applyConfigToUI(d) {
  if (d.window_seconds != null)
    document.getElementById("cfg-window").value = d.window_seconds;
  if (d.window_samples != null && d.model_sample_rate_hz != null)
    document.getElementById("cfg-window-hint").textContent =
      `${d.window_seconds}s = ${d.window_samples} samples @ ${d.model_sample_rate_hz}Hz`;
  if (d.acc_sensor_type)
    document.getElementById("cfg-sensor-type").value = d.acc_sensor_type;
  if (d.hardware_sample_rate)
    document.getElementById("cfg-sample-rate").value = String(d.hardware_sample_rate);
  if (d.resampling_method)
    document.getElementById("cfg-resample-method").value = d.resampling_method;
}

function onDataSourceChange(source) {
  const csvPanel = document.getElementById("csv-panel");
  if (source === "csv") {
    csvPanel.style.display = "block";
    loadFileList();
  } else {
    csvPanel.style.display = "none";
  }
}

async function applyConfig() {
  const status  = document.getElementById("config-status");
  const payload = {
    window_seconds:       parseFloat(document.getElementById("cfg-window").value),
    acc_sensor_type:      document.getElementById("cfg-sensor-type").value,
    hardware_sample_rate: parseInt(document.getElementById("cfg-sample-rate").value, 10),
    resampling_method:    document.getElementById("cfg-resample-method").value,
  };
  status.textContent = "Applying…";
  const res = await mlFetchRaw("/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res) { status.textContent = "Cannot reach ml_server."; return; }
  if (res.ok) {
    const d = await res.json();
    _applyConfigToUI(d);
    status.textContent =
      `Applied — window: ${d.window_seconds}s (${d.window_samples} smp), ` +
      `sensor: ${d.acc_sensor_type}, rate: ${d.hardware_sample_rate}Hz, ` +
      `resample: ${d.resampling_method}`;
  } else {
    const d = await res.json().catch(() => ({}));
    status.textContent = `Failed (${res.status}): ${d.detail || "unknown error"}`;
  }
}

// ---- MinIO / CSV Datalake ----

let _csvFiles = [];   // cached file list

async function loadFileList() {
  const sel = document.getElementById("csv-file-select");
  sel.innerHTML = "<option value=''>— loading… —</option>";
  const data = await mlFetch("/datalake/files");
  if (!data) {
    sel.innerHTML = "<option value=''>MinIO unavailable</option>";
    return;
  }
  _csvFiles = data.files || [];
  if (_csvFiles.length === 0) {
    sel.innerHTML = "<option value=''>No CSV files in bucket yet — upload one above</option>";
    return;
  }
  sel.innerHTML = _csvFiles.map(f =>
    `<option value="${f.name}">${f.name} (${(f.size_bytes / 1024).toFixed(1)} KB)</option>`
  ).join("");
  sel.onchange = () => {
    const f = _csvFiles.find(x => x.name === sel.value);
    document.getElementById("csv-file-meta").textContent = f
      ? `Last modified: ${new Date(f.last_modified).toLocaleString()}`
      : "";
  };
  sel.dispatchEvent(new Event("change"));
}

async function uploadCsv() {
  const input    = document.getElementById("csv-upload-input");
  const statusEl = document.getElementById("upload-status");
  if (!input.files.length) { statusEl.textContent = "Select a .csv file first."; return; }
  const file   = input.files[0];
  const formData = new FormData();
  formData.append("file", file);
  statusEl.textContent = `Uploading ${file.name}…`;
  try {
    const res = await fetch(`${ML_API}/datalake/upload`, {
      method: "POST",
      headers: { "X-API-Key": API_KEY || "" },
      body: formData,
    });
    if (res.ok) {
      const d = await res.json();
      statusEl.textContent = `Uploaded: ${d.filename}`;
      input.value = "";
      loadFileList();
    } else {
      const d = await res.json().catch(() => ({}));
      statusEl.textContent = `Upload failed (${res.status}): ${d.detail || "unknown"}`;
    }
  } catch (e) {
    statusEl.textContent = `Upload error: ${e.message}`;
  }
}

async function runReplay() {
  const filename  = document.getElementById("csv-file-select").value;
  const window_s  = parseFloat(document.getElementById("cfg-window").value);
  const step_s    = parseFloat(document.getElementById("cfg-step").value);
  const statusEl  = document.getElementById("replay-status");
  const resultsEl = document.getElementById("replay-results");

  if (!filename) { statusEl.textContent = "Select a CSV file first."; return; }

  statusEl.textContent = "Running replay… (may take a moment for long recordings)";
  document.getElementById("btn-replay").disabled = true;
  resultsEl.style.display = "none";

  const params = new URLSearchParams({
    filename,
    window_seconds: window_s,
    step_seconds:   step_s,
  });
  const res = await mlFetchRaw(`/datalake/replay?${params}`);
  document.getElementById("btn-replay").disabled = false;

  if (!res) { statusEl.textContent = "Cannot reach ml_server."; return; }
  if (!res.ok) {
    const d = await res.json().catch(() => ({}));
    statusEl.textContent = `Replay failed (${res.status}): ${d.detail || "unknown"}`;
    return;
  }

  const data = await res.json();
  statusEl.textContent = "";

  // Summary line
  const fallRate = data.total_windows > 0
    ? ((data.falls_detected / data.total_windows) * 100).toFixed(1)
    : "0";
  document.getElementById("replay-summary").textContent =
    `${data.total_windows} windows | ${data.falls_detected} falls (${fallRate}%) | model: ${data.model_version?.toUpperCase()} | file: ${data.filename}`;

  // Results table
  const tbody = document.getElementById("replay-tbody");
  tbody.innerHTML = data.predictions.map(p => {
    const t = p.window_start_ms ? new Date(p.window_start_ms).toISOString().substring(11, 23) : "—";
    if (p.error) return `<tr><td>${p.window_index}</td><td>${t}</td><td colspan="3" style="color:#888">Error: ${p.error}</td></tr>`;
    return `
      <tr>
        <td>${p.window_index}</td>
        <td style="font-size:0.8rem">${t}</td>
        <td class="${p.fall_detected ? "fall-yes" : "fall-no"}">${p.fall_detected ? "FALL" : "No"}</td>
        <td>${p.confidence != null ? (p.confidence * 100).toFixed(1) + "%" : "—"}</td>
        <td>${p.latency_ms != null ? p.latency_ms + "ms" : "—"}</td>
      </tr>`;
  }).join("");

  resultsEl.style.display = "block";
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
