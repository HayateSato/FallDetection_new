/**
 * Operator Dashboard — app.js
 *
 * Panels:
 *   1. Model info + hot-switch via POST /api/ml/model/switch
 *   2. Live server health via GET /api/ml/health
 *   3. Grafana links (opens in new tab)
 *   4. Recent inference log via GET /api/caregiver/... (reads Postgres)
 */

const ML_API   = "/api/ml";
const CG_API   = "/api/caregiver";
const API_KEY  = localStorage.getItem("op_api_key") || promptForApiKey();

function promptForApiKey() {
  const key = prompt("Enter operator API key:");
  if (key) localStorage.setItem("op_api_key", key);
  return key;
}

// ---- Boot ----

window.addEventListener("DOMContentLoaded", () => {
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
  if (!version) return;

  statusEl.textContent = "Switching...";
  const data = await mlFetch("/model/switch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ version }),
  });
  if (!data) {
    statusEl.textContent = "Switch failed.";
    return;
  }
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

// ---- Recent Inference Log (reads from caregiver API → Postgres) ----

async function loadRecentLog() {
  // We read all patients' fall history — just show the last 20 any-patient predictions
  // This requires caregiver API to expose a /inferences endpoint (add if needed)
  // For now, show a placeholder that links to Grafana
  const tbody = document.getElementById("log-tbody");
  tbody.innerHTML = `
    <tr>
      <td colspan="6" class="loading">
        For full inference history, see
        <a href="/grafana/d/fall-events-timeline" target="_blank">Fall Events Timeline</a>
        in Grafana (PostgreSQL datasource).
      </td>
    </tr>
  `;
}

// ---- Helpers ----

async function mlFetch(path, options = {}) {
  try {
    const res = await fetch(`${ML_API}${path}`, {
      ...options,
      headers: {
        "X-API-Key": API_KEY || "",
        ...(options.headers || {}),
      },
    });
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    return null;
  }
}
