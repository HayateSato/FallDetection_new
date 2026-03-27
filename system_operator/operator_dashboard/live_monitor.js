"use strict";

const ML_API = "/api/ml";
const MAX_TERMINAL_LINES = 2000;

// ── State ──────────────────────────────────────────────────────────────────

let _logLines = [];
let _sse = null;
let _statusInterval = null;
let _autoScroll = true;

// ── DOM helpers ────────────────────────────────────────────────────────────

function getApiKey() {
  return localStorage.getItem("operatorApiKey") || "";
}

function setActionStatus(msg, colour) {
  const el = document.getElementById("action-status");
  el.textContent = msg;
  el.style.color = colour || "#aaa";
}

function setStatusDot(state) {
  const dot  = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  dot.className = "status-dot " + state.dotClass;
  text.textContent = state.label;
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function lineClass(text) {
  const lower = text.toLowerCase();
  if (lower.includes("error") || lower.includes("traceback") || lower.includes("exception")) return "line-err";
  if (lower.includes("warning") || lower.includes("warn")) return "line-warn";
  if (text.startsWith("---")) return "line-sep";
  return "line-info";
}

// ── Terminal ───────────────────────────────────────────────────────────────

function appendLine(text, extraClass) {
  _logLines.push(text);
  if (_logLines.length > MAX_TERMINAL_LINES) {
    _logLines.shift();
    renderTerminal();
    return;
  }
  const el = document.getElementById("terminal");
  const div = document.createElement("div");
  div.className = extraClass || lineClass(text);
  div.textContent = text;
  el.appendChild(div);
  if (_autoScroll) el.scrollTop = el.scrollHeight;
}

function renderTerminal() {
  const el = document.getElementById("terminal");
  el.innerHTML = "";
  for (const line of _logLines) {
    const div = document.createElement("div");
    div.className = lineClass(line);
    div.textContent = line;
    el.appendChild(div);
  }
  if (_autoScroll) el.scrollTop = el.scrollHeight;
}

function clearTerminal() {
  _logLines = [];
  document.getElementById("terminal").innerHTML = "";
}

// ── Status polling ─────────────────────────────────────────────────────────

async function pollStatus() {
  try {
    const res = await fetch(`${ML_API}/client/status`);
    if (!res.ok) throw new Error(res.status);
    const data = await res.json();

    if (data.running) {
      setStatusDot({ dotClass: "running", label: `Running  (PID ${data.pid})` });
    } else {
      const rc = data.returncode != null ? `  exit ${data.returncode}` : "";
      setStatusDot({ dotClass: "stopped", label: `Stopped${rc}` });
    }

    // Update server health badge
    const badge = document.getElementById("server-status");
    badge.textContent = "ML Server OK";
    badge.className = "badge badge-green";
  } catch {
    setStatusDot({ dotClass: "error", label: "Cannot reach ml_server" });
    const badge = document.getElementById("server-status");
    badge.textContent = "ML Server unreachable";
    badge.className = "badge badge-red";
  }
}

// ── SSE log stream ─────────────────────────────────────────────────────────

function connectLogStream() {
  if (_sse) {
    _sse.close();
    _sse = null;
  }

  const es = new EventSource(`${ML_API}/client/logs`);
  _sse = es;

  es.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.keepalive) return;
      if (msg.eof) {
        appendLine(msg.line || "--- process ended ---", "line-eof");
        return;
      }
      appendLine(msg.line || "");
    } catch {
      appendLine(evt.data);
    }
  };

  es.onerror = () => {
    // SSE disconnected — retry after 3s
    es.close();
    _sse = null;
    setTimeout(connectLogStream, 3000);
  };
}

// ── Control buttons ────────────────────────────────────────────────────────

async function startClient() {
  const apiKey = getApiKey();
  if (!apiKey) {
    setActionStatus("Set API key on the main dashboard first.", "#e74c3c");
    return;
  }
  setActionStatus("Starting…", "#aaa");
  try {
    const res = await fetch(`${ML_API}/client/start`, {
      method: "POST",
      headers: { "X-API-Key": apiKey },
    });
    const data = await res.json();
    if (!res.ok) {
      setActionStatus("Error: " + (data.detail || res.status), "#e74c3c");
      return;
    }
    if (data.status === "already_running") {
      setActionStatus(`Already running (PID ${data.pid})`, "#ffd166");
    } else {
      setActionStatus(`Started (PID ${data.pid})`, "#4caf50");
    }
    await pollStatus();
  } catch (e) {
    setActionStatus("Request failed: " + e.message, "#e74c3c");
  }
}

async function stopClient() {
  const apiKey = getApiKey();
  if (!apiKey) {
    setActionStatus("Set API key on the main dashboard first.", "#e74c3c");
    return;
  }
  setActionStatus("Stopping…", "#aaa");
  try {
    const res = await fetch(`${ML_API}/client/stop`, {
      method: "POST",
      headers: { "X-API-Key": apiKey },
    });
    const data = await res.json();
    if (!res.ok) {
      setActionStatus("Error: " + (data.detail || res.status), "#e74c3c");
      return;
    }
    if (data.status === "not_running") {
      setActionStatus("Was not running.", "#ffd166");
    } else {
      setActionStatus(`Stopped (PID ${data.pid})`, "#aaa");
    }
    await pollStatus();
  } catch (e) {
    setActionStatus("Request failed: " + e.message, "#e74c3c");
  }
}

// ── Auto-scroll toggle ─────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  const terminal = document.getElementById("terminal");
  terminal.addEventListener("scroll", () => {
    const atBottom = terminal.scrollHeight - terminal.scrollTop - terminal.clientHeight < 40;
    _autoScroll = atBottom;
  });

  pollStatus();
  _statusInterval = setInterval(pollStatus, 10000);
  connectLogStream();
});
