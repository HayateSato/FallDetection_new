/**
 * Patient Dashboard — app.js
 *
 * Flow:
 *  1. Connect to SSE at /api/ml/patient/stream?participant=<name>
 *  2. On fall event: show popup step 1 (10-second countdown)
 *     - Patient answers YES  → show popup step 2 (need help? 10s countdown)
 *     - Patient answers NO   → submit user_fall=2, need_help=2 → no alert sent
 *     - 10s timeout (step 1) → handled server-side (user_fall=3, need_help=3 → alert)
 *  3. Step 2: need help?
 *     - YES  → submit user_fall=1, need_help=1 → emergency alert sent
 *     - NO   → submit user_fall=1, need_help=2 → no emergency alert
 *     - 10s timeout (step 2) → submit user_fall=1, need_help=3 → emergency alert sent
 *
 * NOTE: The 10-second server-side timer for step 1 is started automatically when
 * the fall is detected (ml_server asyncio task). The feedback POST either confirms
 * or cancels it. Step 2 timeout is client-managed because the server already has
 * user_fall set from step 1 — we just POST need_help=3 on client timeout.
 */

const ML_API = "/api/ml";   // nginx proxies to ml_server:8001
const POPUP_TIMEOUT_MS = 10_000;

let patientName = localStorage.getItem("patient_name") || "";
let currentInferenceId = null;
let currentEvent = null;
let step1Timer = null;
let step2Timer = null;
let step1Remaining = 10;
let step2Remaining = 10;
let step1Interval = null;
let step2Interval = null;
let sseConn = null;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

window.addEventListener("DOMContentLoaded", () => {
  if (patientName) {
    document.getElementById("patient-name-input").value = patientName;
    document.getElementById("standby-patient").textContent = `Monitoring: ${patientName}`;
  }
  connectSSE();
});

// ---------------------------------------------------------------------------
// Patient name
// ---------------------------------------------------------------------------

function setPatientName() {
  const name = document.getElementById("patient-name-input").value.trim();
  patientName = name;
  localStorage.setItem("patient_name", name);
  document.getElementById("standby-patient").textContent =
    name ? `Monitoring: ${name}` : "Monitoring active";
  // Reconnect SSE with the new patient name filter
  connectSSE();
}

// ---------------------------------------------------------------------------
// SSE Connection
// ---------------------------------------------------------------------------

function connectSSE() {
  if (sseConn) { sseConn.close(); sseConn = null; }

  const url = patientName
    ? `${ML_API}/patient/stream?participant=${encodeURIComponent(patientName)}`
    : `${ML_API}/patient/stream`;

  sseConn = new EventSource(url);
  setStatus("connecting");

  sseConn.onopen = () => setStatus("connected");

  sseConn.onmessage = (ev) => {
    try {
      const data = JSON.parse(ev.data);
      // Only show popup for real-time detections (server filters out replay)
      if (data.fall_detected) {
        handleFallEvent(data);
      }
    } catch (e) {
      console.error("SSE parse error", e);
    }
  };

  sseConn.onerror = () => {
    setStatus("error");
    // EventSource auto-reconnects after ~3s — no manual logic needed
  };
}

function setStatus(state) {
  const el = document.getElementById("connection-status");
  el.className = `status-indicator ${state}`;
  el.textContent = state === "connecting" ? "Connecting..."
                 : state === "connected"  ? "Connected — monitoring active"
                 : "Connection error — retrying...";
}

// ---------------------------------------------------------------------------
// Fall event handling
// ---------------------------------------------------------------------------

function handleFallEvent(data) {
  // Ignore if a popup is already showing (edge case: double detection)
  if (!document.getElementById("popup-step1").classList.contains("hidden")) return;
  if (!document.getElementById("popup-step2").classList.contains("hidden")) return;

  currentEvent = data;
  currentInferenceId = data.inference_id;

  // Show timestamp in popup
  const ts = data.timestamp ? new Date(data.timestamp).toLocaleString() : new Date().toLocaleString();
  document.getElementById("popup-timestamp").textContent = `Detected at: ${ts}`;

  showStep1();
}

// ---------------------------------------------------------------------------
// Step 1 — Did you fall?
// ---------------------------------------------------------------------------

function showStep1() {
  clearTimers();
  step1Remaining = 10;
  document.getElementById("popup-step1").classList.remove("hidden");
  updateTimerBar("timer-bar", "countdown", step1Remaining);

  // Countdown display every second
  step1Interval = setInterval(() => {
    step1Remaining -= 1;
    updateTimerBar("timer-bar", "countdown", step1Remaining);
    if (step1Remaining <= 3) {
      document.getElementById("timer-bar").classList.add("urgent");
    }
  }, 1000);

  // Auto-close at 10s — server-side timer handles the DB update + emergency alert
  step1Timer = setTimeout(() => {
    clearTimers();
    document.getElementById("popup-step1").classList.add("hidden");
    showToast("No response recorded. Emergency contact will be alerted.");
    // No POST needed — ml_server _delayed_emergency_alert fires automatically
  }, POPUP_TIMEOUT_MS);
}

function respondFall(fell) {
  clearTimers();
  document.getElementById("popup-step1").classList.add("hidden");

  if (!fell) {
    // Patient says they did NOT fall → cancel emergency, submit feedback
    submitFeedback(2, 2).then(() => {
      showToast("Response recorded — no alert sent.");
    });
    return;
  }

  // Patient says they DID fall → ask if they need help (step 2)
  showStep2();
}

// ---------------------------------------------------------------------------
// Step 2 — Do you need help?
// ---------------------------------------------------------------------------

function showStep2() {
  clearTimers();
  step2Remaining = 10;
  document.getElementById("popup-step2").classList.remove("hidden");
  updateTimerBar("timer-bar2", "countdown2", step2Remaining);

  step2Interval = setInterval(() => {
    step2Remaining -= 1;
    updateTimerBar("timer-bar2", "countdown2", step2Remaining);
    if (step2Remaining <= 3) {
      document.getElementById("timer-bar2").classList.add("urgent");
    }
  }, 1000);

  // Client-side timeout for step 2: user_fall=1 (confirmed), need_help=3 (no_answer) → alert
  step2Timer = setTimeout(() => {
    clearTimers();
    document.getElementById("popup-step2").classList.add("hidden");
    submitFeedback(1, 3).then(() => {
      showToast("No response — emergency contact alerted.");
    });
  }, POPUP_TIMEOUT_MS);
}

function respondHelp(needsHelp) {
  clearTimers();
  document.getElementById("popup-step2").classList.add("hidden");

  if (needsHelp) {
    submitFeedback(1, 1).then(() => {
      showToast("Emergency contact alerted — help is on the way.");
    });
  } else {
    submitFeedback(1, 2).then(() => {
      showToast("Response recorded — no alert sent.");
    });
  }
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

async function submitFeedback(userFall, needHelp) {
  if (!currentInferenceId) return;
  try {
    await fetch(`${ML_API}/patient/feedback/${currentInferenceId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_fall: userFall, need_help: needHelp }),
    });
  } catch (e) {
    console.error("Failed to submit feedback:", e);
  } finally {
    currentInferenceId = null;
    currentEvent = null;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function updateTimerBar(barId, countdownId, remaining) {
  const pct = (remaining / 10) * 100;
  document.getElementById(barId).style.width = `${pct}%`;
  document.getElementById(countdownId).textContent = remaining;
}

function clearTimers() {
  clearTimeout(step1Timer);
  clearTimeout(step2Timer);
  clearInterval(step1Interval);
  clearInterval(step2Interval);
  step1Timer = step2Timer = step1Interval = step2Interval = null;
  // Reset timer bars for next use
  ["timer-bar", "timer-bar2"].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.style.width = "100%"; el.classList.remove("urgent"); }
  });
}

let toastTimeout = null;
function showToast(msg) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.classList.remove("hidden");
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toast.classList.add("hidden"), 4000);
}
