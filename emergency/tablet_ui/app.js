/**
 * Emergency Tablet UI — app.js
 *
 * Connects to the emergency notification service SSE stream.
 * Shows a full-screen flashing alert when a fall is detected.
 * EventSource auto-reconnects — no manual retry loop needed.
 */

const SSE_URL = "/api/emergency/stream";   // routed by nginx to emergency_svc:8003/stream
let currentAlert = null;

// ---- SSE Connection ----

function connect() {
  const statusEl = document.getElementById("connection-status");
  statusEl.textContent = "Connecting...";
  statusEl.className = "status-badge connecting";

  const es = new EventSource(SSE_URL);

  es.onopen = () => {
    statusEl.textContent = "Connected";
    statusEl.className = "status-badge connected";
  };

  es.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.fall_detected) {
        showAlert(data);
      } else {
        // Log non-fall event in last-event area
        const lastEl = document.getElementById("last-event");
        lastEl.textContent = `Last check: ${data.patient_id} — clear (${new Date(data.timestamp).toLocaleTimeString()})`;
      }
    } catch (e) {
      // ignore parse errors (keep-alive comments come through as empty)
    }
  };

  es.onerror = () => {
    statusEl.textContent = "Reconnecting...";
    statusEl.className = "status-badge disconnected";
    // EventSource handles reconnection automatically — no manual code needed
  };
}

// ---- Alert Display ----

function showAlert(event) {
  currentAlert = event;

  const conf = event.confidence != null
    ? `Confidence: ${(event.confidence * 100).toFixed(0)}%`
    : "";
  const model = event.model_version ? `Model: ${event.model_version}` : "";

  document.getElementById("alert-patient").textContent = event.patient_id;
  document.getElementById("alert-detail").textContent = [conf, model].filter(Boolean).join("  |  ");
  document.getElementById("alert-time").textContent = new Date(event.timestamp).toLocaleString();

  document.getElementById("standby-screen").hidden = true;
  document.getElementById("alert-screen").hidden = false;
  document.body.className = "alert-active";

  // Try browser notification (requires user permission)
  requestBrowserNotification(event);
}

function acknowledge() {
  currentAlert = null;
  document.getElementById("alert-screen").hidden = true;
  document.getElementById("standby-screen").hidden = false;
  document.body.className = "standby";
  document.getElementById("last-event").textContent =
    "Last alert acknowledged at " + new Date().toLocaleTimeString();
}

// ---- Browser Notifications (optional, requires user permission) ----

function requestBrowserNotification(event) {
  if (!("Notification" in window)) return;
  if (Notification.permission === "granted") {
    new Notification("FALL DETECTED", {
      body: `Patient: ${event.patient_id}`,
      icon: "/emergency/icon.png",
      requireInteraction: true,
    });
  } else if (Notification.permission !== "denied") {
    Notification.requestPermission().then(perm => {
      if (perm === "granted") requestBrowserNotification(event);
    });
  }
}

// ---- Start ----
connect();
