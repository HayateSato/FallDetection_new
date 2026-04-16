# Integration Guide — Patient Dashboard (for Isa)

This document covers everything needed to integrate the fall detection backend
into the Patient Dashboard. The fall detection side is built by Hayate; this
guide describes what endpoints to call, what data you get back, and how to
receive real-time fall alerts.

---

## Overview

The Patient Dashboard is one web app shown to the caregiver. It combines two
data sources:

| Panel | Data source | Who provides it |
|-------|-------------|-----------------|
| Patient info — demographics (height, weight), biosignals (HR) | FHIR server + InfluxDB | FOCUS side (you already have this) |
| Fall panel — fall history + real-time fall alerts | `fall_dashboard` REST API + SSE | Hayate's backend (:8002) |

This guide covers the **fall panel** integration only.

---

## Base URL

| Environment | URL |
|-------------|-----|
| Local dev | `http://localhost:8002` |
| Production | TBD (Kubernetes service in our namespace) |

---

## Endpoints

### `GET /api/patients`

Returns all registered patients with their current fall count.

**Response:**
```json
{
  "patients": [
    {
      "patient_id": "patient-001",
      "mac_id":     "AA:BB:CC:DD:EE:FF",
      "fall_count": 3
    }
  ]
}
```

---

### `GET /api/falls`

Returns fall history rows.

**Query parameters:**

| Parameter | Default | Notes |
|-----------|---------|-------|
| `patient_id` | (all patients) | Filter by patient |
| `only_falls` | `true` | `true` = confirmed fall events only; `false` = all rows |
| `limit` | `200` | Max rows returned (up to 2000) |

**Example:**
```
GET /api/falls?patient_id=patient-001&limit=50
```

**Response:**
```json
{
  "falls": [
    {
      "id":                1,
      "patient_id":        "patient-001",
      "mac_id":            "AA:BB:CC:DD:EE:FF",
      "fall_detected":     true,
      "patient_confirmed": "yes",
      "needs_help":        true,
      "observation_id":    "a1b2c3d4-...",
      "detection_time":    "2026-04-16T10:00:10+00:00",
      "alert_time":        "2026-04-16T10:00:21+00:00"
    }
  ]
}
```

**`patient_confirmed` values:**

| Value | Meaning |
|-------|---------|
| `yes` | Patient confirmed they fell |
| `no` | Patient said they did not fall (false positive) |
| `not_answered` | Confirmation popup timed out (10s) — treated as a fall |

---

### `GET /api/stream`

Server-Sent Events (SSE) stream. Emits one event per confirmed fall alert the
instant it arrives from MQTT. Use this to drive the real-time alert banner.

**Connect:**
```js
const source = new EventSource('http://localhost:8002/api/stream');

source.addEventListener('connected', () => {
  console.log('SSE connected');
});

source.onmessage = (e) => {
  const event = JSON.parse(e.data);
  // event has the same fields as /api/falls rows
  showFallAlert(event);
};

source.onerror = () => {
  // Browser auto-reconnects on error — no manual handling needed
};
```

**Event payload** (same shape as `/api/falls` rows):
```json
{
  "fall_id":          1,
  "patient_id":       "patient-001",
  "mac_id":           "AA:BB:CC:DD:EE:FF",
  "fall_detected":    true,
  "patient_confirmed": "yes",
  "needs_help":       true,
  "observation_id":   "a1b2c3d4-...",
  "timestamp":        "2026-04-16T10:00:10+00:00"
}
```

The stream also sends a `: keepalive` comment every 15 seconds to keep the
connection alive through proxies.

---

## MQTT — direct subscription (alternative to SSE)

If the React Native app needs to consume fall alerts directly (without going
through the SSE backend), it can subscribe to the MQTT broker over WebSocket.

**Broker:** same host as the fall_dashboard, port **9001** (WebSocket)
— port 1883 is raw TCP and does not work from a browser or React Native.

**Topic:** `fall/alert/#` (all patients) or `fall/alert/<patient_id>`

**npm package:** `mqtt` (MQTT.js — works in React Native without native modules)

```js
import mqtt from 'mqtt';

const client = mqtt.connect('ws://<broker-host>:9001');

client.on('connect', () => {
  client.subscribe('fall/alert/#');
});

client.on('message', (topic, message) => {
  const event = JSON.parse(message.toString());
  // event.patient_id, event.patient_confirmed, event.needs_help, etc.
  showFallAlert(event);
});
```

**Note:** No authentication is configured on the broker in the current dev setup
(anonymous connections allowed). Auth will be added before production.

---

## CORS

The fall_dashboard API allows all origins (`*`) in dev. No CORS config needed
on your side for local development.

---

## Suggested integration pattern

```
On dashboard load:
  1. GET /api/patients          → render patient list with fall counts
  2. GET /api/falls             → render fall history table
  3. Open EventSource /api/stream

On SSE message:
  4. Show alert banner (patient name, time, needs_help)
  5. Re-fetch GET /api/patients  → update fall count badge
  6. Prepend new row to fall history table
```

---

## Open questions (to confirm with Hayate)

- [ ] Final production URL for fall_dashboard (Kubernetes service name in our namespace)
- [ ] Auth on the API — JWT header required? Currently open (no auth in dev)
- [ ] Auth on MQTT broker — username/password before production
- [ ] Patient ID format — confirm the `patient_id` values match the FHIR Patient identifiers you are already using
