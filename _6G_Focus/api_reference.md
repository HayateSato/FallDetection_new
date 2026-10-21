# Fall Detection — API Reference (Fall Dashboard)

The fall dashboard exposes a JSON API used by the caregiver browser UI, plus an SSE stream
for live fall events.

| Service | Owner | Base URL |
|---------|-------|----------|
| **Fall Dashboard** | FOCUS | `https://<subdomain FOCUS will set>` |

- No authentication — the service is only reachable through Traefik HTTPS (port 443)
and is intended for internal FOCUS/hospital network use.
- Once the subdomain is made, Charite needs to know the URL when they want to monitor the patients.

---

## 1. Fall Dashboard HTTP API

Hosted by FOCUS in K3s. The caregiver opens the browser UI which calls these endpoints
automatically. FOCUS DevOps does not need to call them manually in normal operation.

---

### GET /

**Who calls it:** caregiver — opens the fall-dashboard URL in a browser

Returns the caregiver dashboard HTML page. All further API calls are made automatically
by the browser.

---

### GET /api/patients

**Who calls it:** browser (on page load and after every live fall event)

**Response:**
```json
{
  "patients": [
    {
      "patient_id": "patient-001",
      "name": "John Doe",
      "mac_id": "AA:BB:CC:DD:EE:FF",
      "fall_count": 3,
      "fall_count_today": 1
    }
  ]
}
```

| Field | Notes |
|-------|-------|
| `fall_count` | Falls in the last 30 days (shown on patient card badge) |
| `fall_count_today` | Falls in the last 24 hours (shown in the header "Falls Today" stat) |
| `mac_id` | SmarKo device MAC address — optional, shown as a badge on the card |

---

### POST /api/patients

**Who calls it:** browser — caregiver clicks "+ Add Patient"

**Request body:**

| Field | Type | Required |
|-------|------|----------|
| `patient_id` | string | yes |
| `name` | string | no |
| `mac_id` | string | no |

**Response (201):**
```json
{ "patient_id": "patient-001", "created": true }
```

Patient list is stored in SQLite on a PVC — survives pod restarts and upgrades.

---

### DELETE /api/patients/{patient_id}

**Who calls it:** browser — caregiver clicks the X button on a patient card

**Response:**
```json
{ "patient_id": "patient-001", "deleted": true }
```

Removes the patient from the local SQLite store. Fall history in InfluxDB is not affected.

Returns `404` if the patient was not found.

---

### GET /api/falls

**Who calls it:** browser — when the caregiver opens a patient's detail/history view

**Query parameters:**

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `patient_id` | string | none | Filter to one patient. Omit to return all |
| `only_falls` | bool | `true` | If `true`, returns only rows where `fall_detected = true` |
| `limit` | int | 200 | Maximum number of records to return (max 2000) |
| `hours` | int | 720 | How far back to look (default 720h = 30 days; use 24 for "today") |

**Response:**
```json
{
  "falls": [
    {
      "id": 0,
      "patient_id": "patient-001",
      "fall_detected": true,
      "patient_confirmed": 1,
      "needs_help": true,
      "confidence": 0.8734,
      "detection_time": "2026-07-01T12:34:56+00:00"
    }
  ]
}
```

`patient_confirmed` is an integer:

| Value | Meaning |
|-------|---------|
| `1` | Patient confirmed it was a fall ("yes") |
| `0` | Patient denied — false positive ("no") |
| `-1` | No response within 10-second timeout ("not_answered") |

---

### GET /api/stream

**Who calls it:** browser — persistent SSE connection opened on page load

Long-lived [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) connection. The browser keeps this open while the page is visible and receives push notifications without polling.

#### How alerts travel from the mobile app to the browser

The browser has no direct connection to the MQTT broker. The fall-dashboard acts as a bridge:

```
Mobile app
  |
  | MQTT over WSS (port 443, via Traefik)
  v
mosquitto broker  (port 9001 WebSocket, cluster-internal)
  |
  | MQTT TCP (port 1883, cluster-internal)
  v
fall-dashboard  <-- subscribes to fall/possible/# and fall/alert/#
  |
  | Server-Sent Events (GET /api/stream, HTTPS port 443)
  v
Caregiver browser
```

When the mobile app publishes a message to the broker, the fall-dashboard's internal MQTT client receives it on port 1883, converts it to a JSON event, and pushes it onto every open `/api/stream` connection. The browser receives it as a standard SSE message and updates the UI immediately — no page reload needed.

#### Event types

| Event | When | Data |
|-------|------|------|
| `event: connected` | On first connect | `{}` |
| `data: {...}` (default) | On every fall event | Fall event JSON (see below) |
| `: keepalive` | Every 15 seconds if no events | Comment line — keeps proxy connections alive |

#### Fall event: possible (status: pending)

Triggered when the mobile app publishes to `fall/possible/<patient_id>` — immediately after inference, before the patient has confirmed. The caregiver sees a "Possible fall, wait for confirmation" banner on the patient card.

```json
{
  "patient_id": "patient-001",
  "observation_id": "550e8400-e29b-41d4-a716-446655440000",
  "fall_detected": true,
  "confidence": 0.8734,
  "status": "pending"
}
```

#### Fall event: confirmed (status: confirmed)

Triggered when the mobile app publishes to `fall/alert/<patient_id>` — after the patient responded to the popup (or the 10-second timeout fired). The banner upgrades to "Fall is confirmed" or "Help is requested" depending on `needs_help`.

```json
{
  "patient_id": "patient-001",
  "observation_id": "550e8400-e29b-41d4-a716-446655440000",
  "fall_detected": true,
  "confidence": 0.8734,
  "patient_confirmed": "yes",
  "needs_help": true,
  "alert_time": "2026-07-01T12:35:06+00:00",
  "status": "confirmed"
}
```

| Field | Notes |
|-------|-------|
| `status` | `"pending"` (pre-confirmation) or `"confirmed"` (post-confirmation) |
| `patient_confirmed` | `"yes"` / `"no"` / `"not_answered"` — only present on confirmed events |
| `needs_help` | `true` → "Help is requested" banner; `false` → "Fall is confirmed" banner |
| `observation_id` | UUID linking this alert back to the inference server log |
