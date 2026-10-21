# Fall Detection — API Reference

Two services expose HTTP APIs. This document covers both.

| Service | Owner | Base URL |
|---------|-------|----------|
| **Inference Server** | MCS | `https://fall-api.smarko-health.de` |
| **Fall Dashboard** | FOCUS | `https://<Subdomain FOCUS will set>` |

Additionally, the **MQTT broker** (FOCUS) exposes topics that the mobile app publishes to
and the fall-dashboard subscribes to. Those topic contracts are documented at the end.

---

## 1. Inference Server (MCS)

Hosted by MCS on Hetzner. FOCUS does not need to manage or touch this service.

The mobile app is the primary caller. MCS uses the admin endpoints internally.

### Authentication

If the server is configured with API keys, all `POST` endpoints require:

```
X-API-Key: <key>
```

`GET /health` and `GET /model/info` are always public (no key needed).

---

### GET /health

**Who calls it:** monitoring, mobile app startup check

**Response:**
```json
{
  "status": "ok",
  "model_version": "v3",
  "uses_barometer": true,
  "sensor_type": "bosch",
  "sample_rate_hz": 50,
  "uptime_seconds": 3821.4
}
```

---

### GET /model/info

**Who calls it:** mobile app — called once at startup to know whether to send barometer data

**Response:**
```json
{
  "name": "XGBoost_ACC_BARO_v3",
  "uses_barometer": true,
  "num_features": 45,
  "loaded_as": "v3"
}
```

> If `uses_barometer` is `true`, the mobile app must include `pressure` and
> `pressure_timestamps_ms` in every `/predict` call. If `false`, those fields are ignored.

---

### POST /predict

**Who calls it:** mobile app — after every 9-second sensor window

**Request body:**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `patient_id` | string | yes | Identifies the patient. Used in FHIR output and MQTT alerts |
| `device_id` | string | no | SmarKo wearable identifier |
| `acc_x` | float[] | yes | Raw LSB integers from accelerometer X axis |
| `acc_y` | float[] | yes | Raw LSB integers from accelerometer Y axis |
| `acc_z` | float[] | yes | Raw LSB integers from accelerometer Z axis |
| `timestamps_ms` | float[] | yes | Unix epoch timestamps in milliseconds, one per ACC sample |
| `hardware_sample_rate` | int | no | ACC hardware rate in Hz. Overrides server default (50 Hz). Set to `25` for SmarKo Bosch sensor |
| `pressure` | float[] | no* | Barometer values in Pa. *Required if model `uses_barometer = true` |
| `pressure_timestamps_ms` | float[] | no* | Timestamps for barometer samples in ms. *Required with `pressure` |

All arrays in one request must have the same length. `acc_x`, `acc_y`, `acc_z`, and
`timestamps_ms` must always have equal length.

Minimum data required: `9s × hardware_rate` samples per axis (e.g. 225 samples at 25 Hz).

**Example request:**
```json
{
  "patient_id": "patient-001",
  "device_id": "AA:BB:CC:DD:EE:FF",
  "hardware_sample_rate": 25,
  "acc_x": [512, 518, 505, "...225 values total..."],
  "acc_y": [480, 477, 490, "..."],
  "acc_z": [1024, 1020, 1030, "..."],
  "timestamps_ms": [1751020800000, 1751020800040, "..."],
  "pressure": [101325.0, 101324.8, "..."],
  "pressure_timestamps_ms": [1751020800000, 1751020800040, "..."]
}
```

**Response:**

| Field | Type | Notes |
|-------|------|-------|
| `observation_id` | string (UUID) | Cross-reference key — include in MQTT payloads and `/confirm` call |
| `patient_id` | string | Echoed from request |
| `device_id` | string\|null | Echoed from request |
| `timestamp` | string (ISO 8601) | Server-side detection time in UTC |
| `inference.fall_detected` | bool | `true` if fall detected |
| `inference.confidence` | float | Model confidence score (0–1) |
| `inference.threshold` | float | Decision threshold used |
| `inference.result` | string | Human-readable label (e.g. `"High confidence fall"`) |
| `inference.model_version` | string | Model version that ran the inference |
| `inference.window_size` | int | Number of ACC samples used (should be 450) |
| `fhir_observation` | object | Full FHIR R4 Observation resource — can be POSTed to a FHIR server as-is |
| `fhir_pushed` | bool | `true` if the server already pushed to the configured FHIR server |

**Example response:**
```json
{
  "observation_id": "550e8400-e29b-41d4-a716-446655440000",
  "patient_id": "patient-001",
  "device_id": "AA:BB:CC:DD:EE:FF",
  "timestamp": "2026-07-01T12:34:56+00:00",
  "inference": {
    "fall_detected": true,
    "confidence": 0.8734,
    "threshold": 0.5,
    "result": "High confidence fall",
    "model_version": "v3",
    "window_size": 450
  },
  "fhir_observation": { "resourceType": "Observation", "..." : "..." },
  "fhir_pushed": false
}
```

**Mobile app must do after a `fall_detected: true` response:**
1. Publish to `fall/possible/<patient_id>` on MQTT (immediate — before popup)
2. Show "Did you fall?" popup to patient (10-second countdown)
3. Show "Do you need help?" popup if patient said yes (separate 10-second countdown)
4. Publish to `fall/alert/<patient_id>` on MQTT with patient's answers
5. Write fall event to FOCUS InfluxDB
6. Call `POST /inference/{observation_id}/confirm` with patient's answers

---

### POST /inference/{observation_id}/confirm

**Who calls it:** mobile app — after the patient confirmation popup closes (or times out)

`observation_id` is the UUID returned by `/predict`.

**Request body:**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `patient_confirmed` | string | yes | `"yes"` / `"no"` / `"not_answered"` |
| `needs_help` | bool | no | Whether a rescue alert was sent to the caregiver |

**Example request:**
```json
{
  "patient_confirmed": "yes",
  "needs_help": true
}
```

**Response:**
```json
{
  "status": "accepted",
  "observation_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

This call updates the MCS retraining database. It is non-blocking — the server returns
`202 accepted` immediately and writes the record in the background.

---

### Admin-only endpoints (MCS internal)

These are used by MCS for model management. The mobile app and FOCUS do not call these.

| Method | Path | What it does |
|--------|------|--------------|
| `GET` | `/model/list` | Lists model versions available on disk |
| `POST` | `/model/switch` | Hot-swaps the loaded model (no server restart) |
| `GET` | `/docs` | Swagger UI with live try-it-out |
| `GET` | `/metrics` | Prometheus metrics (if enabled) |

---

## 2. Fall Dashboard (FOCUS)

Hosted by FOCUS in K3s. The caregiver opens the browser UI which calls these endpoints
automatically. FOCUS DevOps does not need to call them manually in normal operation.

No authentication — the service is only reachable through Traefik HTTPS (port 443) and
is intended for internal hospital network use.

---

### GET /

**Who calls it:** caregiver — opens `https://fall.focus-hospital.de` in a browser

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

Long-lived Server-Sent Events connection. The browser keeps this open while the page is visible.
The server pushes one JSON event per MQTT message received from the broker.

**Event types:**

| Event | When | Data |
|-------|------|------|
| `event: connected` | On first connect | `{}` |
| `data: {...}` (default) | On every fall event | Fall event JSON (see below) |
| `: keepalive` | Every 15 seconds if no events | Comment line — no data |

**Fall event JSON (status: pending — from `fall/possible/<patient_id>`):**
```json
{
  "patient_id": "patient-001",
  "observation_id": "550e8400-e29b-41d4-a716-446655440000",
  "fall_detected": true,
  "confidence": 0.8734,
  "status": "pending"
}
```

**Fall event JSON (status: confirmed — from `fall/alert/<patient_id>`):**
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

The dashboard shows a "Possible fall, wait for confirmation" banner on the patient card when
`status = "pending"`, then escalates to "Fall is confirmed / Help is requested" when
`status = "confirmed"` arrives.

---

## 3. MQTT Topics

The MQTT broker (mosquitto) runs in FOCUS K3s. The mobile app connects externally via WSS on
port 443. The fall-dashboard connects internally via TCP on port 1883.

**Broker address:** `wss://mqtt.focus-hospital.de` (mobile app) / `mosquitto:1883` (fall-dashboard, cluster-internal)

---

### fall/possible/`<patient_id>`

**Published by:** mobile app — immediately after `/predict` returns `fall_detected: true`, before showing the popup

**Subscribed by:** fall-dashboard

**Payload:**
```json
{
  "patient_id": "patient-001",
  "device_id": "AA:BB:CC:DD:EE:FF",
  "timestamp": "2026-07-01T12:34:56+00:00",
  "observation_id": "550e8400-e29b-41d4-a716-446655440000",
  "fall_detected": true,
  "confidence": 0.8734,
  "model_version": "v3"
}
```

> `fall_detected` must be `true` — the fall-dashboard ignores messages where it is `false`.

---

### fall/alert/`<patient_id>`

**Published by:** mobile app — after the patient confirmation popup closes (or the 10-second timeout fires)

**Subscribed by:** fall-dashboard

**Payload** (all fields from `fall/possible` plus):

```json
{
  "patient_id": "patient-001",
  "device_id": "AA:BB:CC:DD:EE:FF",
  "timestamp": "2026-07-01T12:34:56+00:00",
  "observation_id": "550e8400-e29b-41d4-a716-446655440000",
  "fall_detected": true,
  "confidence": 0.8734,
  "model_version": "v3",
  "alert_time": "2026-07-01T12:35:06+00:00",
  "patient_confirmed": "yes",
  "needs_help": true
}
```

| Field | Type | Notes |
|-------|------|-------|
| `alert_time` | string (ISO 8601) | When the alert was published (after popup / timeout) |
| `patient_confirmed` | string | `"yes"` / `"no"` / `"not_answered"` |
| `needs_help` | bool\|null | `true` if patient (or timeout) triggered rescue alert |

`needs_help: true` causes the fall-dashboard to show the "Help is requested" banner on the
patient card in red. `needs_help: false` shows "Fall is confirmed" without the help banner.

> `observation_id` must match the UUID returned by `/predict` — this is how the MCS
> retraining pipeline links the MQTT alert back to the inference log record.
