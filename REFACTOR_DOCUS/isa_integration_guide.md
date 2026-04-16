# Integration Guide — Patient Dashboard (for Isa)

This document covers everything Isa needs to integrate our fall detection backend
into the Patient Dashboard. Hayate maintains this guide.

---

## What is what — naming

| Name | What it is | Who builds it |
|------|-----------|---------------|
| **Patient Dashboard** | The unified web app shown to the caregiver | Isa |
| **fall_dashboard** | Our FastAPI backend (:8002) — provides fall history + real-time alerts | Hayate |
| **Mock FHIR server** | Local-dev simulation of FOCUS's FHIR server (:8003) — provides demographics | Hayate (mock only; replaced by real FOCUS FHIR in production) |

The Patient Dashboard combines three data sources:

| Panel | Data source | Host (local dev) |
|-------|-------------|------------------|
| Demographics (name, DOB, height, weight) | FHIR server | `http://localhost:8003` (mock) |
| Biosignals (HR, SpO₂) | InfluxDB | FOCUS side — you already have this |
| Fall history + real-time alerts | fall_dashboard REST API + SSE | `http://localhost:8002` |

---

## Mock FHIR Server (local dev)

During development you do not need to wait for FOCUS's real FHIR server.
Hayate's mock server runs at **`:8003`** and serves synthetic patient demographics.

**Start it:**
```powershell
# From _6G_Integration_v2_mqtt/ as cwd:
uvicorn focus_mock.fhir_server:app --host 0.0.0.0 --port 8003
# OR via docker-compose — focus_mock_fhir service
```

### `GET /fhir/Patient`

Returns a FHIR R4 Bundle of all patients.

```json
{
  "resourceType": "Bundle",
  "type": "searchset",
  "total": 2,
  "entry": [
    { "resource": { "resourceType": "Patient", "id": "test_patient-002", ... } }
  ]
}
```

### `GET /fhir/Patient/{patient_id}`

Returns a single FHIR R4 Patient resource.

```json
{
  "resourceType": "Patient",
  "id": "test_patient-002",
  "name": [{ "family": "Schmidt", "given": ["Margarete"] }],
  "gender": "female",
  "birthDate": "1938-11-03",
  "address": [{ "text": "Ward B, Room 7, Charité Berlin" }],
  "extension": [
    { "url": "http://hl7.org/fhir/StructureDefinition/patient-ward", "valueString": "Ward B" }
  ]
}
```

### `GET /fhir/Observation?patient={patient_id}`

Returns a Bundle of vital-sign Observations (height, weight, heart rate).
LOINC codes: `8302-2` = body height, `29463-7` = body weight, `8867-4` = heart rate.

```json
{
  "resourceType": "Bundle",
  "type": "searchset",
  "total": 3,
  "entry": [
    {
      "resource": {
        "resourceType": "Observation",
        "code": { "coding": [{ "system": "http://loinc.org", "code": "8302-2", "display": "Body height" }] },
        "valueQuantity": { "value": 158, "unit": "cm" }
      }
    }
  ]
}
```

**In production:** replace `http://localhost:8003` with the real FOCUS FHIR server URL.
The response shape is standard FHIR R4 — no code changes needed when switching.

---

## Fall Dashboard REST API

Base URL: `http://localhost:8002` (local dev) · TBD Kubernetes service URL (production)

### `GET /api/patients`

Returns all registered patients with their current fall count.

```json
{
  "patients": [
    {
      "patient_id": "test_patient-002",
      "mac_id":     "6c:1d:eb:04:a9:e6",
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
| `only_falls` | `true` | `true` = confirmed events only; `false` = all rows |
| `limit` | `200` | Max rows returned (up to 2000) |

**Example:** `GET /api/falls?patient_id=test_patient-002&limit=50`

**Response:**
```json
{
  "falls": [
    {
      "id":                1,
      "patient_id":        "test_patient-002",
      "mac_id":            "6c:1d:eb:04:a9:e6",
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

| Value | Meaning | Show alert? |
|-------|---------|------------|
| `yes` | Patient confirmed they fell | Yes |
| `no` | Patient said they did not fall (false positive) | No |
| `not_answered` | Popup timed out (10s) — treated as a fall | Yes |

---

### `GET /api/stream`

Server-Sent Events (SSE) stream. Emits one event per confirmed fall alert the
instant it arrives from MQTT. Use this to drive the real-time alert banner.

```js
const source = new EventSource('http://localhost:8002/api/stream');

source.addEventListener('connected', () => {
  console.log('SSE connected');
});

source.onmessage = (e) => {
  const event = JSON.parse(e.data);
  showFallAlert(event);  // same fields as /api/falls rows
};

// Browser auto-reconnects on error — no manual handling needed
source.onerror = () => {};
```

**Event payload:**
```json
{
  "fall_id":           1,
  "patient_id":        "test_patient-002",
  "mac_id":            "6c:1d:eb:04:a9:e6",
  "fall_detected":     true,
  "patient_confirmed": "yes",
  "needs_help":        true,
  "observation_id":    "a1b2c3d4-...",
  "timestamp":         "2026-04-16T10:00:10+00:00"
}
```

The stream sends a `: keepalive` comment every 15 seconds to keep the connection
alive through proxies.

---

## MQTT — React Native compatibility

**Yes, Eclipse Mosquitto is fully compatible with React Native.**

React Native cannot use raw TCP sockets (port 1883). Use **MQTT over WebSocket**
on port **9001** instead — already enabled in `mosquitto.conf`:

```
listener 9001
protocol websockets
```

**npm package:** `mqtt` (MQTT.js) — works in React Native without any native modules:

```bash
npm install mqtt
```

```js
import mqtt from 'mqtt';

// port 9001 = WebSocket (works in React Native)
// port 1883 = raw TCP   (does NOT work in React Native)
const client = mqtt.connect('ws://<broker-host>:9001');

client.on('connect', () => {
  // Subscribe to all patients, or a specific one:
  client.subscribe('fall/alert/#');
  // client.subscribe('fall/alert/test_patient-002');
});

client.on('message', (topic, message) => {
  const event = JSON.parse(message.toString());
  // event.patient_id, event.fall_detected, event.patient_confirmed,
  // event.needs_help, event.confidence, event.observation_id
  showFallAlert(event);
});

client.on('error', (err) => {
  console.error('MQTT error:', err);
});
```

**Broker connection details:**

| Setting | Local dev | Production |
|---------|-----------|------------|
| Host | `localhost` | Kubernetes service name (TBD) |
| Port | `9001` (WebSocket) | `9001` (WebSocket) |
| Auth | None (anonymous) | Username/password (to be set before prod) |
| Topic | `fall/alert/#` | `fall/alert/#` |

**SSE vs MQTT — which to use in the Patient Dashboard:**

- **Patient Dashboard (web app)** → use SSE (`/api/stream`). Simpler, no extra library, works in any browser.
- **Mobile app (React Native)** → use MQTT over WebSocket. Direct broker subscription, lower latency.

---

## CORS

The fall_dashboard API allows all origins (`*`) in dev. No CORS config needed.

---

## Suggested integration pattern

```
On dashboard load:
  1. GET /fhir/Patient/{id}             → render demographics panel (name, DOB, ward)
  2. GET /fhir/Observation?patient={id} → render vitals (height, weight, HR)
  3. GET /api/patients                  → render patient list with fall count badges
  4. GET /api/falls?patient_id={id}     → render fall history table
  5. Open EventSource /api/stream       → listen for real-time fall alerts

On SSE message:
  6. Show alert banner (patient name, time, needs_help)
  7. Re-fetch GET /api/patients         → update fall count badge
  8. Prepend new row to fall history table
```

---

## Cross-namespace URL (Kubernetes — future)

When deployed to Kubernetes, replace localhost URLs with:

| Service | Kubernetes URL |
|---------|----------------|
| fall_dashboard | `http://fall-dashboard.fall-detection.svc.cluster.local:8002` |
| Mock FHIR (replaced by real FOCUS FHIR in prod) | real FOCUS FHIR URL in their namespace |

---

## Open questions (to confirm with Hayate)

- [ ] Final Kubernetes service name for fall_dashboard
- [ ] JWT auth on the API — required before production
- [ ] MQTT broker auth — username/password before production
- [ ] Confirm `patient_id` values match the FHIR Patient `id` field you use on the FOCUS side
