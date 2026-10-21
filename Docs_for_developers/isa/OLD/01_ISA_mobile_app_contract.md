# Mobile App Integration Contract — for Isa

**Purpose:** the contract the SmarKo mobile app must implement to integrate with our fall-detection backend.
**Scope:** what the mobile app sends to our inference API, what it receives back, how it shows the patient confirmation popup, how it publishes the alert via MQTT, and what (optional) marker it can write to InfluxDB.
**Reference implementation:** `_6G_Integration_v2_mqtt/local_dev/mock_app/` — same flow, just with InfluxDB instead of BLE input. Read this code if anything below is unclear.

---

## 1. The 4 things the mobile app does

```
1. Read raw ACC from SmarKo wearable via BLE
       │
       ▼
2. POST /predict to inference_server (HTTP)        ← this doc, section 3
       │  receives fall_detected, confidence, observation_id
       ▼
3. If fall_detected=True → show patient popup      ← this doc, section 4
       │  patient answers Yes/No + needs_help
       ▼
4. PUBLISH fall/alert/<patient_id> via MQTT        ← this doc, section 5
   (optional) write marker to FOCUS InfluxDB        ← this doc, section 6
```

You only need to do steps 2–4 in production. Step 1 is BLE which you already do.
Step 6 is optional — the data already lives in our Postgres, so the InfluxDB
marker is purely for FOCUS-side visualisation if Isa wants it.

---

## 2. Where the inference server lives

| Environment | URL |
|-------------|-----|
| Local dev (Hayate's laptop) | `http://localhost:8001` |
| Production (in our K8s namespace) | `http://inference-server.<our-namespace>.svc.cluster.local:8001` (or via ingress — exact hostname TBD by FOCUS DevOps) |

All endpoints require an `X-API-Key` header. The current dev key is in `_6G_Integration_v2_mqtt/.env` — `INFERENCE_API_KEY`. We will rotate this for production.

---

## 3. POST /predict — the inference call

### Request

```http
POST /predict HTTP/1.1
Content-Type: application/json
X-API-Key: <key from .env INFERENCE_API_KEY>

{
  "patient_id":             "charite-patient-001",
  "device_id":              "6c:1d:eb:04:a9:e6",
  "acc_x":                  [-512, -498, -510, ...],
  "acc_y":                  [128, 134, 121, ...],
  "acc_z":                  [16300, 16280, 16310, ...],
  "timestamps_ms":          [1712345678000, 1712345678020, ...],
  "pressure":               [101325.0, 101322.5, ...],          // optional
  "pressure_timestamps_ms": [1712345678000, 1712345678020, ...] // optional
}
```

**Field rules:**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `patient_id` | string | **yes** | the FHIR Patient identifier this person is registered as |
| `device_id` | string | no | usually the wearable MAC; useful for debugging |
| `acc_x`, `acc_y`, `acc_z` | array of int | **yes** | **raw LSB integers, NOT converted to g**. The server does the LSB → g conversion internally. |
| `timestamps_ms` | array of int | **yes** | Unix epoch milliseconds, one per sample. Same length as acc_x/y/z. |
| `pressure` | array of float | no | barometer in Pascals. Only needed if the loaded model uses barometer (check `GET /model/info` → `uses_barometer`). |
| `pressure_timestamps_ms` | array of int | no | required if `pressure` is set |

**How much data per call:**
- At least **450 samples per axis** = 9 seconds at 50 Hz.
- If the wearable runs at a different rate, send your hardware rate — the server will resample. Tell us the rate so `HARDWARE_ACC_SAMPLE_RATE` in `.env` matches.

### Response

```json
{
  "observation_id": "3a0a603e-cc5a-4355-ad8c-f53f2e4de1b9",
  "patient_id":     "charite-patient-001",
  "device_id":      "6c:1d:eb:04:a9:e6",
  "timestamp":      "2026-04-28T10:00:00+00:00",
  "inference": {
    "fall_detected": true,
    "confidence":    0.8234,
    "threshold":     0.5,
    "result":        "High confidence fall",
    "model_version": "v0",
    "window_size":   450
  },
  "fhir_observation": { /* FHIR R4 Observation — optional consumer */ },
  "fhir_pushed":    false
}
```

**The critical field is `observation_id`** — a UUID generated server-side per call.
**Save it.** You must include it in the MQTT payload (section 5) so we can link the patient's confirmation back to the original prediction in our Postgres. Without it, the retraining pipeline cannot use the fall as a labelled example.

`fhir_observation` is a FHIR R4 Observation resource encoding the fall result (SNOMED 217082002). It's there for FOCUS to consume if they want; we do not require you to do anything with it. If `FHIR_SERVER_URL` is configured on our side, we'll auto-POST this to FOCUS's FHIR server ourselves.

### Error responses

| Status | Cause | Mobile app behaviour |
|--------|-------|---------------------|
| 200 | success | proceed to step 3 (popup if fall_detected=True) |
| 401 | missing/wrong X-API-Key | retry once with refreshed key; alert ops if persistent |
| 422 | bad payload (length mismatch, missing required field) | log and skip — do not retry the same window |
| 5xx | server error | retry up to 3× with exponential backoff |

### Other useful endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` (no auth) | liveness check, returns `model_version` |
| `GET /model/info` (no auth) | tells you `uses_barometer` so you know whether to send pressure |

---

## 4. Patient confirmation popup

When `inference.fall_detected === true`:

1. **Wake the screen** and show a high-priority popup with a 10-second countdown.
2. **Question 1:** "Did you fall? Yes / No"
   - **Yes →** show Question 2.
   - **No →** publish MQTT alert with `patient_confirmed="no"`, `needs_help=null`. The caregiver is **not** notified.
   - **Timeout (no response in 10s) →** treat as "not_answered". Publish MQTT with `patient_confirmed="not_answered"`, `needs_help=null`. The caregiver **is** notified (conservative: maybe the patient can't answer because they are hurt).
3. **Question 2:** "Do you need help? Yes / No"
   - **Yes →** publish MQTT with `patient_confirmed="yes"`, `needs_help=true`. Caregiver notified.
   - **No →** publish MQTT with `patient_confirmed="yes"`, `needs_help=false`. Stored for retraining but **caregiver is NOT alerted** (patient confirmed but said they're okay).

### Reference UI (mock)

`local_dev/mock_app/patient_server.py` runs a browser-based version of this popup
at `http://localhost:8005/` for development. The Yes/No flow there is exactly what
the native popup should mirror. The HTML at the bottom of that file shows the
layout.

### Caregiver alert filter — important

Our `fall_dashboard` (the backend the caregiver dashboard talks to) only fans out
the live alert in two of the four cases:

| `patient_confirmed` | `needs_help` | Stored in DB | Caregiver alerted? |
|---------------------|--------------|:------------:|:------------------:|
| `not_answered` | n/a | yes | **yes** |
| `yes` | `true` | yes | **yes** |
| `yes` | `false` | yes | no — patient said they're fine |
| `no` | n/a | yes | no — false positive |

So your popup MUST publish all four cases via MQTT (we need them for retraining),
but only two of them will surface as alerts to the caregiver.

---

## 5. MQTT publish — telling the backend

After the confirmation popup completes (Yes/No/timeout), publish to MQTT.

### Connection

| | Local dev | Production |
|---|---|---|
| Host | `localhost` | `mqtt-broker.<our-namespace>.svc.cluster.local` (TBD by FOCUS DevOps) |
| Port | `1883` (TCP) or `9001` (WebSocket — useful if the mobile app is React Native and prefers WS) | same |
| Auth | none in dev | TBD — likely username/password in production |

### Topic

```
fall/alert/<patient_id>
```

The patient_id replaces the wildcard. Our `fall_dashboard` subscribes to `fall/alert/#` (everything under fall/alert/) so it sees all patients.

### Payload — exact JSON contract

```json
{
  "observation_id":    "3a0a603e-cc5a-4355-ad8c-f53f2e4de1b9",
  "patient_id":        "charite-patient-001",
  "device_id":         "6c:1d:eb:04:a9:e6",
  "timestamp":         "2026-04-28T10:00:00+00:00",
  "alert_time":        "2026-04-28T10:00:15+00:00",
  "fall_detected":     true,
  "confidence":        0.8234,
  "model_version":     "v0",
  "fhir_observation":  null,
  "patient_confirmed": "yes",
  "needs_help":        true
}
```

| Field | Source | Required |
|-------|--------|----------|
| `observation_id` | from the /predict response (section 3) | **yes — mandatory** |
| `patient_id` | from /predict | yes |
| `device_id` | from /predict | yes |
| `timestamp` | `inference.timestamp` from /predict response | yes |
| `alert_time` | when the popup completed (now) | yes |
| `fall_detected` | always `true` (only published on confirmed fall flow) | yes |
| `confidence` | from /predict response | yes |
| `model_version` | from /predict response | yes |
| `fhir_observation` | from /predict response, or null | no — can omit |
| `patient_confirmed` | `"yes"` / `"no"` / `"not_answered"` | **yes** |
| `needs_help` | `true` / `false` / `null` | yes |

QoS: `1` (at-least-once) is safe. `2` is overkill. Retain flag: `false`.

### What the backend does with this

```
mobile app
   │  PUBLISH fall/alert/charite-patient-001
   ▼
[MQTT broker]
   │  routes to subscribers of fall/alert/#
   ▼
fall_dashboard (port 8002 in our namespace)
   │
   ├─ writes one row to Postgres fall_history
   │  (so it's queryable forever, drives the Fall Dashboard UI)
   │
   └─ if (patient_confirmed == "not_answered") OR
         (patient_confirmed == "yes" AND needs_help == true):
        SSE fan-out to connected caregiver browsers
        (live banner alert appears on the Fall Dashboard)
```

`fall_dashboard` is the **only** MQTT subscriber on our side. It is the bridge between MQTT (real-time) and Postgres + SSE (durable + browser).

---

## 6. (OPTIONAL) Inject a marker into FOCUS InfluxDB

This is a **convenience for FOCUS-side visualisation only**. The same data is
already in our Postgres `fall_history` table after step 5, so functionally
nothing changes if you skip this. But if FOCUS wants the fall events to appear
inline with the biosignal time series in InfluxDB (so a Grafana panel can show
"HR + fall markers on the same timeline"), the mobile app should write a marker
to the FOCUS-hosted InfluxDB.

### Suggested measurement

```
measurement:    fall_event
tag:            patient_id   = "charite-patient-001"
tag:            device_id    = "6c:1d:eb:04:a9:e6"
field:          confidence   = 0.8234   (float)
field:          confirmed    = "yes"     (string)
field:          needs_help   = true      (bool)
field:          observation_id = "<uuid>"
timestamp:      detection_time
```

Use the InfluxDB write API the mobile app already has for biosignals.

**This write is fire-and-forget** from our perspective — we do not retry it, we
do not depend on it. If it fails, the data is still in our Postgres and the
caregiver alert flow still works.

---

## 7. Common pitfalls — read this once

- **`observation_id` is mandatory in the MQTT payload.** Without it, the patient's confirmation cannot be linked back to the prediction. Retraining pipeline drops these as ambiguous.
- **Aliases are case-sensitive.** Don't typo. `Production` ≠ `production`. (Probably won't affect the mobile app, but if you ever set aliases via API, watch for it.)
- **`acc_x/y/z` must be raw LSB integers**, not converted to g. The server does the conversion. If you send g values, the model will see scaled-down inputs and produce nonsense.
- **Timestamps must match samples 1:1.** `len(acc_x) == len(acc_y) == len(acc_z) == len(timestamps_ms)`. Mismatch returns 422.
- **Hot-swap is in-memory only.** If the inference pod restarts, it boots back to whatever `MODEL_VERSION` is in our `.env`. The mobile app doesn't need to know about this — it sees consistent /predict responses regardless — but if you see a sudden change in model behaviour, check with us whether a hot-swap was rolled back by a restart.

---

## 8. End-to-end smoke test

To verify the integration:

1. POST a known-fall ACC window (e.g. someone dropping the wearable to the floor) to `/predict` → expect `fall_detected: true`.
2. Show the popup, hit "Yes I fell" → "Yes I need help".
3. Watch the caregiver dashboard at `http://localhost:8002/` — within 1–2 seconds the live banner should appear: "FALL ALERT — charite-patient-001 (patient confirmed — needs help, confidence 82%)".
4. Refresh the Fall History tab — the new row should be there with `patient_confirmed=yes`, `needs_help=Yes`.
5. Run `docker exec -it fall_postgres psql -U fall_user -d fall_detection -c "SELECT id, patient_id, patient_confirmed, needs_help, observation_id FROM fall_history ORDER BY id DESC LIMIT 1;"` to confirm the row matches the observation_id you got from /predict.

If steps 1–5 all pass, the mobile app is correctly integrated.

---
