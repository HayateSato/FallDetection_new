# Mobile App Integration Handover

**Audience:** Isa (SmarKo) — the mobile-app developer responsible for integrating the SmarKo wearable phone app with our fall-detection backend.
**Repo / branch:** `_6G_Integration_v2_mqtt/` on branch `6G-integration_with_MQTT`.
**Reference implementation:** `local_dev/mock_app/` — same flow, with InfluxDB replacing BLE as the input source. Read this code if anything below is unclear.
**Scope:**
- Data contract (exact JSON the mobile app sends and receives)
- What the mobile app should do (the 4-step responsibility)
- What is missing (what's not yet implemented on the mobile-app side)
- Where each existing piece lives + how the mobile app talks to each component

---

## 1. The mobile app's job — at a glance

```
1. Read raw ACC + barometer from SmarKo wearable via BLE
       │
       ▼
2. Build a 9-second window @ 50 Hz, POST /predict to inference-server
       │  receive: fall_detected, observation_id, confidence
       ▼
3. If fall_detected=True → show patient confirmation popup (10s timeout)
       │  patient answers Yes/No (and Yes/No need_help if applicable)
       ▼
4. PUBLISH fall/alert/<patient_id> via MQTT
       └ payload includes observation_id + patient_confirmed + needs_help
   (optional) write a marker to FOCUS InfluxDB for time-series visualisation
```

You only need steps 2–4 in production. Step 1 (BLE) you already do. Step 4's InfluxDB marker is optional.

---

## 2. Data contract

### 2.1 Where the inference server lives

| Environment | URL |
|-------------|-----|
| Local dev (Hayate's laptop) | `http://localhost:8001` |
| FOCUS production | `https://<focus-ingress-host>/predict` (ingress hostname TBD by FOCUS DevOps) |
| In-cluster service DNS | `http://inference-server.mcs-fall-detection.svc.cluster.local:8001` (irrelevant for the phone — phone is outside the cluster) |

Auth: every request must include `X-API-Key: <key>`. The dev key is in `_6G_Integration_v2_mqtt/.env` as `INFERENCE_API_KEY`. We will rotate per environment.

### 2.2 POST /predict — the inference call

#### Request

```http
POST /predict HTTP/1.1
Content-Type: application/json
X-API-Key: <INFERENCE_API_KEY>

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

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `patient_id` | string | **yes** | The FHIR Patient identifier |
| `device_id` | string | no | Wearable MAC, useful for debugging |
| `acc_x`, `acc_y`, `acc_z` | array of int | **yes** | **Raw LSB integers, NOT g.** Server does the LSB→g conversion. |
| `timestamps_ms` | array of int (Unix ms) | **yes** | One per ACC sample, same length as acc_x/y/z |
| `pressure` | array of float (Pa) | no | Required when the loaded model uses barometer (currently `v3`) |
| `pressure_timestamps_ms` | array of int (Unix ms) | required if `pressure` is set | One per pressure sample |

**Window sizing:**
- At least **450 ACC samples** per axis = 9 seconds @ 50 Hz.
- The wearable's hardware ACC rate is configured server-side via `HARDWARE_ACC_SAMPLE_RATE` in `.env`. Currently `50`. If the SmarKo BLE feed is at a different rate, tell us and we'll set the env var; the server will resample upward internally.
- Barometer rate is 25 Hz hardware → 50 Hz interpolated server-side.

**How to detect "the model needs barometer"** — call `GET /model/info` (no auth). Response:

```json
{ "model_version": "v3", "uses_barometer": true, "window_size": 450 }
```

If `uses_barometer=false`, omit `pressure` and `pressure_timestamps_ms`. We currently default to v3, so include barometer.

#### Response

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
    "model_version": "v3",
    "window_size":   450
  },
  "fhir_observation": { /* FHIR R4 Observation — optional */ },
  "fhir_pushed":     false
}
```

**The critical field is `observation_id`.** UUID generated server-side per call. **Save it.** You must include it in the MQTT payload (Section 2.4) so the patient's confirmation can be linked back to the original prediction. Without it, retraining cannot use the fall as a labelled example.

`fhir_observation` is the FHIR R4 Observation (SNOMED 217082002). It's there for FOCUS to consume. The mobile app does **not** need to do anything with it. If `FHIR_SERVER_URL` is configured server-side, we POST it for you.

#### Errors

| Status | Cause | Mobile app behaviour |
|--------|-------|----------------------|
| 200 | success | proceed to popup if `fall_detected=true` |
| 401 | bad/missing X-API-Key | refresh the key, retry once, alert ops if persistent |
| 422 | bad payload (length mismatch, missing required field) | log + skip — do not retry the same window |
| 5xx | server error | retry up to 3× with exponential backoff |

#### Useful side endpoints

| Endpoint | Auth | Purpose |
|----------|------|---------|
| `GET /health` | none | liveness — returns `{"ok":true,"model_version":"v3"}` |
| `GET /model/info` | none | tells you `uses_barometer`, `window_size`, current `model_version` |

---

### 2.3 Patient confirmation popup — UX contract

When `inference.fall_detected === true`, the mobile app must:

1. **Wake the screen** + show a high-priority popup with a 10-second countdown.
2. **Question 1:** "Did you fall? Yes / No"
   - **Yes →** show Question 2.
   - **No →** publish MQTT with `patient_confirmed="no"`, `needs_help=null`. Caregiver is **not** notified.
   - **Timeout (no response in 10s) →** publish MQTT with `patient_confirmed="not_answered"`, `needs_help=null`. Caregiver **is** notified (conservative).
3. **Question 2:** "Do you need help? Yes / No"
   - **Yes →** publish MQTT with `patient_confirmed="yes"`, `needs_help=true`. Caregiver notified.
   - **No →** publish MQTT with `patient_confirmed="yes"`, `needs_help=false`. Stored for retraining, **caregiver NOT notified** (patient said they're okay).

#### Reference UI

`local_dev/mock_app/patient_server.py` runs a browser-based version at `http://localhost:8005/`. The Yes/No flow there is exactly what the native popup should mirror. The HTML at the bottom of that file shows the layout.

#### Caregiver alert filter — important

The fall_dashboard backend SSE-fans-out only **two of the four cases** to the caregiver:

| `patient_confirmed` | `needs_help` | Stored in DB | Caregiver notified |
|---------------------|--------------|:------------:|:------------------:|
| `not_answered` | n/a | yes | **yes (amber)** |
| `yes` | `true` | yes | **yes (red)** |
| `yes` | `false` | yes | no — patient said they're fine |
| `no` | n/a | yes | no — false positive |

So the popup **MUST publish all four cases** (we need them for retraining), but only two surface as live alerts.

---

### 2.4 MQTT publish — telling the backend the patient confirmed

After the popup completes (Yes/No/timeout), publish to MQTT.

#### Connection

| | Local dev | FOCUS production |
|---|-----------|------------------|
| Host | `localhost` | `mqtt-broker.<our-namespace>.svc.cluster.local` (TBD by FOCUS DevOps) — or via ingress |
| Port (TCP) | `1883` | `1883` |
| Port (WebSocket) | `9001` (use this if React Native — easier than native MQTT lib) | `9001` |
| Auth | none | TBD — likely user/pass at production cutover |

#### Topic

```
fall/alert/<patient_id>
```

The patient_id substitutes for the wildcard. fall_dashboard subscribes to `fall/alert/#` so it sees all patients.

#### Payload — exact JSON contract

```json
{
  "observation_id":    "3a0a603e-cc5a-4355-ad8c-f53f2e4de1b9",
  "patient_id":        "charite-patient-001",
  "device_id":         "6c:1d:eb:04:a9:e6",
  "timestamp":         "2026-04-28T10:00:00+00:00",
  "alert_time":        "2026-04-28T10:00:15+00:00",
  "fall_detected":     true,
  "confidence":        0.8234,
  "model_version":     "v3",
  "fhir_observation":  null,
  "patient_confirmed": "yes",
  "needs_help":        true
}
```

| Field | Source | Required |
|-------|--------|----------|
| `observation_id` | from the /predict response | **mandatory** |
| `patient_id` | from /predict | yes |
| `device_id` | from /predict | yes |
| `timestamp` | `inference.timestamp` from /predict response | yes |
| `alert_time` | when the popup completed (now) | yes |
| `fall_detected` | always `true` (only published on confirmed-fall flow) | yes |
| `confidence` | from /predict response | yes |
| `model_version` | from /predict response | yes |
| `fhir_observation` | from /predict response, or `null` | no — can omit |
| `patient_confirmed` | `"yes"` / `"no"` / `"not_answered"` | **mandatory** |
| `needs_help` | `true` / `false` / `null` | yes |

QoS: 1 is safe. 2 is overkill. Retain flag: false.

---

### 2.5 (Optional) InfluxDB marker

Convenience for FOCUS-side time-series visualisation. The same data is already in our Postgres `fall_history` after the MQTT publish, so functionally nothing changes if you skip this. Useful only if Charite wants "HR + fall markers on the same Grafana panel".

```
measurement:    fall_event
tag:            patient_id     = "charite-patient-001"
tag:            device_id      = "6c:1d:eb:04:a9:e6"
field:          confidence     = 0.8234   (float)
field:          confirmed      = "yes"     (string)
field:          needs_help     = true      (bool)
field:          observation_id = "<uuid>"  (string)
timestamp:      detection_time
```

Use the InfluxDB write API the mobile app already has for biosignals. Fire-and-forget — we do not retry, do not depend on this.

---

## 3. What should happen — the responsibility list

Concretely, the mobile app must:

| # | Responsibility | Frequency |
|---|----------------|-----------|
| 1 | Read raw ACC samples from BLE at the wearable's hardware rate (currently 50 Hz) | continuous |
| 2 | Read barometer samples from BLE at 25 Hz | continuous |
| 3 | Buffer 9-second windows of ACC + barometer | continuous |
| 4 | POST every window to `/predict` (with API key) | every 9 seconds (or your sliding-window cadence — overlap is fine) |
| 5 | On `fall_detected=true`, save observation_id + show popup | per fall |
| 6 | On popup answer / timeout, publish to MQTT `fall/alert/<patient_id>` | per fall |
| 7 | (Optional) Write the InfluxDB marker | per fall |
| 8 | Continue biosignal writes to FOCUS InfluxDB (existing functionality, unchanged) | continuous |
| 9 | Handle network errors with retries (Section 2.2 errors table) | as needed |

What the app must NOT do:

- Convert ACC LSB to g — server does this. If you send g values, the model gets scaled-down inputs and produces nonsense.
- Send features. Send raw ACC. The server has the feature pipeline.
- Subscribe to MQTT. The mobile app is publish-only on `fall/alert/<pid>`.
- Talk to fall_dashboard, MLflow, or Postgres directly. Only inference-server (HTTP) and MQTT broker.

---

## 4. What is missing (gaps to close)

What the mobile app does NOT yet implement (as of 2026-04-29):

| Gap | Status | Reference |
|-----|:------:|-----------|
| Native fall confirmation popup with 10s countdown | not started | mock UI in `local_dev/mock_app/patient_server.py` |
| Two-question flow (fall? need help?) | not started | same |
| MQTT WebSocket client (or native TCP) wiring | not started | reference: `local_dev/mock_app/poller.py` (Python paho-mqtt) |
| `observation_id` plumbing: receive from /predict, include in MQTT payload | not started | this is the linchpin — see Section 2.2 |
| API-key handling — fetch from secure store, attach as header | not started | dev key in `.env`, prod rotation TBD |
| Retry/backoff on /predict 5xx | not started | Section 2.2 errors table |
| Sliding window buffer of ACC samples | likely partially done already (you have BLE) | confirm |
| InfluxDB marker write (optional) | not started | Section 2.5 — skip for now |

You can validate each gap by running our `local_dev/mock_app/` and reading the same flow in Python — porting it to Swift / Kotlin / React Native is the implementation work.

What is **already** implemented elsewhere and you don't need to touch:

- inference-server (`/predict`, `/model/switch`, `/model/info`)
- MQTT broker (we ship Mosquitto in the Helm chart)
- fall_dashboard (subscribes, writes DB, fans out SSE)
- Patient Dashboard fall panel (FOCUS DevOps owns — see `05_web_app_integration.md`)

---

## 5. Where existing pieces live + how to talk to each

### 5.1 Reference implementation in this repo

```
local_dev/mock_app/                ← simulates exactly what your mobile app should do
├── influx_fetcher.py              ← (mock-only) reads raw ACC from a cloud InfluxDB. Your real app reads from BLE instead.
├── api_caller.py                  ← HTTP POST /predict — copy this verbatim, swap the data source
├── poller.py                      ← polling loop; orchestrates fetch → predict → popup → MQTT publish
├── patient_server.py              ← browser-based popup at :8005 (HTML inline at the bottom)
└── main.py                        ← entry point: python -m local_dev.mock_app.main
```

The Python loop in `poller.py` is the simplest single-file example of the full mobile-app flow. ~150 lines. If you read one file, read this one.

### 5.2 Component-by-component talk-to map

| You call | Component | Protocol | Where it lives in production |
|----------|-----------|----------|------------------------------|
| `POST /predict`, `GET /health`, `GET /model/info` | inference-server | HTTPS | `https://<ingress>/predict` (Ingress → ClusterIP `inference-server.mcs-fall-detection.svc.cluster.local:8001`) |
| `PUBLISH fall/alert/<pid>` | MQTT broker | MQTT/WS (port 9001) or MQTT/TCP (port 1883) | `mqtt-broker.mcs-fall-detection.svc.cluster.local` (or via ingress / NodePort — TBD by FOCUS DevOps) |
| Biosignal write (existing) | FOCUS InfluxDB | InfluxDB HTTP API | FOCUS namespace (your existing flow — unchanged) |
| (Optional) `fall_event` marker | FOCUS InfluxDB | same InfluxDB HTTP API | same |

### 5.3 What you do NOT call

- fall_dashboard (`:8002`) — that's the **Patient Dashboard's** dependency, not yours
- MLflow, MinIO, Postgres, Prometheus, Grafana — internal-only
- FOCUS FHIR server — only the inference-server pushes there (optionally)

---

## 6. End-to-end smoke test

```
1. POST a known-fall ACC window to /predict → expect fall_detected=true.
   (Use a recorded "phone-on-floor" window from your test rig.)

2. Show the popup, click "Yes I fell" → "Yes I need help".

3. Watch the caregiver dashboard at http://localhost:8002/ — within 1–2 seconds
   the live banner should appear:
   "FALL ALERT — charite-patient-001 (patient confirmed — needs help, confidence 82%)".

4. Refresh the Fall History tab — the new row should be there with
   patient_confirmed=yes, needs_help=Yes.

5. Run:
     docker exec -it fall_postgres psql -U fall_user -d fall_detection \
       -c "SELECT id, patient_id, patient_confirmed, needs_help, observation_id
           FROM fall_history ORDER BY id DESC LIMIT 1;"
   Confirm the row's observation_id matches the UUID you got from /predict.

If steps 1–5 all pass, the mobile app is correctly integrated.
```

Repeat with the other 3 popup answers (`No`, `Yes/No I'm okay`, `timeout`) and verify the alert filter behaviour:
- `Yes/No I'm okay` → DB row appears, but no caregiver banner.
- `No` → DB row appears, no caregiver banner.
- `timeout` → DB row appears with `patient_confirmed=not_answered`, amber banner appears.

---

## 7. Common pitfalls (read this once)

- **`observation_id` is mandatory in the MQTT payload.** Without it, the patient confirmation cannot be linked back to the prediction. Retraining drops these as ambiguous.
- **`acc_x/y/z` must be raw LSB integers**, not g values. The server does the conversion.
- **Timestamp lengths must match samples 1:1.** `len(acc_x) == len(acc_y) == len(acc_z) == len(timestamps_ms)`. Mismatch returns 422.
- **Hot-swap is in-memory.** If the inference pod restarts, it boots back to whatever `MODEL_VERSION` is in the chart. Mobile app sees consistent /predict responses regardless — but if you see a sudden behaviour change, ask MCS whether a hot-swap was rolled back.
- **Patient ID must match across /predict and MQTT.** They're correlated via `patient_id` plus `observation_id`. A mismatch and the dashboard shows the alert against a different patient (or no patient).
- **MQTT topic is `fall/alert/<patient_id>`, not `falls/alert/<pid>` or `alert/fall/<pid>`.** Easy typo.
- **WebSocket port 9001, TCP port 1883.** React Native usually wants WebSocket. Use it.
- **Don't retain MQTT messages.** Retain flag must be `false`. Otherwise a re-subscribing dashboard receives stale alerts.
- **API key is per environment.** Local dev key ≠ FOCUS prod key. Don't hardcode.

---

## 8. Cross-references

- [`02_fall_detection_algorithm.md`](02_fall_detection_algorithm.md) — what the model does with what you send
- [`03_fall_detection_system.md`](03_fall_detection_system.md) — system overview, sequence diagrams
- [`05_web_app_integration.md`](05_web_app_integration.md) — what the Patient Dashboard does after your MQTT publish
- [`06_user_flow_patient.md`](06_user_flow_patient.md) — patient-side experience (popup details)
- Existing handover docs at `handover_docs/ISA_mobile_app_contract.md` — earlier draft of this same content

---

## 9. Contact

| For | Reach out to |
|-----|--------------|
| /predict contract, MQTT contract, observation_id, model behaviour | Hayate (MCS) |
| Production hostname / ingress / TLS / API-key rotation | FOCUS DevOps |
| Patient Dashboard rendering of your alerts | FOCUS DevOps (UI team) |
| Wearable BLE characteristics (sample rate, axis convention) | SmarKo internal |

A 30-minute sync once you've started integration usually clears 80% of remaining questions.
