# Integration Plan — Fall Detection into FOCUS System
**Updated from:** Meeting agenda + meeting minutes (2026-04-13)  
**Status:** Planning — awaiting InfluxDB field names from Isa

---

## What We Know (Confirmed)

| Topic | Confirmed detail |
|-------|----------------|
| Wearable | SmarKo (same as our training hardware) |
| Sample rate | TBC — ask Isa (assume 25Hz until confirmed) |
| Barometer | Not used — model v0 (ACC only) |
| InfluxDB location | Same Kubernetes namespace as their existing system |
| InfluxDB tags | `macAddress` + `Patient ID` |
| InfluxDB field names | TBC — ask **Isa** |
| Deployment | Kubernetes + Helm Charts. We deliver a separate Helm chart that plugs into their existing namespace |
| Real-time event transport | **MQTT** (replaces Redis/SSE in our original design) |
| Dashboard roles | Admin (service health + ML model status, no patient data) and Caregiver (patient data scoped to their group/floor) |
| Dashboard developer | Possibly **Andreea** — needs confirmation |
| Machine | 12 cores, 32GB RAM |

## What Is Still Unknown

| Item | Who to ask | Blocking? |
|------|-----------|-----------|
| InfluxDB bucket name + measurement name + ACC field names | Isa | Yes — needed before any data fetcher code |
| SmarKo sample rate setting (25Hz or 100Hz) | Isa / Charite | Yes — one `.env` line but must be correct |
| Do they have a FHIR server? Is FHIR format required? | FOCUS | Yes — determines output format |
| Where should the detection result land? (FHIR DB / dashboard DB / MQTT event only) | FOCUS | Yes — core architecture decision |
| MQTT broker details (host, port, topic names, auth) | FOCUS DevOps | Yes — needed for Event Publisher/Subscriber |
| Patient ID format used in their system | FOCUS | Yes — must match what goes in FHIR Observation |
| Is Andreea the dashboard developer? Will she add our event subscriber? | FOCUS | Yes — determines scope of our dashboard work |
| Helm chart namespace + any naming conventions | FOCUS DevOps | Yes — needed before writing the Helm chart |

---

## Architecture Overview

![Architecture Overview](6G_architecture_overview.png)

**Their existing system (left box):**
- SmarKo → Mobile App → InfluxDB
- Existing Data Fetcher (for their dashboard) — do NOT touch this
- FHIR DB
- Dashboard

**Our additional components (right box — all delivered as one Helm chart):**

| Component | What it does | Maps to our existing code |
|-----------|-------------|--------------------------|
| Data Fetcher (for inference) | Polls InfluxDB for ACC data per patient | `fall_dashboard/influx_poller.py` (to be refactored) |
| API Caller | Sends sensor data to Inference API | `fall_dashboard/inference_client.py` |
| Inference API | Runs XGBoost model → FHIR result | `inference_server/server.py` |
| Event Publisher | Publishes fall event to MQTT broker | Currently Redis publish in `server.py` → **replace with MQTT** |
| Event Subscriber | Subscribes to MQTT → triggers dashboard alert | Currently `redis_listener.py` → **replace with MQTT** |
| Model Updater | Allows model version updates without redeployment | Exists as `POST /model/switch` endpoint |

**Key architecture rule from meeting:**
> The Data Fetcher for inference must be **completely separate** from their existing Data Fetcher for the dashboard. Do not modify or touch their existing fetcher.

---

## Implementation Steps

### Step 1 — Clarify open questions (before writing any code)

Do not proceed to code until these are answered:

- [ ] Get InfluxDB field names from **Isa** (bucket, measurement, ACC field names, macAddress tag format) 
- [ ] Confirm sample rate with Isa (25Hz or 100Hz in this trial) --> 50hz
- [ ] Confirm with FOCUS: do they have a FHIR server, and is FHIR the required output format?
- [ ] Confirm with FOCUS: where should the detection result land? (FHIR DB? Dashboard DB? MQTT only?)
- [ ] Get MQTT broker details from FOCUS DevOps (host, port, topic naming convention, authentication)
- [ ] Confirm Helm chart namespace and naming conventions from FOCUS DevOps
- [ ] Confirm Andreea is the dashboard developer and she will integrate the event subscriber

---

### Step 2 — Replace Redis with MQTT in the inference server

**File to change:** `_6G_Integration_v2/inference_server/server.py`

Currently (lines 295–310) the server calls `_redis_client.publish()` after a fall is detected. This needs to become an MQTT publish instead.

```python
# CURRENT (Redis)
await _redis_client.publish("fall_events", json.dumps(payload))

# TARGET (MQTT — exact topic name TBC from FOCUS)
mqtt_client.publish("fall_events/<patient_id>", json.dumps(payload))
```

**What you need from FOCUS before doing this:**
- MQTT broker host + port
- Topic naming convention (e.g. `fall_events/patient_id`? or a single topic for all?)
- Authentication (username/password? TLS certificate?)
- Which MQTT library to use — `paho-mqtt` is the standard Python library

**Note on MQTT vs Redis:**
MQTT is designed for IoT/mobile bidirectional messaging. The Mobile App can also publish to the same MQTT broker. Redis would only have served backend-to-backend. MQTT is the right choice here.

---

### Step 3 — Replace Redis subscriber with MQTT subscriber in caregiver client

**File to change:** `_6G_Integration_v2/fall_dashboard/redis_listener.py`

This file creates a `FallEventBroker` that subscribes to Redis and fans out to SSE clients. It needs to be rewritten to subscribe to MQTT instead, then push to whatever the dashboard uses (SSE or a direct MQTT consumer in the browser).

**Key question:** Does the dashboard browser directly subscribe to MQTT, or does it go through our backend?
- If their dashboard has an MQTT client built in → we publish to MQTT and they consume it directly, no SSE needed
- If their dashboard expects HTTP/SSE → we subscribe to MQTT on the backend and forward to SSE as before

Confirm with Andreea (the dashboard developer).

---

### Step 4 — Separate the Data Fetcher from the caregiver client

**Current state:** `influx_poller.py` is tightly coupled inside `fall_dashboard/` — it polls InfluxDB AND calls the inference server AND writes to the local DB in one component.

**Required state (per meeting):** The data fetcher must be a standalone component, separate from the rest of the pipeline.

**Refactor plan:**
1. Extract `influx_poller.py` into its own module (e.g. `data_fetcher/influx_fetcher.py`)
2. It should only do one thing: fetch sensor data from InfluxDB and pass it to the API Caller
3. The API Caller (`inference_client.py`) then calls the Inference API
4. Results flow to Event Publisher, not back through the fetcher

This separation also makes the Helm chart cleaner — each component can be its own Kubernetes pod with its own resource limits.

---

### Step 5 — Configure InfluxDB connection with confirmed field names

**Status: SKIPPED — current `.env` settings are kept as-is**

The existing `.env` already has working InfluxDB credentials and field names
from the test setup (`fd_test` bucket, `bosch_acc_x/y/z` fields, `25Hz`).
These will be left unchanged until Isa confirms the production values.
No code changes are needed at this stage.

When Isa provides confirmed values, update `_6G_Integration_v2_mqtt/.env`:
```env
INFLUXDB_BUCKET=<confirmed>
ACC_FIELD_X=<confirmed>
ACC_FIELD_Y=<confirmed>
ACC_FIELD_Z=<confirmed>
HARDWARE_ACC_SAMPLE_RATE=<confirmed>
```

---

### Step 6 — Implement two-role dashboard

**Confirmed roles:**
- **Admin**: sees service health (is the inference server up?), ML model status (which version is loaded, last prediction time), no patient names or identifiers
- **Caregiver**: sees patients assigned to their group/floor, fall history, real-time alerts

**Scope question to clarify:** Are we building the dashboard, or are we providing the backend API that Andreea's team integrates into their existing dashboard?

If we build it:
- Admin page: expose `GET /health` and `GET /model/info` from inference server
- Caregiver page: existing `GET /api/patients` and `GET /api/falls` from caregiver client (already built)
- Add role-based auth (JWT — already exists in the full system at `shared/auth/jwt_utils.py`)

If they integrate:
- We document our API endpoints
- They build the UI
- We only need to ensure our API returns the right data in the right format

---

### Step 7 — Write the Helm chart

This is the **deployment deliverable** — a single Helm chart that deploys all our additional components into their existing Kubernetes namespace.

**Services to include in the chart:**
```
fall-detection/
├── inference-api/          (inference_server/server.py)
├── data-fetcher/           (influx_poller — separated in Step 4)
├── api-caller/             (inference_client.py)
├── event-publisher/        (MQTT publish on fall)
├── event-subscriber/       (MQTT subscribe → dashboard)
└── model-updater/          (POST /model/switch endpoint)
```

**What you need from FOCUS DevOps before doing this:**
- Their Kubernetes namespace name
- Whether they have a container registry we should push images to, or if they pull from Docker Hub
- Any resource quota limits per pod
- Their ingress controller (nginx? traefik?) — needed if any of our services need to be reachable from outside the cluster
- Helm values naming conventions they follow

**Note on complexity:** Writing a Helm chart is primarily a DevOps task, not a data science task. If FOCUS DevOps can take the lead on this given the docker-compose equivalent we provide, that is a reasonable division of labour. Raise this in the next meeting.

---

### Step 8 — End-to-end integration test

Before go-live:

1. Point our Data Fetcher at their InfluxDB (same namespace — just a service name, no external URL)
2. Trigger a test fall (manually inject data into InfluxDB or use CSV replay)
3. Verify: Inference API returns FHIR Observation with correct patient ID
4. Verify: MQTT event published to broker with correct topic
5. Verify: Dashboard receives alert in real time
6. Verify: Fall history is stored and retrievable via API
7. Verify: Admin sees service health; Caregiver sees only their patients

---

## Open Decisions That Need a Second Meeting

| Decision | Options | Who decides |
|----------|---------|-------------|
| FHIR output — is it required? | Yes (FHIR R4 Observation) / No (plain JSON) | FOCUS |
| Where does the result land? | FHIR DB / dashboard DB / MQTT event only | FOCUS |
| Dashboard — who builds it? | Us / Andreea's team / both | FOCUS + us |
| Helm chart — who writes it? | Us / FOCUS DevOps / both | FOCUS DevOps + us |
| Patient feedback loop | In scope for this trial? | Charite + FOCUS |
| Model retraining data pipeline | In scope? Data sharing agreement needed | Charite + FOCUS + us |

---

## Still Open — Remaining Agenda Questions

From the original meeting agenda that were not answered yet:

- [ ] Bucket name + measurement name + ACC field names → **Isa**
- [ ] FHIR server: does it exist, is it required? → **FOCUS**
- [ ] Any other DB for dashbaord → **Isa**
- [ ] MQTT broker: host, port, topic convention, auth → **FOCUS DevOps** --> we decide this
- [ ] Patient ID + MacAddress format in dashboard → **Isa**
- [ ] Is Andreea the dashboard developer? → **FOCUS**
- [ ] Helm chart namespace + conventions → **FOCUS DevOps** --> we built this for now
- [ ] Helm chart namespace, should it be one? or running two (exsting one + additional coponents)? → **FOCUS**
- [ ] Number of patients in trial → **Charite**

---

## Reference Documents

| Document | Contents |
|----------|---------|
| `6G_architecture_overview.png` | Architecture diagram shown in meeting — use as reference for all conversations |
| `partner_meeting_prep.md` | Model performance numbers, full Q&A, FHIR output format |
| `sequence_diagram copy.md` | UML sequence diagrams with color-coded hosting groups |
| `fundamental_questions.md` | The 5 highest-priority questions for the next meeting |
