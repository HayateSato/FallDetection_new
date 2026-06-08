# Claude Code Session Context
**Last updated:** 2026-06-08
**Project:** Fall Detection System — Charite/FOCUS Integration (active branch: `mcs-docker-version`)

Read this document before starting work. It captures the full current state so you can contribute without re-deriving what has already been decided.

---

## 1. What This System Is

A fall detection system for elderly patients wearing a **SmarKo wearable** (BLE sensor). The wearable collects ACC + barometer data. The mobile app reads it via Bluetooth, sends it to an ML inference server, and if a fall is detected, notifies the caregiver.

**This repo has three separate use cases in different branches/folders. The only active one right now is the FOCUS/Charite integration (branch `mcs-docker-version`).** The other two (`complete_system` branch, `_EcoSystem_Integration/` folder) are not in scope.

branch `6G_intergation_with_MQTT` was initially built according to the original request from the tech partner (FOCUS), which contained 10 services in all one k3s stack to communicate with their exsiting k3s stack. However, due to their hardware constraints, 2 out of 10 services will be hosted as a new k3s stack and the other 8 services will be hosted on our Hetzner stack. `mcs-docker-version` is a branch to acoomodate these changes and test the separarted system.

---

## 2. The Three Stakeholders

| Person | Role | Responsibility |
|--------|------|---------------|
| **Hayate (you)** | MCS developer | Inference server, fall dashboard, ML pipeline, coordination |
| **Isa** | MCS developer | Mobile app (React Native)  |
| **Mohammed** | MCS ops | Hetzner deployment, TLS setup |

**FOCUS DevOps** (separate team) will host the caregiver layer after we wrap it K3s for production. Mohammed's FOCUS counterpart handles that.

**FOCUS developer** will need to update the patient dashabord that is running in their exsiting k3s stack by integrating fall_dashboard component - we will need to provide code and instructions.

---

## 3. System Architecture (Current — 2026-06-08)


### Local Testing in Two separate machines / Docker Compose stacks

```
[MCS Network — Laptop 1]                    [MCS Network — Laptop 2]                    
_6G_integration_v3_docker_mcs/               _6G_integration_v3_docker_focus/
  8 services:                                  2 services:
  - inference-server  :8001                    - mqtt (Mosquitto) :1883 / :9001
  - ml-dashboard      :8004                    - fall-dashboard   :8002 (python version)
  - server-health     :8006
  - postgres (mcs_fall_postgres)
  - mlflow
  - minio
  - prometheus
  - grafana


[Cloud Network]
 - InfluxDB (of MCS)
```


### Real deployment k3s stacks

```
[MCS Network / Hetzner]                    [FOCUS k3s stack / FOCUS server]
_6G_integration_v3_docker_mcs/               
  8 services:                                  Existing pods (FOCUS-owned):
  - inference-server  :8001                     - influx DB (of FOCUS)
  - ml-dashboard      :8004                     - patient-dashboard (Flutter)
  - server-health     :8006                    
  - postgres (mcs_fall_postgres)               New pods added to same cluster (_6G_integration_v3_k3s):
  - mlflow                                      - mqtt (Mosquitto) :1883 / :9001
  - minio                                       - fall-dashboard   :8002  
  - prometheus                                  
  - grafana                                    
```

**Key deployment decision (2026-06-08):** mqtt and fall-dashboard are deployed as **new pods inside FOCUS's existing k3s cluster** — not a separate cluster. Their existing Traefik picks up new IngressRoute rules automatically:
- `https://focus-dashboard/` → patient-dashboard pod (existing)
- `https://focus-dashboard/falls` → fall-dashboard pod (new)

This avoids cross-cluster networking complexity. FOCUS DevOps only manages one cluster. See §6 for the integration strategy and future upgrade path.

**External to both stacks:**
- SmarKo wearable (BLE, physical device)
- Mobile app (React Native, Isa's app — runs on patient's phone)
- FOCUS InfluxDB (already running in FOCUS infra — stores biosignals + fall events)
- FOCUS FHIR server (opted out — not used)

### How a fall flows through the system

```
1. SmarKo wearable --BLE--> Mobile app (collects 9s ACC window at 50Hz = 450 samples)
2. Mobile app --HTTPS POST /predict--> inference_server :8001 (MCS)
3. inference_server responds: { fall_detected: true, observation_id: <UUID>, confidence: 0.998 }
4. Mobile app publishes MQTT to broker: fall/possible/<patient_id>  (immediate pre-alert)
5. Mobile app shows popup to patient: "Did you fall? Do you need help?" (10s timeout)
6. Mobile app publishes MQTT to broker: fall/alert/<patient_id>  (with patient response)
7. fall_dashboard receives alert via MQTT -> SSE fan-out to caregiver browser
8. [NOT YET] Mobile app writes fall_events point to FOCUS InfluxDB
9. [NOT YET] Mobile app calls POST /inference/{observation_id}/confirm on inference_server
```

---

## 4. MQTT — Critical Architecture Decision

**The mobile app is React Native. React Native cannot open raw TCP sockets in standard JS. Port 1883 TCP does not work. MQTT must go over WebSocket.**

Mosquitto runs TWO listeners:
- Port `1883` — plain TCP, **internal only** (fall_dashboard connects to broker inside the Docker network)
- Port `9001` — WebSocket, **external** (mobile app connects as `ws://<host>:9001`)

For production with TLS: port 9001 WebSocket is routed through Traefik HTTPS as WSS on port 443.

Mobile app library: **MQTT.js** (not react-native-mqtt, which would need native build config).

```js
// Mobile app connection (LAN/local)
mqtt.connect('ws://<caregiver-machine-ip>:9001')

// Production (with TLS at Traefik)
mqtt.connect('wss://focus-server.hospital.de')
```

### MQTT topics

| Topic | Published by | When | Dashboard reaction |
|-------|-------------|------|--------------------|
| `fall/possible/<patient_id>` | Mobile app | Immediately on fall detection | Card turns pale red, shows "Possible fall" notice |
| `fall/alert/<patient_id>` | Mobile app | After patient answers popup (or 10s timeout) | Full caregiver alert if `patient_confirmed=not_answered` or `needs_help=true` |

### MQTT payload (fall/alert) — must match exactly

```json
{
  "patient_id":        "patient_test_4",
  "observation_id":    "<UUID from /predict response>",
  "fall_detected":     true,
  "patient_confirmed": "yes",
  "needs_help":        true,
  "confidence":        0.998,
  "alert_time":        "2026-06-08T10:36:50.840Z",
  "model_version":     "v1.3"
}
```

`patient_confirmed` is a **string** (`"yes"` / `"no"` / `"not_answered"`) on MQTT. The fall_dashboard converts it to int (1/0/-1) internally for InfluxDB and SSE.

---

## 5. What Was Just Tested (2026-06-08)

**Two-laptop + real mobile app test PASSED.**

- Laptop 1: `_6G_integration_v3_docker_mcs/` (inference layer, no mock app)
- Laptop 2: `_6G_integration_v3_docker_focus/` (caregiver layer: mqtt + fall-dashboard)
- Isa's real mobile app on the same network

Verified working:
- Mobile app → `/predict` → fall_detected response
- Mobile app → MQTT WebSocket :9001 → broker → fall_dashboard → SSE → caregiver UI
- Patient popup all three response paths (yes+help, yes+no-help, timeout)

**NOT working yet (open gap):**
- Mobile app does NOT write to InfluxDB after confirmation
- Therefore: fall history tab on the dashboard is empty (no data to show)
- `/confirm` endpoint call from mobile app not yet tested

---

## 6. Fall Dashboard Integration Strategy (decided 2026-06-08)

The caregiver needs to see patient info (from the existing FOCUS patient dashboard) and fall alerts + fall history (from fall_dashboard) in one place — without opening two separate browser tabs.

### Current approach: Q2 — deploy as new pod in FOCUS's existing k3s cluster

fall_dashboard and mqtt run as new pods inside FOCUS's existing k3s cluster. Their Traefik routes by path:

```
Caregiver opens: https://focus-dashboard/          → patient-dashboard (Flutter, existing)
Caregiver navigates to: https://focus-dashboard/falls  → fall-dashboard (our Python/HTML, new pod)
```

**What we deliver to FOCUS DevOps:**
- Docker image for fall_dashboard (pushed to their registry `registry-smarko-health.de`)
- k8s manifests from `_6G_integration_v3_k3s/`: Deployment + Service + IngressRoute for both mqtt and fall-dashboard
- `.env` variable list documented

**UX limitation of Q2:** Real-time fall alerts only appear when the caregiver is on the `/falls` page. The alert cannot show as a badge on the patient list in the Flutter app because the two UIs are separate pages with different frontends. The caregiver must actively navigate to see alerts.

**Privacy is satisfied:** fall_dashboard runs inside FOCUS's network. It is the only component that holds patient names linked to fall events. MCS's inference server sees only ACC arrays — never names.

---

### Future upgrade: Q3 — Isa integrates fall features directly into the Flutter source

**Prerequisite:** FOCUS grants source code access to the patient dashboard Flutter repo.

If source access is granted, Isa (MCS, knows Flutter) modifies the Flutter app directly:
- Adds a "Fall History" tab — Flutter calls `GET /api/falls` on fall_dashboard
- Adds a real-time alert badge on patient cards — Flutter subscribes to `GET /api/stream` SSE
- Single unified app, consistent UI, alerts appear inline without navigating away

When Q3 is implemented, Q2's path routing (`/falls`) becomes optional (kept for debugging) and the caregiver never needs to leave the Flutter patient dashboard.

**What fall_dashboard becomes in Q3:** Backend-only API service. The HTML/JS frontend it currently serves is no longer used by caregivers.

**Status:** Waiting for FOCUS to confirm whether they will grant source repo access.

---

## 7. Open Tasks (Priority Order)

### Blocking (needs Isa)
1. **[Isa] InfluxDB `fall_events` write** — after the patient confirmation popup, the mobile app must write one point to FOCUS InfluxDB. Without this, the fall dashboard's fall history tab has no data.
   - Measurement: `fall_events`
   - Tags: `patient_id`, `device_id`
   - Fields: `fall_detected` (bool), `patient_confirmed` (**int**: 1=yes / 0=no / -1=not_answered), `needs_help` (bool), `observation_id` (str UUID), `confidence` (float), `model_version` (str)
   - Timestamp: detection time
   - Note: `patient_confirmed` is **int** in InfluxDB, **string** in the MQTT payload — different encodings, same concept.

2. **[Isa] POST /inference/{observation_id}/confirm** — call this after the confirmation popup so the retraining pipeline gets the ground-truth label.
   - Endpoint: `POST https://<inference-server>/inference/{observation_id}/confirm`
   - Payload: `{"patient_confirmed": "yes", "needs_help": true}`
   - Same `X-API-Key` header as `/predict`

### Blocking (needs Mohammed)
3. **[Mohammed] Hetzner deployment** — deploy `_6G_integration_v3_docker_mcs/` on Hetzner with TLS (Nginx + Certbot / Let's Encrypt). Inference server needs a public HTTPS endpoint so the mobile app can reach it.

### Not blocking but open
4. **K3s manifests for FOCUS** — package mqtt + fall-dashboard from `_6G_integration_v3_k3s/` as k8s manifests (Deployment + Service + IngressRoute). Deliver to FOCUS DevOps for deployment into their existing cluster. See §6 above and `__Refactoring_docs/TODO.md` Section 8.
5. **Ask FOCUS for patient dashboard source repo access** — prerequisite for Q3 upgrade. Until confirmed, Q2 is the plan.
6. **FOCUS instruction document** — spec for FOCUS DevOps: MQTT broker setup, both topic subscriptions, InfluxDB Flux query examples. Template format: `handover_docs_2/`.
7. **Auth gate** — ml_dashboard and server_health have no authentication yet (deferred). See TODO Section 4.

---

## 8. Key Inference Server Endpoints

Base URL (local dev): `http://localhost:8001`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Main inference. Body: `{patient_id, model_version, acc_data: [[x,y,z]...]}` (450 rows). Returns: `{fall_detected, confidence, observation_id}` |
| `/inference/{observation_id}/confirm` | POST | Records patient answer. Body: `{patient_confirmed: "yes"/"no"/"not_answered", needs_help: bool}` |
| `/model/switch` | POST | Hot-swap model. Body: `{mlflow_stage: "Production"}` |
| `/model/list` | GET | Lists available models |
| `/health` | GET | Health probe |
| `/metrics` | GET | Prometheus metrics |

Auth: `X-API-Key` header on all endpoints (value from `.env` `API_KEY`).

---

## 9. Database Schema (MCS Postgres — `mcs_fall_postgres`)

Two logical databases:
- `fall_detection` — our application data
- `mlflow` — MLflow internal tables

Key tables in `fall_detection`:

| Table | Written by | Contents |
|-------|-----------|----------|
| `inference_log` | inference_server | Every /predict call: patient_id, model_version, fall_detected, confidence, latency_ms, detection_time, observation_id, patient_confirmed, needs_help |
| `feature_snapshot` | inference_server | 450 ACC feature values per inference (for retraining) |

`fall_history` table was **removed** in migration 0003. Labels are now directly on `inference_log`.

Alembic migration chain: `0001 → 0002 → 0003 → 0004 → 0005`
Migrations run automatically via the `db-migrate` service in `inference_posttraining_layer/docker-compose.yml`.

**Caregiver layer (FOCUS side) has NO Postgres:**
- Patient list: SQLite (`fall_dashboard/patient_store`, persisted in a named Docker volume)
- Fall history: queried from FOCUS InfluxDB (`fall_events` measurement)

---

## 10. InfluxDB — Two Instances, Completely Different Roles

| Instance | Hosted by | Role | Used by |
|----------|-----------|------|---------|
| **FOCUS-hosted InfluxDB** | FOCUS (their existing infra) | Stores biosignals written by real mobile app; stores `fall_events` points | Mobile app (writes), fall_dashboard (reads fall history), FOCUS Flutter Patient Dashboard (reads biosignals) |
| **MCS cloud InfluxDB** (`ecosystem-influxdb.smarko-health.de`) | MCS/SmarKo | Dev testing only — `fd_test` bucket used by mock_app as fake BLE input | mock_app only. Completely absent in production. |

**Our inference pipeline never reads from any InfluxDB at runtime.** In production: mobile app reads BLE → POSTs directly to `/predict`. InfluxDB is not in the inference path.

---

## 11. Active Codebase Location

```
C:\Users\hayat\Documents\6G\FallDetection_new\
  _6G_integration_v3_docker_mcs/     ← MCS inference layer (ACTIVE, Hetzner target)
  _6G_integration_v3_docker_focus/   ← FOCUS caregiver layer (ACTIVE, FOCUS server target)
  _6G_integration_v3_k3s             ← FOCUS caregiver layer (next step, FOCUS server target, k3s version of above)
  _6G_Integration_v2_mqtt/           ← previous version (reference only, frozen)
  __Refactoring_docs/                ← planning docs, architecture, TODOs
    TODO.md                          ← main task tracker
    deployment_architecture.md       ← full architecture diagram + data sources
    mobile_app_publisher_contract_ISA.md  ← spec for Isa: MQTT topics, payloads, InfluxDB schema
    mqtt_broker_logs_and_qos.md      ← how to read mosquitto logs, QoS explanation
    react_mqtt.md                    ← why WebSocket (port 9001) not TCP (port 1883)
    network_config.md                ← Windows firewall rules for cross-machine testing
    _ai_wiki/                        ← this folder
```

Current git branch: `mcs-docker-version`

---

## 12. Key Known Gotchas

- **MQTT host must be `127.0.0.1` not `localhost`** in `.env` on Windows — Windows resolves `localhost` to `::1` (IPv6) but port-forward binds IPv4 only.
- **Port 9001 WebSocket** is the only port the mobile app can use. Port 1883 is internal Docker network only (fall_dashboard → broker). Do NOT expose 1883 to the mobile app.
- **MinIO console runs on port 9002** (not 9001) — 9001 is taken by the MQTT WebSocket listener.
- **`patient_confirmed` encoding asymmetry**: string on MQTT (`"yes"`/`"no"`/`"not_answered"`), int in InfluxDB (`1`/`0`/`-1`). The fall_dashboard does the conversion — the mobile app must send a string on MQTT and an int to InfluxDB.
- **Caregiver alert only fires for `not_answered` OR (`yes` AND `needs_help=true`)** — confirmed falls where the patient says they're OK produce no caregiver alert (stored to history only).
- **Docker Compose in inference layer uses `--env-file .env`** — the compose file is in a subdirectory, so the flag is required.
- **`fall_dashboard` has no Postgres** — any old code referencing SQLAlchemy/psycopg2/Alembic in that component is outdated. Patient data = SQLite, fall data = InfluxDB.
- **PowerShell 5.1 cannot parse non-ASCII characters** (em-dash, smart quotes) in `.ps1` files saved as UTF-8 without BOM — use plain ASCII in `.ps1` files only.

---

## 13. Retraining Pipeline (Background — Not Immediate Priority)

Data source: **MCS Postgres only** (inference_log + feature_snapshot). InfluxDB is NOT in the retraining loop.

```sql
SELECT il.*, fs.feature_name, fs.feature_value
FROM inference_log il
JOIN feature_snapshot fs ON fs.inference_id = il.id
WHERE il.fall_detected = TRUE AND il.patient_confirmed = 'yes'
```

Trigger: technical user (MCS) clicks "Retrain" in ml_dashboard → runs `python -m retrain.retrain` → MLflow logs run → model registered → hot-swap via `/model/switch`.

Models: XGBoost. Best model is v3 (ACC + barometer). Window: 9s @ 50Hz = 450 samples.
