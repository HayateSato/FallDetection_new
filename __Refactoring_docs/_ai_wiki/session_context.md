# Claude Code Session Context
**Last updated:** 2026-06-09
**Project:** Fall Detection System — Charite/FOCUS Integration (active branch: `mcs-docker-version`)

Read this document before starting work. It captures the full current state so you can contribute without re-deriving what has already been decided.

---

## 1. What This System Is

A fall detection system for elderly patients wearing a **SmarKo wearable** (BLE sensor). The wearable collects ACC + barometer data. The mobile app reads it via Bluetooth, sends it to an ML inference server, and if a fall is detected, notifies the caregiver.

**This repo has three separate use cases in different branches/folders. The only active one right now is the FOCUS/Charite integration (branch `mcs-docker-version`).** The other two (`complete_system` branch, `_EcoSystem_Integration/` folder) are not in scope.

Branch `6G_intergation_with_MQTT` was initially built according to the original request from the tech partner (FOCUS), which contained 10 services in all one k3s stack to communicate with their existing k3s stack. However, due to their hardware constraints, 2 out of 10 services will be hosted as a new k3s stack and the other 8 services will be hosted on our Hetzner stack. `mcs-docker-version` is a branch to accommodate these changes and test the separated system.

---

## 2. The Three Stakeholders

| Person | Role | Responsibility |
|--------|------|---------------|
| **Hayate (you)** | MCS developer | Inference server, fall dashboard, ML pipeline, coordination |
| **Isa** | MCS developer | Mobile app (React Native) |
| **Mohammed** | MCS ops | Hetzner deployment, TLS setup, K3s chart delivery to FOCUS |

**FOCUS DevOps** (separate team) will host the caregiver layer after we wrap it in K3s for production. Mohammed's FOCUS counterpart handles that.

**FOCUS developer** will need to update the patient dashboard that is running in their existing k3s stack by integrating the fall_dashboard component — we will need to provide code and instructions.

---

## 3. System Architecture (Current — 2026-06-09)

### Local Testing — Two separate machines / Docker Compose stacks

```
[MCS Network — Laptop 1]                    [MCS Network — Laptop 2]
_6G_integration_v3_docker_mcs/               _6G_integration_v3_docker_focus/
  8 services:                                  2 services:
  - inference-server  :8001                    - mqtt (Mosquitto) :1883 / :9001
  - ml-dashboard      :8004                    - fall-dashboard   :8002
  - server-health     :8006
  - postgres (mcs_fall_postgres)
  - mlflow
  - minio
  - prometheus
  - grafana

[Cloud Network]
 - InfluxDB (MCS cloud, for mock_app testing only)
```

### Local Testing — K3s variant (also tested, now verified)

```
[MCS Network — Laptop 1 / K3s]              [MCS Network — Laptop 2 / Docker]
_6G_integration_v3_k3s/ (Helm chart)         _6G_integration_v3_docker_mcs/
  K3s pods:                                    8 services (same as above)
  - mosquitto pod  :1883 (internal)
                   NodePort 30901 (WS, LAN)
  - fall-dashboard :8002 (internal)
                   NodePort 30802 (LAN)
  - Traefik (pre-existing, handles ingress)

  mock-app runs separately as docker run on Laptop 2
```

### Real Deployment

```
[MCS / Hetzner]                    [FOCUS k3s cluster / FOCUS server]
_6G_integration_v3_docker_mcs/
  8 services:                        Existing pods (FOCUS-owned):
  - inference-server  :8001           - influxDB (FOCUS)
  - ml-dashboard      :8004           - patient-dashboard (Flutter)
  - server-health     :8006
  - postgres                         New pods added to same cluster (_6G_integration_v3_k3s):
  - mlflow                            - mqtt (Mosquitto) :1883 / :9001
  - minio                             - fall-dashboard   :8002
  - prometheus
  - grafana
```

**Key deployment decision (2026-06-08):** mqtt and fall-dashboard are deployed as **new pods inside FOCUS's existing k3s cluster** — not a separate cluster. Their existing Traefik picks up new IngressRoute rules automatically:
- `https://focus-dashboard/` → patient-dashboard pod (existing)
- `https://focus-dashboard/falls` → fall-dashboard pod (new)

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
8. [NOT YET — Isa] Mobile app writes fall_events point to FOCUS InfluxDB
9. [NOT YET — Isa] Mobile app calls POST /inference/{observation_id}/confirm on inference_server
```

---

## 4. MQTT — Critical Architecture Decision

**The mobile app is React Native. React Native cannot open raw TCP sockets in standard JS. Port 1883 TCP does not work. MQTT must go over WebSocket.**

Mosquitto runs TWO listeners:
- Port `1883` — plain TCP, **internal only** (fall_dashboard connects to broker inside the Docker/K8s network)
- Port `9001` — WebSocket, **external** (mobile app connects as `ws://<host>:9001`)

For production with TLS: port 9001 WebSocket is routed through Traefik HTTPS as WSS on port 443.

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

## 5. Test Status (updated 2026-06-09)

### Two-laptop Docker test — PASSED (2026-06-08)
- Laptop 1: `_6G_integration_v3_docker_mcs/` (inference layer)
- Laptop 2: `_6G_integration_v3_docker_focus/` (caregiver layer: mqtt + fall-dashboard)
- Isa's real mobile app on the same network
- Verified: `/predict`, MQTT WebSocket :9001, SSE, all three patient response paths

### K3s two-laptop test — PASSED (2026-06-09)
- Laptop 1: `_6G_integration_v3_k3s/` (K3s Helm chart, Docker Desktop Kubernetes + Traefik)
- Laptop 2: `_6G_integration_v3_docker_mcs/` (inference layer) + mock-app as standalone `docker run`
- Verified: `fall/possible/<pid>` pre-alert (pale-red card animation) + `fall/alert/<pid>` confirmed alert + dashboard animations — all working end-to-end
- Key fix for K3s test: mock-app must use `host.docker.internal:8001` (not `localhost`) for inference server, `MQTT_TRANSPORT=websockets`, port `30901` (NodePort), no `--network host`

### Docker MCS ↔ K3s caregiver cross-layer connectivity — PASSED (2026-06-09)
- server_health (Docker, Laptop 2) now probes all K3s services on Laptop 1
- fall-dashboard probe: `http://<laptop1-ip>:30802` via NodePort ✓
- mqtt_broker TCP probe: `<laptop1-ip>:30901` via WebSocket NodePort ✓
- Key fixes applied:
  1. `server-health` docker-compose was missing 5 env vars — added `DATABASE_URL`, `MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL`, `MQTT_BROKER_HOST`, `MQTT_BROKER_PORT`
  2. `FALL_DASHBOARD_URL` and `MQTT_BROKER_HOST` now set to Laptop 1 IP in `.env`
  3. `fall-dashboard` K3s Service was ClusterIP (NodePort never applied) — fixed by running `helm upgrade`
- `.env.example` updated: `FALL_DASHBOARD_URL=http://<LAPTOP1_IP>:30802`, `MQTT_BROKER_HOST=<LAPTOP1_IP>`, `MQTT_BROKER_PORT=30901`

### NOT working yet (open gap — needs Isa)
- Mobile app does NOT write `fall_events` to InfluxDB after confirmation
- Therefore: fall history tab on the dashboard is empty (no data to show)
- `/confirm` endpoint call from mobile app not yet implemented

---

## 6. Fall Dashboard Integration Strategy (decided 2026-06-08)

### Current approach: Q2 — deploy as new pod in FOCUS's existing k3s cluster

fall_dashboard and mqtt run as new pods inside FOCUS's existing k3s cluster. Traefik routes by path:

```
Caregiver opens: https://focus-dashboard/          → patient-dashboard (Flutter, existing)
Caregiver navigates to: https://focus-dashboard/falls  → fall-dashboard (our Python/HTML, new pod)
```

**UX limitation of Q2:** Real-time fall alerts only appear when the caregiver is on the `/falls` page.

**Privacy:** fall_dashboard runs inside FOCUS's network. MCS inference server sees only ACC arrays — never names.

### Future upgrade: Q3 — Isa integrates fall features into the Flutter source

**Prerequisite:** FOCUS grants source code access to the patient dashboard Flutter repo.
If granted, Isa modifies the Flutter app directly: adds fall history tab + real-time SSE alerts inline.
**Status:** Waiting for FOCUS confirmation.

---

## 7. Open Tasks (Priority Order)

### Blocking (needs Isa)
1. **[Isa] InfluxDB `fall_events` write** — after confirmation popup, mobile app must write one point to FOCUS InfluxDB. Schema documented in `__Refactoring_docs/influxdb_schema.md`.
   - Tags: `patient_id` (str), `device_id` (str/MAC)
   - Fields: `fall_detected` (bool, always true), `patient_confirmed` (**int**: 1/0/-1), `needs_help` (bool), `observation_id` (str UUID), `confidence` (float), `model_version` (str)
   - Timestamp: detection time (from `/predict` response), NOT current time
   - **Important:** `patient_confirmed` is **int** in InfluxDB, **string** in MQTT — different encodings

2. **[Isa] POST /inference/{observation_id}/confirm** — call after confirmation popup.
   - Endpoint: `POST https://<inference-server>/inference/{observation_id}/confirm`
   - Payload: `{"patient_confirmed": "yes", "needs_help": true}`
   - Same `X-API-Key` header as `/predict`

### Blocking (needs Mohammed)
3. **[Mohammed] Hetzner deployment** — deploy `_6G_integration_v3_docker_mcs/` on Hetzner with TLS. Inference server needs a public HTTPS endpoint so the mobile app can reach it.
   - Config file: `_6G_integration_v3_docker_mcs/.env` (copy from `.env.example`, fill in CHANGE_ME values)
   - Reference: `FOCUS_devs_handover/production_config_checklist.md` Part 1

4. **[Mohammed] K3s chart delivery to FOCUS DevOps** — deliver `_6G_integration_v3_k3s/` Helm chart. FOCUS DevOps installs it into their existing k3s cluster.
   - Production config: `_6G_integration_v3_k3s/helm/values_production.yaml` — fill in all `CHANGE_ME` values
   - Reference: `FOCUS_devs_handover/production_config_checklist.md` Part 2

### Not blocking but open
5. **Ask FOCUS for patient dashboard source repo access** — prerequisite for Q3 upgrade.
6. **Auth gate** — ml_dashboard and server_health have no authentication yet (deferred).

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

Auth: `X-API-Key` header on all endpoints (value from `.env` `API_KEYS`).

---

## 9. Database Schema (MCS Postgres — `mcs_fall_postgres`)

Two logical databases:
- `fall_detection` — our application data
- `mlflow` — MLflow internal tables

Key tables in `fall_detection`:

| Table | Written by | Contents |
|-------|-----------|----------|
| `inference_log` | inference_server | Every /predict call: patient_id, model_version, fall_detected, confidence, latency_ms, detection_time, observation_id, patient_confirmed, needs_help |
| `feature_snapshot` | inference_server | ACC feature values per inference (for retraining) |

`fall_history` table was **removed** in migration 0003. Labels are now directly on `inference_log`.

Alembic migration chain: `0001 → 0002 → 0003 → 0004 → 0005`
Migrations run automatically via the `db-migrate` service on `docker compose up`.

**Caregiver layer (FOCUS side) has NO Postgres:**
- Patient list: SQLite (`fall_dashboard/patient_store`, persisted in a named Docker volume)
- Fall history: queried from FOCUS InfluxDB (`fall_events` measurement)

---

## 10. InfluxDB — Two Instances, Completely Different Roles

| Instance | Hosted by | Role | Used by |
|----------|-----------|------|---------|
| **FOCUS-hosted InfluxDB** | FOCUS (their existing infra) | Stores biosignals + `fall_events` points | Mobile app (writes fall_events), fall_dashboard (reads fall history), Flutter Patient Dashboard (reads biosignals) |
| **MCS cloud InfluxDB** (`ecosystem-influxdb.smarko-health.de`) | MCS/SmarKo | Dev testing only — `fd_test` bucket used by mock_app as fake BLE input | mock_app only. Completely absent in production. |

**Our inference pipeline never reads from any InfluxDB at runtime.**

**MCS `.env` does NOT need InfluxDB credentials for Hetzner production** — no service in `_6G_integration_v3_docker_mcs/docker-compose.yml` uses InfluxDB. The InfluxDB creds in `.env.example` were leftover from when mock_app was part of that stack. They are now commented out.

---

## 11. Active Codebase Location

```
C:\Users\hayat\Documents\6G\FallDetection_new\
  _6G_integration_v3_docker_mcs/       MCS inference layer (ACTIVE, Hetzner target)
    .env.example                        Reference config — copy to .env, fill in CHANGE_ME
  _6G_integration_v3_docker_focus/     FOCUS caregiver layer, Docker version (ACTIVE)
    mock_app/influx_writer.py           Writes fall_events to InfluxDB (fields: observation_id + model_version added 2026-06-09)
    fall_dashboard/mqtt_listener.py     5-retry startup loop (fixed 2026-06-09)
  _6G_integration_v3_k3s/              FOCUS caregiver layer, K3s Helm chart (ACTIVE, tested 2026-06-09)
    helm/values.yaml                    Local testing values (NodePorts 30901/30802, local image)
    helm/values_production.yaml         Production values template — fill in all CHANGE_ME values
    fall_dashboard/mqtt_listener.py     5-retry startup loop (same fix as docker_focus)
    fall_dashboard/patient_store.py     SQLite CRUD — upsert_patient() + delete_patient() (added 2026-06-09)
    fall_dashboard/web.py               POST /api/patients + DELETE /api/patients/{id} (added 2026-06-09)
    fall_dashboard/dashboard/           UI: "+ Add Patient" button + modal + per-card delete (added 2026-06-09)
  _6G_Integration_v2_mqtt/             Previous version (reference only, frozen)
  __Refactoring_docs/
    TODO.md                             Main task tracker
    deployment_architecture.md          Full architecture diagram + data sources
    influxdb_schema.md                  NEW (2026-06-09): fall_events measurement schema, Flux queries, visualization tips
    port_management.md                  NEW (2026-06-09): Port mapping across all 4 environments with diagrams
    service_test.md                     How to run service tests (Docker + K3s variants)
    k3s_two_laptop_test.md              Step-by-step K3s two-laptop test guide (updated 2026-06-09)
    mobile_app_publisher_contract_ISA.md  Spec for Isa: MQTT topics, payloads, InfluxDB schema
    mqtt_broker_logs_and_qos.md         How to read mosquitto logs, QoS explanation
    react_mqtt.md                       Why WebSocket (port 9001) not TCP (port 1883)
    network_config.md                   Windows firewall rules for cross-machine testing
    _ai_wiki/                           This folder
  FOCUS_devs_handover/                  NEW (2026-06-09): Handover documents for Mohammed + FOCUS DevOps
    production_config_checklist.md      All values that change from local testing to production, split by owner
```

Current git branch: `mcs-docker-version`

---

## 12. Key Known Gotchas

- **MQTT host must be `127.0.0.1` not `localhost`** in `.env` on Windows — Windows resolves `localhost` to `::1` (IPv6) but port-forward binds IPv4 only.
- **Port 9001 WebSocket** is the only port the mobile app can use. Port 1883 is internal Docker/K8s network only (fall_dashboard → broker). Do NOT expose 1883 to the mobile app.
- **MinIO console runs on port 9002** (not 9001) — 9001 is taken by the MQTT WebSocket listener.
- **`patient_confirmed` encoding asymmetry**: string on MQTT (`"yes"`/`"no"`/`"not_answered"`), int in InfluxDB (`1`/`0`/`-1`). The fall_dashboard does the conversion — the mobile app must send string on MQTT and int to InfluxDB.
- **Caregiver alert only fires for `not_answered` OR (`yes` AND `needs_help=true`)** — confirmed falls where the patient says they're OK produce no caregiver alert (stored to history only).
- **`fall_dashboard` has no Postgres** — any old code referencing SQLAlchemy/psycopg2/Alembic in that component is outdated. Patient data = SQLite, fall data = InfluxDB.
- **PowerShell 5.1 cannot parse non-ASCII characters** (em-dash, smart quotes) in `.ps1` files saved as UTF-8 without BOM — use plain ASCII in `.ps1` files only.
- **K3s mock-app must NOT use `--network host`** — Docker Desktop on Windows runs containers in a Linux VM; `--network host` binds to the VM NIC, not Windows. Use `host.docker.internal` to reach Windows host services and `-p 8005:8005` for port mapping.
- **K3s fall-dashboard MQTT startup race**: fall-dashboard pod may start before mosquitto is ready. Code retries 5 times with 5s delay (25s total). If it still fails, `kubectl delete pod` to restart — K8s recreates it and mosquitto will be ready by then.
- **Docker Desktop K8s does NOT include Traefik** — unlike real k3s, Traefik must be installed separately (`helm install traefik traefik/traefik --namespace kube-system`). The K3s Helm chart creates `IngressRoute` CRDs which require Traefik.
- **MCS `.env` does NOT need InfluxDB credentials** — no service in `_6G_integration_v3_docker_mcs/` uses InfluxDB. Those vars in `.env.example` are now commented out with an explanation.
- **`imagePullPolicy: IfNotPresent` required for local K3s testing** — without it, K8s tries to pull from `registry-smarko-health.de` and fails. Set to `Always` in `values_production.yaml` only.
- **`imagePullSecrets` must be conditional** — a blank `imagePullSecret: ""` in values.yaml must NOT produce an `imagePullSecrets: [{name: ""}]` block in the deployment (K8s rejects it). The template wraps it in `{{- if .Values.imagePullSecret }}`.
- **`helm upgrade` required after every values.yaml change** — Kubernetes does not watch the file. A Service that was installed as ClusterIP stays ClusterIP until `helm upgrade` is run, even if the template now says NodePort. Rule: after any edit to `values.yaml` or a Helm template, re-run `helm upgrade caregiver helm --namespace fall-dashboard --values helm/values.yaml` (or just run `install.ps1` which is idempotent).
- **`server-health` docker-compose needs 5 env vars explicitly** — `DATABASE_URL`, `MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL`, `MQTT_BROKER_HOST`, `MQTT_BROKER_PORT`. Without them, all five probes fall back to `localhost:...` defaults which fail inside the container. Only `INFERENCE_SERVER_URL` and `FALL_DASHBOARD_URL` were originally set.
- **`FALL_DASHBOARD_URL` and `MQTT_BROKER_HOST` in Laptop 2 `.env`** must point to Laptop 1's LAN IP. Ports: `30802` (fall-dashboard NodePort), `30901` (MQTT WebSocket NodePort). Port 1883 is K3s-internal only and not reachable from Laptop 2.
- **Patient add/delete is now dynamic from the UI** — `POST /api/patients` and `DELETE /api/patients/{id}` endpoints added to fall_dashboard. No longer need to edit `values.yaml` / `PATIENT_IDS` env var and restart the pod to add patients. The SQLite store on the PVC persists across restarts.

---

## 13. Production Config — Who Sets What

Three parties are involved. See `FOCUS_devs_handover/production_config_checklist.md` for full details.

**Values that must be exchanged between parties:**

| What | From | To | Where |
|---|---|---|---|
| Inference server URL | Mohammed (Hetzner domain) | Isa | Mobile app config |
| `API_KEYS` value | Mohammed | Isa | Mobile app `X-API-Key` header |
| MQTT broker domain (`mosquitto.ingress.host`) | FOCUS DevOps | Isa | Mobile app MQTT connection |
| MQTT username + password | FOCUS DevOps | Isa | Mobile app MQTT auth |
| FOCUS InfluxDB credentials | FOCUS DevOps | Isa | Mobile app InfluxDB write |
| Fall-dashboard URL (`fallDashboard.ingress.host`) | FOCUS DevOps | Mohammed | `FALL_DASHBOARD_URL` in MCS `.env` |

**MQTT auth warning:** `fallDashboard.mqtt.username/password` in K3s values.yaml and Isa's mobile app must be **identical**. A mismatch causes silent connection failure — the mobile app cannot publish alerts.

---

## 14. Retraining Pipeline (Background — Not Immediate Priority)

Data source: **MCS Postgres only** (inference_log + feature_snapshot). InfluxDB is NOT in the retraining loop.

```sql
SELECT il.*, fs.feature_name, fs.feature_value
FROM inference_log il
JOIN feature_snapshot fs ON fs.inference_id = il.id
WHERE il.fall_detected = TRUE AND il.patient_confirmed = 'yes'
```

Trigger: technical user (MCS) clicks "Retrain" in ml_dashboard → runs `python -m retrain.retrain` → MLflow logs run → model registered → hot-swap via `/model/switch`.

Models: XGBoost. Best model is v3 (ACC + barometer). Window: 9s @ 50Hz = 450 samples.
