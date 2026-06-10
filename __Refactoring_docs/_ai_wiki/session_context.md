# Claude Code Session Context
**Last updated:** 2026-06-10
**Project:** Fall Detection System — Charite/FOCUS Integration (active branch: `mcs-docker-deployment`)

Read this document before starting work. It captures the full current state so you can contribute without re-deriving what has already been decided.

---

## 1. What This System Is

A fall detection system for elderly patients wearing a **SmarKo wearable** (BLE sensor). The wearable collects ACC + barometer data. The mobile app reads it via Bluetooth, sends it to an ML inference server, and if a fall is detected, notifies the caregiver.

**This repo has three separate use cases in different branches/folders. The only active one right now is the FOCUS/Charite integration (branch `mcs-docker-deployment`).** The other two (`complete_system` branch, `_EcoSystem_Integration/` folder) are not in scope.

The system was originally designed as one combined 10-service k3s stack (branch `6G_intergation_with_MQTT`). Due to FOCUS hardware constraints (≤15 GB RAM), it was split: 2 services run in FOCUS's existing k3s cluster, 8 services run on MCS's Hetzner Docker stack.

---

## 2. The Three Stakeholders

| Person | Role | Responsibility |
|--------|------|---------------|
| **Hayate (you)** | MCS developer | Inference server, fall dashboard, ML pipeline, coordination |
| **Isa** | MCS developer | Mobile app (React Native) |
| **Mohammed** | MCS ops | Hetzner deployment, TLS setup, K3s chart delivery to FOCUS |

**FOCUS DevOps** (separate team) deploys the caregiver layer (2 pods) into their existing k3s cluster using the Helm chart we deliver.

---

## 3. System Architecture (Current — 2026-06-10)

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
```

### Local Testing — K3s variant (verified 2026-06-09)

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

**Key deployment decision (2026-06-08):** mqtt and fall-dashboard are deployed as **new pods inside FOCUS's existing k3s cluster** — not a separate cluster. Traefik routes by domain (not path):
- `https://fall.focus-hospital.de` → fall-dashboard pod (new)
- `https://mqtt.focus-hospital.de` → mosquitto pod (new, WSS)

### How a fall flows through the system

```
1. SmarKo wearable --BLE--> Mobile app (collects 9s ACC window at 50Hz = 450 samples)
2. Mobile app writes raw sensor data to FOCUS InfluxDB (SMART_DATA measurement)
3. Mobile app --HTTPS POST /predict--> inference_server :8001 (MCS)
4. inference_server responds: { fall_detected: true, observation_id: <UUID>, confidence: 0.998 }
5. Mobile app publishes MQTT to broker: fall/possible/<patient_id>  (immediate pre-alert)
6. Mobile app shows popup to patient: "Did you fall?" (10s) → "Do you need help?" (10s)
7. Mobile app publishes MQTT to broker: fall/alert/<patient_id>  (with patient response)
8. fall_dashboard receives alert via MQTT -> SSE fan-out to caregiver browser
9. [NOT YET — Isa] Mobile app writes fall_events point to FOCUS InfluxDB
10. [NOT YET — Isa] Mobile app calls POST /inference/{observation_id}/confirm on inference_server
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
mqtt.connect('wss://mqtt.focus-hospital.de')
```

### MQTT topics

| Topic | Published by | When | Dashboard reaction |
|-------|-------------|------|--------------------|
| `fall/possible/<patient_id>` | Mobile app | Immediately on fall detection | Card turns pale red, shows "Possible fall" notice |
| `fall/alert/<patient_id>` | Mobile app | After patient answers popup (or 10s timeout) | Full caregiver alert |

### MQTT payload (fall/alert) — must match exactly

```json
{
  "patient_id":        "patient_test_4",
  "observation_id":    "<UUID from /predict response>",
  "fall_detected":     true,
  "patient_confirmed": "yes",
  "needs_help":        true,
  "confidence":        0.998,
  "timestamp":         "<detection ISO 8601>",
  "alert_time":        "<confirmation ISO 8601>",
  "model_version":     "v0"
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

### Docker MCS ↔ K3s caregiver cross-layer connectivity — PASSED (2026-06-09)
- server_health (Docker, Laptop 2) probes all K3s services on Laptop 1
- fall-dashboard probe: `http://<laptop1-ip>:30802` ✓
- mqtt_broker TCP probe: `<laptop1-ip>:30901` ✓

### NOT working yet (open gap — needs Isa)
- Mobile app does NOT write `fall_events` to InfluxDB after confirmation
- Mobile app does NOT call `/confirm` endpoint
- Fall history tab on dashboard is therefore empty

---

## 6. Fall Dashboard Integration Strategy (decided 2026-06-08)

fall_dashboard and mqtt run as new pods inside FOCUS's existing k3s cluster. Each gets its own subdomain via Traefik:
- `https://fall.focus-hospital.de` → fall-dashboard
- `https://mqtt.focus-hospital.de` → mosquitto (WSS)

**Future upgrade (Q3):** If FOCUS grants source repo access, Isa integrates fall features directly into the Flutter patient dashboard. Status: waiting for FOCUS confirmation.

---

## 7. Open Tasks (Priority Order)

### Blocking (needs Isa)
1. **[Isa] InfluxDB `fall_events` write** — after confirmation popup, mobile app must write one point to FOCUS InfluxDB. Full spec in `isa/todos_isa.md`.
   - Tags: `patient_id` (str), `device_id` (str/MAC)
   - Fields: `fall_detected` (bool), `patient_confirmed` (**int**: 1/0/-1), `needs_help` (bool), `observation_id` (str UUID), `confidence` (float), `model_version` (str)
   - Timestamp: detection time (from `/predict` response), NOT current time
   - Also write `SMART_DATA` measurement continuously (biosignal feed for patient dashboard)

2. **[Isa] POST /inference/{observation_id}/confirm** — call after confirmation popup.
   - Payload: `{"patient_confirmed": "yes"/"no"/"not_answered", "needs_help": bool}`
   - Same `X-API-Key` header as `/predict`

### Blocking (needs Mohammed)
3. **[Mohammed] Hetzner deployment** — deploy `_6G_integration_v3_docker_mcs/` on Hetzner with TLS. Full steps in `mohammed/mohammed_handover.md`.

4. **[Mohammed] K3s chart delivery to FOCUS DevOps** — deliver `_6G_integration_v3_k3s/` + filled `values_production.yaml`.

### Not blocking but open
5. **Ask FOCUS for patient dashboard source repo access** — prerequisite for Q3 upgrade.
6. **Auth gate** — ml_dashboard and server_health have no authentication yet (deferred).

---

## 8. Key Inference Server Endpoints

Base URL (local dev): `http://localhost:8001`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Main inference. Body: `{patient_id, device_id, acc_x[], acc_y[], acc_z[], timestamps_ms[], pressure[] (opt)}`. Returns: `{inference: {fall_detected, confidence, model_version}, observation_id, timestamp}` |
| `/inference/{observation_id}/confirm` | POST | Records patient answer. Body: `{patient_confirmed: "yes"/"no"/"not_answered", needs_help: bool}` |
| `/model/info` | GET | Returns `{uses_barometer: bool, model_version: ...}` — mobile app queries this once on startup |
| `/model/switch` | POST | Hot-swap model. Body: `{mlflow_stage: "Production"}` |
| `/model/list` | GET | Lists available models |
| `/health` | GET | Health probe |
| `/metrics` | GET | Prometheus metrics |

Auth: `X-API-Key` header on all endpoints (value from `.env` `API_KEYS`).

---

## 9. Database Schema (MCS Postgres — `mcs_fall_postgres`)

Two logical databases: `fall_detection` (app data) and `mlflow` (MLflow internals).

| Table | Written by | Contents |
|-------|-----------|----------|
| `inference_log` | inference_server | Every /predict call: patient_id, model_version, fall_detected, confidence, latency_ms, detection_time, observation_id, patient_confirmed (1/0/-1/NULL), needs_help |
| `feature_snapshot` | inference_server | ACC feature values per inference (for retraining) |

Removed tables: `fall_history` (migration 0003), `participant_session` (migration 0005).
Migration chain: `0001 → 0002 → 0003 → 0004 → 0005` — run automatically by `db-migrate` on `docker compose up`.

**Caregiver layer (FOCUS side) has NO Postgres:**
- Patient list: SQLite (PVC, managed dynamically from UI)
- Fall history: queried from FOCUS InfluxDB (`fall_events` measurement)

---

## 10. InfluxDB — Two Instances, Completely Different Roles

| Instance | Hosted by | Role | Used by |
|----------|-----------|------|---------|
| **FOCUS-hosted InfluxDB** | FOCUS (their existing infra) | Stores biosignals (`SMART_DATA`) + `fall_events` points | Mobile app (writes both), fall_dashboard (reads fall history), Flutter Patient Dashboard (reads biosignals) |
| **MCS cloud InfluxDB** (`ecosystem-influxdb.smarko-health.de`) | MCS/SmarKo | Dev testing only — `fd_test` bucket used by mock_app as fake BLE input | mock_app only. Absent in production. |

**The inference pipeline never reads from any InfluxDB at runtime.**
**MCS `.env` does NOT need InfluxDB credentials for Hetzner production.**

---

## 11. Active Codebase Location

```
C:\Users\hayat\Documents\6G\FallDetection_new\
  README.md                               Project root — start here. Links to all handover docs.
  system_overview.md                      NEW (2026-06-10): Three-layer system overview with hosting codes (F1/F2/F3/M), ASCII diagram, component tables, cross-network data flows.

  _6G_integration_v3_docker_mcs/         MCS inference layer (ACTIVE, Hetzner target)
    .env.example                          Reference config — copy to .env, fill in CHANGE_ME

  _6G_integration_v3_docker_focus/       FOCUS caregiver layer, Docker version (local testing backup)
    mock_app/                             Simulates mobile app — reference for Isa's implementation
      main.py                             Entry point — full flow documented in comments
      api_caller.py                       /predict and /confirm HTTP calls
      influx_writer.py                    fall_events write (schema, field types, encoding)
      poller.py                           Polling loop, MQTT publish (possible + alert), patient popup
      patient_server.py                   Browser-based patient confirmation popup (port 8005)
      influx_fetcher.py                   Reads SMART_DATA from InfluxDB (mock of BLE read)

  _6G_integration_v3_k3s/               FOCUS caregiver layer, K3s Helm chart (ACTIVE, production target)
    fall_dashboard/db.py                  InfluxDB client is now INLINED here — ml_pipeline/ removed (2026-06-10)
    helm/values.yaml                      Local testing values (NodePorts 30901/30802, local image)
    helm/values_production.yaml           Production values template — fill in all CHANGE_ME values
    helm/build.sh / install.sh            Bash equivalents of .ps1 scripts — use these on Linux (FOCUS server)
    helm/test.sh / teardown.sh            Smoke test (7 probes) + teardown — bash versions for Linux
    NOTE: ml_pipeline/ and config/ folders DELETED (2026-06-10) — contained MCS ML IP not needed by FOCUS
    NOTE: traefik-mqtts-entrypoint.yaml (extras/) is DEPRECATED — do NOT apply
    NOTE: mosquitto-ingressroutetcp.yaml (templates/) is DEPRECATED — do NOT apply

  mohammed/                               Handover docs for Mohammed
    mohammed_handover.md                  What Mohammed must do, step-by-step Hetzner deployment
    config_checklist_mohammed.md          Numbered config checklist for MCS .env (8 sections)

  FOCUS_devs_handover/                    Handover docs for FOCUS DevOps
    focus_devops_handover.md              What FOCUS DevOps must do, Step 0 = firewall, step-by-step k3s deploy
    config_checklist_focus_devops.md      Numbered config checklist for values_production.yaml (12 sections)
    production_config_checklist.md        Pointer index → above two files
    draft/                                Topic-by-topic instruction docs (in progress, 2026-06-10)
      01_system_overview.md               What FOCUS is adding and why — two pods, how they fit existing stack
      02_mqtt_broker.md                   MQTT broker setup, port design, config, log reading
      2.1_mqtt_connection_test.md         MQTT test commands (split from 02)
      03_k3s_values_and_secrets.md        values_production.yaml field-by-field guide + install steps (bash)
      04_pulling_fall_dashboard_image.md  Registry pull secret setup + troubleshooting
      05_firewall_and_ports.md            Port reference, ufw/firewalld commands, k3s+ufw gotcha
      06_influxdb_schema_and_queries.md   (stub — not yet written)
      07_fall_dashboard_user_guide.md     Caregiver UI guide: cards, alerts, add/delete patients, history tab
      08_debug_guide.md                   8 failure modes with kubectl commands and fixes
      helm_scripts_reference.md           What each .sh script does + who runs it

  isa/
    todos_isa.md                          NEW (2026-06-10): Full mobile app to-do list with checkboxes, derived from mock_app comparison. Covers BLE read, InfluxDB SMART_DATA write, /predict, MQTT possible publish, confirmation popup, /confirm, MQTT alert publish, fall_events write.
    todos_isa.txt                         Old plain-text version (kept for reference)

  __Refactoring_docs/
    TODO.md                               Main task tracker
    deployment_architecture.md            Full architecture diagram + data sources
    influxdb_schema.md                    fall_events measurement schema, Flux queries
    port_management.md                    Port mapping across all 4 environments
    service_test.md                       How to run service tests
    k3s_two_laptop_test.md                Step-by-step K3s two-laptop test guide
    mobile_app_publisher_contract_ISA.md  Spec for Isa: MQTT topics, payloads, InfluxDB schema
    mqtt_broker_logs_and_qos.md           Mosquitto logs, QoS explanation
    react_mqtt.md                         Why WebSocket not TCP
    network_config.md                     Windows firewall rules for cross-machine testing
    _ai_wiki/                             This folder
```

Current git branch: `mcs-docker-deployment`

**Git tracked folders only** (allowlist .gitignore as of 2026-06-10):
`_6G_integration_v3_docker_focus/`, `_6G_integration_v3_docker_mcs/`, `_6G_integration_v3_k3s/`, `FOCUS_devs_handover/`, `isa/`, `mohammed/`, `__Refactoring_docs/`, `.gitignore`, `README.md`, `system_overview.md`

---

## 12. Key Known Gotchas

- **MQTT host must be `127.0.0.1` not `localhost`** in `.env` on Windows — Windows resolves `localhost` to `::1` (IPv6) but port-forward binds IPv4 only.
- **Port 9001 WebSocket** is the only port the mobile app can use. Port 1883 is internal Docker/K8s network only. Do NOT expose 1883 to the mobile app.
- **MinIO console runs on port 9002** (not 9001) — 9001 is taken by the MQTT WebSocket listener.
- **`patient_confirmed` encoding asymmetry**: string on MQTT (`"yes"`/`"no"`/`"not_answered"`), int in InfluxDB (`1`/`0`/`-1`). The fall_dashboard converts — the mobile app must send string on MQTT and int to InfluxDB.
- **Confirmation popup: 10 seconds PER question**, not 10 seconds shared across both questions. Q1 ("Did you fall?") gets 10s. If yes, Q2 ("Do you need help?") gets another 10s.
- **`fall_dashboard` has no Postgres** — patient data = SQLite (PVC), fall data = InfluxDB.
- **PowerShell 5.1 cannot parse non-ASCII characters** (em-dash, smart quotes) in `.ps1` files saved as UTF-8 without BOM — use plain ASCII in `.ps1` files only.
- **K3s mock-app must NOT use `--network host`** — Docker Desktop on Windows runs containers in a Linux VM; use `host.docker.internal` to reach Windows host services.
- **K3s fall-dashboard MQTT startup race**: fall-dashboard pod may start before mosquitto is ready. Code retries 5 times with 5s delay. If still failing, `kubectl delete pod` to restart.
- **Docker Desktop K8s does NOT include Traefik** — must install separately for IngressRoute CRDs to work.
- **`.ps1` scripts are Windows-only** — FOCUS runs Linux. Bash equivalents (`build.sh`, `install.sh`, `test.sh`, `teardown.sh`) are in `_6G_integration_v3_k3s/helm/`. Use `bash helm/install.sh` on the FOCUS server.
- **`traefik-mqtts-entrypoint.yaml` is DEPRECATED** — the extras/ file in the k3s chart must NOT be applied. MQTT now goes over WebSocket on standard port 443 via IngressRoute. Applying the old file adds an unused entrypoint.
- **`ml_pipeline/` and `config/` removed from `_6G_integration_v3_k3s/`** (2026-06-10) — only `influx_client_manager` was used; it is now inlined in `fall_dashboard/db.py`. The folders contained MCS ML pipeline code that should not be shipped to FOCUS.
- **MCS `.env` does NOT need InfluxDB credentials** — no service in `_6G_integration_v3_docker_mcs/` uses InfluxDB.
- **`imagePullPolicy: IfNotPresent` required for local K3s testing** — set to `Always` in `values_production.yaml` only.
- **`helm upgrade` required after every values.yaml change** — K8s does not watch the file.
- **`server-health` docker-compose needs 5 env vars explicitly** — `DATABASE_URL`, `MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL`, `MQTT_BROKER_HOST`, `MQTT_BROKER_PORT`.
- **Patient add/delete is now dynamic from the UI** — no longer need to edit `values.yaml` or `PATIENT_IDS` to add patients.
- **FOCUS firewall: only ports 80 and 443 need to be open** — no separate MQTT port needed. WSS traffic goes through Traefik on 443. If FOCUS already has Traefik serving other services, no new firewall rules are likely needed. See Step 0 in `FOCUS_devs_handover/focus_devops_handover.md`.

---

## 13. Production Config — Who Sets What

Three parties are involved. See `mohammed/config_checklist_mohammed.md` and `FOCUS_devs_handover/config_checklist_focus_devops.md` for full details.

**Values that must be exchanged between parties:**

| What | From | To | Where |
|---|---|---|---|
| Inference server URL | Mohammed (Hetzner domain) | Isa | Mobile app config |
| `API_KEYS` value | Mohammed | Isa | Mobile app `X-API-Key` header |
| MQTT broker domain (`mosquitto.ingress.host`) | FOCUS DevOps | Isa + Mohammed | Mobile app MQTT / MCS `.env` |
| MQTT username + password | FOCUS DevOps | Isa | Mobile app MQTT auth |
| FOCUS InfluxDB write credentials | FOCUS DevOps | Isa | Mobile app InfluxDB write |
| Fall-dashboard domain (`fallDashboard.ingress.host`) | FOCUS DevOps | Mohammed | `FALL_DASHBOARD_URL` in MCS `.env` |
| Registry credentials (`registry-smarko-health.de`) | Mohammed | FOCUS DevOps | `kubectl create secret docker-registry mcs-labs ...` |

**MQTT auth warning:** `fallDashboard.mqtt.username/password` in K3s values.yaml and Isa's mobile app must be **identical**. A mismatch causes silent connection failure.

---

## 14. Retraining Pipeline (Background — Not Immediate Priority)

Data source: **MCS Postgres only** (inference_log + feature_snapshot). InfluxDB is NOT in the retraining loop.

Labels come from `inference_log.patient_confirmed` directly (1 = confirmed fall, used as positive label).

Trigger: MCS admin clicks "Retrain" in ml_dashboard → runs retraining → MLflow logs run → model registered → hot-swap via `/model/switch`.

Models: XGBoost. Best model is v3 (ACC + barometer). Window: 9s @ 50Hz = 450 samples.
