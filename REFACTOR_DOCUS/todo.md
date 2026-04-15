# To-Do List — 6G / Charite Integration

**Owners:** Hayate (H), Isa (I)  
**Status key:** [ ] not started · [~] in progress · [x] done · [!] blocked · [-] skipped

---

## 0 — Blockers: Ask These First (H)

Nothing in Steps 2–7 can start until these are answered.

| # | Question | Ask | Blocking what |
|---|----------|-----|---------------|
| 0.1 | [x] InfluxDB field names / bucket — N/A: Isa owns the InfluxDB side (FOCUS-hosted); mock_app uses our own InfluxDB with existing settings | — | ~~Step 4~~ |
| 0.2 | [x] Sample rate — confirmed 50Hz; data in InfluxDB already at 50Hz (no resampling needed) | — | ~~Step 4~~ |
| 0.3 | [x] MQTT broker: host, port, topic naming convention, auth | FOCUS DevOps | ~~Step 1, 2~~ |
| 0.4 | [ ] Patient ID format in their system (what goes in FHIR + local DB) | FOCUS | Step 5 |
| 0.5 | [ ] Do they have a FHIR server, and is FHIR format required? | FOCUS | Step 5 |
| 0.6 | [ ] Where should the detection result land — FHIR DB / dashboard DB / MQTT only? | FOCUS | Step 5 |
| 0.7 | [ ] Kubernetes namespace + naming conventions | FOCUS DevOps | Step 9 |
| 0.8 | [ ] Container registry — do they have one, or should images be on Docker Hub? | FOCUS DevOps | Step 9 |
| 0.9 | [x] Isa is the dashboard developer — will integrate our event subscriber | FOCUS | ~~Step 2~~ |

---

## Step 1 — Inference Server: HTTP-only, no MQTT (H)

**Decision made:** The inference server has no MQTT client. It returns the fall result
in the HTTP response. The mobile app reads the result there and handles the rest.

- [x] 1.1 Remove `paho-mqtt` from `requirements.txt`
- [x] 1.2 Remove MQTT client, `_connect_mqtt()`, `_disconnect_mqtt()`, `_publish_fall_event()` from `server.py`
- [x] 1.3 Remove `MQTT_FALL_TOPIC` and MQTT config globals from `server.py`
- [x] 1.4 Update `.env`: removed `MQTT_FALL_TOPIC`; added `MQTT_ALERT_TOPIC`, `MOCK_PATIENT_RESPONSE_TIMEOUT`
- [x] 1.5 Delete `influx_marker_writer.py` (colleague writes InfluxDB markers directly)

---

## Step 2 — Mobile App: Patient Confirmation + MQTT Alert (H)

Simulates the real mobile app. Fetches from InfluxDB instead of BLE wearable.

- [x] 2.1 Create `mock_app/influx_fetcher.py` — InfluxDB query logic (moved from caregiver_client)
- [x] 2.2 Create `mock_app/api_caller.py` — HTTP client to inference server `/predict`
- [x] 2.3 Create `mock_app/poller.py` — polling loop; on fall: spawns `_simulate_patient_confirmation` daemon thread
- [x] 2.4 `_simulate_patient_confirmation`: waits `MOCK_PATIENT_RESPONSE_TIMEOUT` seconds, then publishes to `fall/alert/<patient_id>` with `patient_confirmed="not_answered"`, `needs_help=True`
- [x] 2.5 Create `mock_app/client.py` — entry point; creates paho MQTT publisher; graceful shutdown

---

## Step 3 — Caregiver Client: Subscribe to Confirmed Alerts Only (H)

- [x] 3.1 Create `caregiver_client/mqtt_listener.py` — `FallEventBroker` subscribes to `fall/alert/#`
- [x] 3.2 `on_fall` callback wired in `client.py` before `broker.start()` (avoids race condition)
- [x] 3.3 `_on_fall_mqtt` reads `patient_confirmed` from the alert payload (set by mobile app)
- [x] 3.4 Removed `start_auto_confirm_timer` — patient confirmation is now the mobile app's responsibility
- [x] 3.5 `broker.start()` moved from `web.py` to `client.py` startup hook

---

## Step 4 — InfluxDB Config (N/A)

**Status: CLOSED** — InfluxDB polling is done by the mobile app in the real system.
Isa owns the FOCUS-hosted InfluxDB side. For local testing we use our own InfluxDB
with existing `.env` settings. Sample rate confirmed as 50Hz — `HARDWARE_ACC_SAMPLE_RATE=50`
updated in `.env` (server no longer resamples).

- [-] 4.1 N/A — Isa handles field names on her side
- [-] 4.2 N/A — `macAddress` tag used in mock_app only for local testing
- [-] 4.3 N/A — measurement name only matters for our local InfluxDB test setup

---

## Step 5 — FHIR / Output Format (H)

**Blocked on: 0.5, 0.6**

- [ ] 5.1 If FHIR server required → confirm `FHIR_SERVER_URL` is set in `.env` (already implemented in `server.py`)
- [ ] 5.2 If they want FHIR stored in local DB → add JSON column to `FallHistory` in `db.py`
- [ ] 5.3 If no FHIR required → confirm plain JSON output from `/predict` is sufficient
- [ ] 5.4 Check with FOCUS whether LOINC `72514-3` will pass their FHIR validator

---

## Step 6 — MLOps: Prometheus + Grafana (H)

### 6a — Prometheus metrics

- [x] 6.1 Created `inference_server/services/metrics_collector.py`
      Provides: `fall_detections_total`, `inference_latency_seconds`, `model_confidence`
- [x] 6.2 Added `prometheus-client` and `prometheus-fastapi-instrumentator` to `requirements.txt`
- [x] 6.3 Wired `Instrumentator().instrument(app).expose(app)` in `server.py`
- [x] 6.4 `_record_prediction()` called inside `/predict` after inference

### 6b — Postgres inference log (optional — confirm if needed for this partner)

- [ ] 6.5 Copy `_OLD/system_operator/ml_server/services/db_writer.py` → `inference_server/services/db_writer.py`
- [ ] 6.6 Copy `_OLD/shared/db/models.py` + `session.py` → `shared/db/`
- [ ] 6.7 Add `BackgroundTasks` DB write call in `/predict`
- [ ] 6.8 Add `DATABASE_URL` to `.env`

### 6c — Grafana dashboards

- [ ] 6.9 Add Prometheus + Grafana to Helm chart (Step 9)
- [ ] 6.10 Wire 3 dashboards: `ml_server_overview`, `model_performance`, `fall_events_timeline`

---

## Step 7 — Model Hot-Swap Endpoint (H)

- [x] 7.1 Ported `POST /model/switch` into `inference_server/server.py` (with `threading.Lock`)
- [x] 7.2 Ported `GET /model/list` (returns available versions from disk)
- [ ] 7.3 Confirm model files will be packaged in the Helm chart image or mounted as a volume

---

## Step 8 — Two-Role Dashboard (I — with API from H)

### H provides:
- [ ] 8.1 Document `GET /health` and `GET /model/info` endpoints — for Admin view
- [ ] 8.2 Confirm `GET /api/patients` and `GET /api/falls` return correct fields for Caregiver view
- [ ] 8.3 Add role-based auth if needed (copy `_OLD/shared/auth/jwt_utils.py`)

### Isa builds:
- [ ] 8.4 **Admin view:** service health, model version loaded, last prediction time
- [ ] 8.5 **Caregiver view:** patient list, fall history, real-time alerts via SSE
- [ ] 8.6 Integrate SSE endpoint (`/api/stream`) — caregiver subscribes, no MQTT in browser

---

## Step 9 — Helm Chart (H + FOCUS DevOps)

**Blocked on: 0.7, 0.8**

- [ ] 9.1 Write `Dockerfile` for each component (inference server, mock_app/real mobile app backend, caregiver client)
- [ ] 9.2 Create `helm/fall-detection/` chart with one `values.yaml`
- [ ] 9.3 Each component = one Kubernetes `Deployment` + `Service`
- [ ] 9.4 Confirm resource limits per pod with FOCUS DevOps
- [ ] 9.5 Confirm ingress controller type for exposing inference API
- [ ] 9.6 Consider: FOCUS DevOps may prefer to write the Helm chart themselves — provide `docker-compose.yml` as reference

---

## Step 10 — End-to-End Integration Test (H + I)

- [ ] 10.1 Point mock_app at their InfluxDB (service name in same namespace)
- [ ] 10.2 Trigger a test fall (manually inject data into InfluxDB or use CSV replay)
- [ ] 10.3 Verify: Inference API returns FHIR Observation with correct patient ID format
- [ ] 10.4 Verify: mock_app publishes `fall/alert/<patient_id>` after 10s confirmation window
- [ ] 10.5 Verify: Caregiver dashboard (Isa) receives real-time alert via SSE
- [ ] 10.6 Verify: Fall history retrievable via `GET /api/falls`
- [ ] 10.7 Verify: Admin sees service health; Caregiver sees only their patients
- [ ] 10.8 Verify: Prometheus `/metrics` endpoint scraped; Grafana shows inference latency + fall rate

---

## Step 11 — MLflow: Retraining on Charite Data (H)

**Pre-condition:** Charite data sharing agreement must be in place before any patient
sensor data can be used for retraining. All sub-steps blocked until then.

### 11a — MLflow tracking server

- [ ] 11.1 Add `mlflow` to requirements
- [ ] 11.2 Decide where to run MLflow server (local dev: SQLite; production: Postgres + MinIO)
- [ ] 11.3 Set `MLFLOW_TRACKING_URI` in `.env`

### 11b — Instrument the training script

- [ ] 11.4 Wrap training script with `mlflow.start_run()` context
- [ ] 11.5 Log parameters: `window_seconds`, `sample_rate`, `model_version`, `threshold`, `feature_set`
- [ ] 11.6 Log metrics: `accuracy`, `precision`, `recall`, `f1`, `auc`, confusion matrix
- [ ] 11.7 Log trained `.pkl` as MLflow artifact
- [ ] 11.8 Tag runs with `dataset=charite` vs `dataset=original`

### 11c — Model registry

- [ ] 11.9 Register best Charite-trained model in MLflow Model Registry
- [ ] 11.10 Use stages: `Staging` → evaluate → `Production`
- [ ] 11.11 Wire `POST /model/switch` to load by MLflow registry name/stage

### 11d — Retraining data pipeline

- [ ] 11.12 Decide data source: InfluxDB export to CSV, or confirmed falls from caregiver DB
- [ ] 11.13 Write `retrain.py` script
- [ ] 11.14 Define retraining trigger: manual / scheduled / threshold-based

---

## Summary by Owner

### Hayate (H)
Steps 0 (send questions), 5 (FHIR — blocked), 6b/6c (Grafana), 7.3, 8.1–8.3 (API docs), 9, 10, 11.

### Isa (I)
Step 8 (dashboard UI) once H confirms API endpoints. Step 10.5 — dashboard alert test.

---

## What Can Start Now (no blockers)

| Task | Notes |
|------|-------|
| Step 6b (Postgres inference log) | Optional — only if confirmed needed by partner |
| Step 8.1–8.3 (document API for Isa) | Write up current endpoints + response shapes |
| Step 11a (MLflow tracking server) | Self-contained — blocked only on data sharing agreement |
