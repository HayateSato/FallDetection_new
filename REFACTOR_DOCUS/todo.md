# To-Do List — 6G / Charite Integration

**Owners:** Hayate (H), Isa (I)  
**Status key:** [ ] not started · [~] in progress · [x] done · [!] blocked · [-] skipped

---

## 0 — Blockers: Ask These First (H)

| # | Question | Ask | Blocking what |
|---|----------|-----|---------------|
| 0.1 | [x] InfluxDB field names / bucket — N/A: Isa owns the InfluxDB side (FOCUS-hosted); mock_app uses our own InfluxDB with existing settings | — | ~~Step 4~~ |
| 0.2 | [x] Sample rate — confirmed 50Hz; data in InfluxDB already at 50Hz (no resampling needed) | — | ~~Step 4~~ |
| 0.3 | [x] MQTT broker: host, port, topic naming convention, auth — we control this | — | ~~Step 1, 2~~ |
| 0.4 | [ ] Patient ID format in their system (what goes in FHIR + local DB) | FOCUS | Step 5 |
| 0.5 | [ ] Do they have a FHIR server, and is FHIR format required? | FOCUS | Step 5 |
| 0.6 | [ ] Where should the detection result land — FHIR DB / dashboard DB / MQTT only? | FOCUS | Step 5 |
| 0.7 | [ ] Kubernetes namespace names + naming conventions | FOCUS DevOps | Step 9 |
| 0.8 | [ ] Container registry — do they have one, or should images be on Docker Hub? | FOCUS DevOps | Step 9 |
| 0.9 | [x] Isa is the dashboard developer — will integrate our event subscriber | FOCUS | ~~Step 2~~ |

---

## Step 1 — Inference Server: HTTP-only, no MQTT (H) ✓

**Decision:** Inference server has no MQTT client. Returns fall result in HTTP response.

- [x] 1.1 Remove `paho-mqtt` from `requirements.txt`
- [x] 1.2 Remove MQTT client, connect/disconnect hooks, `_publish_fall_event()` from `server.py`
- [x] 1.3 Remove `MQTT_FALL_TOPIC` and MQTT config globals
- [x] 1.4 Update `.env`: removed `MQTT_FALL_TOPIC`; added `MQTT_ALERT_TOPIC`, `MOCK_PATIENT_RESPONSE_TIMEOUT`
- [x] 1.5 Delete `influx_marker_writer.py` (Isa writes InfluxDB markers directly from mobile app)

---

## Step 2 — Mobile App: Patient Confirmation + MQTT Alert (H) ✓

- [x] 2.1 Create `mock_app/influx_fetcher.py`
- [x] 2.2 Create `mock_app/api_caller.py`
- [x] 2.3 Create `mock_app/poller.py` with `_simulate_patient_confirmation` (10s timeout)
- [x] 2.4 Publish to `fall/alert/<patient_id>` after timeout with `patient_confirmed`, `needs_help`
- [x] 2.5 Create `mock_app/client.py` with paho MQTT publisher + graceful shutdown

---

## Step 3 — Caregiver Client: Subscribe to Confirmed Alerts Only (H) ✓

- [x] 3.1 Create `caregiver_client/mqtt_listener.py` — subscribes to `fall/alert/#`
- [x] 3.2 `on_fall` callback wired before `broker.start()` (avoids race condition)
- [x] 3.3 `_on_fall_mqtt` reads `patient_confirmed` from alert payload
- [x] 3.4 Removed `start_auto_confirm_timer` — patient confirmation is mobile app's responsibility
- [x] 3.5 `broker.start()` moved to `client.py` startup hook

---

## Step 4 — InfluxDB Config (N/A — CLOSED) ✓

Isa owns the FOCUS-hosted InfluxDB. Mock_app uses our own InfluxDB with existing settings.
Sample rate confirmed 50Hz → `HARDWARE_ACC_SAMPLE_RATE=50` set in `.env`.

---

## Step 5 — FHIR / Output Format (H)

**Blocked on: 0.5, 0.6**

- [ ] 5.1 Confirm whether FHIR server exists and `FHIR_SERVER_URL` is needed
- [ ] 5.2 If FHIR stored in DB → add JSON column to `fall_history` in `db.py`
- [ ] 5.3 Confirm plain JSON from `/predict` is sufficient if no FHIR required
- [ ] 5.4 Check with FOCUS whether LOINC `72514-3` passes their FHIR validator

---

## Step 6a — Prometheus Metrics (H) ✓

- [x] 6.1 Created `inference_server/services/metrics_collector.py`
- [x] 6.2 Added `prometheus-client` and `prometheus-fastapi-instrumentator` to requirements
- [x] 6.3 Wired `Instrumentator().instrument(app).expose(app)` in `server.py`
- [x] 6.4 `_record_prediction()` called inside `/predict` after inference

---

## Step 6b — Postgres: Shared Inference Log + Fall History DB (H) ← REQUIRED

**Decision (2026-04-15):** Not optional. Required for:
1. Retraining pipeline — confirmed falls (from `fall_history`) joined with ACC windows
   (from `inference_log`) produce clean labelled training examples without InfluxDB archaeology.
2. Model drift analysis — `feature_snapshot` shows which features are shifting over time.

**Design:** One Postgres instance, two logical databases:
- `fall_detection` — our tables (inference_log, feature_snapshot, fall_history, participant_session)
- `mlflow` — MLflow's internal tracking tables (kept separate to avoid migration conflicts)

Both `inference_server` and `caregiver_client` point at the same Postgres instance but
write to different tables. Both live in the same Kubernetes namespace so no cross-namespace
DB traffic.

The `fall_history.inference_id` FK links each confirmed alert to its `inference_log` row —
makes the retraining JOIN a single FK lookup.

- [ ] 6.5 Copy `_OLD/.../db_writer.py` → `inference_server/services/db_writer.py`
- [ ] 6.6 Create `shared/db/models.py` with updated schema:
        `inference_log`, `feature_snapshot`, `fall_history` (add `inference_id` FK + `needs_help`),
        `participant_session`
- [ ] 6.7 Create `shared/db/session.py` — `SessionLocal` factory + `get_db()` dependency
- [ ] 6.8 Set up Alembic in `shared/db/migrations/`
- [ ] 6.9 Add `BackgroundTasks` DB write in `/predict` (inference_server)
- [ ] 6.10 Update `caregiver_client/db.py` to use the shared models (drop standalone SQLite schema)
- [ ] 6.11 Inference server puts its `inference_log.id` in the MQTT alert payload so caregiver_client
         can set `fall_history.inference_id` on arrival
- [ ] 6.12 Update `DATABASE_URL` in `.env` to point at Postgres
         (keep SQLite fallback for local dev without Docker)

---

## Step 6c — Grafana Dashboards (H)

- [ ] 6.13 Add Prometheus + Grafana to Helm chart (Step 9)
- [ ] 6.14 Wire 3 dashboards: `ml_server_overview`, `model_performance`, `fall_events_timeline`

---

## Step 7 — Model Hot-Swap Endpoint (H) ✓

- [x] 7.1 Ported `POST /model/switch` with `threading.Lock`
- [x] 7.2 Ported `GET /model/list`
- [ ] 7.3 Confirm whether model files go in Docker image or mounted volume (ask FOCUS DevOps)

---

## Step 8 — Two-Role Dashboard (I — with API from H)

### H provides:
- [ ] 8.1 Document `GET /health` and `GET /model/info` — for Admin view
- [ ] 8.2 Confirm `GET /api/patients` and `GET /api/falls` return correct fields for Caregiver view
- [ ] 8.3 Add role-based auth if needed (copy `_OLD/shared/auth/jwt_utils.py`)

### Isa builds:
- [ ] 8.4 **Admin view:** service health, model version, last prediction time
- [ ] 8.5 **Caregiver view:** patient list (from FHIR), fall history (from Postgres via our API), real-time alerts (SSE)
- [ ] 8.6 Integrate SSE endpoint (`/api/stream`)

---

## Step 9 — Helm Chart (H + FOCUS DevOps)

**Blocked on: 0.7, 0.8**

**Two namespaces confirmed:**
- **FOCUS namespace** — FHIR server, InfluxDB (eventually), mobile app, FOCUS data fetcher
- **Our namespace** — everything we build (see deployment_architecture.md for full breakdown)

- [ ] 9.1 Write `Dockerfile` for each component (inference_server, caregiver_client, MLflow, Prometheus, Grafana)
- [ ] 9.2 Create `helm/fall-detection/` chart with one `values.yaml`
- [ ] 9.3 Each component = one Kubernetes `Deployment` + `Service`
- [ ] 9.4 Postgres as a `StatefulSet` with persistent volume (one instance, two databases)
- [ ] 9.5 MQTT broker (eclipse-mosquitto) as a `Deployment` in our namespace
- [ ] 9.6 Confirm resource limits per pod with FOCUS DevOps
- [ ] 9.7 Confirm ingress controller for exposing inference API across namespace boundary
- [ ] 9.8 Plan InfluxDB migration: currently using our cloud InfluxDB; eventually package
         a new InfluxDB instance inside our Helm namespace

---

## Step 10 — End-to-End Integration Test (H + I)

- [ ] 10.1 Point mock_app at FOCUS InfluxDB (service name in their namespace)
- [ ] 10.2 Trigger a test fall (manually inject data or CSV replay)
- [ ] 10.3 Verify: Inference API returns FHIR Observation with correct patient ID
- [ ] 10.4 Verify: mock_app publishes `fall/alert/<patient_id>` after confirmation window
- [ ] 10.5 Verify: Caregiver dashboard receives real-time SSE alert
- [ ] 10.6 Verify: Fall history in Postgres, retrievable via `GET /api/falls`
- [ ] 10.7 Verify: `inference_id` FK correctly links `fall_history` → `inference_log`
- [ ] 10.8 Verify: Admin sees service health; Caregiver sees only their patients
- [ ] 10.9 Verify: Prometheus `/metrics` scraped; Grafana shows latency + fall rate

---

## Step 11 — MLflow: Retraining on Charite Data (H)

**Pre-condition:** Charite data sharing agreement required before using patient data.

### 11a — MLflow tracking server

- [ ] 11.1 Add `mlflow` to requirements
- [ ] 11.2 MLflow tracking server as a pod; backed by `mlflow` Postgres database + MinIO artifact store
         (MinIO needed only when inference server loads models from registry across pods)
- [ ] 11.3 Set `MLFLOW_TRACKING_URI` in `.env`

### 11b — Instrument training script

- [ ] 11.4 Wrap with `mlflow.start_run()` context
- [ ] 11.5 Log parameters: `window_seconds`, `sample_rate`, `model_version`, `threshold`, `feature_set`
- [ ] 11.6 Log metrics: `accuracy`, `precision`, `recall`, `f1`, `auc`, confusion matrix
- [ ] 11.7 Log trained `.pkl` as MLflow artifact
- [ ] 11.8 Tag runs: `dataset=charite` vs `dataset=original`

### 11c — Model registry

- [ ] 11.9 Register best model in MLflow Model Registry
- [ ] 11.10 Stages: `Staging` → evaluate → `Production`
- [ ] 11.11 Wire `POST /model/switch` to load from registry by name/stage

### 11d — Retraining data pipeline

- [ ] 11.12 Query: `SELECT il.acc_features, fh.patient_confirmed FROM inference_log il JOIN fall_history fh ON fh.inference_id = il.id WHERE il.fall_detected = TRUE`
- [ ] 11.13 Write `retrain.py` with feature extraction + XGBoost training
- [ ] 11.14 Define retraining trigger: manual / scheduled cron / confidence drift threshold

---

## Summary by Owner

### Hayate (H)
Step 5 (FHIR — blocked), **Step 6b (Postgres — start now)**, Step 6c, Step 7.3,
Step 8.1–8.3 (API docs for Isa), Step 9 (Helm), Step 10, Step 11.

### Isa (I)
Step 8 (dashboard UI). Step 10.5 (dashboard alert test).
Also: writes fall detection result + patient confirmation to FOCUS InfluxDB from mobile app.

---

## What Can Start Now (no blockers)

| Task | Notes |
|------|-------|
| **Step 6b (Postgres)** | Required — start with shared models + Alembic setup |
| Step 8.1–8.3 (API docs for Isa) | Write up current endpoints + response field shapes |
| Step 11a (MLflow tracking server) | Blocked only on data sharing agreement with Charite |
