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
- [x] 2.5 Create `mock_app/main.py` with paho MQTT publisher + graceful shutdown

---

## Step 3 — Caregiver Client: Subscribe to Confirmed Alerts Only (H) ✓

- [x] 3.1 Create `fall_dashboard/mqtt_listener.py` — subscribes to `fall/alert/#`
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

**Naming clarification (2026-04-16):**
- **Patient Dashboard** = Isa's unified web app (lives in FOCUS namespace). Combines:
  - Demographics panel → `GET /fhir/Patient/{id}` from FOCUS FHIR server (or `mock_focus` locally)
  - Biosignals panel → InfluxDB (FOCUS side, not our code)
  - Fall panel → our `fall_dashboard` API (`GET /api/falls`, SSE `/api/stream`)
- **fall_dashboard** = our backend service (:8002). Not a standalone UI — feeds the fall panel only.

**Mock FHIR server implemented (2026-04-16):**
`mock_focus/fhir_server.py` — FastAPI, port 8003. Simulates FOCUS namespace for local dev.
- `GET /fhir/Patient` — Bundle of all patients
- `GET /fhir/Patient/{id}` — single Patient resource (name, DOB, gender, ward)
- `GET /fhir/Observation?patient={id}` — height, weight, heart rate
Run: `uvicorn mock_focus.fhir_server:app --host 0.0.0.0 --port 8003`
Or via docker-compose (mock_focus_fhir service).
**Replace with real FOCUS FHIR URL in K8s — this mock never ships.**

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

## Step 6b — Postgres: Shared Inference Log + Fall History DB (H) ✓

**Decision (2026-04-15):** Not optional. Required for:
1. Retraining pipeline — confirmed falls (from `fall_history`) joined with ACC windows
   (from `inference_log`) produce clean labelled training examples without InfluxDB archaeology.
2. Model drift analysis — `feature_snapshot` shows which features are shifting over time.

**Design:** One Postgres instance, two logical databases:
- `fall_detection` — our tables (inference_log, feature_snapshot, fall_history, participant_session)
- `mlflow` — MLflow's internal tracking tables (kept separate to avoid migration conflicts)

Cross-reference key: `observation_id` (UUID, not integer FK) is generated at the start of
every `/predict` call, returned in the HTTP response, carried through MQTT alert payload,
and stored in both `inference_log.observation_id` and `fall_history.observation_id`.
This allows the retraining JOIN without a synchronous DB call in the HTTP handler.

- [x] 6.5 Created `inference_server/services/db_writer.py` — BackgroundTask write; never raises
- [x] 6.6 Created `shared/db/models.py` — InferenceLog, FeatureSnapshot, FallHistory, ParticipantSession
- [x] 6.7 Created `shared/db/session.py` — SessionLocal factory, get_db(), init_db()
- [x] 6.8 Set up Alembic: `alembic.ini` + `shared/db/migrations/` + `versions/0001_initial_schema.py`
- [x] 6.9 Added `BackgroundTasks` DB write (step 10) in `/predict`; `observation_id` in PredictResponse
- [x] 6.10 Rewrote `fall_dashboard/db.py` to import from shared models; added `observation_id`, `needs_help`
- [x] 6.11 `mock_app/poller.py` includes `observation_id` from HTTP response in MQTT alert payload;
          `fall_dashboard/main.py` reads it and passes to `record_fall()`
- [x] 6.12 `DATABASE_URL=sqlite:///./caregiver.db` in `.env` (SQLite default; Postgres in production)

---

## Step 6c — Grafana Dashboards (H) ✓

**Infrastructure:** `infrastructure/` folder — Docker Compose spins up Postgres, MQTT, Prometheus, Grafana.
Python services (inference_server, fall_dashboard, mock_app) still run manually in terminals.

Run order:
```powershell
# 1 — Start infrastructure (from _6G_Integration_v2_mqtt/):
docker-compose -f infrastructure/docker-compose.yml up

# 2 — Switch DATABASE_URL in .env to Postgres (see .env comment), then run migrations:
$env:DATABASE_URL = "postgresql+psycopg2://fall_user:fall_pass@localhost:5432/fall_detection"
alembic upgrade head

# 3 — Start Python services as normal (see README run order)
```

- [x] 6.13 Add Prometheus + Grafana to Docker Compose (`infrastructure/docker-compose.yml`)
       Helm chart wiring deferred to Step 9 (blocked on FOCUS DevOps)
- [x] 6.14 Wired 3 dashboards (auto-provisioned via `infrastructure/grafana/provisioning/`):
       - `ml_server_overview.json` — request rate, error rate, p95 latency, falls/hour, confidence buckets
       - `model_performance.json`  — fall rate trend, confidence distribution, low-confidence ratio (drift alert), per-version breakdown
       - `fall_events_timeline.json` — SQL-backed: falls today, recent events table, confidence scatter, falls per patient
- [x] 6.15 Created `infrastructure/postgres/init.sql` — creates `fall_detection` + `mlflow` databases
- [x] 6.16 Created `infrastructure/mosquitto/mosquitto.conf` — MQTT broker config (replaces manual docker run command)
- [x] 6.17 Created `infrastructure/prometheus/prometheus.yml` — scrapes `host.docker.internal:8001/metrics`
- [x] 6.18 Created `infrastructure/grafana/provisioning/datasources/datasources.yaml` — Prometheus + PostgreSQL datasources auto-provisioned

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

- [ ] 9.1 Write `Dockerfile` for each component (inference_server, fall_dashboard, MLflow, Prometheus, Grafana)
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

## Step 11 — MLflow: Retraining on Charite Data (H) ✓ (pipeline implemented; data pending)

**Pre-condition:** Charite data sharing agreement required before using patient data.
**Data source for retraining:** Postgres only (`feature_snapshot` JOIN `fall_history`). InfluxDB is NOT in the retraining loop — features are pre-computed by inference_server at prediction time and stored in Postgres.

**For testing (when no live inference_server has run yet):** Use `retrain/seed_test_data.py` to populate Postgres:
- `--synthetic N` — generates fake feature distributions directly in Postgres (no external deps)
- `--influxdb` — dev utility: fetches historical ACC windows from our own InfluxDB, runs feature extraction locally, writes results to Postgres. Simulates what Postgres would contain after days of live inference. **NOT part of the production retraining flow.**

### 11a — MLflow tracking server ✓

- [x] 11.1 Added `mlflow>=2.10` to `retrain/requirements.txt`
- [ ] 11.2 MLflow tracking server as a pod; backed by `mlflow` Postgres database + MinIO artifact store
         (MinIO needed only when inference server loads models from registry across pods) — deferred to Step 9 (Helm)
- [x] 11.3 Added `MLFLOW_TRACKING_URI=./mlruns` to `.env` (local file store; change to `http://mlflow:5000` in production)

### 11b — Instrument training script ✓

- [x] 11.4 `retrain/retrain.py` — wraps training with `mlflow.start_run()`
- [x] 11.5 Logs params: `model_version`, `n_features`, `n_train`, `n_test`, `scale_pos_weight`, `threshold`
- [x] 11.6 Logs metrics: `accuracy`, `precision`, `recall`, `f1`, `auc`, `tp`, `fp`, `tn`, `fn`
- [x] 11.7 Logs trained `.pkl` as MLflow artifact via `mlflow.xgboost.log_model()`
- [x] 11.8 Tags runs: `dataset=our_data` vs `dataset=charite`; `model_version`, `feature_set`, `window_seconds`

### 11c — Model registry (partial)

- [x] 11.9 `--register` flag in `retrain.py` registers model in MLflow Model Registry as `fall-detection-xgboost`
- [ ] 11.10 Stages: `Staging` → evaluate → `Production` — manual via MLflow UI; no code needed
- [x] 11.11 Wire `POST /model/switch` to load from registry by name/stage — implemented (2026-04-17):
           `{"mlflow_stage": "Production"}` downloads latest .pkl from MLflow registry and hot-swaps it.
           File-based `{"version": "v0_retrained"}` still works as before. mlflow>=2.10 added to inference_server/requirements.txt.

### 11d — Retraining data pipeline ✓

- [x] 11.12 `retrain/data_pipeline.py` — JOIN query; pivot feature_snapshot long→wide; label assignment
- [x] 11.13 `retrain/retrain.py` — full training script (load → split → XGBoost fit → MLflow log → save .pkl)
- [x] 11.14 Trigger: manual (`python -m retrain.retrain`). Scheduled / drift-based trigger deferred.
- [x] 11.15 `retrain/seed_test_data.py` — seeds Postgres for testing without Charite data:
           `--synthetic N` (no InfluxDB needed) or `--influxdb` (dev utility: fetches from our InfluxDB,
           runs feature extraction, writes to Postgres — simulates what a live inference_server would produce)

### How to test the pipeline now (no Charite data needed)

```powershell
# From _6G_Integration_v2_mqtt/ as cwd, with venv active:
pip install -r retrain/requirements.txt

# Seed with 100 synthetic labelled windows:
python -m retrain.seed_test_data --synthetic 100 --model-version v3

# Check dataset stats:
python -m retrain.retrain --dry-run

# Train + log to MLflow:
python -m retrain.retrain --model-version v3 --dataset our_data

# View results:
mlflow ui --backend-store-uri ./mlruns
# → http://localhost:5000
```

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
| **Step 8.1–8.3 (API docs for Isa)** | Write up current endpoints + response field shapes |
| **Step 10 (End-to-end test)** | Steps 1–3, 6b, 6c, 11 complete — can run full local test now |
| **Test MLflow pipeline** | `seed_test_data.py --synthetic 100` then `retrain.py` — no blockers |
| **Test Grafana** | `docker-compose -f infrastructure/docker-compose.yml up`, then start inference_server, run a few /predict calls |

