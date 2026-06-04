# Fall Detection — 6G / Charite Integration (MQTT)

Fall detection stack designed for the FOCUS / Charite monitoring ecosystem.
The inference server returns FHIR R4 Observation resources over HTTP. The mobile
app (mock or real) handles patient confirmation and publishes a confirmed alert
over MQTT. The fall dashboard subscribes to those alerts, writes them to a database,
and feeds the fall panel inside the Patient Dashboard (Isa's unified caregiver UI).

---

## Folder structure

```
_6G_Integration_v2_mqtt/
├── .env                            ← single shared config for all components
├── README.md                       ← you are here
├── alembic.ini                     ← Alembic config (script_location = shared_db/db/migrations)
├── ml_pipeline/                    ← shared ML pipeline (do not edit)
├── config/                         ← settings.py reads .env
├── model/                          ← XGBoost .pkl files (v0, v3, v5_lsb, ...)
│
├── inference_server/               ← HTTP-only ML server (:8001) — ships to K8s
│   ├── server.py                   ← FastAPI: /predict, /health, /metrics, /model/*
│   ├── services/
│   │   ├── metrics_collector.py    ← Prometheus counters + histograms
│   │   └── db_writer.py            ← BackgroundTask: writes inference_log + feature_snapshot
│   └── requirements.txt
│
├── fall_dashboard/                 ← MQTT subscriber + caregiver fall API (:8002) — ships to K8s
│   ├── README.md                   ← see this for endpoints + filter rules
│   ├── main.py                     ← entry point: python -m fall_dashboard.main
│   ├── mqtt_listener.py, web.py, db.py, ...
│   └── dashboard/                  ← local-test HTML/JS (replaced by Isa's web app in prod)
│
├── ml_dashboard/                   ← ADMIN UI: retrain + promote + hot-swap (:8004) — ships to K8s
│   ├── README.md                   ← see this for the operator playbook
│   ├── main.py, web.py             ← FastAPI app + MLflow client + subprocess runner
│   └── dashboard/                  ← HTML page with playbook, retrain, versions, hot-swap
│
├── server_health/                  ← ADMIN UI: aggregate service status (:8006) — ships to K8s
│   ├── README.md                   ← see this for probe list + auto-refresh behaviour
│   ├── main.py, web.py, checks.py  ← parallel probes against 6 services
│   └── dashboard/                  ← traffic-light banner + per-service cards
│
├── shared_db/                      ← shared code imported by both inference_server and fall_dashboard
│   └── db/
│       ├── models.py               ← SQLAlchemy ORM: InferenceLog, FeatureSnapshot, FallHistory, ParticipantSession
│       ├── session.py              ← SessionLocal factory, get_db(), init_db()
│       └── migrations/             ← Alembic migration files
│           ├── env.py
│           └── versions/
│               └── 0001_initial_schema.py
│
├── retrain/                        ← MLflow retraining pipeline — ships to K8s
│   ├── data_pipeline.py            ← JOIN query: inference_log + feature_snapshot + fall_history → DataFrame
│   ├── retrain.py                  ← train XGBoost + log to MLflow (CLI)
│   ├── seed_test_data.py           ← seed Postgres for testing: --synthetic N or --influxdb
│   └── requirements.txt
│
├── local_dev/                      ← LOCAL TESTING ONLY — never ships to Kubernetes
│   ├── mock_app/                   ← simulates the SmarKo mobile app
│   │   ├── main.py                 ← entry point: python -m local_dev.mock_app.main
│   │   ├── poller.py               ← fetch ACC from InfluxDB → infer → confirm → MQTT publish
│   │   ├── patient_server.py       ← patient confirmation popup server (:8005) — browser UI at http://localhost:8005/
│   │   ├── influx_fetcher.py       ← queries our cloud InfluxDB for raw ACC windows
│   │   ├── api_caller.py           ← HTTP client to inference_server /predict
│   │   └── requirements.txt
│   └── dev_scripts/
│       └── switch_model.ps1        ← hot-swap inference server model via /model/switch API
│
└── infrastructure/                 ← Docker Compose for supporting services — local dev only
    ├── docker-compose.yml          ← postgres, mqtt, prometheus, grafana, minio
    ├── postgres/
    │   └── init.sql                ← creates fall_detection + mlflow databases
    ├── mosquitto/
    │   └── mosquitto.conf          ← MQTT broker config (listener 1883, WebSocket 9001)
    ├── prometheus/
    │   └── prometheus.yml          ← scrapes host.docker.internal:8001/metrics
    └── grafana/
        ├── provisioning/
        │   ├── datasources/        ← auto-provisions Prometheus + PostgreSQL datasources
        │   └── dashboards/         ← loads JSON files from dashboards/
        └── dashboards/
            ├── ml_server_overview.json     ← request rate, latency, error rate, falls/hour
            ├── model_performance.json      ← confidence distribution, drift alert, per-version breakdown
            └── fall_events_timeline.json   ← SQL: falls today, recent events table, per-patient
```

---

## Architecture

### How it fits into the wider system

The Patient Dashboard is one web app shown to the caregiver (built by Isa). It has two panels:

| Panel | Data source | Owned by |
|-------|-------------|----------|
| Patient info — demographics (height, weight), biosignals (HR) | FHIR server + InfluxDB | FOCUS |
| Fall panel — fall history + real-time alerts | `fall_dashboard` API (this repo) | Us |

The `fall_dashboard` service (:8002) is the backend that feeds the fall panel. It is not a standalone end-user product.

### Message flow

```
[our cloud InfluxDB — local testing only]
    │  fetch raw ACC window (50 Hz)
    ▼
[local_dev/mock_app]  ──── HTTP POST /predict ────►  [inference_server :8001]
    ◄─────────── HTTP response ──────────  fall_detected, confidence,
    │                                      observation_id (UUID), FHIR
    │                                          │ BackgroundTask (non-blocking)
    │                                          ▼
    │                                   [Postgres: inference_log
    │                                    + feature_snapshot]
    │
    │  patient confirmation window (MOCK_PATIENT_RESPONSE_TIMEOUT seconds)
    │  (real app: native popup on patient's phone)
    │  (mock: browser popup at http://localhost:8005/ — Yes/No/countdown)
    │
    │  MQTT PUBLISH  fall/alert/<patient_id>
    │  payload: { observation_id, patient_confirmed, needs_help, ... }
    ▼
[MQTT broker :1883]
    │  route to subscribers of fall/alert/#
    ▼
[fall_dashboard :8002]  ← real system uses the actual SmarKo app instead of mock_app
    │  DB write → fall_history (Postgres/SQLite)   ← always, for all confirmed alerts
    │  SSE fan-out → caregiver browser             ← only when caregiver action needed:
    │                                                   patient_confirmed="not_answered"
    │                                                   OR confirmed="yes" + needs_help=True
    ▼
[Patient Dashboard — caregiver browser]
    fall panel: "patient-001 fell at 10:00:10, confirmed, needs help"
```

### MQTT clients (only 2)

| Component | Role | Topic |
|-----------|------|-------|
| `local_dev/mock_app` (local) / real SmarKo app (production) | publisher | `fall/alert/<patient_id>` |
| `fall_dashboard` | subscriber | `fall/alert/#` |

The inference server has **no MQTT client**. Fall result is returned directly in the HTTP response.

### observation_id cross-reference

A UUID is generated at the start of every `/predict` call. It flows:

```
inference_server generates obs_id
    → returned in HTTP response (PredictResponse.observation_id)
    → mock_app includes it in MQTT alert payload
    → fall_dashboard stores it in fall_history.observation_id
    → also stored in inference_log.observation_id (BackgroundTask)
```

This links `inference_log` ↔ `fall_history` without a synchronous DB round-trip in the HTTP handler, and enables the retraining JOIN.

---

## Quick start

### Step 1 — Start infrastructure services

```powershell
cd _6G_Integration_v2_mqtt

# Start Postgres, MQTT broker, Prometheus, Grafana
docker-compose -f infrastructure/docker-compose.yml up
```

Service URLs once running:
- Grafana: http://localhost:3000 (admin / admin)
- Prometheus: http://localhost:9090
- MQTT broker: localhost:1883
- Postgres: localhost:5432 (fall_user / fall_pass)

### Step 2 — Install Python dependencies

```powershell
pip install -r inference_server/requirements.txt
pip install -r fall_dashboard/requirements.txt

# Local testing only (mock_app):
pip install -r local_dev/mock_app/requirements.txt
```

### Step 3 — Run database migrations

**SQLite (default)** — no extra steps. Tables are created automatically on first run.

**Postgres** (using the compose stack):

```powershell
# Install the Postgres driver if not already installed
pip install psycopg2-binary

# Point Alembic at the compose Postgres instance
$env:DATABASE_URL = "postgresql+psycopg2://fall_user:fall_pass@localhost:5432/fall_detection"

# Run migrations (from _6G_Integration_v2_mqtt/ — where alembic.ini lives)
alembic upgrade head
```

Verify the tables were created:

```powershell
docker exec -it mcs_fall_postgres psql -U fall_user -d fall_detection -c "\dt"
```

Expected output:
```
 public | fall_history        | table | fall_user
 public | feature_snapshot    | table | fall_user
 public | inference_log       | table | fall_user
 public | participant_session | table | fall_user
```

### Step 4 — Edit .env

Copy `.env.example` to `.env` (or edit `.env` directly). Minimum required:

```
PATIENT_IDS=patient-001
MAC_IDS=AA:BB:CC:DD:EE:FF
INFLUXDB_URL=...
INFLUXDB_TOKEN=...
INFLUXDB_ORG=...
INFLUXDB_BUCKET=...
MQTT_BROKER_HOST=localhost
```

### Step 5 — Start Python services

Five terminals (only the first three are required for the data path; the last two are admin UIs).

```powershell
# Terminal 1 — inference server
uvicorn inference_server.server:app --host 0.0.0.0 --port 8001

# Terminal 2 — fall dashboard (caregiver view)
python -m fall_dashboard.main
# API: http://localhost:8002/api/patients
# Local test UI: http://localhost:8002/

# Terminal 3 — mock mobile app (local testing only — simulates the SmarKo app)
python -m local_dev.mock_app.main
# Patient confirmation popup: http://localhost:8005/   ← open this in a browser when a fall fires

# Terminal 4 — ml_dashboard (admin UI for retrain + hot-swap)
python -m ml_dashboard.main
# Web UI: http://localhost:8004/    (see ml_dashboard/README.md)

# Terminal 5 — server_health (admin status dashboard)
python -m server_health.main
# Web UI: http://localhost:8006/    (see server_health/README.md)
```

### Verify

```powershell
curl.exe http://localhost:8001/health
curl.exe http://localhost:8001/model/info       # check uses_barometer flag
curl.exe http://localhost:8002/api/patients     # should list configured patients
curl.exe http://localhost:8002/api/falls        # fall history (empty until a fall fires)
```

---

## Inference Server API (port 8001)

### `POST /predict` — header: `X-API-Key: <key>`

```json
{
  "patient_id":              "charite-patient-007",
  "device_id":               "smarko-wearable-42",
  "acc_x":                   [-512, -498, "..."],
  "acc_y":                   [128, 134, "..."],
  "acc_z":                   [16300, 16280, "..."],
  "timestamps_ms":           [1712345678000, 1712345678020, "..."],
  "pressure":                [101325.0, 101322.5, "..."],
  "pressure_timestamps_ms":  [1712345678000, 1712345678020, "..."]
}
```

Input ACC values must be **raw LSB integers** as recorded by the SmarKo app. The server converts LSB → g and resamples from hardware rate to 50 Hz internally.

Response includes `fall_detected`, `confidence`, `observation_id` (UUID), and a full FHIR R4 Observation in `fhir_observation`. If `FHIR_SERVER_URL` is set, the server also POSTs the observation there automatically.

### `GET /model/info`

Returns loaded model metadata including the `uses_barometer` flag. mock_app reads this to decide whether to fetch barometer data from InfluxDB.

### `GET /model/list`

Returns all model versions available on disk.

### `POST /model/switch` — header: `X-API-Key: <key>`

Hot-swaps the loaded model without restarting the server.

```json
{ "version": "v3" }
```

### `GET /health`

Liveness check. Returns model version, uptime, sensor config.

### `GET /metrics`

Prometheus metrics endpoint. Exposes: `fall_detections_total`, `inference_latency_seconds`, `model_confidence`.

---

## Fall Dashboard API (port 8002)

This API is consumed by the Patient Dashboard (Isa's UI) to populate the fall panel.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/patients` | Patient list with fall counts and MAC address |
| GET | `/api/falls` | Fall history (`?patient_id=&only_falls=true&limit=200`) |
| GET | `/api/stream` | Server-Sent Events — live confirmed fall alerts |
| GET | `/` | Standalone local test dashboard (HTML/JS) |

### Response fields — `/api/falls`

| Field | Type | Notes |
|-------|------|-------|
| `id` | int | Row PK |
| `patient_id` | string | FHIR Patient identifier |
| `mac_id` | string | MAC address (from MAC_IDS env var) |
| `fall_detected` | bool | True if model flagged a fall |
| `patient_confirmed` | string | `yes` / `no` / `not_answered` |
| `needs_help` | bool / null | Set by patient in confirmation popup |
| `observation_id` | string | UUID linking to `inference_log` |
| `detection_time` | datetime | When the fall was detected |
| `alert_time` | datetime | When the MQTT alert was received |

---

## Database schema

Shared Postgres instance (two logical databases). Managed by Alembic.

**Database: `fall_detection`**

| Table | Written by | Purpose |
|-------|-----------|---------|
| `inference_log` | inference_server (BackgroundTask) | One row per /predict call |
| `feature_snapshot` | inference_server (BackgroundTask) | One row per feature value per call |
| `fall_history` | fall_dashboard (on MQTT arrival) | One row per confirmed alert |
| `participant_session` | fall_dashboard (on startup) | One row per registered patient |

**Database: `mlflow`** — MLflow internal tracking tables (kept separate to avoid migration conflicts).

Cross-reference: `observation_id` (UUID string) is the join key between `inference_log` and `fall_history`. It is not an integer FK — this avoids a synchronous DB call inside the `/predict` HTTP handler.

Local dev uses SQLite (`DATABASE_URL=sqlite:///./caregiver.db`) — no Docker needed. Switch to Postgres by changing `DATABASE_URL` only.

---

## MLflow retraining pipeline

Test the pipeline without Charite data using synthetic Postgres seeds:

```powershell
pip install -r retrain/requirements.txt

# Seed Postgres with 100 synthetic labelled windows:
python -m retrain.seed_test_data --synthetic 100 --model-version v3

# Check dataset stats (dry run):
python -m retrain.retrain --dry-run

# Train + log to MLflow:
python -m retrain.retrain --model-version v3 --dataset our_data

# View results in MLflow UI:
mlflow ui --backend-store-uri ./mlruns
# → http://localhost:5000
```

Retraining data comes from **Postgres only** (feature_snapshot + fall_history joined on observation_id). InfluxDB is not needed for retraining — features are pre-computed and stored at inference time.

---

## Configuration

All components share a single `.env` in this folder.

MOCK APP = `local_dev/mock_app` — local testing only, never runs in production.

| Variable | SERVER | MOCK APP | FALL DASHBOARD | Notes |
|----------|:------:|:--------:|:--------------:|-------|
| `MODEL_VERSION` | X | | | Model loaded on startup |
| `ACC_SENSOR_TYPE` | X | X | | Must match on both sides |
| `HARDWARE_ACC_SAMPLE_RATE` | X | X | | 50 Hz for Charite InfluxDB data |
| `RESAMPLING_METHOD` | X | | | Server resamples to 50 Hz |
| `INFLUXDB_URL` | | X | | Our cloud InfluxDB — local testing only |
| `INFLUXDB_TOKEN` | | X | | Our cloud InfluxDB — local testing only |
| `INFLUXDB_ORG` | | X | | Our cloud InfluxDB — local testing only |
| `INFLUXDB_BUCKET` | | X | | Our cloud InfluxDB — local testing only |
| `PATIENT_IDS` | | X | X | Comma-separated |
| `MAC_IDS` | | X | X | Positional, 1:1 with PATIENT_IDS |
| `POLL_INTERVAL_SECONDS` | | X | | How often mock_app polls InfluxDB |
| `POLL_LOOKBACK_SECONDS` | | X | | Seconds of history per poll |
| `INFERENCE_SERVER_URL` | | X | | Where mock_app POSTs /predict |
| `INFERENCE_API_KEY` | | X | | Must match `API_KEYS` on server |
| `API_KEYS` | X | | | Accepted X-API-Key values |
| `MOCK_PATIENT_RESPONSE_TIMEOUT` | | X | | Seconds before treating as not_answered |
| `MOCK_PATIENT_SERVER_PORT` | | X | | Patient popup server port (default 8005) |
| `DATABASE_URL` | X | | X | SQLite default; Postgres in production |
| `MQTT_BROKER_HOST` | | X | X | Broker hostname or IP |
| `MQTT_BROKER_PORT` | | X | X | Default 1883 |
| `MQTT_ALERT_TOPIC` | | X | X | Default `fall/alert` |
| `MQTT_USERNAME` | | X | X | Leave empty if no broker auth |
| `MQTT_PASSWORD` | | X | X | Leave empty if no broker auth |
| `FHIR_SERVER_URL` | X | | | Optional — server auto-pushes observations |
| `MLFLOW_TRACKING_URI` | | | | Default `./mlruns`; set to `http://mlflow:5000` in production |
| `CAREGIVER_HOST` | | | X | Default `0.0.0.0` |
| `CAREGIVER_PORT` | | | X | Default 8002 |
| `SERVER_PORT` | X | | | Default 8001 |

---

## Postgres — interactive debugging

Enter the container and stay in a live psql session:

```powershell
docker exec -it mcs_fall_postgres psql -U fall_user -d fall_detection
```

You'll get a `fall_detection=#` prompt. Useful queries:

```sql
-- Check recent inference results
SELECT id, patient_id, fall_detected, confidence, detection_time
FROM inference_log
ORDER BY id DESC LIMIT 5;

-- Check fall history (written by fall_dashboard on MQTT arrival)
SELECT id, patient_id, patient_confirmed, needs_help, detection_time
FROM fall_history
ORDER BY id DESC LIMIT 5;

-- Count rows per table
SELECT 'inference_log' AS tbl, COUNT(*) FROM inference_log
UNION ALL
SELECT 'fall_history',         COUNT(*) FROM fall_history
UNION ALL
SELECT 'feature_snapshot',     COUNT(*) FROM feature_snapshot
UNION ALL
SELECT 'participant_session',  COUNT(*) FROM participant_session;

-- List tables
\dt

-- Describe a table's columns
\d inference_log

-- Exit
\q
```

One-liner (run query without entering the container):

```powershell
docker exec -it mcs_fall_postgres psql -U fall_user -d fall_detection -c "SELECT id, patient_id, fall_detected, confidence, detection_time FROM inference_log ORDER BY id DESC LIMIT 5;"
```

---

## Development notes

- `MAC_IDS` uses positional mapping to `PATIENT_IDS` (comma-separated, same order). Do not use `key:value` format — MAC addresses contain `:` which breaks parsing.
- `python-dotenv` does not strip inline `#` comments. Put comments on their own line.
- The inference server must run `--workers 1`. Prometheus counters are per-process; multiple workers would give split counts.
- If `MQTT_BROKER_HOST` is empty: mock_app skips publishing; fall_dashboard SSE sends keepalives only; fall_history is never written.
- Old `caregiver.db` (pre Step 6b) has an incompatible schema. Delete it on first run after upgrading — the correct schema is created automatically by `init_db()`.
- SQLite can report "database is locked" if an external DB viewer holds the file open. Fix: close the viewer, or set `connect_args={"timeout": 30}`.
- `local_dev/dev_scripts/switch_model.ps1` — developer-only helper to hot-swap the inference server model via `POST /model/switch`. Not part of any automated pipeline. Run from `_6G_Integration_v2_mqtt/` as cwd.

### Local Helm testing (without a real registry)

Docker Desktop's built-in Kubernetes shares the local Docker image cache. Build images with the placeholder registry tag and set `imagePullPolicy: Never` in `values.yaml`:

```powershell
# Build (from _6G_Integration_v2_mqtt/ as cwd)
docker build -t registry.example.com/inference-server:latest -f inference_server/Dockerfile .
docker build -t registry.example.com/fall-dashboard:latest  -f fall_dashboard/Dockerfile  .

# Install (uses local images — no push needed)
helm install fall-detection ./helm/fall-detection `
  --set images.pullPolicy=Never `
  --namespace fall-detection --create-namespace
```

Switch `images.pullPolicy` back to `Always` (or `IfNotPresent`) when deploying to the real cluster with a real registry.
