# Fall Detection — 6G / Charite Integration (MQTT)

Fall detection stack for the FOCUS / Charite monitoring ecosystem.
The system is split into two independently hosted layers.

---

## Two-layer architecture

```
┌─────────────────────────────────────┐   ┌──────────────────────────────────────────┐
│  CAREGIVER LAYER  (FOCUS network)   │   │  INFERENCE & POST-TRAINING (MCS/Hetzner) │
│  caregiver_layer/                   │   │  inference_posttraining_layer/            │
│                                     │   │                                           │
│  mqtt           :1883  MQTT broker  │   │  inference-server  :8001  /predict API   │
│  influxdb       :8086  Fall events  │◄──┤  ml-dashboard      :8004  Retrain UI     │
│  fall-dashboard :8002  SSE + API    │   │  server-health     :8006  Status probes  │
│  postgres       :5432  Sessions     │   │  postgres          :5432  inference_log  │
└─────────────────────────────────────┘   │  mlflow            :5000  Model registry │
                                          │  minio             :9000  Artifact store │
                                          │  prometheus        :9090  Metrics        │
                                          │  grafana           :3000  Dashboards     │
                                          └──────────────────────────────────────────┘
```

**FOCUS** hosts their InfluxDB + Flutter caregiver dashboard (not in this repo).
**MCS** hosts the inference server and all ML/post-training components on Hetzner.

---

## Folder structure

```
_6G_Integration_v2_mqtt/
├── .env                            ← shared config for local dev (mock_app etc.)
├── alembic.ini                     ← Alembic config (script_location = shared_db/db/migrations)
│
├── inference_posttraining_layer/   ← MCS / Hetzner deployment (docker compose up)
│   ├── docker-compose.yml          ← 8 services: inference-server, ml-dashboard, server-health,
│   │                                             postgres, mlflow, minio, prometheus, grafana
│   ├── .env.example                ← copy to .env and fill in secrets
│   ├── prometheus.yml              ← scrapes inference-server:8001 by service name
│   └── README.md                   ← deployment guide for Mohammed (TLS, Hetzner, commands)
│
├── caregiver_layer/                ← FOCUS mock / second laptop
│   ├── docker-compose.yml          ← 5 services: mqtt, influxdb, fall-dashboard, postgres, db-migrate
│   ├── .env.example                ← InfluxDB credentials, MQTT auth
│   ├── mosquitto.conf              ← Mosquitto broker config
│   └── README.md                   ← two-laptop test guide with exact IP swap instructions
│
├── inference_server/               ← FastAPI :8001 — HTTP only, no MQTT client
│   ├── server.py                   ← /predict, /health, /metrics, /model/*, /inference/{id}/confirm
│   ├── services/
│   │   ├── metrics_collector.py    ← Prometheus counters + histograms
│   │   └── db_writer.py            ← write_inference_log() + write_confirmation() BackgroundTasks
│   ├── Dockerfile
│   └── requirements.txt
│
├── fall_dashboard/                 ← FastAPI :8002 — MQTT subscriber + caregiver API
│   ├── main.py                     ← entry point: python -m fall_dashboard.main
│   ├── mqtt_listener.py            ← FallEventBroker: fall/alert/# → SSE fan-out
│   ├── web.py                      ← /api/stream SSE + /api/falls + /api/patients
│   ├── db.py                       ← participant_session writes; list_falls() queries InfluxDB
│   ├── Dockerfile
│   └── requirements.txt
│
├── ml_dashboard/                   ← Admin UI :8004 — retrain + model hot-swap
│   ├── main.py, web.py
│   ├── Dockerfile
│   └── README.md
│
├── server_health/                  ← Admin UI :8006 — aggregate service health probes
│   ├── main.py, web.py, checks.py
│   ├── Dockerfile
│   └── README.md
│
├── shared_db/                      ← SQLAlchemy ORM + Alembic migrations
│   └── db/
│       ├── models.py               ← InferenceLog (+ patient_confirmed/needs_help),
│       │                              FeatureSnapshot, ParticipantSession
│       │                              (FallHistory removed — migration 0003)
│       ├── session.py              ← SessionLocal factory, init_db()
│       └── migrations/versions/
│           ├── 0001_initial_schema.py
│           ├── 0002_widen_model_version.py
│           └── 0003_drop_fall_history_add_confirmation.py
│
├── retrain/                        ← MLflow retraining pipeline
│   ├── data_pipeline.py            ← reads inference_log + feature_snapshot → labelled DataFrame
│   ├── retrain.py                  ← train XGBoost + log to MLflow (CLI)
│   ├── seed_test_data.py           ← seed Postgres for testing: --synthetic N or --influxdb
│   └── requirements.txt
│
├── local_dev/                      ← LOCAL TESTING ONLY — never ships to production
│   ├── mock_app/                   ← simulates the SmarKo mobile app
│   │   ├── main.py                 ← entry point: python -m local_dev.mock_app.main
│   │   ├── poller.py               ← poll InfluxDB → /predict → MQTT + InfluxDB write + /confirm
│   │   ├── influx_fetcher.py       ← reads raw ACC windows from MCS cloud InfluxDB
│   │   ├── influx_writer.py        ← writes fall_events to InfluxDB after patient confirmation
│   │   ├── api_caller.py           ← /predict + /inference/{id}/confirm HTTP calls
│   │   ├── patient_server.py       ← patient confirmation popup (:8005) — browser UI
│   │   └── requirements.txt
│   └── dev_scripts/
│       └── switch_model.ps1        ← hot-swap model via /model/switch API
│
├── ml_pipeline/                    ← shared ML pipeline (feature extraction, inference engine)
├── config/                         ← settings.py reads from .env
├── model/                          ← XGBoost .pkl files (v0, v3, ...)
└── infrastructure/                 ← Docker Compose for local dev only (Postgres, MQTT, Prometheus, Grafana)
    ├── docker-compose.yml          ← supporting infra only — Python services run manually
    ├── postgres/init.sql
    ├── mosquitto/mosquitto.conf
    ├── prometheus/prometheus.yml
    └── grafana/
```

---

## Message flow

```
[MCS cloud InfluxDB — ACC sensor data]
    │  fetch raw ACC window (50 Hz)
    ▼
[local_dev/mock_app]  ──── HTTPS POST /predict ────────►  [inference-server :8001]
    ◄──────────── HTTP response ──────────  fall_detected, confidence, observation_id
    │                                                │ BackgroundTask (non-blocking)
    │                                                ▼
    │                                     [Postgres: inference_log + feature_snapshot]
    │
    │  patient confirmation popup (10s timeout)
    │
    ├── MQTT PUBLISH fall/alert/<patient_id>  ────────────►  [MQTT broker :1883]
    │   payload: { observation_id, patient_confirmed, needs_help }        │
    │                                                                      ▼
    │                                                         [fall_dashboard :8002]
    │                                                             SSE fan-out → caregiver UI
    │
    ├── InfluxDB write (fall_events measurement)  ─────────►  [InfluxDB :8086]
    │   fields: patient_id, observation_id, patient_confirmed,     ▲
    │           needs_help, confidence, model_version              │
    │                                                    fall_dashboard /api/falls queries here
    │
    └── HTTPS POST /inference/{observation_id}/confirm  ──►  [inference-server :8001]
        payload: { patient_confirmed, needs_help }               │ BackgroundTask
                                                                  ▼
                                                    [Postgres: inference_log.patient_confirmed
                                                               inference_log.needs_help]
                                                               (used as retraining labels)
```

### MQTT clients (only 2)

| Component | Role | Topic |
|-----------|------|-------|
| mobile app / mock_app | publisher | `fall/alert/<patient_id>` |
| fall_dashboard | subscriber | `fall/alert/#` |

The inference server has **no MQTT client**. Fall result is returned in the HTTP response.

### InfluxDB fall_events schema

| Key | Type | Value |
|-----|------|-------|
| measurement | — | `fall_events` |
| tag: patient_id | string | patient identifier |
| tag: device_id | string | MAC address |
| field: fall_detected | bool | always True |
| field: patient_confirmed | string | `yes` / `no` / `not_answered` |
| field: needs_help | bool | whether rescue alert was sent |
| field: observation_id | string | UUID — links to MCS inference_log |
| field: confidence | float | model confidence score |
| field: model_version | string | e.g. `v0` |
| timestamp | — | detection time |

---

## Quick start

### Option A — Docker Compose (recommended)

#### MCS layer (inference + ML observability)

```powershell
cd _6G_Integration_v2_mqtt

# First time: copy and fill in .env
copy inference_posttraining_layer\.env.example inference_posttraining_layer\.env
# Edit .env: fill in POSTGRES_PASSWORD, API_KEYS, MINIO_USER, MINIO_PASSWORD, GF_SECURITY_ADMIN_PASSWORD

docker compose -f inference_posttraining_layer/docker-compose.yml --env-file inference_posttraining_layer/.env up -d
```

#### Caregiver layer (FOCUS mock — run on second laptop or locally)

```powershell
# Copy env (defaults work for local testing)
copy caregiver_layer\.env.example caregiver_layer\.env

docker compose -f caregiver_layer/docker-compose.yml --env-file caregiver_layer/.env up -d
```

#### Verify

```powershell
curl.exe http://localhost:8001/health          # inference-server
curl.exe http://localhost:8002/api/patients    # fall-dashboard
curl.exe http://localhost:8004/                # ml-dashboard
curl.exe http://localhost:8006/                # server-health
curl.exe http://localhost:5000/                # MLflow
curl.exe http://localhost:9090/-/healthy       # Prometheus
curl.exe http://localhost:3000/api/health      # Grafana
curl.exe http://localhost:9000/minio/health/live  # MinIO
curl.exe http://localhost:8086/health          # InfluxDB (caregiver layer)
```

See `inference_posttraining_layer/README.md` and `caregiver_layer/README.md` for full details
including two-laptop cross-machine testing.

---

### Option B — Manual (for development / debugging)

Start infrastructure only:

```powershell
docker compose -f infrastructure/docker-compose.yml up
```

Run Python services in separate terminals:

```powershell
# inference server
uvicorn inference_server.server:app --host 0.0.0.0 --port 8001

# fall dashboard
python -m fall_dashboard.main

# mock mobile app (local testing — simulates SmarKo app)
python -m local_dev.mock_app.main
# Patient popup: http://localhost:8005/

# ml_dashboard (admin UI)
python -m ml_dashboard.main          # http://localhost:8004/

# server_health (status dashboard)
python -m server_health.main         # http://localhost:8006/
```

---

## Inference Server API (port 8001)

### `POST /predict` — `X-API-Key: <key>`

```json
{
  "patient_id":   "charite-patient-007",
  "device_id":    "smarko-wearable-42",
  "acc_x":        [-512, -498, "..."],
  "acc_y":        [128, 134, "..."],
  "acc_z":        [16300, 16280, "..."],
  "timestamps_ms": [1712345678000, "..."],
  "pressure":     [101325.0, "..."],
  "pressure_timestamps_ms": [1712345678000, "..."]
}
```

Input ACC values must be **raw LSB integers**. The server converts LSB → g and resamples to 50 Hz.
Response includes `fall_detected`, `confidence`, `observation_id` (UUID), and a FHIR R4 Observation.

### `POST /inference/{observation_id}/confirm` — `X-API-Key: <key>`

Called by the mobile app after the patient responds to the confirmation popup.
Updates `patient_confirmed` and `needs_help` on the `inference_log` row (used as retraining labels).

```json
{ "patient_confirmed": "yes", "needs_help": true }
```

`patient_confirmed` must be `"yes"`, `"no"`, or `"not_answered"`.

### `GET /model/info` / `GET /model/list`

Returns loaded model metadata. `uses_barometer` flag tells mock_app whether to fetch pressure data.

### `POST /model/switch` — `X-API-Key: <key>`

Hot-swap the loaded model without restarting.

```json
{ "version": "v3" }              // file-based
{ "mlflow_stage": "Production" } // registry-based (downloads from MLflow + MinIO)
```

### `GET /health` / `GET /metrics`

Liveness check and Prometheus metrics (`fall_detections_total`, `inference_latency_seconds`).

---

## Fall Dashboard API (port 8002)

Consumed by the FOCUS Flutter caregiver dashboard.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/patients` | Patient list with fall counts |
| GET | `/api/falls` | Fall history from InfluxDB (`?patient_id=&only_falls=true&limit=200`) |
| GET | `/api/stream` | Server-Sent Events — live fall alerts from MQTT |

`/api/falls` returns fields: `patient_id`, `fall_detected`, `patient_confirmed`, `needs_help`,
`observation_id`, `detection_time`.

---

## Database schema

**MCS Postgres — `fall_detection` database**

| Table | Written by | Purpose |
|-------|-----------|---------|
| `inference_log` | inference_server (BackgroundTask) | One row per /predict call; `patient_confirmed` + `needs_help` updated via /confirm |
| `feature_snapshot` | inference_server (BackgroundTask) | One row per feature value per call |
| `participant_session` | fall_dashboard (on startup) | One row per registered patient |

`fall_history` table was **removed** (migration 0003). Fall events are now stored in FOCUS InfluxDB
(`fall_events` measurement) by the mobile app. Retraining labels come from `inference_log.patient_confirmed`.

**MCS Postgres — `mlflow` database** — MLflow internal tracking tables.

Alembic migrations: `0001_initial_schema` → `0002_widen_model_version` → `0003_drop_fall_history_add_confirmation`

---

## MLflow retraining pipeline

```powershell
pip install -r retrain/requirements.txt

# Seed Postgres with synthetic labelled windows (no Charite data needed)
python -m retrain.seed_test_data --synthetic 100 --model-version v3

# Check dataset stats
python -m retrain.retrain --dry-run

# Train + log to MLflow
python -m retrain.retrain --model-version v3 --dataset our_data
```

Retraining reads from **Postgres only** (inference_log + feature_snapshot).
Labels come from `inference_log.patient_confirmed` (set via `/inference/{id}/confirm`).
InfluxDB is not in the retraining loop.

---

## Configuration

Key variables in `.env`:

| Variable | Inference Server | mock_app | fall_dashboard | Notes |
|----------|:---:|:---:|:---:|-------|
| `MODEL_VERSION` | X | | | |
| `ACC_SENSOR_TYPE` | X | X | | Must match on both sides |
| `HARDWARE_ACC_SAMPLE_RATE` | X | X | | 50 Hz for Charite data |
| `DATABASE_URL` | X | | X | |
| `API_KEYS` | X | | | Comma-separated accepted keys |
| `INFLUXDB_URL` | | X | X | mock_app: MCS cloud; fall_dashboard: FOCUS InfluxDB |
| `INFLUXDB_TOKEN` | | X | X | |
| `INFLUXDB_ORG` | | X | X | |
| `INFLUXDB_BUCKET` / `INFLUXDB_FALL_EVENTS_BUCKET` | | X | X | |
| `MQTT_BROKER_HOST` | | X | X | IP of caregiver layer machine in two-laptop test |
| `MQTT_BROKER_PORT` | | X | X | Default 1883 |
| `PATIENT_IDS` | | X | X | Comma-separated |
| `MAC_IDS` | | X | X | Positional 1:1 with PATIENT_IDS |
| `INFERENCE_SERVER_URL` | | X | | Where mock_app POSTs /predict |
| `MOCK_PATIENT_RESPONSE_TIMEOUT` | | X | | Seconds before treating as not_answered |

---

## Postgres — interactive debugging

```powershell
docker exec -it mcs_fall_postgres psql -U fall_user -d fall_detection
```

```sql
-- Recent inferences
SELECT id, patient_id, fall_detected, confidence, patient_confirmed, detection_time
FROM inference_log ORDER BY id DESC LIMIT 5;

-- Feature count per inference
SELECT inference_id, COUNT(*) FROM feature_snapshot GROUP BY inference_id ORDER BY inference_id DESC LIMIT 5;

-- Active patient sessions
SELECT participant_name, fall_count, start_time FROM participant_session WHERE end_time IS NULL;

-- Table row counts
SELECT 'inference_log' AS tbl, COUNT(*) FROM inference_log
UNION ALL SELECT 'feature_snapshot', COUNT(*) FROM feature_snapshot
UNION ALL SELECT 'participant_session', COUNT(*) FROM participant_session;
```

---

## Development notes

- `MQTT_BROKER_HOST` must be `127.0.0.1` (not `localhost`) on Windows — `localhost` resolves to `::1` (IPv6) but the broker only binds IPv4.
- `MAC_IDS` uses positional mapping to `PATIENT_IDS`. Do not use `key:value` format — MAC addresses contain `:` which breaks parsing.
- The inference server must run `--workers 1`. Prometheus counters are per-process.
- `python-dotenv` does not strip inline `#` comments — put comments on their own line.
- `local_dev/dev_scripts/switch_model.ps1` — developer helper to hot-swap model via `POST /model/switch`.
