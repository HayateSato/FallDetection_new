# Fall Detection System — CLAUDE.md

Project reference for AI-assisted development. Describes architecture, data flows, key files, and planned next steps.

> **Current state (2026-03-20):** Full multi-user system implemented.
> New folder structure: `patient/`, `caregiver/`, `system_operator/`, `emergency/`, `shared/`, `infrastructure/`.
> Root `server.py` and `main.py` still work as-is (backward-compatible wrappers).

---

## Multi-User System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  patient/            Wearable (SmarKo) → BT → Smartphone App → InfluxDB             │
│                      (no code here — see patient/README.md)                         │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                                           │ InfluxDB query
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  system_operator/                                                                   │
│    client/           Flask :8000 — queries InfluxDB, sends to ml_server             │
│    ml_server/        FastAPI :8001 — XGBoost inference + Prometheus /metrics        │
│                        → writes to PostgreSQL (inference_log, feature_snapshot)      │
│                        → publishes to Redis 'fall_events' channel                   │
│    operator_dashboard/ Vanilla HTML/JS — model switcher, Grafana links, health      │
└──────┬────────────────────────────────────────────────────┬───────────────────────┬─┘
       │ Redis pub/sub                                       │ Redis pub/sub         │
       ▼                                                     ▼                      │
┌──────────────────────────┐              ┌──────────────────────────┐              │
│  caregiver/              │              │  emergency/              │              │
│    api/  FastAPI :8002   │              │    notification_service/ │              │
│    dashboard/ HTML/JS    │              │    FastAPI :8003 (SSE)   │              │
│    (patient list, falls) │              │    tablet_ui/ HTML/JS    │              │
└──────────────────────────┘              └──────────────────────────┘              │
                                                                                    │
┌───────────────────────────────────────────────────────────────────────────────────┘
│  infrastructure/                                                                    │
│    docker-compose.yml  — all 10 services (postgres, redis, influxdb, nginx, ...    │
│    prometheus/         — scrape config + alert rules                                │
│    grafana/            — 3 dashboards: server overview, model perf, fall timeline  │
│    alertmanager/       — alert routing (email, webhook)                             │
│    nginx/              — reverse proxy, SSE support, static dashboard files        │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Original System Architecture (still applies to core ML pipeline)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          HARDWARE LAYER                                         │
│                                                                                 │
│   ┌──────────────────────┐           ┌──────────────────────────────────────┐  │
│   │   SmarKo Wearable    │           │         SmarKo Mobile App            │  │
│   │  (IMU Watch)         │  ──BT──►  │         (Android / iOS)              │  │
│   │                      │           │                                      │  │
│   │  • Bosch ACC 25Hz    │           │  • Receives sensor stream via BT     │  │
│   │  • Barometer 25Hz    │           │  • Uploads to InfluxDB over Wi-Fi    │  │
│   │  • Non-Bosch ACC     │           │  • Fields: bosch_acc_x/y/z,          │  │
│   │    100Hz variant     │           │    acc_x/y/z, bmp_pressure           │  │
│   └──────────────────────┘           └──────────────────┬───────────────────┘  │
│                                                         │ HTTPS / HTTP          │
└─────────────────────────────────────────────────────────┼─────────────────────-┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA STORAGE LAYER                                    │
│                                                                                 │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                         InfluxDB                                         │  │
│   │              (Time-Series Database — cloud or local Docker)              │  │
│   │                                                                          │  │
│   │   Measurements: bosch_acc_x, bosch_acc_y, bosch_acc_z                   │  │
│   │                 acc_x, acc_y, acc_z  (non-Bosch 100Hz)                  │  │
│   │                 bmp_pressure  (barometer, Pa)                            │  │
│   │                                                                          │  │
│   │   Lookback window used by client: last 15–30 seconds                    │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────┬───────────────────────┘
                                                          │ InfluxDB query
                                                          │ (influxdb-client Python SDK)
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE CLIENT  (This Windows Laptop)                      │
│                         main.py  —  Flask :8000                                 │
│                                                                                 │
│  Startup:  python main.py                                                       │
│  Key env:  INFERENCE_MODE=remote | local                                        │
│            REMOTE_SERVER_URL=https://xxxx.ngrok-free.app                        │
│            REMOTE_API_KEY=<key>                                                 │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │  ContinuousMonitor (background thread, every N seconds)                  │  │
│  │    1. Query InfluxDB → raw acc_data[3×N], pressure[M]                    │  │
│  │    2a. LOCAL mode  → PipelineSelector.predict() on this machine          │  │
│  │    2b. REMOTE mode → POST /predict to API Server (via ngrok)             │  │
│  │    3. Queue fall notification (SSE or polling) → Web UI                  │  │
│  │    4. CSV export (if recording active)                                   │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  Routes (Flask):  POST /trigger      — manual one-shot detection               │
│                   GET  /events       — SSE stream for fall alerts               │
│                   POST /recording/state — participant info + recording toggle   │
│                   POST /manual_truth/marker — write truth label to InfluxDB     │
│                   GET  /             — Web dashboard UI (index.html)            │
│                                                                                 │
│  Outputs:  results/exported_flaskApp_data/<date>/*.csv                         │
│            results/logs/<date>/*.log                                            │
└─────────────────────────────────────────────────────────┬───────────────────────┘
                                                          │ POST /predict
                                                          │ JSON: {acc_x, acc_y, acc_z,
                                                          │        timestamps_ms,
                                                          │        pressure, ...}
                                                          │ Header: X-API-Key
                                                          │
                                              ┌───────────▼──────────┐
                                              │      ngrok tunnel     │
                                              │  (or Cloudflare CDN) │
                                              │  HTTPS public URL     │
                                              └───────────┬───────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  INFERENCE API SERVER  (Another Windows/Linux Laptop)           │
│                        server.py  —  FastAPI + uvicorn :8000                    │
│                                                                                 │
│  Startup:  python server.py                                                     │
│  Expose:   ngrok http 8000  → copy URL to client's REMOTE_SERVER_URL           │
│  Key env:  MODEL_VERSION=v0 | v3 | ...                                          │
│            PUBLIC_ENDPOINT_ENABLED=true                                         │
│            API_KEYS=<shared key>                                                │
│                                                                                 │
│  Preprocessing pipeline (runs on every /predict request):                      │
│    raw acc arrays (LSB, 25Hz)                                                  │
│        → AccelerometerResampler (25Hz → 50Hz)                                  │
│        → convert_lsb_to_g  (if model expects g units)                          │
│        → convert_acc_nparray_to_df  (numpy → DataFrame)                        │
│        → compose_detection_window  (extract last 9s = 450 samples)             │
│        → PipelineSelector.extract_features (16–22 features)                    │
│        → XGBoost.predict  → {is_fall, confidence}                              │
│                                                                                 │
│  Endpoints:                                                                     │
│    POST /predict    — full inference pipeline, returns prediction JSON          │
│    GET  /model/info — currently loaded model metadata                          │
│    GET  /health     — uptime + model version                                   │
│    GET  /docs       — auto-generated OpenAPI / Swagger UI                      │
│                                                                                 │
│  Models available (model/ dir):                                                 │
│    v0         — ACC only, 16 features (no barometer)                           │
│    v0_lsb_int — ACC only, raw LSB integers                                     │
│    v3         — ACC + BARO, 20 features  [BEST]                                │
│    v5_lsb     — raw features, 22 features                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
Wearable
  │ Bluetooth
  ▼
SmarKo App
  │ Wi-Fi / HTTPS  →  InfluxDB  (stores raw sensor time-series)
                          │
                          │ influxdb-client query (lookback 15–30s)
                          ▼
                    Inference Client (main.py)
                          │
                 INFERENCE_MODE=remote?
                    Yes │                  No │
                        ▼                     ▼
                 POST /predict           PipelineSelector
                 to API Server           .predict() locally
                        │                     │
                        └──────────┬──────────┘
                                   ▼
                         {fall_detected, confidence}
                                   │
                          ┌────────┴─────────┐
                          │                  │
                          ▼                  ▼
                     Web UI alert       CSV export
                  (SSE / polling)    results/<date>/
```

---

## Key Files

| File | Role | Machine |
|------|------|---------|
| `main.py` | Flask client entry point | Client |
| `server.py` | FastAPI server entry point | Server |
| `app/core/inference_engine.py` | `PipelineSelector` — unified XGBoost inference | Both |
| `app/services/remote_inference.py` | HTTP client that POSTs to server | Client |
| `app/services/continuous_monitoring.py` | Background polling thread | Client |
| `app/data_input/data_loader/influx_data_fetcher.py` | Query InfluxDB | Client |
| `app/data_input/data_converter.py` | numpy↔DataFrame, LSB→g, windowing | Both |
| `app/data_input/accelerometer_processor/acc_resampler.py` | 25/100Hz → 50Hz | Both |
| `app/data_input/barometer_processor/` | EMA and slope-limit filtering | Both |
| `config/settings.py` | All .env settings loaded here | Both |
| `config/hardware_config.py` | Hardware profiles (SmarKo, AIDAPT-Trial, Custom) | Both |
| `model/model_v*/` | XGBoost .pkl model files | Server |
| `.env.example` | Client .env template | Client |
| `.env.server.example` | Server .env template | Server |
| `requirements.txt` | Full client dependencies | Client |
| `requirements.server.txt` | Lightweight server dependencies | Server |

---

## Environment Variables (key ones)

### Client (`.env`)

```env
# Switch between local and remote inference
INFERENCE_MODE=remote                        # 'local' or 'remote'
REMOTE_SERVER_URL=https://xxxx.ngrok-free.app
REMOTE_API_KEY=<shared key>

# Sensor hardware
ACC_SENSOR_TYPE=bosch                        # 'bosch' or 'non_bosch'
HARDWARE_ACC_SAMPLE_RATE=25                  # 25, 50, or 100

# InfluxDB connection
INFLUXDB_URL=https://...
INFLUXDB_TOKEN=...
INFLUXDB_ORG=...
INFLUXDB_BUCKET=...
```

### Server (`.env`)

```env
MODEL_VERSION=v0                             # v0, v3, v5_lsb, ...
HARDWARE_ACC_SAMPLE_RATE=25                  # must match client's hardware
PUBLIC_ENDPOINT_ENABLED=true
API_KEYS=<shared key>
SERVER_PORT=8000
```

---

## How to Run

### Server laptop

```bash
git clone <repo>
cd FallDetection_new
pip install -r requirements.server.txt
cp .env.server.example .env
# Edit .env: set MODEL_VERSION, HARDWARE_ACC_SAMPLE_RATE, API_KEYS
python server.py                             # starts on 0.0.0.0:8000
ngrok http 8000                              # copy the https://xxxx.ngrok-free.app URL
```

### Client laptop (this machine)

```bash
# In .env set:
# INFERENCE_MODE=remote
# REMOTE_SERVER_URL=https://xxxx.ngrok-free.app
# REMOTE_API_KEY=<same key as server API_KEYS>
python main.py
```

---

## Model Versions

| Version | ACC Features | BARO | Total Features | Status |
|---------|-------------|------|----------------|--------|
| v0 | Statistical (16) | No | 16 | Active |
| v0_lsb_int | Statistical (16), raw LSB | No | 16 | Active |
| v3 | Statistical (16) | Slope-limit (4) | 20 | Active — BEST |
| v5_lsb | Raw (16), LSB | Raw (6) | 22 | Active |
| v1, v2, v4, v1_tuned, v3_tuned | — | — | — | Model files missing |

All models: XGBoost, threshold = 0.5, window = 9s @ 50Hz (450 samples).

---

## Preprocessing Pipeline Detail

```
InfluxDB records  (raw LSB integers, 25Hz)
      │
      │  fetch_and_preprocess_sensor_data()
      │  → converts FluxRecord list → numpy acc_data[3×N], acc_time[N]
      │
      ▼
[CLIENT sends to SERVER as JSON]
      │
      ▼  (on server)
AccelerometerResampler (if 25Hz → 50Hz via linear interpolation)
      │
convert_lsb_to_g()      (LSB ÷ 16384 → g, skip if model uses LSB)
      │
convert_acc_nparray_to_df()  → DataFrame [timestamp, x, y, z]
      │
compose_detection_window()   → last 450 rows (9s × 50Hz)
      │
PipelineSelector.extract_features()
      │   v1_features: min/max/mean/var per axis + magnitude = 16 features
      │   v2_paper:    magnitude peaks + impact events = 6 features
      │   + barometer: EMA or slope-limit = 4–6 features
      ▼
XGBoost.predict_proba()  →  confidence score
      │
threshold (0.5)  →  {is_fall: bool, confidence: float}
```

---

## New Key Files (multi-user expansion)

| File | Role |
|------|------|
| `system_operator/ml_server/server.py` | FastAPI ML server with Prometheus, Postgres write, Redis publish, model hot-swap |
| `system_operator/ml_server/services/db_writer.py` | Background-task Postgres writer (never blocks inference) |
| `system_operator/ml_server/services/metrics_collector.py` | Prometheus metrics: fall_detections_total, inference_latency_seconds, model_confidence |
| `system_operator/operator_dashboard/` | Operator HTML/JS — model switcher, health, Grafana links |
| `caregiver/api/server.py` | FastAPI REST API for care-givers — reads Postgres, SSE from Redis |
| `caregiver/dashboard/` | Care-giver HTML/JS — patient list, live fall alert banner |
| `emergency/notification_service/server.py` | SSE fan-out service — Redis → tablets + optional webhook |
| `emergency/tablet_ui/` | Emergency tablet HTML/JS — large-text fall alert, auto-reconnect |
| `shared/db/models.py` | SQLAlchemy ORM: inference_log, feature_snapshot, participant_session, api_request_log |
| `shared/db/migrations/` | Alembic migrations. Run: `alembic upgrade head` |
| `shared/redis_client.py` | Async/sync Redis helpers, subscribe_fall_events() generator |
| `shared/auth/jwt_utils.py` | JWT create/verify, bcrypt password hashing, require_role() FastAPI dependency |
| `infrastructure/docker-compose.yml` | Full 10-service stack |
| `infrastructure/prometheus/` | Prometheus config + alert rules (latency, drift, downtime) |
| `infrastructure/grafana/dashboards/` | 3 pre-built dashboards: server overview, model performance, fall timeline |
| `infrastructure/alertmanager/` | Alert routing to email/webhook |
| `infrastructure/nginx/nginx.conf` | Reverse proxy with SSE support (proxy_buffering off) |
| `alembic.ini` | Alembic config (project root) |

## How to Run (full stack)

```bash
# 1. Configure environment
cp system_operator/ml_server/.env.example .env    # fill in API_KEYS, JWT_SECRET_KEY, etc.

# 2. Start all services
# IMPORTANT: always pass --env-file so Docker Compose reads the root .env
# (compose file lives in infrastructure/ so it won't find .env automatically)
docker-compose -f infrastructure/docker-compose.yml --env-file .env up -d

# Alternative: copy .env into infrastructure/ so the flag is not needed every time
#   Copy-Item .env infrastructure/.env   (PowerShell)
#   cp .env infrastructure/.env          (bash)
# Downside: you must keep both files in sync when you change a value.

# 3. Run Alembic migrations (first time only)
DATABASE_URL=postgresql://falldetect:falldetect@localhost:5432/falldetect alembic upgrade head

# 4. Access services
#   Operator dashboard:   http://localhost/operator/
#   Care-giver dashboard: http://localhost/caregiver/
#   Emergency tablet:     http://localhost/emergency/
#   Grafana:              http://localhost/grafana/  (admin / see GRAFANA_ADMIN_PASSWORD)
#   ML Server API docs:   http://localhost/api/ml/docs
```

## How to Run (development, no Docker)

```bash
# Terminal 1 — ML Server
DATABASE_URL=postgresql://... REDIS_URL=redis://localhost:6379/0 \
python system_operator/ml_server/server.py

# Terminal 2 — Caregiver API
DATABASE_URL=postgresql://... REDIS_URL=redis://localhost:6379/0 \
JWT_SECRET_KEY=mysecret python -m uvicorn caregiver.api.server:app --port 8002

# Terminal 3 — Emergency Service
REDIS_URL=redis://localhost:6379/0 \
python -m uvicorn emergency.notification_service.server:app --port 8003

# Terminal 4 — Flask Client (unchanged)
python main.py
```

---

## Next Steps (remaining)

### 1. PostgreSQL — Inference History & Audit Log

Currently, inference results are only written to local CSV files on the client. Adding PostgreSQL would give:

**What to store:**

| Table | Columns | Purpose |
|-------|---------|---------|
| `inference_log` | `id, timestamp, model_version, fall_detected, confidence, window_size, num_features, inference_mode, latency_ms` | Track every prediction |
| `api_request_log` | `id, timestamp, client_ip, endpoint, status_code, response_time_ms, api_key_hash` | Server-side request audit |
| `feature_snapshot` | `inference_id (FK), feature_name, feature_value` | Store feature vectors for retraining / debugging |
| `participant_session` | `id, participant_name, gender, start_time, end_time, fall_count` | Session tracking |

**Where to add it:**
- Server-side: `server.py` saves to PostgreSQL after every `/predict` call
- Client-side: `continuous_monitoring.py` and `detection.py` save session metadata

**Stack:**
```
PostgreSQL (local or cloud e.g. Supabase)
    │
psycopg2 or SQLAlchemy ORM
    │
server.py / main.py
```

**Example schema:**
```sql
CREATE TABLE inference_log (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version   VARCHAR(20),
    fall_detected   BOOLEAN,
    confidence      FLOAT,
    window_size     INT,
    inference_mode  VARCHAR(10),   -- 'local' or 'remote'
    latency_ms      INT,
    participant     VARCHAR(100)
);
```

**Why this matters for MLOps:**
- Lets you query "how many falls detected per day?"
- Track model confidence distributions over time
- Identify drift (model starts returning unusual confidence scores)
- Feed historical results back into model retraining pipeline

### 2. Docker

Containerize `server.py` so the inference server is fully portable:
```
docker build -t fall-detection-server .
docker run -p 8000:8000 --env-file .env fall-detection-server
```

### 3. Model Versioning

Store model metadata in PostgreSQL and serve different versions via `/predict?model=v3` query param.

### 4. Automated Retraining Pipeline

Use PostgreSQL `feature_snapshot` table as training data source. Add a `retrain.py` script that:
1. Queries confirmed falls from `inference_log`
2. Fetches feature vectors from `feature_snapshot`
3. Retrains XGBoost
4. Saves new model to `model/` with version bump

---

## Branch / Git Info

- Main branch: `main`
- Current work branch: `complete_system`
- Recent commits: REST API client-server split, file cleaning
