# Fall Detection System — CLAUDE.md

Project reference for AI-assisted development. Describes architecture, data flows, key files, and planned next steps.

> **Current state (2026-03-27):** Full multi-user system implemented. All three priorities complete.
> New folder structure: `patient/`, `caregiver/`, `system_operator/`, `emergency/`, `shared/`, `infrastructure/`, `datalake/`.
> Root `server.py` and `main.py` still work as-is (backward-compatible wrappers).
> MinIO datalake + CSV offline replay implemented. Model comparison page at `/operator/model_comparison.html`.
> Patient feedback loop: fall → `patient_alerts` Redis → patient popup (10s) → conditional `fall_events` → emergency.
> Live Monitor page at `/operator/live_monitor.html`: start/stop `main.py` from the browser, stream terminal output via SSE.
> Patient SSE uses queue + keepalive pattern — stays "Connected" even when no falls occur.
> ml_server runs with `--workers 1` so asyncio timer tasks and feedback POSTs always share the same process.

---

## Multi-User System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  patient/                                                                           │
│    dashboard/  HTML/JS — standby screen + 10s fall popup + YES/NO feedback flow    │
│                SSE from /api/ml/patient/stream  →  POST /api/ml/patient/feedback   │
└──────────────────────────────────────────┬──────────────────────────────────────────┘
                        ▲ Redis 'patient_alerts' SSE │ feedback POST
                        │                            ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  system_operator/                                                                   │
│    client/           Flask :8000 — queries InfluxDB, sends to ml_server             │
│    ml_server/        FastAPI :8001 — XGBoost inference + Prometheus /metrics        │
│                        → writes to PostgreSQL (inference_log + user_fall/need_help) │
│                        → publishes to Redis 'patient_alerts' on every fall          │
│                        → starts 12s asyncio timer; cancels on feedback              │
│                        → publishes to Redis 'fall_events' only when alert needed    │
│    operator_dashboard/ Vanilla HTML/JS — model switcher, config, replay, comparison │
└──────┬────────────────────────────────────────────────────┬───────────────────────┬─┘
       │ Redis 'fall_events'                                 │ Redis 'fall_events'   │
       ▼                                                     ▼                      │
┌──────────────────────────┐              ┌──────────────────────────┐              │
│  caregiver/              │              │  emergency/              │              │
│    api/  FastAPI :8002   │              │    notification_service/ │              │
│    dashboard/ HTML/JS    │              │    FastAPI :8003 (SSE)   │              │
│    (patient list, falls  │              │    tablet_ui/ HTML/JS    │              │
│     history + feedback)  │              │    (alert ONLY when      │              │
└──────────────────────────┘              │     patient didn't       │              │
                                          │     cancel timer)        │              │
                                          └──────────────────────────┘              │
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
| `system_operator/ml_server/server.py` | FastAPI ML server with Prometheus, Postgres write, Redis publish, model hot-swap, replay, model comparison API, patient feedback endpoints, client subprocess management (`/client/start`, `/client/stop`, `/client/status`, `/client/logs`) |
| `system_operator/ml_server/services/db_writer.py` | Background-task Postgres writer (never blocks inference) |
| `system_operator/ml_server/services/metrics_collector.py` | Prometheus metrics: fall_detections_total, inference_latency_seconds, model_confidence |
| `system_operator/operator_dashboard/index.html` | Operator main page — model switcher, health, config, replay, links to Grafana + comparison + Live Monitor |
| `system_operator/operator_dashboard/model_comparison.html` | Model comparison sub-page — interactive Plotly charts sourced from `GET /model/comparison` |
| `system_operator/operator_dashboard/model_comparison.js` | Comparison page logic — fetches API, renders 5 Plotly charts + tables |
| `system_operator/operator_dashboard/live_monitor.html` | Live Monitor sub-page — Start/Stop `main.py` + terminal log viewer |
| `system_operator/operator_dashboard/live_monitor.js` | Status polling, SSE log stream, API key save/load from localStorage, auto-scroll terminal |
| `caregiver/api/server.py` | FastAPI REST API for care-givers — reads Postgres, SSE from Redis; includes `/falls/history` with time range + feedback filters |
| `caregiver/dashboard/` | Care-giver HTML/JS — two-tab layout: patient list + fall history table with filters and feedback columns |
| `emergency/notification_service/server.py` | SSE fan-out service — Redis `fall_events` → tablets + optional webhook |
| `emergency/tablet_ui/` | Emergency tablet HTML/JS — large-text fall alert, auto-reconnect |
| `patient/dashboard/` | Patient HTML/JS — standby screen + 10s fall detection popup + two-step YES/NO feedback flow |
| `shared/db/models.py` | SQLAlchemy ORM: inference_log (+ user_fall/need_help), feature_snapshot, participant_session, api_request_log |
| `shared/db/migrations/versions/0002_add_feedback_and_missing_columns.py` | Adds user_fall, need_help columns + step_seconds, resampling_method, acc_sensor_type |
| `shared/redis_client.py` | Async/sync Redis helpers; two channels: `patient_alerts` + `fall_events`; subscribe generators |
| `shared/auth/jwt_utils.py` | JWT create/verify, bcrypt password hashing, require_role() FastAPI dependency |
| `infrastructure/docker-compose.yml` | Full 10-service stack; nginx mounts patient/dashboard |
| `infrastructure/prometheus/` | Prometheus config + alert rules (latency, drift, downtime) |
| `infrastructure/grafana/dashboards/` | 3 pre-built dashboards: server overview, model performance, fall timeline |
| `infrastructure/alertmanager/` | Alert routing to email/webhook |
| `infrastructure/nginx/nginx.conf` | Reverse proxy; SSE locations for `/api/ml/patient/stream` + `/api/caregiver/patients/stream`; `/patient/` static files |
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

## Development Priorities (as of 2026-03-25)

The primary goal is **automated model comparison** — comparing XGBoost model versions, feature
processing pipelines, and time window sizes systematically, replacing manual CSV inspection.

### Priority 1 — Operator Dashboard + ML server communication — **COMPLETE**
- Operator dashboard fully functional: model switching, recent inference log, live health
- Processing config panel: window, sensor type, sample rate, resampling method
- Data source toggle: InfluxDB (live) or CSV/MinIO (offline replay)
- All operator → ml_server API calls working end-to-end

### Priority 2 — Model comparison — **COMPLETE**

**Goal:** automated model comparison — same CSV, different model versions, compare results.

**What was implemented:**
- `GET /model/comparison` endpoint in `system_operator/ml_server/server.py`:
  - Queries `inference_log WHERE inference_mode='replay'` aggregated by model_version
  - Returns: fall rate %, confidence percentiles (p10–p95), latency avg+p95, confidence buckets, per-recording × model matrix, recent session list, raw timeseries for scatter
- `system_operator/operator_dashboard/model_comparison.html` — standalone sub-page at `/operator/model_comparison.html`
- `system_operator/operator_dashboard/model_comparison.js` — 5 interactive Plotly.js charts:
  - Fall rate by model (horizontal bar)
  - Latency avg + p95 (grouped bar)
  - Confidence box plot — falls vs non-falls per model
  - Confidence histogram — distribution across all windows per model
  - Confidence scatter — every window coloured by fall/no-fall
  - Confidence percentile table (p10/p25/median/p75/p90/p95/mean/stddev + uncertainty %)
  - Per-recording × model table (fall rate colour-coded low/medium/high)
  - Recent sessions audit table

**Decision:** model comparison is built into the operator dashboard (not Grafana). Grafana remains for server health metrics (ml_server_overview, model_performance, fall_events_timeline dashboards).

**Workflow for model comparison (end-to-end):**
1. Upload SmarKo CSV to MinIO via operator dashboard → Processing Configuration panel
2. Switch to model v0 → Run Replay on the CSV
3. Switch to model v3 → Run Replay on same CSV
4. Click "View Full Model Comparison →" button (appears after replay) or open `/operator/model_comparison.html`

### Priority 3 — Patient feedback + Caregiver dashboard + Emergency alerts — **COMPLETE**

**What was implemented:**

**Patient feedback loop (ml_server + new patient/dashboard/):**
- On every real-time fall (not replay): publishes to Redis `patient_alerts` channel + starts 12-second `asyncio` timer
- `GET /patient/stream` SSE endpoint — queue-based with 15s keepalive; stays "Connected" even when no falls occur; retries Redis internally on error
- ml_server runs with `--workers 1` — asyncio timer tasks and feedback POSTs always hit the same process
- Patient popup shows for 10s: "Strong impact detected — did you fall?" with animated countdown
  - YES → second popup: "Do you need help?" (10s countdown)
    - YES → `POST /patient/feedback` with `user_fall=1, need_help=1` → cancels timer → publishes `fall_events` → emergency alerted
    - NO  → `user_fall=1, need_help=2` → cancels timer → no emergency alert
    - 10s timeout → `user_fall=1, need_help=3` → publishes `fall_events` → emergency alerted
  - NO  → `user_fall=2, need_help=2` → cancels timer → no emergency alert
  - 12s server-side timeout (no response at all) → `user_fall=3, need_help=3` → publishes `fall_events` → emergency alerted
- Feedback values logged to `inference_log`: `user_fall` (0=pending/1=yes/2=no/3=no_answer), `need_help` (same)
- Emergency service receives events only from `fall_events` channel (conditional) — NOT on every fall

**Caregiver dashboard (full rewrite):**
- Fixed CSS `.hidden` class bug (`display:none !important`) — login screen no longer dominates after login
- Two-tab layout: **Patients** (patient cards with fall count badges) and **Fall History**
- Fall History tab: filter by patient name, time range (from/to), `user_fall` status, `need_help` status
- All fall tables show "Patient confirmed" and "Help requested" columns (Yes / No / No answer / Pending)
- Header stats: falls today, confirmed falls, help requested, avg confidence
- `GET /falls/history` API endpoint: paginated, all filters, returns feedback columns

**Caregiver API fixes:**
- Fixed route ordering: `/patients/stream` declared before `/patients/{name}/...` (FastAPI matching bug)
- Added `user_fall`/`need_help` to all fall history responses
- New `GET /falls/history` endpoint with time range + feedback filters + pagination

### Live Monitor — main.py integration — **COMPLETE** (2026-03-27)

Operator can start/stop `main.py` from the browser and watch its terminal output in real time.

**Backend (ml_server):**
- `POST /client/start` — spawns `python /app/main.py` as subprocess, captures stdout+stderr
- `POST /client/stop` — terminates subprocess (SIGTERM → SIGKILL after 5s)
- `GET /client/status` — `{ running, pid, returncode }`
- `GET /client/logs` — SSE stream: sends buffered history (last 2000 lines) on connect, then live lines; 25s keepalive; retries on subscriber disconnect

**Frontend (`operator_dashboard/live_monitor.html` + `live_monitor.js`):**
- API key field (pre-fills from localStorage if already set on main dashboard)
- Start/Stop buttons with action status
- Terminal-style log viewer: colour-coded (errors red, warnings yellow), auto-scroll, manual scroll to pause
- Status dot (green = running, grey = stopped)
- Polls status every 10s

**nginx:** Added `/api/ml/client/logs` SSE location (proxy_buffering off) before generic `/api/ml/` block.

### MinIO datalake + CSV offline replay — IMPLEMENTED
- MinIO runs as Docker service (`fall_minio`), S3-compatible, ports 9000 (API) and 9001 (console)
- `datalake/minio_client.py` — boto3 S3 client helpers (list/upload/download)
- `datalake/csv_converter.py` — SmarKo CSV → sliding windows for inference (filters `is_accelerometer_bosch==1`, `is_pressure==1`)
- ml_server new endpoints: `GET /datalake/files`, `POST /datalake/upload`, `POST /datalake/replay`
- Operator dashboard: CSV radio shows file picker, upload button, replay button, results table
- Replay runs the full inference pipeline server-side (resample → LSB→g → window → XGBoost) for every window
- Replay saves all predictions to `inference_log` with `inference_mode='replay'` and `participant=filename`
- **Stack:** MinIO + boto3 + pandas (all in Dockerfile.python)
- **MinIO console:** http://localhost:9001 (minioadmin / minioadmin by default)

### Processing configuration via API — IMPLEMENTED
- `POST /config` accepts: `window_seconds`, `acc_sensor_type`, `hardware_sample_rate`, `resampling_method`
- `GET /config` returns current values
- Operator dashboard config panel sends all 4 fields and shows confirmation

---

## Known Issues / Gotchas

- Docker Compose must always use `--env-file .env` (compose file is in `infrastructure/` subdirectory)
- Bcrypt hashes contain `$` — use `infrastructure/caregiver_secrets.env` with `$$` for Docker; `caregiver/api/.env` with single `$` for dev mode
- `GF_SECURITY_ADMIN_PASSWORD` is ignored after first Grafana startup — use `grafana cli admin reset-admin-password` to change password
- nginx `proxy_pass http://grafana/;` (trailing slash) breaks Grafana sub-path — must be `http://grafana;` (no slash)
- Grafana 10.x requires datasource as `{"type": "...", "uid": "..."}` object — plain string silently gives no data
- Grafana datasource UIDs are fixed after first creation (NOT updated by datasources.yml): PostgreSQL=`PCC52D03280B7034C`, Prometheus=`PBFA97CFB590B2093`
- `grafana_ro` Postgres user must be created manually after first `docker-compose up`
- ml_server must run `--workers 1` — asyncio timer tasks (`_pending_emergency_tasks`) are per-process; feedback POSTs on a different worker cannot cancel the timer
- nginx SSE locations (`/api/ml/patient/stream`, `/api/caregiver/patients/stream`, `/api/ml/client/logs`) MUST be declared before their parent `/api/ml/` and `/api/caregiver/` blocks
- Patient SSE generator uses a queue + background task pattern with 15s keepalive — do not replace with a simple `async for` loop over Redis or the browser will show "Connection error" if Redis briefly hiccups on startup
- Alembic must be run inside the ml_server container (`docker exec fall_ml_server alembic upgrade head`) — Postgres port 5432 is not published to the host
- If `step_seconds`/`resampling_method`/`acc_sensor_type` columns already exist in the DB (ORM was created before migration 0002), migration uses `IF NOT EXISTS` — safe to re-run

---

## Potential Next Steps

### 1. Automated Retraining Pipeline

`inference_log` now stores confirmed fall labels (`user_fall=1`) and `feature_snapshot` stores the full feature vectors. This is ready to use as training data.

```python
# retrain.py (not yet written)
# 1. SELECT * FROM inference_log WHERE user_fall=1 (confirmed falls)
# 2. JOIN feature_snapshot ON inference_id to get feature vectors
# 3. Retrain XGBoost
# 4. Save to model/ with version bump
```

### 2. Wire participant_session writes

`participant_session` table exists but is never written to. The caregiver `/patients` list will be empty until `main.py` writes session rows on recording start/stop.

### 3. AlertManager notification targets

`infrastructure/alertmanager/alertmanager.yml` has placeholder receivers. Add real email or webhook to get alerts from `ConfidenceDrift`, `HighInferenceLatency`, etc.

### 4. Production hardening

- Close ports 8001, 8002, 9090 on host (dev-only — route through nginx)
- Add SSL certificate to nginx
- Replace plaintext `.env` secrets with Docker secrets or Vault

---

## Branch / Git Info

- Main branch: `main`
- Current work branch: `complete_system`
- Recent commits: REST API client-server split, file cleaning
