# Fall Detection System

Real-time fall detection using XGBoost + SmarKo wearable sensor.
Supports multi-user operation with separate roles for system operators, caregivers, and emergency responders.

For full setup instructions see [Docu/HOW_TO_RUN.md](Docu/HOW_TO_RUN.md).

---

## How to Run

Two options:

**Option A — Full stack in Docker (recommended)**
All 10 services start with one command. Requires Docker Desktop.
```powershell
docker-compose -f infrastructure/docker-compose.yml --env-file .env up --build -d
```

**Option B — Individual services (development)**
Start only Postgres + Redis in Docker, run each Python service in its own terminal.
Useful when actively editing code — supports `--reload`.
```powershell
docker-compose -f infrastructure/docker-compose.yml --env-file .env up -d postgres redis
python system_operator/ml_server/server.py      # Terminal 1
python -m uvicorn caregiver.api.server:app --port 8002 --reload  # Terminal 2
python main.py                                  # Terminal 3 (Flask client)
```

See [Docu/HOW_TO_RUN.md](Docu/HOW_TO_RUN.md) for the complete step-by-step guide including
first-time setup, migrations, credentials, and troubleshooting.

---

## Services & Ports

### Current setup (development)

All 10 containers and how they are currently reachable:

| Container (`docker ps` name) | Service | Internal port | Host port | Via nginx |
|------------------------------|---------|:---:|:---:|:---:|
| `fall_nginx` | Reverse proxy | 80, 443 | **:80, :443** | — (is nginx) |
| `fall_ml_server` | XGBoost inference (FastAPI) | 8001 | **:8001** ⚠️ | `/api/ml/` |
| `fall_caregiver_api` | Caregiver REST API (FastAPI) | 8002 | **:8002** ⚠️ | `/api/caregiver/` |
| `fall_emergency_svc` | Emergency SSE fan-out (FastAPI) | 8003 | — | `/api/emergency/` |
| `fall_prometheus` | Prometheus metrics store | 9090 | **:9090** ⚠️ | — |
| `fall_grafana` | Grafana dashboards | 3000 | — | `/grafana/` |
| `fall_influxdb` | InfluxDB time-series (sensor data) | 8086 | **:8086** | — |
| `fall_postgres` | PostgreSQL (inference history) | 5432 | — | — |
| `fall_redis` | Redis pub/sub (fall events) | 6379 | — | — |
| `fall_alertmanager` | AlertManager (alert routing) | 9093 | — | — |
| `fall_minio` | MinIO datalake (sensor CSV files) | 9000, 9001 | **:9000, :9001** | — |

⚠️ = port published for **development/testing only** — see production table below.

---

### How each service is accessed

**Via nginx only** — browser requests go to `localhost:80`, nginx routes by URL path:

| URL path | Routed to | What it serves |
|----------|-----------|----------------|
| `localhost/operator/` | static files | Operator dashboard HTML/JS (served directly by nginx from disk) |
| `localhost/caregiver/` | static files | Caregiver dashboard HTML/JS (served directly by nginx from disk) |
| `localhost/emergency/` | static files | Emergency tablet UI HTML/JS (served directly by nginx from disk) |
| `localhost/api/caregiver/` | `fall_caregiver_api:8002` | Caregiver REST API (patient list, fall history, SSE stream) |
| `localhost/api/emergency/` | `fall_emergency_svc:8003` | Emergency SSE fan-out (live fall alerts to tablet UI) |
| `localhost/grafana/` | `fall_grafana:3000` | Grafana dashboard UI |

**Directly exposed to host** (bypass nginx):

| URL | Container | Why exposed directly |
|-----|-----------|----------------------|
| `localhost:8001` | `fall_ml_server` | `main.py` runs on Windows (outside Docker) and cannot go through nginx |
| `localhost:8002` | `fall_caregiver_api` | Dev health checks and `curl.exe` testing |
| `localhost:9090` | `fall_prometheus` | No nginx route defined — direct browser access for debugging |
| `localhost:8086` | `fall_influxdb` | SmarKo mobile app writes sensor data from outside the network |
| `localhost:9000` | `fall_minio` | S3 API — used by ml_server (Docker) and boto3 tools (host) |
| `localhost:9001` | `fall_minio` | MinIO web console — upload/browse CSV files |

**Internal Docker network only** (not reachable from Windows host):

| Container | Why internal is enough |
|-----------|----------------------|
| `fall_postgres` | Only accessed by Python services inside Docker |
| `fall_redis` | Only accessed by Python services inside Docker |
| `fall_grafana` | Accessed via nginx — no reason to expose port 3000 directly |
| `fall_alertmanager` | Receives alerts from Prometheus (same network) — no external access needed |

---

### Production vs current setup

What would change when moving to a real deployment:

| Container | Current (dev) | Production |
|-----------|--------------|------------|
| `fall_ml_server` | Port 8001 published to host ⚠️ | Port closed — route through nginx only |
| `fall_caregiver_api` | Port 8002 published to host ⚠️ | Port closed — route through nginx only |
| `fall_prometheus` | Port 9090 published to host ⚠️ | Port closed — IP allowlist or VPN only |
| `fall_influxdb` | Port 8086 published to host | Keep open — SmarKo app writes from outside |
| `fall_nginx` | HTTP only (port 80) | Add SSL cert, redirect HTTP → HTTPS |
| `fall_grafana` | No login enforcement beyond password | Add IP allowlist in nginx.conf |
| `fall_alertmanager` | Email/Slack receivers are empty placeholders | Fill in real receivers in `alertmanager.yml` |
| All services | Single `.env` with real secrets in plain text | Use Docker secrets or a secrets manager (e.g. Vault) |

---

## Access URLs (Docker Compose running)

| What | URL | Auth | Notes |
|------|-----|------|-------|
| Operator dashboard | http://localhost/operator/ | none | Static HTML, served by nginx |
| Caregiver dashboard | http://localhost/caregiver/ | username + password (JWT) | Static HTML + API calls to `/api/caregiver/` |
| Emergency tablet UI | http://localhost/emergency/ | none | Static HTML, SSE stream from `/api/emergency/stream` |
| Grafana | http://localhost/grafana/ | `admin` / `GRAFANA_ADMIN_PASSWORD` | Via nginx |
| Prometheus | http://localhost:9090 | none | Direct — dev/debug only |
| MinIO web console | http://localhost:9001 | `minioadmin` / `minioadmin` | Browse and upload sensor CSV files |
| MinIO S3 API | http://localhost:9000 | access/secret key | Used by ml_server internally and boto3 tools |
| ML server API docs | http://localhost:8001/docs | none | Dev only — Swagger UI |
| Caregiver API docs | http://localhost:8002/docs | none | Dev only — Swagger UI |
| ML server health | http://localhost:8001/health | none | Dev only |
| Caregiver health | http://localhost:8002/health | none | Dev only |

---

## Backend Architecture — What Each Server Actually Is

The system uses a **Backend-For-Frontend (BFF)** pattern: each user type has its own dedicated API server. They do not share endpoints.

### ml_server — FastAPI :8001 — inference engine + operator API

This is the core of the system. It owns the XGBoost model and is the only service that does inference.

| Endpoint | Purpose |
|----------|---------|
| `POST /predict` | Run inference pipeline, return `{ fall_detected, confidence }` |
| `GET /model/info` | Currently loaded model name, version, feature count |
| `GET /model/list` | All available model versions found in `model/` directory |
| `POST /model/switch` | Hot-swap the loaded model without restart (API key required) |
| `GET /inferences` | Recent inference history from PostgreSQL (for operator dashboard) |
| `GET /health` | Uptime, model version |
| `GET /metrics` | Prometheus metrics endpoint (scraped every 15s) |

After every `/predict` call, three things happen as background tasks:
- Write row to PostgreSQL `inference_log`
- Update Prometheus counters
- If fall detected: publish to Redis `fall_events` channel

**Who calls it:** `main.py` (Flask client, for inference) and the operator dashboard (for model management and recent log).

---

### caregiver/api — FastAPI :8002 — caregiver data API

This server is for the caregiver dashboard only. It does not do inference. It reads historical data from PostgreSQL (written by ml_server) and streams live fall events from Redis.

| Endpoint | Purpose |
|----------|---------|
| `POST /auth/login` | Username + password → JWT token |
| `GET /patients` | List all participant sessions with summary stats |
| `GET /patients/{name}/falls` | Paginated fall history for one patient |
| `GET /patients/{name}/stream` | SSE stream of live fall events for one patient |
| `GET /patients/stream` | SSE stream of all fall events (any patient) |
| `GET /stats/summary` | Aggregate stats for today (falls count, avg confidence) |

**Who calls it:** The caregiver dashboard only.

---

### emergency/notification_service — FastAPI :8003 — SSE fan-out only

The simplest service. It subscribes to Redis `fall_events` and fans out to all connected SSE clients (emergency tablets). No database, no auth, no inference.

| Endpoint | Purpose |
|----------|---------|
| `GET /stream` | SSE stream — any connected client receives every fall event in real time |
| `GET /health` | Uptime check |

**Who calls it:** The emergency tablet UI only.

---

### Why this separation?

Each user type sees different data and has different access rights:
- The operator needs model control and raw inference metrics — ml_server owns those.
- The caregiver needs patient history and live alerts per patient — caregiver_api owns those.
- The emergency responder only needs a loud real-time alert — emergency_svc owns that.

None of these services share an endpoint. If you need data from a different user's domain, you are looking at the wrong server.

---

## Data Flow

### Full pipeline — from sensor to all three dashboards

```
SmarKo Wearable
  │  Bluetooth
  ▼
SmarKo Mobile App
  │  HTTPS / Wi-Fi
  ▼
InfluxDB :8086  ←─────────────────────────────── sensor time-series stored here
  │
  │  influxdb-client query (lookback ~15s)
  ▼
main.py — Flask :8000  (runs on Windows, outside Docker)
  │
  │  POST /predict  +  X-API-Key header
  ▼
┌──────────────────────────────────────────────────────────────────┐
│  ml_server  FastAPI :8001 (inference engine + operator API)      │
│                                                                  │
│  1. AccelerometerResampler  (25Hz → 50Hz)                        │
│  2. convert_lsb_to_g                                             │
│  3. compose_detection_window  (last 9s = 450 samples)            │
│  4. PipelineSelector.extract_features  (16–22 features)          │
│  5. XGBoost.predict_proba  →  { fall_detected, confidence }      │
│                                                                  │
│  After every prediction — three things happen in parallel:       │
│                                                                  │
│  A. Background task → PostgreSQL (inference_log, feature_snapshot)│
│  B. Prometheus counters updated → /metrics endpoint              │
│  C. If fall: Redis PUBLISH → channel "fall_events"               │
└──────┬──────────────────────┬──────────────────────┬─────────────┘
       │ A                    │ B                    │ C
       ▼                      ▼                      ▼
  PostgreSQL             Prometheus             Redis pub/sub
  :5432                  :9090                  :6379
  (persists forever)     (30-day retention)     (real-time only,
                                                 no persistence)
       │                      │                      │
       │              scrapes every 15s              │  subscribers
       │                      │                      │
       │                      ▼                      ├──────────────────┐
       │                  Grafana                    │                  │
       │                  :3000                      ▼                  ▼
       │              reads BOTH sources:       caregiver_api      emergency_svc
       │              • Prometheus metrics      :8002              :8003
       │              • PostgreSQL history      │                  │
       │                      │                 │  SSE stream      │  SSE stream
       │                      │                 ▼                  ▼
       │                      ▼            Caregiver          Emergency
       │                 Operator          Dashboard          Tablet UI
       │                 Dashboard         /caregiver/        /emergency/
       │                 /operator/        (patient list,     (large-text
       │                 (model switch,    fall history,      fall alert,
       │                  health, metrics  live alerts)       auto-reconnect)
       │                  links to Grafana)
       │                      ▲
       └──────────────────────┘
         operator dashboard reads
         PostgreSQL via ml_server
         GET /inferences endpoint
```

---

### What each dashboard sees and how

| Dashboard | URL | Data source | What it shows |
|-----------|-----|-------------|---------------|
| **Operator** | `/operator/` | ml_server API (direct) + Grafana links | Active model, health, uptime, links to Grafana dashboards |
| **Caregiver** | `/caregiver/` | caregiver_api → PostgreSQL + Redis SSE | Patient list, fall history per patient, live fall alert banner |
| **Emergency tablet** | `/emergency/` | emergency_svc → Redis SSE | Large-text real-time fall alert, patient name, confidence, auto-reconnect |
| **Grafana** | `/grafana/` | Prometheus (metrics) + PostgreSQL (history) | 3 dashboards: server health, model performance/drift, fall events timeline |

---

### PostgreSQL tables

All written by `ml_server` after every `/predict` call. Read by `caregiver_api` and Grafana.

| Table | Written by | Read by | What is stored |
|-------|-----------|---------|----------------|
| `inference_log` | ml_server (background task) | caregiver_api, Grafana | One row per prediction: `timestamp`, `model_version`, `fall_detected`, `confidence`, `window_size`, `latency_ms`, `participant` |
| `feature_snapshot` | ml_server (background task) | Grafana (for retraining analysis) | One row per feature per prediction: `inference_id` (FK → inference_log), `feature_name`, `feature_value` — stores the full 16–22 feature vector |
| `participant_session` | main.py (recording toggle) | caregiver_api, Grafana | One row per recording session: `participant_name`, `gender`, `start_time`, `end_time`, `fall_count` |
| `api_request_log` | ml_server (every request) | Grafana (audit) | One row per HTTP request: `client_ip`, `endpoint`, `status_code`, `response_time_ms`, `api_key_hash` (SHA-256, never raw key) |

**Key relationship:** `feature_snapshot.inference_id` → `inference_log.id`
Every prediction has exactly one `inference_log` row and N `feature_snapshot` rows (one per feature).
This lets you replay any past prediction by fetching its feature vector.

---

### Redis — what is published (not persisted)

Channel: `fall_events`. Published only when `fall_detected = true`.

```json
{
  "patient_id":    "alice",
  "fall_detected": true,
  "confidence":    0.93,
  "model_version": "v3",
  "timestamp":     "2026-03-25T14:32:01+00:00",
  "inference_id":  42
}
```

`inference_id` links back to the `inference_log` row in PostgreSQL so subscribers can fetch full details.

---



```
FallDetection_new/
├── main.py                         Flask client — queries InfluxDB, sends to ml_server
├── server.py                       Legacy server entry point (backward compat)
├── .env                            All environment variables (single root file)
│
├── system_operator/
│   ├── ml_server/                  FastAPI :8001 — XGBoost inference, Prometheus metrics
│   └── operator_dashboard/         HTML/JS — model switcher, health overview
│
├── caregiver/
│   ├── api/                        FastAPI :8002 — patient list, fall history, SSE stream
│   └── dashboard/                  HTML/JS — patient list, live fall alert banner
│
├── emergency/
│   ├── notification_service/       FastAPI :8003 — Redis → SSE fan-out to tablets
│   └── tablet_ui/                  HTML/JS — large-text fall alert, auto-reconnect
│
├── shared/
│   ├── db/models.py                SQLAlchemy ORM (inference_log, feature_snapshot, ...)
│   ├── db/migrations/              Alembic migration scripts
│   ├── redis_client.py             Async/sync Redis helpers
│   └── auth/jwt_utils.py           JWT + bcrypt auth helpers
│
├── infrastructure/
│   ├── docker-compose.yml          Full 10-service stack
│   ├── Dockerfile.python           Shared base image for Python services
│   ├── caregiver_secrets.env       Bcrypt hashes for Docker ($ escaped as $$)
│   ├── prometheus/                 Scrape config + alert rules
│   ├── grafana/                    3 pre-built dashboards + provisioning
│   ├── alertmanager/               Alert routing (email / webhook)
│   └── nginx/                      Reverse proxy config (SSE-aware)
│
├── datalake/
│   ├── minio_client.py             boto3 S3 helpers — list/upload/download CSV files in MinIO
│   └── csv_converter.py            SmarKo CSV → sliding windows for offline inference replay
│
├── app/                            Core ML pipeline (shared by main.py and ml_server)
├── model/                          XGBoost .pkl files (v0, v3, v5_lsb, ...)
├── config/                         Settings and hardware profiles
└── Docu/                           Documentation
    ├── HOW_TO_RUN.md               Full setup and run guide (start here)
    ├── STATUS.md                   What is done vs. still to implement
    ├── GRAFANA_PROMETHEUS_PLAN.md  Monitoring integration plan
    └── REDIS.md                    Redis architecture notes
```
