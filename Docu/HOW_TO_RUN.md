# How to Run — Fall Detection System

Complete guide for running the full multi-user system.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [First-Time Setup](#2-first-time-setup)
3. [Option A — Full Stack with Docker Compose](#3-option-a--full-stack-with-docker-compose)
4. [Option B — Development Mode (no Docker)](#4-option-b--development-mode-no-docker)
5. [Running the Original Client (main.py)](#5-running-the-original-client-mainpy)
6. [Verifying Everything Works](#6-verifying-everything-works)
7. [Accessing Dashboards](#7-accessing-dashboards)
8. [Grafana Setup (first time)](#8-grafana-setup-first-time)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

### Required software

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | python.org |
| Docker Desktop | latest | docker.com |
| Git | any | git-scm.com |

### Required Python packages (new for multi-user system)

Install these once from the project root:

```bash
pip install sqlalchemy psycopg2-binary alembic redis \
            prometheus-client prometheus-fastapi-instrumentator \
            python-jose[cryptography] passlib[bcrypt] httpx
```

Or install per-service (from the project root):

```bash
pip install -r system_operator/ml_server/requirements.txt
pip install -r caregiver/api/requirements.txt
pip install -r emergency/notification_service/requirements.txt
```

---

## 2. First-Time Setup

### Step 1 — Generate secrets

```bash
# Generate API key (for ml_server ↔ client auth)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate JWT secret (shared by ml_server and caregiver_api)
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Save these — you'll use them in the `.env` files below.

### Step 2 — Create .env files

**ML Server** (`system_operator/ml_server/.env`):
```env
MODEL_VERSION=v0
HARDWARE_ACC_SAMPLE_RATE=25
ACC_SENSOR_TYPE=bosch
PUBLIC_ENDPOINT_ENABLED=true
API_KEYS=<your-api-key-from-step-1>
DATABASE_URL=postgresql://falldetect:falldetect@localhost:5432/falldetect
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=<your-jwt-secret-from-step-1>
SERVER_PORT=8001
```

**Caregiver API** (`caregiver/api/.env`):
```env
DATABASE_URL=postgresql://falldetect:falldetect@localhost:5432/falldetect
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=<same-jwt-secret-as-ml-server>
CAREGIVER_PORT=8002
```

> To add caregiver users, generate bcrypt hashes:
> ```bash
> python -c "from shared.auth.jwt_utils import hash_password; print(hash_password('mypassword'))"
> ```
> Then add to `.env`:
> ```env
> CAREGIVER_USERS=alice:$2b$12$...,bob:$2b$12$...
> ```

**Emergency Service** (`emergency/notification_service/.env`):
```env
REDIS_URL=redis://localhost:6379/0
EMERGENCY_PORT=8003
# WEBHOOK_URL=https://your-pager-service.example.com/alert  # optional
```

**Root `.env`** (for Flask client / `main.py`):
```env
INFERENCE_MODE=remote
REMOTE_SERVER_URL=http://localhost:8001
REMOTE_API_KEY=<same-api-key-as-ml-server>
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=local-dev-token-fall-detection-2024
INFLUXDB_ORG=fall-detection
INFLUXDB_BUCKET=fd_test
ACC_SENSOR_TYPE=bosch
HARDWARE_ACC_SAMPLE_RATE=25
```

### Step 3 — Start PostgreSQL and Redis (Docker only for these two)

If you don't want to run the full stack, you can start just the databases:

```bash
docker run -d --name fall_postgres \
  -e POSTGRES_USER=falldetect \
  -e POSTGRES_PASSWORD=falldetect \
  -e POSTGRES_DB=falldetect \
  -p 5432:5432 \
  postgres:16-alpine

docker run -d --name fall_redis \
  -p 6379:6379 \
  redis:7-alpine
```

### Step 4 — Run database migrations

From the project root (with `DATABASE_URL` set in environment or in `.env`):

```bash
# Load .env so alembic picks up DATABASE_URL
set DATABASE_URL=postgresql://falldetect:falldetect@localhost:5432/falldetect   # Windows
export DATABASE_URL=postgresql://falldetect:falldetect@localhost:5432/falldetect  # Linux/Mac

alembic upgrade head
```

Expected output:
```
INFO  [alembic.runtime.migration] Running upgrade  -> 0001, Initial schema
```

Verify tables were created:
```bash
docker exec fall_postgres psql -U falldetect -d falldetect -c "\dt"
```

Should show: `inference_log`, `feature_snapshot`, `participant_session`, `api_request_log`.

### Step 5 — Create Grafana read-only Postgres user (for Grafana dashboards)

```bash
docker exec fall_postgres psql -U falldetect -d falldetect -c "
  CREATE USER grafana_ro WITH PASSWORD 'grafana_ro';
  GRANT SELECT ON inference_log, participant_session TO grafana_ro;
"
```

---

## 3. Option A — Full Stack with Docker Compose

Starts all 10 services: postgres, redis, influxdb, ml_server, caregiver_api, emergency_svc, prometheus, grafana, alertmanager, nginx.

### Build and start

```bash
# From project root
docker-compose -f infrastructure/docker-compose.yml up --build
```

For background (detached):
```bash
docker-compose -f infrastructure/docker-compose.yml up --build -d
```

### First-time only: run migrations inside the container

```bash
docker-compose -f infrastructure/docker-compose.yml exec ml_server \
  alembic upgrade head
```

### Check all containers are healthy

```bash
docker-compose -f infrastructure/docker-compose.yml ps
```

All services should show `Up` or `healthy`.

### Stop

```bash
docker-compose -f infrastructure/docker-compose.yml down
# To also delete volumes (wipes Postgres + InfluxDB data):
docker-compose -f infrastructure/docker-compose.yml down -v
```

---

## 4. Option B — Development Mode (no Docker)

Run each service in a separate terminal. Requires Postgres and Redis already running (see Step 3 above).

### Terminal 1 — ML Server

```bash
cd c:\Users\hayat\Documents\6G\FallDetection_new

set DATABASE_URL=postgresql://falldetect:falldetect@localhost:5432/falldetect
set REDIS_URL=redis://localhost:6379/0

python system_operator/ml_server/server.py
# Starts on http://localhost:8001
# API docs: http://localhost:8001/docs
```

### Terminal 2 — Caregiver API

```bash
cd c:\Users\hayat\Documents\6G\FallDetection_new

set DATABASE_URL=postgresql://falldetect:falldetect@localhost:5432/falldetect
set REDIS_URL=redis://localhost:6379/0
set JWT_SECRET_KEY=<your-secret>

python -m uvicorn caregiver.api.server:app --host 0.0.0.0 --port 8002 --reload
# Starts on http://localhost:8002
# API docs: http://localhost:8002/docs
```

### Terminal 3 — Emergency Notification Service

```bash
cd c:\Users\hayat\Documents\6G\FallDetection_new

set REDIS_URL=redis://localhost:6379/0

python -m uvicorn emergency.notification_service.server:app --host 0.0.0.0 --port 8003 --reload
# Starts on http://localhost:8003
```

### Terminal 4 — Flask Client (existing behaviour, unchanged)

```bash
cd c:\Users\hayat\Documents\6G\FallDetection_new
python main.py
# Starts on http://localhost:8000
# Dashboard: http://localhost:8000
```

> In development mode, nginx is not running. Access each service directly on its port.
> The dashboards in `caregiver/dashboard/`, `system_operator/operator_dashboard/`, and
> `emergency/tablet_ui/` can be opened directly as HTML files in a browser,
> but the API calls will fail unless you update `API_BASE` constants in the JS files
> to point to the direct service ports (e.g. `http://localhost:8002`).

---

## 5. Running the Original Client (main.py)

The root `main.py` is unchanged and still works exactly as before.

```bash
# Local inference (no server needed)
set INFERENCE_MODE=local
python main.py

# Remote inference (ml_server must be running)
set INFERENCE_MODE=remote
set REMOTE_SERVER_URL=http://localhost:8001
set REMOTE_API_KEY=<your-api-key>
python main.py
```

The Flask dashboard is at `http://localhost:8000`.

---

## 6. Verifying Everything Works

### Check ML server health

```bash
curl http://localhost:8001/health
# Expected: {"status":"ok","model_loaded":true,"model_version":"v0","uptime_seconds":...}
```

### Check ML server metrics (Prometheus endpoint)

```bash
curl http://localhost:8001/metrics
# Expected: Prometheus text format with http_requests_total, inference_latency_seconds, etc.
```

### Send a test prediction

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-api-key>" \
  -d '{
    "acc_x": [100,102,98,105,99,101,103,97,100,102],
    "acc_y": [200,201,199,200,202,198,200,201,199,200],
    "acc_z": [16384,16400,16370,16390,16380,16400,16370,16390,16380,16400],
    "timestamps_ms": [0,40,80,120,160,200,240,280,320,360],
    "participant": "test_patient"
  }'
# Expected: {"fall_detected":false,"confidence":...,"model_version":"v0",...}
```

### Verify PostgreSQL write

```bash
docker exec fall_postgres psql -U falldetect -d falldetect \
  -c "SELECT id, participant, fall_detected, confidence, latency_ms FROM inference_log ORDER BY id DESC LIMIT 3;"
```

### Check Caregiver API

```bash
curl http://localhost:8002/health
# Expected: {"status":"ok","service":"caregiver_api"}

# Login (replace with your configured user)
curl -X POST http://localhost:8002/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","password":"mypassword"}'
# Expected: {"access_token":"eyJ...","token_type":"bearer","role":"caregiver"}
```

### Verify Redis fall event published

```bash
# In one terminal: subscribe to channel
docker exec fall_redis redis-cli subscribe fall_events

# In another terminal: trigger a prediction with a fall-like pattern
# The subscriber terminal should show the published event
```

---

## 7. Accessing Dashboards

### With Docker Compose (nginx running on port 80)

| Dashboard | URL |
|-----------|-----|
| Operator Dashboard | http://localhost/operator/ |
| Care-Giver Dashboard | http://localhost/caregiver/ |
| Emergency Tablet | http://localhost/emergency/ |
| Grafana | http://localhost/grafana/ |
| ML Server API docs | http://localhost/api/ml/docs |
| Caregiver API docs | http://localhost/api/caregiver/docs |
| Prometheus | http://localhost:9090 (direct, not through nginx) |

### Without Docker (development mode)

Open these HTML files directly in a browser:
- `system_operator/operator_dashboard/index.html`
- `caregiver/dashboard/index.html`
- `emergency/tablet_ui/index.html`

> Note: when opened as `file://`, API calls will fail due to CORS.
> Use a simple file server: `python -m http.server 3000` in the dashboard folder.

---

## 8. Grafana Setup (first time)

1. Open Grafana: http://localhost/grafana/ (or http://localhost:3000 direct)
2. Login: admin / `falldetect` (or whatever `GRAFANA_ADMIN_PASSWORD` is set to)
3. Datasources are auto-provisioned from `infrastructure/grafana/provisioning/datasources/datasources.yml`
4. Dashboards are auto-provisioned from `infrastructure/grafana/dashboards/`
5. You should see three dashboards under the "Fall Detection" folder:
   - **ML Server Overview** — request rate, latency, error rate
   - **Model Performance** — confidence distribution, drift detection
   - **Fall Events Timeline** — patient fall history (reads from PostgreSQL)

If dashboards don't appear, trigger a refresh:
```bash
curl -X POST http://admin:falldetect@localhost:3000/api/admin/provisioning/dashboards/reload
```

---

## 9. Troubleshooting

### "DATABASE_URL not configured" in db_writer logs

The `DATABASE_URL` env var is not set or not loaded. Check:
```bash
python -c "import os; print(os.environ.get('DATABASE_URL'))"
```
Make sure `.env` is in the project root and `python-dotenv` is installed.

### Alembic error: "target database is not up to date"

```bash
alembic current     # shows current migration head
alembic upgrade head  # applies all pending migrations
```

### Redis connection refused

Make sure Redis is running:
```bash
docker exec fall_redis redis-cli ping  # should return PONG
```

### SSE stream disconnects immediately in browser

nginx is buffering the response. Verify `proxy_buffering off` is set in
`infrastructure/nginx/nginx.conf` for the `/stream` location block.

### ml_server: "Model not found" on startup

Set `MODEL_VERSION` to one of the models that have `.pkl` files on disk:
```bash
ls model/
```
Valid options with files: `v0`, `v0_lsb_int`, `v3`, `v5_lsb`.

### Caregiver dashboard: API calls return 401

The JWT token in `localStorage` has expired (8h default). Click Logout and log in again.

### Port conflicts (Windows)

If port 8001/8002/8003 is in use:
```bash
netstat -ano | findstr :8001
```
Kill the process or change `SERVER_PORT` / `CAREGIVER_API_PORT` in the `.env` files.
