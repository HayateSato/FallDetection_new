# How to Run — Fall Detection System (Windows)

Complete guide for running the full multi-user system on Windows.

> **Shell note:** All commands below are for **PowerShell**. Do not use `set VAR=value` (that is CMD syntax).
> Use `$env:VAR = "value"` to set environment variables in PowerShell.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [First-Time Setup](#2-first-time-setup)
3. [Option A — Full Stack with Docker Compose](#3-option-a--full-stack-with-docker-compose)
4. [Option B — Development Mode (no Docker for Python services)](#4-option-b--development-mode-no-docker-for-python-services)
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

### Python packages

Install from the project root (activate your venv first):

```powershell
pip install -r requirements.txt
pip install sqlalchemy psycopg2-binary alembic redis httpx `
            prometheus-client prometheus-fastapi-instrumentator `
            python-jose[cryptography] passlib[bcrypt]
```

### Windows curl note

PowerShell has a `curl` alias that maps to `Invoke-WebRequest` and does **not** accept `-H` flags.
Always use `curl.exe` for HTTP testing:

```powershell
curl.exe http://localhost:8001/health        # correct
curl http://localhost:8001/health            # wrong — triggers PS alias
```

For requests with a JSON body, write the body to a file first:

```powershell
'{"key":"value"}' | Out-File -Encoding utf8 body.json
curl.exe -X POST http://localhost:8001/endpoint -H "Content-Type: application/json" -d "@body.json"
```

---

## 2. First-Time Setup

### Step 1 — Generate secrets

```powershell
# Generate API key (ml_server ↔ client auth)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate JWT secret (shared by ml_server and caregiver_api — run again for a different value)
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Save both values — you will use them in `.env` below.

### Step 2 — Configure root .env

The project uses a single root `.env` file for all services. Edit it at the project root:

```env
# Inference
MODEL_VERSION=v0
INFERENCE_MODE=remote
REMOTE_SERVER_URL=http://localhost:8001
REMOTE_API_KEY=<your-api-key-from-step-1>

# ML server
API_KEYS=<same-api-key>
JWT_SECRET_KEY=<your-jwt-secret-from-step-1>
PUBLIC_ENDPOINT_ENABLED=true
SERVER_PORT=8001

# Hardware / sensor
ACC_SENSOR_TYPE=bosch
HARDWARE_ACC_SAMPLE_RATE=25

# InfluxDB
INFLUXDB_URL=https://your-influxdb-url
INFLUXDB_TOKEN=your-token
INFLUXDB_ORG=your-org
INFLUXDB_BUCKET=fd_test

# PostgreSQL
DATABASE_URL=postgresql://falldetect:falldetect@localhost:5432/falldetect

# Redis
REDIS_URL=redis://localhost:6379/0

# Grafana
GRAFANA_ADMIN_PASSWORD=choose_a_password

# Caregiver users (for dev mode / python-dotenv reading)
# Format: username:bcrypt_hash  — generate hash with step below
CAREGIVER_USERS=test_user:$2b$12$...
```

### Step 3 — Generate a caregiver password hash

```powershell
python -c "import bcrypt; h=bcrypt.hashpw(b'yourpassword', bcrypt.gensalt()).decode(); print(h)"
```

Paste the output (the full `$2b$12$...` string) as the hash in `caregiver/api/.env`:

```env
CAREGIVER_USERS=test_user:$2b$12$<hash>
```

**Docker Compose bcrypt note:** Docker Compose interpolates `$` signs in env values.
Bcrypt hashes contain `$` and will be truncated if passed through Docker Compose variable substitution.
The project handles this via `infrastructure/caregiver_secrets.env`, which stores the same hash
with every `$` doubled to `$$` (Docker's escape sequence for a literal `$`):

```env
# infrastructure/caregiver_secrets.env  — Docker only, NOT read by python-dotenv
CAREGIVER_USERS=test_user:$$2b$$12$$<same-hash-with-all-$-doubled>
```

When you change a password, update **both** files:
- `caregiver/api/.env` — `$` as-is (for dev mode)
- `infrastructure/caregiver_secrets.env` — every `$` doubled to `$$` (for Docker)

### Step 4 — Start PostgreSQL and Redis

Use Docker Compose to start only the databases (preferred over standalone `docker run`):

```powershell
docker-compose -f infrastructure/docker-compose.yml --env-file .env up -d postgres redis
```

Verify they are healthy:

```powershell
docker-compose -f infrastructure/docker-compose.yml ps
```

Both should show `Up (healthy)`.

### Step 5 — Run database migrations

```powershell
# Set DATABASE_URL for the current PowerShell session
$env:DATABASE_URL = "postgresql://falldetect:falldetect@localhost:5432/falldetect"

# Verify it is set (should print the URL, not blank)
echo $env:DATABASE_URL

# Run migrations
alembic upgrade head
```

Expected output:
```
INFO  [alembic.runtime.migration] Running upgrade  -> 0001, Initial schema
```

Verify tables were created:

```powershell
docker exec fall_postgres psql -U falldetect -d falldetect -c "\dt"
```

Should show: `inference_log`, `feature_snapshot`, `participant_session`, `api_request_log`.

### Step 6 — Create Grafana read-only PostgreSQL user (first time only)

```powershell
docker exec fall_postgres psql -U falldetect -d falldetect -c "CREATE USER grafana_ro WITH PASSWORD 'grafana_ro'; GRANT CONNECT ON DATABASE falldetect TO grafana_ro; GRANT USAGE ON SCHEMA public TO grafana_ro; GRANT SELECT ON inference_log, participant_session, feature_snapshot TO grafana_ro;"
```

---

## 3. Option A — Full Stack with Docker Compose

Starts all 10 services: postgres, redis, influxdb, ml_server, caregiver_api, emergency_svc, prometheus, grafana, alertmanager, nginx.

### Important: always pass --env-file

The compose file lives in `infrastructure/`, so Docker Compose will not find the root `.env` automatically.
Always use `--env-file .env` from the project root:

```powershell
# From project root
docker-compose -f infrastructure/docker-compose.yml --env-file .env up --build -d
```

### First-time: run migrations inside the stack

```powershell
docker-compose -f infrastructure/docker-compose.yml exec ml_server alembic upgrade head
```

### Check all containers are healthy

```powershell
docker-compose -f infrastructure/docker-compose.yml ps
```

Expected — all services show `Up` or `Up (healthy)`:

| Container | Port published |
|-----------|---------------|
| fall_postgres | internal only (5432) |
| fall_redis | internal only (6379) |
| fall_influxdb | 8086 |
| fall_ml_server | **8001** (published for main.py) |
| fall_caregiver_api | **8002** (published for health checks) |
| fall_emergency_svc | internal (via nginx) |
| fall_prometheus | 9090 |
| fall_grafana | internal (via nginx at /grafana/) |
| fall_alertmanager | internal |
| fall_nginx | **80**, 443 |

> **Why 8001 and 8002 are published:** `main.py` runs on Windows (outside Docker) and talks directly
> to `ml_server`. In production you would remove these and route everything through nginx.

### Stop the stack

```powershell
docker-compose -f infrastructure/docker-compose.yml down

# To also delete all data volumes (wipes Postgres, InfluxDB, Grafana):
docker-compose -f infrastructure/docker-compose.yml down -v
```

### Restart a single service after a config change

```powershell
docker-compose -f infrastructure/docker-compose.yml --env-file .env up -d ml_server
```

---

## 4. Option B — Development Mode (no Docker for Python services)

Run each Python service in a separate terminal. Requires Postgres and Redis running from Docker (Step 4 above).

### Terminal 1 — ML Server

```powershell
cd C:\Users\hayat\Documents\6G\FallDetection_new
$env:DATABASE_URL = "postgresql://falldetect:falldetect@localhost:5432/falldetect"
$env:REDIS_URL    = "redis://localhost:6379/0"
python system_operator/ml_server/server.py
# http://localhost:8001  |  Docs: http://localhost:8001/docs
```

### Terminal 2 — Caregiver API

```powershell
cd C:\Users\hayat\Documents\6G\FallDetection_new
$env:DATABASE_URL    = "postgresql://falldetect:falldetect@localhost:5432/falldetect"
$env:REDIS_URL       = "redis://localhost:6379/0"
$env:JWT_SECRET_KEY  = "your-jwt-secret"
python -m uvicorn caregiver.api.server:app --host 0.0.0.0 --port 8002 --reload
# http://localhost:8002  |  Docs: http://localhost:8002/docs
```

> The caregiver server reads `CAREGIVER_USERS` from the environment. For dev mode, set it or
> rely on `caregiver/api/.env` being loaded by python-dotenv if configured.

### Terminal 3 — Emergency Notification Service

```powershell
cd C:\Users\hayat\Documents\6G\FallDetection_new
$env:REDIS_URL = "redis://localhost:6379/0"
python -m uvicorn emergency.notification_service.server:app --host 0.0.0.0 --port 8003 --reload
# http://localhost:8003
```

### Terminal 4 — Flask Client (main.py)

```powershell
cd C:\Users\hayat\Documents\6G\FallDetection_new
python main.py
# http://localhost:8000
```

> In dev mode, nginx is not running. Access each service on its port directly.
> Dashboard HTML files can be opened with `python -m http.server 3000` in the dashboard folder
> to avoid CORS errors.

---

## 5. Running the Original Client (main.py)

The root `main.py` is unchanged and backward-compatible.

```powershell
# Local inference (no ml_server needed)
$env:INFERENCE_MODE = "local"
python main.py

# Remote inference (ml_server must be running on port 8001)
$env:INFERENCE_MODE      = "remote"
$env:REMOTE_SERVER_URL   = "http://localhost:8001"
$env:REMOTE_API_KEY      = "your-api-key"
python main.py
```

Flask dashboard: `http://localhost:8000`

> If `.env` is configured at the project root, these variables are loaded automatically
> and you don't need to set them manually.

---

## 6. Verifying Everything Works

### Check ML server health

```powershell
curl.exe http://localhost:8001/health
# Expected: {"status":"ok","model_loaded":true,"model_version":"v0",...}
```

### Check ML server Prometheus metrics

```powershell
curl.exe http://localhost:8001/metrics
# Expected: Prometheus text format — fall_detections_total, inference_latency_seconds, etc.
```

### Send a test prediction

Write the body to a file first (avoids PowerShell quoting issues):

```powershell
'{"acc_x":[100,102,98,105,99,101,103,97,100,102],"acc_y":[200,201,199,200,202,198,200,201,199,200],"acc_z":[16384,16400,16370,16390,16380,16400,16370,16390,16380,16400],"timestamps_ms":[0,40,80,120,160,200,240,280,320,360],"participant":"test_patient"}' | Out-File -Encoding utf8 test.json

curl.exe -X POST http://localhost:8001/predict -H "Content-Type: application/json" -H "X-API-Key: your-api-key" -d "@test.json"
# Expected: {"fall_detected":false,"confidence":...,"model_version":"v0"}
```

### Verify PostgreSQL write

```powershell
docker exec fall_postgres psql -U falldetect -d falldetect -c "SELECT id, participant, fall_detected, confidence, latency_ms FROM inference_log ORDER BY id DESC LIMIT 3;"
```

### Check Caregiver API health

```powershell
curl.exe http://localhost:8002/health
# Expected: {"status":"ok","service":"caregiver_api"}
```

### Login to Caregiver API

```powershell
'{"username":"test_user","password":"mypassword123"}' | Out-File -Encoding utf8 login.json
curl.exe -X POST http://localhost:8002/auth/login -H "Content-Type: application/json" -d "@login.json"
# Expected: {"access_token":"eyJ...","token_type":"bearer","role":"caregiver"}
```

### Verify Redis pub/sub

```powershell
# Terminal 1 — subscribe
docker exec fall_redis redis-cli subscribe fall_events

# Terminal 2 — trigger a prediction (use the test.json from above)
curl.exe -X POST http://localhost:8001/predict -H "Content-Type: application/json" -H "X-API-Key: your-api-key" -d "@test.json"

# Terminal 1 should show the published fall event JSON
```

### Check container environment variables (debug)

```powershell
# PowerShell — use Select-String or findstr, NOT grep
docker exec fall_ml_server env | Select-String "API_KEYS"
docker exec fall_ml_server env | findstr "API_KEYS"
```

---

## 7. Accessing Dashboards

### With Docker Compose (nginx on port 80)

| Dashboard | URL |
|-----------|-----|
| Operator Dashboard | http://localhost/operator/ |
| Care-Giver Dashboard | http://localhost/caregiver/ |
| Emergency Tablet | http://localhost/emergency/ |
| Grafana | http://localhost/grafana/ |
| ML Server API docs | http://localhost/api/ml/docs |
| Caregiver API docs | http://localhost/api/caregiver/docs |
| Prometheus | http://localhost:9090 (direct, not via nginx) |

### Without Docker (development mode)

```powershell
# Serve caregiver dashboard on port 3000
cd caregiver/dashboard
python -m http.server 3000
# Open: http://localhost:3000
```

---

## 8. Grafana Setup (first time)

1. Open Grafana: `http://localhost/grafana/`
2. Login: `admin` / value of `GRAFANA_ADMIN_PASSWORD` in `.env` (default: `admin`)
3. Datasources auto-provisioned from `infrastructure/grafana/provisioning/datasources/datasources.yml`
4. Dashboards auto-provisioned from `infrastructure/grafana/dashboards/`
5. Three dashboards under **Fall Detection** folder:
   - **ML Server Overview** — request rate, p95 latency, error rate, falls/hour
   - **Model Performance** — confidence distribution, drift detection, low-confidence ratio
   - **Fall Events Timeline** — patient fall history from PostgreSQL

If dashboards are empty, send a few test predictions first (Prometheus scrapes every 15s).

If dashboards don't appear after restart:

```powershell
curl.exe -X POST "http://admin:your-grafana-password@localhost:3000/api/admin/provisioning/dashboards/reload"
```

---

## 9. Troubleshooting

### PowerShell "grep is not recognized"

```powershell
# Use these instead:
docker exec fall_ml_server env | Select-String "API_KEYS"
docker exec fall_ml_server env | findstr "API_KEYS"
```

### 403 Forbidden on /predict

The API key sent by the client doesn't match `API_KEYS` in the container. Check what the container actually has:

```powershell
docker exec fall_ml_server env | findstr "API_KEYS"
```

If it shows `changeme` instead of your real key, the `.env` wasn't loaded. Always use `--env-file`:

```powershell
docker-compose -f infrastructure/docker-compose.yml --env-file .env up -d ml_server
```

### CAREGIVER_USERS truncated in container (login always fails)

Docker Compose interpolates `$` in env values. Bcrypt hashes like `$2b$12$...` get truncated at the first `$<letter>` pattern.

The fix is already in place via `infrastructure/caregiver_secrets.env` (uses `$$` escaping).
If you regenerate the hash, update **both**:
- `caregiver/api/.env` — with literal `$` (dev mode)
- `infrastructure/caregiver_secrets.env` — with `$$` (Docker)

### Container name conflict on docker-compose up

```
Error: The container name "/fall_redis" is already in use
```

A standalone `docker run` container with the same name exists. Remove it first:

```powershell
docker rm -f fall_redis fall_postgres
docker-compose -f infrastructure/docker-compose.yml --env-file .env up -d
```

### Alembic: "password authentication failed for user 'user'"

The `DATABASE_URL` env var was not set. In PowerShell, `set` (CMD syntax) does nothing.
Use `$env:` syntax:

```powershell
$env:DATABASE_URL = "postgresql://falldetect:falldetect@localhost:5432/falldetect"
echo $env:DATABASE_URL    # verify it is set before running alembic
alembic upgrade head
```

### Port already in use

```powershell
netstat -ano | findstr :8001
# Find the PID in the last column, then:
Stop-Process -Id <PID> -Force
```

### Service keeps restarting in docker-compose ps

Check the logs:

```powershell
docker logs fall_emergency_svc --tail 30
docker logs fall_alertmanager --tail 30
```

Common causes:
- Missing Python package → add to `infrastructure/Dockerfile.python` and rebuild with `--build`
- Wrong YAML field in alertmanager config → check `infrastructure/alertmanager/alertmanager.yml`
- Missing env var → check container env with `docker exec <name> env | findstr <VAR>`

### Redis connection refused

```powershell
docker exec fall_redis redis-cli ping    # should return PONG
```

If the container is not running:

```powershell
docker-compose -f infrastructure/docker-compose.yml --env-file .env up -d redis
```

### Caregiver dashboard: "Internal Server Error" on login

The bcrypt hash in the container is malformed (truncated `$`). See the CAREGIVER_USERS section above.

### ml_server: "Model not found" on startup

Set `MODEL_VERSION` to a model that has `.pkl` files on disk:

```powershell
ls model/
```

Valid options: `v0`, `v0_lsb_int`, `v3`, `v5_lsb`.

### Caregiver dashboard: API calls return 401

JWT token in `localStorage` has expired (8h default). Click Logout and log in again.

### SSE stream disconnects immediately

nginx is buffering. Verify `proxy_buffering off` is set in
`infrastructure/nginx/nginx.conf` for the `/stream` location block.
