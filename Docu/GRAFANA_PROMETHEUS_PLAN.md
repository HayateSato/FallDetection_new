# Grafana & Prometheus Integration Plan — Fall Detection ML Monitoring

> **Goal:** Get the existing Prometheus metrics and Grafana dashboards wired up and working end-to-end.
> This document covers what is already built, what gaps remain, and the exact steps to complete integration.

---

## 1. What Is Already Built

All the code and config files exist. Nothing needs to be written from scratch.

### Metrics (Python side)

| File | What it does |
|------|-------------|
| `system_operator/ml_server/services/metrics_collector.py` | Defines 3 Prometheus metrics and a `record_prediction()` function |
| `system_operator/ml_server/server.py` | Calls `record_prediction()` on every inference, exposes `/metrics` endpoint via `prometheus_fastapi_instrumentator` |

**Three metrics collected:**

| Metric name | Type | Labels | What it tracks |
|-------------|------|--------|----------------|
| `fall_detections_total` | Counter | `model_version`, `confidence_bucket` (high/medium/low) | Every fall detected, grouped by model and certainty |
| `inference_latency_seconds` | Histogram | — | How long each prediction takes (buckets: 50ms → 5s) |
| `model_confidence` | Histogram | `model_version` | XGBoost confidence score distribution (0.0 → 1.0) |

**Plus automatic HTTP metrics** from `prometheus_fastapi_instrumentator`:
- `http_requests_total` (by endpoint, method, status_code)
- `http_request_duration_seconds`

### Prometheus Config

| File | Purpose |
|------|---------|
| `infrastructure/prometheus/prometheus.yml` | Scrape rules — pulls from ml_server:8001, caregiver_api:8002, emergency_svc:8003 every 15s |
| `infrastructure/prometheus/alert_rules.yml` | 4 production alerts (latency, drift, server down, error rate) |

### Grafana Config

| File | Purpose |
|------|---------|
| `infrastructure/grafana/provisioning/datasources/datasources.yml` | Auto-configures Prometheus + PostgreSQL datasources on startup |
| `infrastructure/grafana/provisioning/dashboards/dashboard.yml` | Auto-loads JSON dashboards from `/var/lib/grafana/dashboards` |
| `infrastructure/grafana/dashboards/ml_server_overview.json` | Real-time server health (requests/min, p95 latency, error rate, falls/hour) |
| `infrastructure/grafana/dashboards/model_performance.json` | Drift detection (confidence distribution, low-confidence ratio, falls by model) |
| `infrastructure/grafana/dashboards/fall_events_timeline.json` | Operational view from PostgreSQL (recent falls table, active sessions) |

### AlertManager Config

| File | Purpose |
|------|---------|
| `infrastructure/alertmanager/alertmanager.yml` | Routing rules — critical vs warning alerts, repeat interval 4h |

---

## 2. What Is Not Yet Done (Gaps)

### Gap 1 — AlertManager notification targets are empty

`infrastructure/alertmanager/alertmanager.yml` has placeholder receivers. No real email, Slack, or webhook is configured. Alerts fire inside Prometheus but go nowhere.

**Action:** Fill in at least one receiver (see Step 5 below).

### Gap 2 — Grafana admin password is not set in `.env`

The docker-compose reads `GRAFANA_ADMIN_PASSWORD` from `.env`. If it is blank, Grafana defaults to `admin/admin` which is a security risk.

**Action:** Add `GRAFANA_ADMIN_PASSWORD=<your password>` to `.env`.

### Gap 3 — PostgreSQL read-only Grafana user does not exist yet

The `fall_events_timeline` dashboard reads from PostgreSQL using a `grafana_ro` user. That user must be created manually after the first migration (Alembic does not create it).

**Action:** Run the SQL command in Step 4 below (one time only).

### Gap 4 — caregiver_api and emergency_svc do not expose `/metrics` yet

`prometheus.yml` tries to scrape `caregiver_api:8002/metrics` and `emergency_svc:8003/metrics` but neither service has `prometheus_fastapi_instrumentator` added. Prometheus will show these targets as DOWN.

**Action:** Add the instrumentator to both servers (see Step 6 below).

### Gap 5 — nginx does not proxy `/metrics` endpoints (by design)

The `/metrics` endpoints are intentionally not exposed through nginx to the public. Prometheus scrapes the services directly over the internal Docker network. This is correct — no action needed, just understand why.

---

## 3. Integration Plan — Step by Step

### Step 1 — Set environment variables

Open `.env` at the project root and make sure these are set:

```env
# Grafana
GRAFANA_ADMIN_PASSWORD=choose_a_strong_password

# Prometheus (already works via docker-compose — no extra vars needed)

# AlertManager (if using email)
ALERTMANAGER_EMAIL_FROM=your@email.com
ALERTMANAGER_EMAIL_TO=oncall@email.com
ALERTMANAGER_SMTP_HOST=smtp.gmail.com:587
```

---

### Step 2 — Start the observability stack

Start only the services you need right now (Postgres, Redis, Prometheus, Grafana):

```bash
docker-compose -f infrastructure/docker-compose.yml up -d postgres redis prometheus grafana alertmanager
```

Verify they are running:

```bash
docker-compose -f infrastructure/docker-compose.yml ps
```

Expected: all 5 show `Up` or `Up (healthy)`.

---

### Step 3 — Run Alembic migrations (first time only)

Creates the PostgreSQL tables that Grafana's `fall_events_timeline` dashboard reads from.

```bash
# Windows
set DATABASE_URL=postgresql://falldetect:falldetect@localhost:5432/falldetect
alembic upgrade head

# Verify tables exist
docker exec fall_postgres psql -U falldetect -d falldetect -c "\dt"
```

Expected output: `inference_log`, `feature_snapshot`, `participant_session`, `api_request_log`.

---

### Step 4 — Create the Grafana read-only PostgreSQL user (first time only)

```bash
docker exec fall_postgres psql -U falldetect -d falldetect -c "
  CREATE USER grafana_ro WITH PASSWORD 'grafana_ro';
  GRANT CONNECT ON DATABASE falldetect TO grafana_ro;
  GRANT USAGE ON SCHEMA public TO grafana_ro;
  GRANT SELECT ON inference_log, participant_session, feature_snapshot TO grafana_ro;
"
```

This user is already referenced in `infrastructure/grafana/provisioning/datasources/datasources.yml`. It just needs to exist in the database.

---

### Step 5 — Configure AlertManager (optional but recommended)

Edit `infrastructure/alertmanager/alertmanager.yml`. Replace the placeholder receivers with real ones.

**Option A — Email (Gmail example):**

```yaml
receivers:
  - name: "default"
    email_configs:
      - to: "your@email.com"
        from: "falldetect-alerts@gmail.com"
        smarthost: "smtp.gmail.com:587"
        auth_username: "falldetect-alerts@gmail.com"
        auth_password: "your_app_password"   # Gmail App Password, not account password
        require_tls: true

  - name: "critical"
    email_configs:
      - to: "oncall@email.com"
        from: "falldetect-alerts@gmail.com"
        smarthost: "smtp.gmail.com:587"
        auth_username: "falldetect-alerts@gmail.com"
        auth_password: "your_app_password"
        require_tls: true
        send_resolved: true
```

**Option B — Slack (incoming webhook):**

```yaml
receivers:
  - name: "default"
    slack_configs:
      - api_url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
        channel: "#fall-alerts"
        title: "Fall Detection Alert"
        text: "{{ .CommonAnnotations.summary }}"
```

After editing, restart AlertManager:

```bash
docker-compose -f infrastructure/docker-compose.yml restart alertmanager
```

---

### Step 6 — Add `/metrics` to caregiver_api and emergency_svc

These two services are missing the Prometheus instrumentation. Add 3 lines to each.

**`caregiver/api/server.py`** — find the FastAPI app initialization and add:

```python
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(...)

# Add this after app is created:
Instrumentator().instrument(app).expose(app)
```

**`emergency/notification_service/server.py`** — same pattern:

```python
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(...)
Instrumentator().instrument(app).expose(app)
```

Also add `prometheus-fastapi-instrumentator` to each service's requirements if not already present.

After this change, Prometheus will show all 3 targets as UP.

---

### Step 7 — Start the ML server and verify metrics flow

```bash
# Terminal 1 — start the ML server (dev mode, not Docker)
cd c:\Users\hayat\Documents\6G\FallDetection_new
set DATABASE_URL=postgresql://falldetect:falldetect@localhost:5432/falldetect
set REDIS_URL=redis://localhost:6379/0
python system_operator/ml_server/server.py
```

Verify the `/metrics` endpoint is working:

```bash
curl http://localhost:8001/metrics
```

You should see lines like:
```
# HELP fall_detections_total Total number of fall detection events
# TYPE fall_detections_total counter
fall_detections_total{confidence_bucket="high",model_version="v3"} 0.0
inference_latency_seconds_bucket{le="0.05"} 1.0
```

---

### Step 8 — Verify Prometheus is scraping

Open Prometheus in browser: `http://localhost:9090`

Go to **Status → Targets**. You should see:

| Job | State | Notes |
|-----|-------|-------|
| ml_server | UP | ✓ Working |
| caregiver_api | UP | Only after Step 6 |
| emergency_svc | UP | Only after Step 6 |
| prometheus | UP | Self-monitoring |

If a target shows `DOWN`, click the error link — common causes:
- Service not running
- Wrong port in `prometheus.yml`
- `/metrics` endpoint not wired up (Step 6)

Run a test query in Prometheus to confirm data arrives:

```
fall_detections_total
inference_latency_seconds_count
http_requests_total
```

---

### Step 9 — Verify Grafana dashboards

Open Grafana: `http://localhost:3000`

Login: `admin` / `<GRAFANA_ADMIN_PASSWORD from .env>`

Go to **Dashboards → Fall Detection**. Three dashboards should be pre-loaded automatically (from provisioning):

1. **ML Server Overview** — shows server status, request rate, p95 latency, error rate
2. **Model Performance** — shows confidence distribution, falls per model, drift indicators
3. **Fall Events Timeline** — shows recent fall events from PostgreSQL

**If dashboards are empty (no data):**
- Make a test request to the ML server: `curl -X POST http://localhost:8001/predict -H "X-API-Key: <your key>" -H "Content-Type: application/json" -d '{"test": true}'`
- Wait 30 seconds for Prometheus to scrape
- Refresh Grafana

**If datasources show errors:**
- Check Grafana → Configuration → Data Sources
- Test each datasource connection
- For PostgreSQL: verify `grafana_ro` user was created (Step 4)

---

### Step 10 — Verify alerts fire correctly

Test the `MlServerDown` alert by stopping the ML server:

```bash
# Stop the ML server (Ctrl+C or kill the process)
# Wait 1 minute — alert should fire in Prometheus
```

In Prometheus, go to **Alerts** — `MlServerDown` should show `FIRING` in red after 1 minute.

In AlertManager (`http://localhost:9093`) — the alert should appear in the active alerts list.

Restart the ML server and verify the alert resolves.

---

## 4. Data Flow Reference

```
┌──────────────────────────────────────────────────────────────────┐
│  ML Server (FastAPI :8001)                                       │
│                                                                  │
│  Every /predict request:                                         │
│    1. XGBoost inference → {fall_detected, confidence}           │
│    2. record_prediction() → updates Prometheus counters/histos  │
│    3. db_writer.py (background) → INSERT INTO inference_log     │
│    4. redis.publish("fall_events", payload)                     │
│    5. return JSON response                                       │
│                                                                  │
│  GET /metrics → Prometheus scrapes this every 15s               │
└──────────────────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
┌─────────────────┐       ┌──────────────────────┐
│  Prometheus     │       │  PostgreSQL           │
│  (9090)         │       │  inference_log table  │
│                 │       └──────────┬────────────┘
│  Evaluates      │                  │
│  alert rules    │                  │
│  every 15s      │       ┌──────────▼────────────┐
└────────┬────────┘       │  Grafana (3000)        │
         │                │  Datasource A:         │
         ▼                │    Prometheus metrics  │
┌─────────────────┐       │  Datasource B:         │
│  AlertManager   │       │    PostgreSQL history  │
│  (9093)         │       │                        │
│  Sends email /  │       │  3 Dashboards          │
│  Slack / webhook│       │  auto-provisioned      │
└─────────────────┘       └────────────────────────┘
```

---

## 5. Alert Rules Reference

Defined in `infrastructure/prometheus/alert_rules.yml`:

| Alert | Fires when | Severity | Why it matters |
|-------|-----------|----------|----------------|
| `HighInferenceLatency` | p95 latency > 2s for 2 minutes | warning | Falls may be detected too late; check CPU or DB connection |
| `ConfidenceDrift` | Median confidence on fall predictions < 0.6 for 30 minutes | warning | Model is uncertain — sensor data may have changed (new patient, different hardware) |
| `MlServerDown` | ML server unreachable for 1 minute | critical | No fall detection happening at all |
| `HighErrorRate` | >5% of requests returning 5xx for 5 minutes | warning | Model or dependency (Postgres, Redis) is failing |

---

## 6. Troubleshooting Quick Reference

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Prometheus target shows DOWN | Service not running or wrong port | Check service is up; verify port in prometheus.yml |
| Grafana dashboard empty | No data in Prometheus yet | Make a test inference request; wait 30s |
| `fall_events_timeline` shows DB error | `grafana_ro` user missing | Run Step 4 SQL |
| AlertManager not sending emails | SMTP config missing or wrong | Fill in alertmanager.yml receivers (Step 5) |
| `/metrics` returns 404 on caregiver/emergency | `Instrumentator` not added | Apply Step 6 code change |
| Grafana login fails | Wrong password | Check `GRAFANA_ADMIN_PASSWORD` in `.env` |
| Docker container exits immediately | Missing env var | Run `docker logs <container_name>` |

---

## 7. Remaining Work Summary

| Task | Status | Effort |
|------|--------|--------|
| Metrics collector + ML server wiring | Done | — |
| Prometheus config + alert rules | Done | — |
| Grafana dashboards (3 JSON files) | Done | — |
| Grafana provisioning (auto-load) | Done | — |
| Docker Compose (all 10 services) | Done | — |
| Nginx SSE support | Done | — |
| AlertManager receivers (email/Slack) | **Not done** | 15 min |
| `grafana_ro` PostgreSQL user | **Not done** | 2 min (one SQL command) |
| `/metrics` on caregiver_api | **Not done** | 5 min (3 lines of code) |
| `/metrics` on emergency_svc | **Not done** | 5 min (3 lines of code) |
| `GRAFANA_ADMIN_PASSWORD` in `.env` | **Not done** | 1 min |
