# Project Status — Multi-User System Refactoring

Summary of what was implemented in the refactoring and what still needs to be built.
Last updated: 2026-03-23

---

## What Was Done

### Folder Structure

The project was reorganised from a flat 2-machine setup into per-user-type folders:

```
patient/              Reference docs — no code
caregiver/            Care-giver API + dashboard
system_operator/      ML server + client + operator dashboard
emergency/            Notification service + tablet UI
shared/               DB models, auth, Redis (used by all services)
infrastructure/       Docker Compose, Prometheus, Grafana, nginx
```

Root `main.py` and `server.py` are unchanged and still work as before.

---

### Component Status

#### shared/ — Foundation Layer

| File | Status | Notes |
|------|--------|-------|
| `shared/db/models.py` | Done | 4 SQLAlchemy ORM tables: inference_log, feature_snapshot, participant_session, api_request_log |
| `shared/db/session.py` | Done | SessionLocal factory + get_db() FastAPI dependency |
| `shared/db/migrations/` | Done | Alembic setup + initial migration `0001_initial_schema.py`. Run: `alembic upgrade head` |
| `shared/schemas/` | Done | Pydantic schemas for inference, patient, fall events |
| `shared/auth/jwt_utils.py` | Done | JWT create/verify (HS256), bcrypt password hashing, `require_role()` FastAPI dependency. Fixed 2026-03-23: uses `import bcrypt` directly (passlib 4.x incompatibility removed) |
| `shared/auth/api_key_utils.py` | Done | SHA-256 hashing for API key audit logs |
| `shared/redis_client.py` | Done | Sync + async Redis helpers, `subscribe_fall_events()` async generator |
| `alembic.ini` | Done | Migration config at project root |

#### system_operator/ml_server/ — Inference Server

| File | Status | Notes |
|------|--------|-------|
| `server.py` | Done | Full FastAPI: inference + Prometheus + BackgroundTask DB write + Redis publish + model hot-swap |
| `services/db_writer.py` | Done | Postgres write wrapped in try/except, never blocks inference |
| `services/metrics_collector.py` | Done | 3 Prometheus metrics: fall_detections_total, inference_latency_seconds, model_confidence |
| `.env.example` | Done | Template with all required env vars |
| `requirements.txt` | Done | Extends requirements.server.txt + adds new packages |

#### system_operator/operator_dashboard/ — Operator UI

| File | Status | Notes |
|------|--------|-------|
| `index.html` | Done | 4 panels: model info, model switcher, live health, Grafana links |
| `app.js` | Done | Calls /model/info, /model/list, /model/switch, /health |
| `style.css` | Done | Dark theme matching operator context |

#### caregiver/ — Care-Giver Portal

| File | Status | Notes |
|------|--------|-------|
| `api/server.py` | Done | FastAPI: /auth/login, /patients, /patients/{name}/falls, /patients/stream (SSE), /stats/summary |
| `dashboard/index.html` | Done | Patient list, live alert banner, detail view |
| `dashboard/app.js` | Done | JWT auth flow, SSE consumer, patient list + fall history |
| `dashboard/style.css` | Done | |
| `api/.env.example` | Done | |
| `api/requirements.txt` | Done | |

#### emergency/ — Emergency Notifications

| File | Status | Notes |
|------|--------|-------|
| `notification_service/server.py` | Done | FastAPI SSE fan-out, subscribes to Redis, 30s keep-alive pings |
| `notification_service/channels/sse.py` | Done | ConnectionManager: asyncio.Queue per tablet client |
| `notification_service/channels/webhook.py` | Done | Optional HTTP webhook (httpx POST) |
| `tablet_ui/index.html` | Done | Standby + alert screens |
| `tablet_ui/app.js` | Done | EventSource SSE consumer, browser notification support |
| `tablet_ui/style.css` | Done | Full-screen flashing alert, dark green standby |

#### infrastructure/ — Deployment

| File | Status | Notes |
|------|--------|-------|
| `docker-compose.yml` | Done | 10 services: postgres, redis, influxdb, ml_server, caregiver_api, emergency_svc, prometheus, grafana, alertmanager, nginx |
| `Dockerfile.python` | Done | Base image for all Python services |
| `nginx/nginx.conf` | Done | Reverse proxy + `proxy_buffering off` for SSE |
| `prometheus/prometheus.yml` | Done | Scrapes ml_server, caregiver_api, emergency_svc |
| `prometheus/alert_rules.yml` | Done | 4 rules: HighInferenceLatency, ConfidenceDrift, MlServerDown, HighErrorRate |
| `alertmanager/alertmanager.yml` | Done | Template — needs real email/webhook URLs |
| `grafana/provisioning/` | Done | Auto-provisions Prometheus + PostgreSQL datasources |
| `grafana/dashboards/ml_server_overview.json` | Done | Request rate, latency gauge, error rate, falls count |
| `grafana/dashboards/model_performance.json` | Done | Confidence distribution, drift detection, low-confidence ratio |
| `grafana/dashboards/fall_events_timeline.json` | Done | Patient fall table, confidence scatter, sessions (Postgres datasource) |

---

## What Still Needs To Be Implemented

These are real gaps — either not yet written or known to need work before production use.

### High Priority (system won't work correctly without these)

#### 1. Authentication on caregiver GET endpoints
**File:** `caregiver/api/server.py`

The `/patients`, `/patients/{name}/falls`, and `/stats/summary` endpoints currently have **no JWT auth applied**. Anyone who knows the URL can access patient data.

**Fix:** Add `Depends(require_role("caregiver"))` to each route:
```python
from shared.auth.jwt_utils import require_role

@app.get("/patients", dependencies=[Depends(require_role("caregiver"))])
async def list_patients(db: Session = Depends(get_db)):
    ...
```

#### 2. SSE authentication
**File:** `caregiver/dashboard/app.js`, `caregiver/api/server.py`

Browser `EventSource` does **not support custom headers** — you cannot send a JWT `Authorization: Bearer` header with it. The `/patients/stream` SSE endpoint is currently unprotected.

**Options:**
- **Token in query param**: `new EventSource("/api/caregiver/patients/stream?token=<jwt>")` — validate in the endpoint
- **Session cookie**: Set a `HttpOnly` cookie on login, read it in the SSE endpoint
- **Short-lived SSE token**: Add `GET /auth/sse-token` that returns a one-use token valid for 60s

#### 3. participant_session table is never written to
**File:** `app/routes/recording.py` or `system_operator/ml_server/server.py`

The `participant_session` table was designed but nothing in the codebase creates or updates rows in it. The care-giver dashboard `/patients` endpoint reads from it and will always return an empty list.

**Fix:** When the Flask client starts a recording (`POST /recording/state`), write a row to `participant_session`. When recording stops, set `end_time`. When a fall is logged in `db_writer.py`, increment `fall_count`.

#### 4. ~~Environment files not filled in~~ — **DONE** (2026-03-23)
Root `.env` is filled in with real values (DATABASE_URL, REDIS_URL, API_KEYS, JWT_SECRET_KEY, InfluxDB credentials). Only `CAREGIVER_USERS` remains empty — generate with:
```
python -c "from shared.auth.jwt_utils import hash_password; print(hash_password('mypassword'))"
```
Then set: `CAREGIVER_USERS=alice:<hash>` in `.env`.

---

### Medium Priority (degraded functionality without these)

#### 5. Grafana read-only Postgres user not created
The `infrastructure/grafana/provisioning/datasources/datasources.yml` configures a `grafana_ro` user, but this user doesn't exist yet in Postgres. The "Fall Events Timeline" dashboard will fail.

**Fix:** Run once after `alembic upgrade head`:
```sql
CREATE USER grafana_ro WITH PASSWORD 'grafana_ro';
GRANT SELECT ON inference_log, participant_session TO grafana_ro;
```

#### 6. AlertManager has no real notification target
`infrastructure/alertmanager/alertmanager.yml` has placeholder receivers. Alerts fire in Prometheus but go nowhere.

**Fix:** Add `webhook_configs` or `email_configs` to the receiver blocks:
```yaml
receivers:
  - name: "default"
    email_configs:
      - to: "operator@example.com"
        from: "alerts@example.com"
        smarthost: "smtp.example.com:587"
```

#### 7. `system_operator/client/static/index.html` is empty
The client folder has no `index.html`. The operator client's dashboard is still served from `app/static/index.html` in the original `main.py`.

**Fix:** Either symlink or copy `app/static/index.html` to `system_operator/client/static/`.

#### 8. ~~Missing `__init__.py` files~~ — **DONE** (2026-03-23)
All `__init__.py` files are present: `system_operator/`, `system_operator/client/`, `emergency/notification_service/channels/`, `caregiver/api/`.

---

### Low Priority (nice to have / future features)

#### 9. Automated retraining pipeline
From the original CLAUDE.md next steps. Uses `feature_snapshot` table as training data.

**Files to create:** `retrain.py` at project root:
1. Query confirmed falls from `inference_log`
2. Fetch feature vectors from `feature_snapshot`
3. Retrain XGBoost
4. Save new model to `model/` with version bump

#### 10. Model versioning via `/predict?model=v3`
Currently the model is set at startup via `MODEL_VERSION` env var. Serving different versions per-request via query param is listed in CLAUDE.md but not implemented.

#### 11. `/inferences` endpoint in caregiver API
The operator dashboard `app.js` references a `/inferences` endpoint that doesn't exist. Currently replaced with a Grafana link. Would be useful for the operator log panel.

#### 12. Operator dashboard JWT login page
The operator dashboard prompts for an API key via `window.prompt()` — functional but not polished. A proper login page with JWT (similar to the caregiver dashboard) would be better.

#### 13. Docker health checks for Python services
The `Dockerfile.python` doesn't install `curl`. The `healthcheck` in `docker-compose.yml` uses `curl`, which will fail inside the container.

**Fix:** Add to `Dockerfile.python`:
```dockerfile
RUN apt-get install -y curl
```
Or change healthcheck to use Python:
```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"]
```

#### 14. Tests
No unit or integration tests exist for the new components (`shared/`, `caregiver/`, `emergency/`, `system_operator/ml_server/`). The original `app/` code also has no tests.

---

## Quick Start Checklist

Before running for the first time, check these off:

- [x] Create `.env` files — root `.env` filled in
- [x] `__init__.py` in all service directories
- [x] bcrypt password hashing fixed (direct `import bcrypt`)
- [ ] Set `CAREGIVER_USERS=alice:<bcrypt_hash>` in `.env`
- [ ] Start Postgres + Redis (Docker or local)
- [ ] Run `alembic upgrade head`
- [ ] Create `grafana_ro` Postgres user (for Grafana dashboards)
- [ ] Apply auth to caregiver GET endpoints (item 1 above)
- [ ] Wire `participant_session` writes from recording endpoints (item 3 above)
