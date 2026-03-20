## What the plan covers

**New folder structure** — one top-level folder per user type:

- `patient/` — reference docs only (wearable setup)
- `caregiver/` — FastAPI REST API + HTML/JS dashboard (patient list + live fall alerts)
- `system_operator/` — refactored ML server, Flask client, operator dashboard
- `emergency/` — SSE notification service + tablet UI
- `shared/` — PostgreSQL models, JWT auth, Redis client (shared by all services)
- `infrastructure/` — Docker Compose (10 services), nginx, Prometheus, Grafana, AlertManager

**Existing code stays intact** — `app/`, `config/`, `model/` are untouched. Root `server.py` and `main.py` become thin wrappers so nothing breaks.

**6 implementation phases** (ordered for maximum value):

1. PostgreSQL — write inference results after every prediction (background task, non-blocking)
2. Grafana + Prometheus — 3 dashboards: server health, model confidence drift, fall timeline
3. Caregiver dashboard — patient list + real-time SSE fall alerts via Redis pub/sub
4. Emergency notification — SSE broadcast to tablets via Redis
5. Operator dashboard — model hot-swap UI + Grafana embed
6. nginx — full Docker Compose, TLS, SSE proxy

**Industry-standard components added (vs your sketch):**

| Component | Reason |
| --- | --- |
| **Redis pub/sub** | Decouples ML server from caregiver/emergency — if downstream restarts, events aren't lost |
| **Alembic** | Schema migrations for Postgres without dropping tables |
| **prometheus_fastapi_instrumentator** | 3-line setup auto-instruments all FastAPI routes |
| **AlertManager** | Deduplication + routing of Prometheus alerts to email/webhook |
| **nginx** | TLS termination, `proxy_buffering off` for SSE, blocks direct DB exposure |
| **JWT + bcrypt** | Role-based auth for operator vs caregiver dashboards |
| **BackgroundTasks** | FastAPI built-in — DB + Redis writes after response sent, zero latency cost |

**Frontend**: Vanilla HTML/JS throughout (consistent with existing `app/static/index.html`) — no React/Node build toolchain needed.

**Claude’s Plan** C:\Users\hayat\.claude\plans\resilient-kindling-summit.md


**Update Todos**

- [ ]  Create new folder structure (patient/, caregiver/, system_operator/, emergency/, shared/, infrastructure/)
- [ ]  Phase 1: Write shared/db/models.py and shared/db/session.py (PostgreSQL ORM)
- [ ]  Phase 1: Write system_operator/ml_server/services/db_writer.py
- [ ]  Phase 1: Set up Alembic migrations
- [ ]  Phase 1: Add PostgreSQL to infrastructure/docker-compose.yml
- [ ]  Phase 1: Integrate db_writer into system_operator/ml_server/server.py
- [ ]  Phase 1: Add DATABASE_URL + REDIS_URL + JWT_SECRET_KEY to config/settings.py
- [ ]  Phase 2: Write metrics_collector.py + wire Prometheus into ml_server
- [ ]  Phase 2: Write infrastructure/prometheus/ and infrastructure/grafana/ configs + 3 dashboards
- [ ]  Phase 2: Add Prometheus + Grafana + AlertManager to docker-compose
- [ ]  Phase 3: Write shared/redis_client.py + shared/auth/jwt_utils.py
- [ ]  Phase 3: Write caregiver/api/server.py (FastAPI REST API)
- [ ]  Phase 3: Write caregiver/dashboard/ (HTML/JS frontend)
- [ ]  Phase 4: Write emergency/notification_service/ + emergency/tablet_ui/
- [ ]  Phase 5: Add model switching routes + write operator_dashboard/
- [ ]  Phase 6: Write infrastructure/nginx/nginx.conf + finalize docker-compose.yml
- [ ]  Write patient/README.md
- [ ]  Update CLAUDE.md with new architecture

---
Everything is implemented. Here's a summary of what was built:

---

## What was created

**New folder structure** with per-user separation:

`patient/              README.md — wearable setup, InfluxDB fields
caregiver/
  api/server.py       FastAPI: /patients, /falls, /stream (SSE), /auth/login
  dashboard/          HTML/JS: patient list, live fall alert banner, fall history
system_operator/
  ml_server/server.py FastAPI: all original + Postgres write, Redis publish, model hot-swap
  ml_server/services/ db_writer.py + metrics_collector.py
  operator_dashboard/ HTML/JS: model switcher, health panel, Grafana links
  client/main.py      Thin wrapper around root main.py
emergency/
  notification_service/server.py  SSE broadcast from Redis → tablets + webhook
  tablet_ui/          Full-screen flashing fall alert, auto-reconnect EventSource
shared/
  db/models.py + session.py + migrations/  SQLAlchemy ORM, Alembic
  redis_client.py     subscribe_fall_events() async generator
  auth/jwt_utils.py   JWT create/verify, bcrypt, require_role() FastAPI dependency
infrastructure/
  docker-compose.yml  10 services: postgres, redis, influxdb, ml_server, caregiver_api,
                      emergency_svc, prometheus, grafana, alertmanager, nginx
  prometheus/         Scrape config + 4 alert rules (latency, drift, downtime, error rate)
  grafana/dashboards/ 3 JSON dashboards (server overview, model performance, fall timeline)
  nginx/nginx.conf    Reverse proxy with proxy_buffering off for SSE
alembic.ini           Migration config (run: alembic upgrade head)`

**Root `server.py` and `main.py` are untouched** — still work as before.

**Key industry additions** over your original sketch:

- **Redis pub/sub** — decouples ml_server from caregiver/emergency
- **Alembic** — schema migrations without dropping tables
- **prometheus_fastapi_instrumentator** — 3-line auto-instrumentation
- **AlertManager** — alert deduplication + routing (latency, drift, downtime)
- **JWT + bcrypt** — role-based auth for care-giver dashboard login
- **BackgroundTasks** — DB/Redis writes happen after inference response is sent (zero latency cost)
- **SSE keep-alive pings** — prevents nginx from closing idle tablet connections



Clean. Here's what moved and why:

| Moved | Reason |
| --- | --- |
| `API_helpers/` | ngrok/Cloudflare tunnel scripts — superseded by nginx in `infrastructure/docker-compose.yml` |
| `docker-compose.yml` | Old InfluxDB-only compose — superseded by `infrastructure/docker-compose.yml` (10 services) |


The root is now just the essentials: entry points (`main.py`, `server.py`), the four user folders, shared code, infrastructure, and config.