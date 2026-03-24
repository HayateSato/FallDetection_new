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

### Python services

| Container | Service | Internal port | Exposed to host | Exposed via nginx |
|-----------|---------|:---:|:---:|:---:|
| `fall_ml_server` | XGBoost inference (FastAPI) | 8001 | `localhost:8001` | `localhost/api/ml/` |
| `fall_caregiver_api` | Caregiver REST API (FastAPI) | 8002 | `localhost:8002` | `localhost/api/caregiver/` |
| `fall_emergency_svc` | Emergency SSE fan-out (FastAPI) | 8003 | — | `localhost/api/emergency/` |

> `fall_ml_server` and `fall_caregiver_api` are published directly to the host so that
> `main.py` (running on Windows, outside Docker) and health-check scripts can reach them
> without going through nginx. In a production deployment these ports would be closed
> and all traffic routed through nginx only.

### Infrastructure

| Container | Service | Internal port | Exposed to host |
|-----------|---------|:---:|:---:|
| `fall_postgres` | PostgreSQL 16 | 5432 | — (internal only) |
| `fall_redis` | Redis 7 | 6379 | — (internal only) |
| `fall_influxdb` | InfluxDB 2.7 | 8086 | `localhost:8086` |
| `fall_nginx` | Reverse proxy | 80, 443 | `localhost:80` / `localhost:443` |

### Observability

| Container | Service | Internal port | Exposed to host |
|-----------|---------|:---:|:---:|
| `fall_prometheus` | Prometheus | 9090 | `localhost:9090` (direct, not via nginx) |
| `fall_grafana` | Grafana | 3000 | via nginx at `localhost/grafana/` |
| `fall_alertmanager` | AlertManager | 9093 | — (internal only) |

---

## Access URLs (Docker Compose running)

| What | URL | Auth |
|------|-----|------|
| Operator dashboard | http://localhost/operator/ | none |
| Caregiver dashboard | http://localhost/caregiver/ | username + password |
| Emergency tablet UI | http://localhost/emergency/ | none |
| Grafana | http://localhost/grafana/ | `admin` / `GRAFANA_ADMIN_PASSWORD` |
| Prometheus | http://localhost:9090 | none |
| ML server API docs | http://localhost:8001/docs | none |
| Caregiver API docs | http://localhost:8002/docs | none |
| ML server health | http://localhost:8001/health | none |
| Caregiver health | http://localhost:8002/health | none |

---

## Project Structure

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
├── app/                            Core ML pipeline (shared by main.py and ml_server)
├── model/                          XGBoost .pkl files (v0, v3, v5_lsb, ...)
├── config/                         Settings and hardware profiles
└── Docu/                           Documentation
    ├── HOW_TO_RUN.md               Full setup and run guide (start here)
    ├── STATUS.md                   What is done vs. still to implement
    ├── GRAFANA_PROMETHEUS_PLAN.md  Monitoring integration plan
    └── REDIS.md                    Redis architecture notes
```
