# Fall Detection — MCS Inference Layer (Docker)

Production Docker Compose stack for the MCS / Hetzner server.
Run this on the MCS machine (Laptop 2 in local two-laptop tests, Hetzner in production).

The caregiver layer (MQTT broker + fall-dashboard) runs separately on a different machine
(`_6G_integration_v3_k3s/` for K3s or `_6G_integration_v3_docker_focus/` for Docker).
server-health probes that machine over the LAN — set `FALL_DASHBOARD_URL` and
`MQTT_BROKER_HOST` in `.env` to point at it.

8 long-running services + 2 one-off init jobs.

---

## Services

| Service | Port | Purpose |
|---|---|---|
| inference-server | 8001 | Fall detection API — /predict + /confirm (public, behind reverse proxy) |
| ml-dashboard | 8004 | Admin UI — retrain + model hot-swap (internal) |
| server-health | 8006 | Aggregate health probe dashboard (internal) |
| postgres (mcs_fall_postgres) | 5432 | inference_log + feature_snapshot + MLflow tracking DB |
| mlflow | 5000 | ML experiment tracking + model registry (internal) |
| minio | 9000 | Model artifact store — S3-compatible (internal) |
| prometheus | 9090 | Metrics scraping (internal) |
| grafana | 3000 | ML dashboards (internal) |
| db-migrate | — | One-off: runs Alembic migrations (0001-0005), then exits |
| minio-setup | — | One-off: creates mlflow-artifacts bucket, then exits |

---

## Structure

```
_6G_integration_v3_docker_mcs/
  docker-compose.yml        <- start here
  .env.example              <- copy to .env and fill in
  prometheus.yml            <- Prometheus scrape config
  alembic.ini               <- Alembic config (used by db-migrate container)
  inference_server/         <- /predict + /confirm FastAPI service + Dockerfile
  ml_dashboard/             <- admin UI: retrain + model hot-swap + Dockerfile
  server_health/            <- health probe dashboard + Dockerfile
  shared_db/                <- SQLAlchemy ORM + Alembic migrations (0001-0005)
  ml_pipeline/              <- signal processing + XGBoost inference engine
  config/                   <- hardware config
  model/                    <- XGBoost model files (v0, v3, v5_lsb, ...)
  retrain/                  <- MLflow retraining pipeline (spawned by ml_dashboard)
  infrastructure/
    postgres/init.sql       <- creates fall_detection + mlflow databases
    mlflow/Dockerfile       <- pinned MLflow + psycopg2 + boto3
    grafana/
      provisioning/         <- auto-provisioned datasources + dashboard config
      dashboards/           <- Grafana dashboard JSON files
```

---

## Quick start

### 1. Configure .env

```powershell
copy .env.example .env
```

Open `.env` and set at minimum:
- `POSTGRES_PASSWORD` — strong password for Postgres
- `API_KEYS` — comma-separated API keys for the inference server
- `GF_SECURITY_ADMIN_PASSWORD` — Grafana admin password

For server-health to probe the caregiver layer (K3s NodePorts on the other machine):
- `FALL_DASHBOARD_URL=http://<CAREGIVER_IP>:30802` — fall-dashboard NodePort
- `MQTT_BROKER_HOST=<CAREGIVER_IP>` — same machine as fall-dashboard
- `MQTT_BROKER_PORT=30901` — MQTT WebSocket NodePort (port 1883 is K3s-internal only)

### 2. Start

Run from `_6G_integration_v3_docker_mcs/` as working directory:

```powershell
docker compose up -d
```

db-migrate runs first and applies all Alembic migrations automatically before
inference-server starts.

### 3. Verify

```powershell
docker compose ps
curl.exe http://localhost:8001/health       # inference-server
curl.exe http://localhost:8006/api/status   # server-health (returns JSON for all 6 probes)
# Grafana:    http://localhost:3000  (admin / your password)
# MLflow:     http://localhost:5000
# MinIO:      http://localhost:9002  (minioadmin / minioadmin)
# ml-dashboard: http://localhost:8004
```

---

## Useful commands

Run all commands from `_6G_integration_v3_docker_mcs/` as working directory.

```powershell
# Logs
docker compose logs -f inference-server
docker compose logs -f ml-dashboard
docker compose logs -f server-health
docker compose logs -f mlflow
docker compose logs -f postgres

# Rebuild after code changes
docker compose build --no-cache inference-server
docker compose up -d inference-server

# Run Alembic migrations manually
docker exec fall_inference_server alembic upgrade head

# Postgres shell
docker exec -it mcs_fall_postgres psql -U fall_user -d fall_detection

# Stop (volumes preserved)
docker compose down

# Full reset including volumes (DELETES ALL DATA)
docker compose down -v
```

---

## Notes

- **inference-server runs with `--workers 1`** — Prometheus counters and asyncio timers
  are per-process; multiple workers would give incorrect metrics.
- **Grafana admin password** set via env var is ignored after first startup.
  Reset it with: `docker exec fall_grafana grafana cli admin reset-admin-password <new>`
- **server-health probes the caregiver layer over the LAN.** It needs `FALL_DASHBOARD_URL`,
  `MQTT_BROKER_HOST`, and `MQTT_BROKER_PORT` in `.env` to reach the K3s NodePorts on the
  caregiver machine. Without them, those two probe cards will show "down" (defaults to
  `localhost` which fails inside the container). Use port `30802` for fall-dashboard and
  `30901` for MQTT — port 1883 is K3s-internal only and not reachable externally.
  **In production (Hetzner → FOCUS server), the fall-dashboard and mqtt-broker probes will
  likely show "down" unless a VPN tunnel (e.g. WireGuard) is established between the Hetzner
  server and the FOCUS network. Without it, Hetzner cannot reach the FOCUS K3s NodePorts.
  This is acceptable — the MCS-side probes (inference-server, postgres, mlflow, minio) will
  still be healthy. The caregiver-layer cards can be ignored or left unconfigured in the
  Hetzner `.env` until a VPN is in place.**
- **server-health "View logs" commands** show `docker logs <container>` for services in
  this stack and `kubectl logs` for fall-dashboard and mqtt-broker (K3s on the other machine).
- **FOCUS caregiver services** (MQTT broker + fall-dashboard) run separately:
  `_6G_integration_v3_k3s/` (K3s, production target) or
  `_6G_integration_v3_docker_focus/` (Docker Compose, local testing).

---

## Related

- `_6G_integration_v3_docker_focus/` — FOCUS caregiver layer (Docker Compose, 2nd laptop)
- `_6G_integration_v3_k3s/` — FOCUS caregiver layer (k3s Helm chart, production)
