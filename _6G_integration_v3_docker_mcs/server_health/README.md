# server_health — admin status dashboard

Plain-language traffic-light view of the 6 backing services. Designed for a
clinical admin who needs to know "is anything obviously broken right now?" —
not for SREs who want metrics over time (use Grafana for that).

| | |
|---|---|
| Port | `8006` |
| Run | `python -m server_health.main` |
| Audience | **admin only** (warning banner shown — auth gate deferred per todo 11.5.4) |
| Production target | runs in `mcs-fall-detection` namespace, exposed at `/admin/health` behind ingress |

---

## What it does

```
6 services probed in parallel via asyncio.gather
   │
   ▼
GET /api/status returns:
   { "overall": "healthy" | "degraded" | "down",
     "services": [
       { "name": "inference_server", "status": "healthy", "url": "...", "details": "...", "latency_ms": 12 },
       ...
     ]
   }
   │
   ▼
HTML page renders a traffic-light banner + per-service cards.
Auto-refreshes every 30s.
```

Aggregation rule for `overall`:
- **healthy** — every service is healthy
- **degraded** — no service is "down" but at least one is "degraded" (e.g. file-based MLflow instead of a tracking server)
- **down** — at least one service did not respond or returned an error

---

## Probes

Each probe has a 3-second timeout. Failures translate to `"down"` — they never raise.

| Service | Probe | What "healthy" looks like |
|---------|-------|--------------------------|
| `inference_server` | `GET /health` | parses model_version + uptime |
| `fall_dashboard` | `GET /api/patients` | counts registered patients |
| `postgres` | SQL `SELECT 1` via SQLAlchemy | round-trip succeeds |
| `mqtt_broker` | TCP socket connect to port 1883 | port reachable |
| `mlflow` | `GET /health` (only if HTTP URL — file-based MLflow shows "degraded") | 200 response |
| `minio` | `GET /minio/health/live` | 200 response |

---

## REST API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | dashboard HTML (banner + service cards + help section) |
| GET | `/api/status` | JSON status payload (consumed by the page; also useful for external monitors) |

---

## Files

```
server_health/
├── main.py              ← uvicorn entry + banner + signal handling
├── web.py               ← FastAPI app: /api/status + static mount
├── checks.py            ← per-service async probe functions + run_all()
├── dashboard/
│   ├── index.html       ← banner + cards + help section
│   ├── app.js           ← /api/status fetch + 30s auto-refresh + in-place rendering
│   └── style.css        ← dark theme matching fall_dashboard / ml_dashboard
├── Dockerfile
└── requirements.txt
```

---

## Configuration

| Var | Default | Purpose |
|-----|---------|---------|
| `SERVER_HEALTH_HOST` | `0.0.0.0` | bind address |
| `SERVER_HEALTH_PORT` | `8006` | HTTP port |
| `INFERENCE_SERVER_URL` | `http://localhost:8001` | probe target |
| `FALL_DASHBOARD_URL` | `http://localhost:8002` | probe target |
| `DATABASE_URL` | (required) | Postgres connection string |
| `MQTT_BROKER_HOST` + `MQTT_BROKER_PORT` | `localhost:1883` | TCP probe target |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | probe target (or sqlite for degraded) |
| `MLFLOW_S3_ENDPOINT_URL` | `http://localhost:9000` | MinIO probe target |

---

## What this dashboard does NOT show

| Question | Right tool |
|----------|-----------|
| Latency / throughput / error rate | Grafana — `ml_server_overview` |
| Model confidence drift / fall rate over time | Grafana — `model_performance` |
| Application logs | `kubectl logs` in production, terminal output locally |
| Pod CPU / memory | `kubectl top pods` or Grafana |
| Recent fall events | fall_dashboard `/api/falls` or the caregiver web app |

The page itself surfaces this list under "What does each status mean?" so an
admin opening it for the first time understands the boundaries.

---

## Production notes

- Auth gate not yet enforced — see todo.md 11.5.4 (shared with ml_dashboard)
- Auto-refresh every 30s is appropriate here — probes are cheap (HTTP + a SQL `SELECT 1`)
  and the whole point of the page is "live status"
- This dashboard is a *consumer*, not a *source* — failures here mean other
  services are down, not server_health itself
