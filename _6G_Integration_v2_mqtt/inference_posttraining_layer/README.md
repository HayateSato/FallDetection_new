# Fall Detection â€” Docker Deployment Guide (Hetzner / MCS)

Production deployment for the MCS-hosted inference and ML observability stack.
FOCUS hosts only their InfluxDB and Flutter caregiver dashboard separately.

## Services

| Service | Port | Purpose | Public? |
|---------|------|---------|---------|
| inference-server | 8001 | Fall detection API (mobile app calls this) | Yes â€” put behind reverse proxy |
| fall-dashboard | 8002 | SSE feed + /api/falls for FOCUS Flutter dashboard | Yes |
| ml-dashboard | 8004 | Admin UI: retrain + model hot-swap | Internal only |
| server-health | 8006 | Aggregate health probe dashboard | Internal only |
| postgres | 5432 | Shared database | Internal only |
| mqtt | 1883/8883 | MQTT broker (mobile app publishes fall alerts) | Yes â€” port 1883 or 8883 |
| mlflow | 5000 | ML experiment tracking + model registry | Internal only |
| minio | 9000/9002 | Model artifact store (console at :9002) | Internal only |
| prometheus | 9090 | Metrics scraping | Internal only |
| grafana | 3000 | ML + server dashboards | Internal only |

## Prerequisites

- Docker Engine >= 24 and Docker Compose plugin (v2) installed on the Hetzner server
- Git access to this repository
- FOCUS InfluxDB credentials (URL, token, org, bucket) received from FOCUS DevOps

Check Docker Compose version:

```bash
docker compose version
# must be v2.1 or higher (for service_completed_successfully condition)
```

## Quick Start

### 1. Clone the repository

```bash
git clone <repo-url>
cd FallDetection_new/_6G_Integration_v2_mqtt
```

### 2. Create your .env file

```bash
cp inference_posttraining_layer/.env.example inference_posttraining_layer/.env
```

Open `inference_posttraining_layer/.env` and fill in every `CHANGE_ME` value:

| Variable | Description |
|----------|-------------|
| `POSTGRES_PASSWORD` | Database password â€” choose a strong password |
| `API_KEYS` | API key for the mobile app (share with Isa) |
| `MINIO_USER` | MinIO admin username |
| `MINIO_PASSWORD` | MinIO admin password |
| `GF_SECURITY_ADMIN_PASSWORD` | Grafana admin password |
| `INFLUXDB_URL` | FOCUS InfluxDB URL (e.g. https://influx.focus-domain.de) |
| `INFLUXDB_TOKEN` | FOCUS InfluxDB API token |
| `INFLUXDB_ORG` | FOCUS InfluxDB organisation name |
| `INFLUXDB_FALL_EVENTS_BUCKET` | FOCUS InfluxDB bucket name for fall events |

Generate a secure API key:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 3. Build all images

From `_6G_Integration_v2_mqtt/` as working directory:

```bash
docker compose -f inference_posttraining_layer/docker-compose.yml --env-file inference_posttraining_layer/.env build
```

This builds 5 custom images (inference-server, fall-dashboard, ml-dashboard,
server-health, mlflow). Standard images (postgres, mosquitto, minio, prometheus,
grafana, minio/mc) are pulled automatically.

### 4. Start the stack

```bash
docker compose -f inference_posttraining_layer/docker-compose.yml --env-file inference_posttraining_layer/.env up -d
```

On first run:
- `db-migrate` runs `alembic upgrade head` then exits (applies all 3 migrations)
- `minio-setup` creates the `mlflow-artifacts` bucket then exits
- All 10 services start in dependency order

### 5. Verify everything is running

```bash
docker compose -f inference_posttraining_layer/docker-compose.yml ps
```

All 10 services should show `running`. The two init containers (`db-migrate`,
`minio-setup`) will show `exited (0)` â€” that is correct.

Quick health check:

```bash
curl http://localhost:8001/health
curl http://localhost:8002/api/patients
```

## Service URLs (after deployment)

| Service | URL | Credentials |
|---------|-----|------------|
| Inference API | http://YOUR_SERVER:8001/docs | X-API-Key header |
| Fall Dashboard | http://YOUR_SERVER:8002/ | none |
| ML Dashboard | http://YOUR_SERVER:8004/ | none (add auth â€” see below) |
| Server Health | http://YOUR_SERVER:8006/ | none |
| Grafana | http://YOUR_SERVER:3000/ | admin / GF_SECURITY_ADMIN_PASSWORD |
| MLflow | http://YOUR_SERVER:5000/ | none |
| MinIO Console | http://YOUR_SERVER:9002/ | MINIO_USER / MINIO_PASSWORD |

**Note:** ml-dashboard and server-health have no authentication yet.
Only expose them internally or behind a firewall rule until auth is added.

## Updating a single service

After a code change, rebuild only that image and restart it:

```bash
# From _6G_Integration_v2_mqtt/
docker compose -f inference_posttraining_layer/docker-compose.yml --env-file inference_posttraining_layer/.env build inference-server
docker compose -f inference_posttraining_layer/docker-compose.yml --env-file inference_posttraining_layer/.env up -d --no-deps inference-server
```

## Useful commands

```bash
# View logs for a specific service
docker compose -f inference_posttraining_layer/docker-compose.yml logs -f inference-server

# View logs for all services (last 50 lines each)
docker compose -f inference_posttraining_layer/docker-compose.yml logs --tail=50

# Stop everything (data volumes are preserved)
docker compose -f inference_posttraining_layer/docker-compose.yml down

# Full reset including all data (WARNING: deletes database, models, metrics)
docker compose -f inference_posttraining_layer/docker-compose.yml down -v

# Restart a single service
docker compose -f inference_posttraining_layer/docker-compose.yml restart fall-dashboard

# Run alembic migration manually (e.g. after an upgrade)
docker compose -f inference_posttraining_layer/docker-compose.yml run --rm db-migrate
```

## Shorthand (run from inside inference_posttraining_layer/)

If you cd into the `inference_posttraining_layer/` directory, Docker Compose will automatically pick
up `docker-compose.yml` and `.env` from the current directory:

```bash
cd inference_posttraining_layer/
docker compose up -d
docker compose ps
docker compose logs -f inference-server
```

## HTTPS / TLS certificate (Mohammed's task)

Only `inference-server` (:8001) and `fall-dashboard` (:8002) need to be
reachable from the internet. All other ports should be blocked at the firewall.

**TLS is required** â€” the mobile app will refuse plain HTTP connections.
Use Let's Encrypt (free, auto-renews every 90 days) via Certbot + Nginx:

**Step 1 â€” DNS:** Ask the MCS domain admin to create an A-record pointing
a subdomain to this Hetzner server's IP address.
Example: `fall-api.mcs-labs.de` â†’ `<hetzner-server-ip>`

**Step 2 â€” Install Nginx + Certbot:**
```bash
apt update && apt install -y nginx certbot python3-certbot-nginx
```

**Step 3 â€” Get the certificate:**
```bash
certbot --nginx -d fall-api.mcs-labs.de
```
Certbot edits the Nginx config automatically and schedules auto-renewal.

**Step 4 â€” Add reverse proxy rules** to `/etc/nginx/sites-available/default`:
```nginx
server {
    listen 443 ssl;
    server_name fall-api.mcs-labs.de;
    # (certbot fills in ssl_certificate lines automatically)

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```
Repeat for fall-dashboard on its own subdomain pointing to `:8002`.

After this, Isa's mobile app calls `https://fall-api.mcs-labs.de/predict`
and the FOCUS Flutter dashboard connects to `https://fall-dashboard.mcs-labs.de/api/stream`.

## MQTT broker

The MQTT broker runs on port 1883. The mobile app (Isa) must be able to reach it.
By default there is no authentication â€” set `MQTT_USERNAME` / `MQTT_PASSWORD` in
`.env` and follow the steps in `inference_posttraining_layer/mosquitto.conf` to enable password auth.

## Model management

After the stack is running, log into the ML Dashboard at `:8004` to:
- Trigger a retraining run (uses data from Postgres `inference_log`)
- Promote a retrained model to Production in MLflow
- Hot-swap the running inference-server to the new model

Or use the API directly:

```bash
# Check current model
curl http://localhost:8001/model/info

# Hot-swap to MLflow Production version
curl -X POST http://localhost:8001/model/switch \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"mlflow_stage": "Production"}'
```

## Troubleshooting

**db-migrate exits with error:**
Check Postgres is healthy first. Inspect logs:
```bash
docker compose -f inference_posttraining_layer/docker-compose.yml logs db-migrate
```

**inference-server health check failing:**
Model file may be missing from `model/` directory. Check:
```bash
docker compose -f inference_posttraining_layer/docker-compose.yml logs inference-server
```

**MLflow can't connect to MinIO:**
Check minio-setup completed successfully:
```bash
docker compose -f inference_posttraining_layer/docker-compose.yml logs minio-setup
```
The `mlflow-artifacts` bucket must exist before MLflow starts.

**fall-dashboard list_falls() returns empty:**
Check FOCUS InfluxDB credentials in `.env`. Test connectivity:
```bash
docker compose -f inference_posttraining_layer/docker-compose.yml exec fall-dashboard \
  python -c "from ml_pipeline.data_input.data_loader.influx_client_manager import _get_influxdb_client; print(_get_influxdb_client().health())"
```

**Grafana shows no data:**
Grafana datasources are auto-provisioned. If dashboards are blank, verify
Prometheus is scraping inference-server correctly at http://localhost:9090/targets.
