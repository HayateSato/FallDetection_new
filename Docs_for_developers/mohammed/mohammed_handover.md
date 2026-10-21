# Fall Detection — Mohammed Handover (2026-06-09)

## Context / What Changed Since Last Handover

Last time the architecture was based on `_6G_Integration_v2_mqtt` — one combined stack
where FOCUS hosted almost everything including the inference server in k3s.

**The split is now final:**

| Side | Owner | Runtime | Services |
|------|-------|---------|----------|
| **MCS inference layer** | Mohammed / MCS | Docker Compose on Hetzner | 8 services |
| **FOCUS caregiver layer** | FOCUS DevOps | K3s inside FOCUS network | 2 services |

FOCUS only runs the MQTT broker and the fall dashboard. Everything else — inference,
ML pipeline, model registry, metrics — runs on MCS's own Hetzner server.

---

## Your To-Do List

These are the tasks you need to complete. In order:

- [ ] **1. Set up Hetzner server** — install Docker, Docker Compose, Nginx, Certbot
- [ ] **2. Create DNS A-record** — point the subdomain (e.g. `fall-api.mcs-labs.de`) at the Hetzner IP
- [ ] **3. Get TLS certificate** — `certbot --nginx -d fall-api.mcs-labs.de`
      Nginx forwards HTTPS :443 → inference-server :8001
- [ ] **4. Deploy MCS inference layer** — copy `_6G_integration_v3_docker_mcs/` to Hetzner, fill `.env`, run `docker compose up -d`
- [ ] **5. Verify all 8 services are running** — see verification commands in the section below
- [ ] **6. Push fall-dashboard image to registry** — build from `_6G_integration_v3_k3s/fall_dashboard/`, push to `registry-smarko-health.de/fall-detection/fall-dashboard:latest`
- [ ] **7. Fill in `values_production.yaml`** — all `CHANGE_ME` fields in `_6G_integration_v3_k3s/helm/values_production.yaml`
- [ ] **8. Deliver K3s Helm chart to FOCUS DevOps** — hand over `_6G_integration_v3_k3s/` and the filled `values_production.yaml`
- [ ] **9. Share domains + API key with Isa and Hayate** — once #3 is done, share:
      - Public inference server URL (e.g. `https://fall-api.mcs-labs.de`)
      - API key from `API_KEYS` in `.env`
      Share also the fall-dashboard domain (from FOCUS DevOps) with Isa

After you complete the above, Hayate will run the end-to-end test with Isa's real mobile
app against the Hetzner server.

---

## MCS Inference Layer — What Each Service Does

**Folder:** `_6G_integration_v3_docker_mcs/`
**Start command:** `docker compose up -d` (run from that folder)

### Long-running services (8)

| Service | Container name | Port (host) | What it does |
|---------|---------------|-------------|--------------|
| `inference-server` | `fall_inference_server` | **8001** (public) | The main API. Receives `/predict` from the mobile app, runs XGBoost fall detection, writes result to Postgres. Also receives `/inference/{id}/confirm` from the mobile app after patient popup. |
| `ml-dashboard` | `fall_ml_dashboard` | 8004 (internal) | Admin web UI. Trigger model retraining, inspect MLflow runs, hot-swap the active model without restarting inference-server. Used by only MCS members and will not be opned to FOCUS  |
| `server-health` | `fall_server_health` | 8006 (internal) | Aggregate health dashboard. Probes all 6 services every 30s and shows a status card for each. Also probes the FOCUS caregiver layer over the LAN/internet. Used by only MCS members and will not be opned to FOCUS|
| `postgres` | `mcs_fall_postgres` | 5432 (internal) | Shared database. Stores: `inference_log` (every /predict result), `feature_snapshot` (raw features for retraining), MLflow tracking data (in a separate `mlflow` database). |
| `mlflow` | `fall_mlflow` | 5000 (internal) | ML experiment tracking and model registry. Stores run metadata in Postgres; stores model files (`.ubj` XGBoost binaries) in MinIO. |
| `minio` | `fall_minio` | 9000 (API) / 9002 (console) (internal) | S3-compatible object store. Only used to store MLflow model artifacts. |
| `prometheus` | `fall_prometheus` | 9090 (internal) | Scrapes metrics from inference-server (`/metrics`). Stores time-series data for Grafana. |
| `grafana` | `fall_grafana` | 3000 (internal) | ML dashboards: inference rate, model confidence drift, fall detection counts. Reads from Prometheus and Postgres. Used by only MCS members and will not be opned to FOCUS |

### One-off init jobs (2 — run on startup, then exit)

| Service | What it does |
|---------|--------------|
| `db-migrate` | Runs Alembic migrations (`upgrade head`) before inference-server starts. Creates and updates all Postgres tables. If you add a new server version with new migrations, this runs them automatically on next `docker compose up`. |
| `minio-setup` | Creates the `mlflow-artifacts` bucket in MinIO on first start. Idempotent — safe to run repeatedly. |

### What is NOT in this stack (compared to the old v2 design)

- No MQTT broker — that runs in FOCUS k3s
- No fall-dashboard — that runs in FOCUS k3s
- No mock-app — replaced by Isa's real mobile app
- No InfluxDB — FOCUS has their own; MCS doesn't write to it directly

---

## FOCUS Caregiver Layer — What Each Service Does

**Folder:** `_6G_integration_v3_k3s/` (K3s Helm chart)
**Namespace:** `fall-dashboard`
**Helm release name:** `caregiver`

FOCUS DevOps deploys this into their existing k3s cluster. You do not deploy this —
you just need to push the Docker image and hand over the chart with filled values.

| Service (pod) | Internal port | External access | What it does |
|---------------|--------------|-----------------|--------------|
| `mosquitto` | 1883 (TCP) / 9001 (WebSocket) | 9001 → NodePort 30901 → Traefik HTTPS (WSS) in production | MQTT broker. Receives fall alerts from the mobile app (WebSocket port 9001). Forwards them to fall-dashboard (port 1883, internal). |
| `fall-dashboard` | 8002 | 8002 → NodePort 30802 → Traefik HTTPS in production | Fall alert dashboard for caregivers. Subscribes to MQTT broker. Fans out alerts to browser via SSE. Shows per-patient fall history from FOCUS InfluxDB. Patients are managed dynamically via the "+ Add Patient" UI button (stored in SQLite on a PVC). |

---

## Communication Map — Who Talks to Whom

### Inside the MCS Docker network (`fall-net`)

All these use Docker internal DNS — hostnames are the service names in `docker-compose.yml`.

```
Mobile app (internet)
    -- HTTPS POST /predict --> inference-server:8001
    -- HTTPS POST /inference/{id}/confirm --> inference-server:8001

inference-server
    -- PostgreSQL TCP --> postgres:5432          (writes inference_log, feature_snapshot)
    -- HTTP --> mlflow:5000                      (reads active model from registry)
    -- HTTP --> minio:9000                       (downloads .ubj model file on hot-swap)

ml-dashboard
    -- HTTP --> mlflow:5000                      (reads experiment runs, triggers retrain)
    -- HTTP --> minio:9000                       (model artifact storage)
    -- HTTP --> inference-server:8001            (POST /model/reload on hot-swap)
    -- PostgreSQL TCP --> postgres:5432          (reads feature_snapshot for retraining)

mlflow
    -- PostgreSQL TCP --> postgres:5432          (stores experiment + run metadata)
    -- HTTP --> minio:9000                       (stores model files as S3 objects)

prometheus
    -- HTTP GET /metrics --> inference-server:8001   (scrapes every 15s)

grafana
    -- HTTP --> prometheus:9090                  (Prometheus datasource)
    -- PostgreSQL TCP --> postgres:5432          (PostgreSQL datasource for inference_log)

server-health
    -- HTTP GET /health --> inference-server:8001
    -- PostgreSQL TCP --> postgres:5432          (connectivity probe)
    -- HTTP --> mlflow:5000                      (connectivity probe)
    -- HTTP --> minio:9000                       (connectivity probe)
```

### Cross-network (MCS → FOCUS)

These go over the internet / LAN and are NOT Docker-internal.

```
server-health (Docker on Hetzner)
    -- HTTP GET /health --> fall-dashboard:30802 (FOCUS K3s NodePort or Traefik HTTPS)
    -- TCP connect --> mqtt-broker:30901         (FOCUS K3s NodePort or Traefik HTTPS WSS)

    NOTE: In production, server-health needs FALL_DASHBOARD_URL and MQTT_BROKER_HOST
    set in .env pointing at FOCUS's public domain. Without a VPN or public URL, these
    two probe cards will show "down" on the server-health UI — that is acceptable.
    The 4 MCS-side probes will still be healthy.
```

### Mobile app (Isa) → services

```
Mobile app (Isa)
    -- HTTPS POST /predict --> inference-server (Hetzner, port 443 via Nginx)
    -- HTTPS POST /inference/{id}/confirm --> inference-server (Hetzner, port 443 via Nginx)
    -- InfluxDB write (fall_events point) --> FOCUS InfluxDB directly (not via MCS)
    -- MQTT WebSocket (wss://...) --> mosquitto in FOCUS k3s (port 9001 / Traefik WSS)
```

### Inside FOCUS K3s (`fall-dashboard` namespace)

```
fall-dashboard pod
    -- MQTT TCP --> mosquitto:1883               (internal cluster DNS, subscribes to fall/alert/# and fall/possible/#)
    -- HTTP --> FOCUS InfluxDB                   (queries fall_events for history tab)
    -- SSE stream --> caregiver browser          (fans out MQTT alerts in real time)
```

---

## How to Deploy on Hetzner (Step by Step)

### Prerequisites

- Ubuntu server on Hetzner
- DNS A-record pointing `fall-api.mcs-labs.de` (or your chosen subdomain) to Hetzner IP
- Docker + Docker Compose installed

### 1. Transfer files

Copy `_6G_integration_v3_docker_mcs/` to the Hetzner server.

### 2. Configure .env

```bash
cd _6G_integration_v3_docker_mcs/
cp .env.example .env
```

Open `.env` and fill in (minimum required):

```
POSTGRES_PASSWORD=<strong random password>
API_KEYS=<generate with: python3 -c "import secrets; print(secrets.token_urlsafe(32))">
MINIO_USER=<any username>
MINIO_PASSWORD=<strong random password>
GF_SECURITY_ADMIN_PASSWORD=<strong random password>

# Set these after FOCUS DevOps confirms the caregiver layer is deployed:
FALL_DASHBOARD_URL=https://<fall-dashboard domain from FOCUS>
MQTT_BROKER_HOST=<mqtt domain from FOCUS>
MQTT_BROKER_PORT=443
```

See `FOCUS_devs_handover/config_checklist_mohammed.md` for a complete variable table.

### 3. Set up Nginx + TLS

```bash
apt install nginx certbot python3-certbot-nginx
certbot --nginx -d fall-api.mcs-labs.de
```

Nginx config — add to `/etc/nginx/sites-available/fall-detection`:

```nginx
server {
    listen 443 ssl;
    server_name fall-api.mcs-labs.de;

    # certbot fills in ssl_certificate and ssl_certificate_key automatically

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Reload Nginx after certbot: `nginx -s reload`

### 4. Start the stack

```bash
docker compose up -d
```

db-migrate runs first automatically. Wait ~30 seconds for all services to start.

### 5. Verify

```bash
docker compose ps                               # all 8 services should be "running"
curl https://fall-api.mcs-labs.de/health        # {"status":"ok","model_version":"v0",...}
curl http://localhost:8006/api/status           # server-health JSON (all 4 MCS probes green)
# Grafana: http://localhost:3000 (admin / your GF_SECURITY_ADMIN_PASSWORD)
# MLflow:  http://localhost:5000
# MinIO:   http://localhost:9002
```

### 6. Share with Isa and Hayate

Once inference server is up and TLS works, send Hayate:
- Public URL: `https://fall-api.mcs-labs.de`
- API key value from `API_KEYS` in `.env`

Hayate will share these with Isa so she can update the mobile app config.

---

## How to Push the fall-dashboard Image to Registry

The fall-dashboard Docker image must be in MCS registry so FOCUS k3s can pull it.

```bash
cd _6G_integration_v3_k3s/fall_dashboard/

# Build
docker build -t registry-smarko-health.de/fall-detection/fall-dashboard:latest .

# Push (log in first if needed)
docker login registry-smarko-health.de
docker push registry-smarko-health.de/fall-detection/fall-dashboard:latest
```

Confirm the old image (from the v2 architecture) is removed from the registry if it had
a different name — FOCUS DevOps will need to update their image pull if so.

---

## How to Fill in values_production.yaml

File: `_6G_integration_v3_k3s/helm/values_production.yaml`

Search for `CHANGE_ME` — there are entries for:
- `mosquitto.ingress.host` — domain FOCUS uses for MQTT (WSS)
- `fallDashboard.ingress.host` — domain FOCUS uses for fall-dashboard
- `fallDashboard.influxdb.*` — FOCUS's InfluxDB URL, org, bucket, token
- `imagePullSecret` — set to `mcs-labs` (MCS registry pull secret)
- `fallDashboard.image` — set to `registry-smarko-health.de/fall-detection/fall-dashboard:latest`

The `patientIds` and `macIds` fields in `values.yaml` no longer need to be filled — patient
management is now dynamic via the "+Add Patient" button in the dashboard UI. Patients are
stored in a SQLite file on a PVC and survive pod restarts without any values.yaml edit.

See `FOCUS_devs_handover/config_checklist_focus_devops.md` for the full table.

---

## What Hayate Will Do After Your Deployment Is Done

Once you confirm Hetzner is up and share the URL + API key:

1. Hayate sends the URL and API key to Isa
2. Isa updates her mobile app config and tests against the real Hetzner server
3. End-to-end test: SmarKo → mobile app → inference server (Hetzner) → MQTT → FOCUS caregiver dashboard → alert visible on browser
4. If the test passes → system is ready for production handover to FOCUS

---

## Key Files for Reference

| File | What it contains |
|------|-----------------|
| `_6G_integration_v3_docker_mcs/docker-compose.yml` | All 8 services + startup order |
| `_6G_integration_v3_docker_mcs/.env.example` | Every env variable with descriptions |
| `_6G_integration_v3_docker_mcs/README.md` | Quick-start + useful commands |
| `_6G_integration_v3_k3s/helm/values_production.yaml` | FOCUS caregiver layer config (fill `CHANGE_ME` values) |
| `FOCUS_devs_handover/production_config_checklist.md` | Full table of every config variable for both layers, who sets what, and cross-party agreements |
