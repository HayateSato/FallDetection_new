# Production Configuration Checklist

Two separate deployments. Each has its own set of values to fill in.

| Deployment | Owner | Folder | Config file |
|---|---|---|---|
| **MCS inference layer** | Mohammed | `_6G_integration_v3_docker_mcs/` | `.env` (copy from `.env.example`) |
| **FOCUS caregiver layer** | FOCUS DevOps | `_6G_integration_v3_k3s/` | `helm/values.yaml` |

They are independent stacks. Mohammed sets his up first so the inference server URL is
known before FOCUS DevOps fills in the caregiver values.

---

## Part 1 — MCS Inference Layer (Mohammed / Hetzner)

**File:** `_6G_integration_v3_docker_mcs/.env`
Copy `.env.example` to `.env`, then change every row marked REQUIRED below.

### Secrets — all REQUIRED

| Variable | Local testing value | Production value | Notes |
|---|---|---|---|
| `POSTGRES_PASSWORD` | `CHANGE_ME` | Strong random password | Used by inference-server, mlflow, db-migrate |
| `API_KEYS` | `CHANGE_ME` | Generated secret key | Mobile app (Isa) sends this in `X-API-Key` header. Generate: `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Share with Isa. |
| `MINIO_USER` | `CHANGE_ME` | Any username | MinIO model artifact store admin user |
| `MINIO_PASSWORD` | `CHANGE_ME` | Strong random password | MinIO admin password |
| `GF_SECURITY_ADMIN_PASSWORD` | `CHANGE_ME` | Strong random password | Grafana admin login |

### InfluxDB — NOT needed in MCS .env

No service in `_6G_integration_v3_docker_mcs/docker-compose.yml` uses any `INFLUXDB_*` variable.
The InfluxDB credentials belong entirely to the FOCUS caregiver layer (Part 2).

The `INFLUXDB_*` lines in `.env.example` are leftovers from when `mock_app` was part of this stack.
They can be left commented out or deleted. Do not fill them in for Hetzner.

### Caregiver layer URL — REQUIRED

| Variable | Local testing value | Production value | Notes |
|---|---|---|---|
| `FALL_DASHBOARD_URL` | `http://192.168.x.x:8002` | `https://<FOCUS-domain>/falls` | Used by `server-health` to health-probe fall-dashboard. Set to the Traefik-exposed URL of fall-dashboard in FOCUS k3s (see Part 2 `fallDashboard.ingress.host`). |

### Security hardening — recommended for production

| Variable | Current value | Recommended | Notes |
|---|---|---|---|
| `CORS_ALLOWED_ORIGINS` | `*` | Exact FOCUS dashboard origin | e.g. `https://dashboard.focus-hospital.de` |
| `RATE_LIMIT_PER_MINUTE` | `60` | Keep or increase | 60 req/min per IP is fine for one patient |

### Values that do NOT need to change

| Variable | Value | Why unchanged |
|---|---|---|
| `MODEL_VERSION` | `v0` | Default model. Change only after retraining (via MLflow hot-swap). |
| `ACC_SENSOR_TYPE` | `bosch` | SmarKo hardware is Bosch ACC |
| `HARDWARE_ACC_SAMPLE_RATE` | `50` | InfluxDB data confirmed at 50 Hz |
| `RESAMPLING_METHOD` | `linear` | Standard |
| `MLFLOW_REGISTERED_MODEL` | `fall-detection-xgboost` | Model registry name |
| `MQTT_ALERT_TOPIC` | `fall/alert` | Must match FOCUS k3s values.yaml |
| `PUBLIC_ENDPOINT_ENABLED` | `true` | Inference endpoint must be public |

### How to run on Hetzner

```bash
# from _6G_integration_v3_docker_mcs/ as working directory
cp .env.example .env
# fill in .env
docker compose --env-file .env up -d
```

Check inference server is up:
```bash
curl https://<your-domain>/health
# expected: {"status":"ok","model_version":"v0", ...}
```

---

## Part 2 — FOCUS Caregiver Layer (FOCUS DevOps / K3s)

**File:** `_6G_integration_v3_k3s/helm/values.yaml`
Edit this file directly, then run `helm install` / `helm upgrade`.

### Registry and image pull — REQUIRED

| Key | Local testing value | Production value | Notes |
|---|---|---|---|
| `imagePullSecret` | `""` | `mcs-labs` | Pull secret name for `registry-smarko-health.de` |
| `fallDashboard.image` | `fall-detection/fall-dashboard:local` | `registry-smarko-health.de/fall-detection/fall-dashboard:latest` | Production image from MCS registry |
| `fallDashboard.imagePullPolicy` | `IfNotPresent` | `Always` | Always pull from registry in production |

### NodePorts — remove for production

NodePorts were only needed for local two-laptop testing. In production, Traefik handles all external access.

| Key | Local testing value | Production value | Notes |
|---|---|---|---|
| `mosquitto.wsNodePort` | `30901` | `""` (blank) | Blank = ClusterIP only; Traefik IngressRoute handles external WSS |
| `fallDashboard.httpNodePort` | `30802` | `""` (blank) | Blank = ClusterIP only; Traefik IngressRoute handles external HTTPS |

### Mosquitto ingress (Traefik) — REQUIRED

| Key | Local testing value | Production value | Notes |
|---|---|---|---|
| `mosquitto.ingress.host` | `CHANGE_ME` | e.g. `mqtt.focus-hospital.de` | Domain the real mobile app connects to via WSS |
| `mosquitto.ingress.certResolver` | `""` | e.g. `le` | Traefik Let's Encrypt resolver name. Ask FOCUS DevOps for the resolver name used in their cluster. |

### Fall-dashboard ingress (Traefik) — REQUIRED

| Key | Local testing value | Production value | Notes |
|---|---|---|---|
| `fallDashboard.ingress.host` | `localhost` | e.g. `fall.focus-hospital.de` | Domain Flutter dashboard calls for `/api/falls`, `/api/stream` SSE |
| `fallDashboard.ingress.certResolver` | `""` | e.g. `le` | Same resolver as above |

### InfluxDB — REQUIRED (FOCUS's own instance)

The fall-dashboard reads fall history from FOCUS's InfluxDB. These must match what FOCUS's mobile app writes to.

| Key | Local testing value | Production value |
|---|---|---|
| `fallDashboard.influxdb.url` | `https://ecosystem-influxdb.smarko-health.de` | FOCUS InfluxDB URL |
| `fallDashboard.influxdb.org` | `MCS Datalabs GmbH` | FOCUS InfluxDB organisation name |
| `fallDashboard.influxdb.bucket` | `fd_test` | FOCUS bucket name (raw sensor data) |
| `fallDashboard.influxdb.fallEventsBucket` | `fd_test` | FOCUS bucket where `fall_events` are written (often same as `bucket`) |
| `fallDashboard.influxdb.token` | MCS testing token | Token from FOCUS with read access to both buckets |

### Patients — REQUIRED

| Key | Local testing value | Production value | Notes |
|---|---|---|---|
| `fallDashboard.patientIds` | `"patient_test_50"` | Real patient IDs, comma-separated | Must match the patient IDs in the real mobile app and InfluxDB |
| `fallDashboard.macIds` | `"6c:1d:eb:04:a9:d9"` | Real wearable MAC addresses, comma-separated | Positional mapping to `patientIds` |

### MQTT auth — set if broker requires authentication

| Key | Local testing value | Production value | Notes |
|---|---|---|---|
| `fallDashboard.mqtt.username` | `""` | broker username | Leave blank if `allow_anonymous true` stays in mosquitto config |
| `fallDashboard.mqtt.password` | `""` | broker password | Leave blank if anonymous allowed |

### Values that do NOT need to change

| Key | Value | Why unchanged |
|---|---|---|
| `namespace` | `fall-dashboard` | Standard namespace name |
| `mosquitto.port` | `1883` | Internal TCP port (cluster-only, never changes) |
| `mosquitto.wsPort` | `9001` | Internal WebSocket port (Traefik forwards to this) |
| `mosquitto.image` | `eclipse-mosquitto:2` | Standard image, no custom build |
| `fallDashboard.replicas` | `1` | Must stay 1 — SSE fan-out is in-process |
| `fallDashboard.port` | `8002` | Internal pod port |
| `fallDashboard.mqtt.alertTopic` | `fall/alert` | Must match MCS `.env` `MQTT_ALERT_TOPIC` |
| `fallDashboard.mqtt.possibleTopic` | `fall/possible` | Pre-confirmation alerts |
| `mosquitto.persistence.size` | `500Mi` | Sufficient for MQTT broker storage |
| `fallDashboard.persistence.size` | `500Mi` | Sufficient for SQLite patient store |
| `mosquitto.persistence.storageClass` | `""` | Blank = k3s default (`local-path`) |
| `fallDashboard.persistence.storageClass` | `""` | Blank = k3s default (`local-path`) |
| `registry` | `registry-smarko-health.de` | Already set to production registry |

### How to install on FOCUS k3s

```bash
# from _6G_integration_v3_k3s/ as working directory

# First time
helm install caregiver helm/ \
  --namespace fall-dashboard \
  --create-namespace

# Updates
helm upgrade caregiver helm/ \
  --namespace fall-dashboard
```

Check pods are running:
```bash
kubectl get pods -n fall-dashboard
# Expected:
#   mosquitto-xxxx        1/1   Running
#   fall-dashboard-xxxx   1/1   Running
```

Check fall-dashboard is reachable (from inside the cluster):
```bash
kubectl exec -n fall-dashboard deploy/fall-dashboard -- \
  wget -qO- http://localhost:8002/api/patients
```

---

## Cross-reference: values that must match between the two deployments

| MCS `.env` | FOCUS `values.yaml` | Must be equal |
|---|---|---|
| `MQTT_ALERT_TOPIC=fall/alert` | `fallDashboard.mqtt.alertTopic: fall/alert` | yes |
| `INFLUXDB_FALL_EVENTS_BUCKET=<bucket>` | `fallDashboard.influxdb.fallEventsBucket: <bucket>` | yes — same InfluxDB bucket |
| `FALL_DASHBOARD_URL=https://<host>` | `fallDashboard.ingress.host: <host>` | yes — MCS server-health probes this URL |

---

## What Isa (mobile app) needs from Mohammed

Once the Hetzner server is running, share with Isa:

| Item | Value |
|---|---|
| Inference server URL | `https://<hetzner-domain>/predict` |
| API key | Value of `API_KEYS` from `.env` |
| MQTT broker WebSocket URL | `wss://<mosquitto.ingress.host>` |
| MQTT alert topic | `fall/alert` |
| MQTT possible topic | `fall/possible` |
| InfluxDB measurement name | `fall_events` (see `__Refactoring_docs/influxdb_schema.md`) |
