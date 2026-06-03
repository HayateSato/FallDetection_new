# Caregiver Layer — Local Mock FOCUS Environment

Simulates the services that FOCUS hosts in their own network:
MQTT broker, fall-dashboard (MQTT subscriber + caregiver API), and the mock mobile app.

Run this on the **second Windows laptop** when testing cross-machine communication.

## Services

| Service | Port | Purpose |
|---------|------|---------|
| mock-app | 8005 | Simulates the SmarKo mobile app — patient popup at http://localhost:8005/ |
| mqtt | 1883 | MQTT broker — receives fall alerts from mock-app |
| postgres | 5432 | participant_session table for fall-dashboard |
| fall-dashboard | 8002 | SSE feed + /api/falls + /api/patients |

InfluxDB is **not hosted here** — both mock-app and fall-dashboard connect to an external instance:
- **Local testing:** MCS cloud InfluxDB (`ecosystem-influxdb.smarko-health.de`, `fd_test` bucket)
- **Production:** FOCUS's own InfluxDB in their k3s cluster

## Two-laptop test setup

```
Laptop 1 (MCS)                          Laptop 2 (FOCUS mock / this machine)
──────────────────────────────────       ────────────────────────────────────────
inference_posttraining_layer/            caregiver_layer/
  inference-server  :8001                  mock-app       :8005
  ml-dashboard      :8004                  mqtt           :1883
  server-health     :8006                  fall-dashboard :8002
  postgres          :5432                  postgres       :5432
  mlflow            :5000
  minio             :9000                      MCS cloud InfluxDB (internet)
  prometheus        :9090                        fd_test bucket
  grafana           :3000
```

Communication flow:

```
mock-app (L2)    ──── HTTP POST /predict ──────────► inference-server (Laptop 1 :8001)
mock-app (L2)    ──── MQTT publish fall/alert ─────► mqtt             (same machine :1883)
mock-app (L2)    ──── write fall_events ───────────► MCS cloud InfluxDB
mock-app (L2)    ──── POST /confirm ───────────────► inference-server (Laptop 1 :8001)
fall-dashboard   ──── MQTT subscribe ──────────────► mqtt             (same machine)
fall-dashboard   ──── query fall_events ───────────► MCS cloud InfluxDB
Flutter dashboard──── GET /api/stream SSE ─────────► fall-dashboard   (L2 :8002)
```

## Quick start (Laptop 2)

### 1. Get this laptop's IP address

```powershell
ipconfig | Select-String "IPv4"
# Note down the IP, e.g. 192.168.1.50
```

### 2. Configure .env

```powershell
cd C:\...\FallDetection_new\_6G_Integration_v2_mqtt
copy caregiver_layer\.env.example caregiver_layer\.env
```

Open `caregiver_layer\.env` and set:

```ini
# Point to Laptop 1 (MCS) IP
INFERENCE_SERVER_URL=http://192.168.1.XX:8001
INFERENCE_API_KEY=<same key as in inference_posttraining_layer/.env>
```

Everything else (InfluxDB, MQTT, patient IDs) has working defaults for local testing.

### 3. Build and start

```powershell
docker compose -f caregiver_layer/docker-compose.yml --env-file caregiver_layer/.env up -d
```

### 4. Verify

```powershell
curl.exe http://localhost:8002/api/patients    # fall-dashboard
curl.exe http://localhost:8002/api/falls       # fall history from InfluxDB
# Patient popup (open in browser when a fall fires):
# http://localhost:8005/
```

## Configure Laptop 1 (MCS) to point to this machine

On the MCS laptop, in `inference_posttraining_layer/.env` set:

```ini
# URL for server-health to probe fall-dashboard on Laptop 2
FALL_DASHBOARD_URL=http://192.168.1.50:8002
```

Then restart server-health:

```powershell
docker compose -f inference_posttraining_layer/docker-compose.yml restart server-health
```

## Expected end-to-end flow

1. mock-app fetches ACC data from MCS cloud InfluxDB
2. mock-app POSTs to inference-server on Laptop 1 → gets `observation_id`
3. Fall detected → patient confirmation popup opens at http://localhost:8005/
4. mock-app publishes `fall/alert/<patient_id>` to local MQTT broker
5. fall-dashboard receives MQTT → fans out SSE to Flutter dashboard
6. mock-app writes `fall_events` point to MCS cloud InfluxDB
7. mock-app POSTs `/inference/{observation_id}/confirm` to Laptop 1
8. fall-dashboard `/api/falls` returns the event from InfluxDB

## Useful commands

```powershell
# Logs
docker compose -f caregiver_layer/docker-compose.yml logs -f mock-app
docker compose -f caregiver_layer/docker-compose.yml logs -f fall-dashboard

# Stop (data preserved)
docker compose -f caregiver_layer/docker-compose.yml down

# Full reset
docker compose -f caregiver_layer/docker-compose.yml down -v
```
