# Fall Detection — FOCUS Caregiver Layer (Docker)

Simulates the services that FOCUS hosts in their own network:
MQTT broker, fall-dashboard (MQTT subscriber + caregiver API), and the mock mobile app.

Run this on the **second Windows laptop** when testing cross-machine communication.

**No local database.** The patient list comes from the `PATIENT_IDS` env var.
Fall history and fall counts are read from InfluxDB (external, not hosted here).

## Services

| Service | Port | Purpose |
|---------|------|---------|
| mock-app | 8005 | Simulates the SmarKo mobile app — patient popup at http://localhost:8005/ |
| mqtt | 1883 | MQTT broker — receives fall alerts from mock-app |
| fall-dashboard | 8002 | SSE feed + /api/falls + /api/patients |

InfluxDB is **not hosted here** — both mock-app and fall-dashboard connect to an external instance:
- **Local testing:** MCS cloud InfluxDB (`ecosystem-influxdb.smarko-health.de`, `fd_test` bucket)
- **Production:** FOCUS's own InfluxDB in their k3s cluster

## Two-laptop test setup

```
Laptop 1 (MCS)                          Laptop 2 (FOCUS mock / this machine)
──────────────────────────────────       ────────────────────────────────────────
inference_posttraining_layer/            _6G_integration_v3_docker_focus/
  inference-server  :8001                  mock-app       :8005
  ml-dashboard      :8004                  mqtt           :1883
  server-health     :8006                  fall-dashboard :8002
  postgres          :5432
  mlflow            :5000
  minio             :9000                      MCS cloud InfluxDB (internet)
  prometheus        :9090                        fd_test bucket
  grafana           :3000
```

Communication flow:

```
mock-app (L2)    ──── HTTP POST /predict ──────────► inference-server (Laptop 1 :8001)
mock-app (L2)    ──── MQTT publish fall/possible ──► mqtt             (same machine :1883)
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

Run from `_6G_integration_v3_docker_focus/` as working directory:

```powershell
docker compose up -d
```

### 4. Verify

```powershell
curl.exe http://localhost:8002/api/patients    # patient list (from PATIENT_IDS env var)
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
3. Fall detected → mock-app publishes `fall/possible/<patient_id>` to MQTT → caregiver dashboard shows amber "Possible fall" badge on patient card
4. Patient confirmation popup opens at http://localhost:8005/
5. After patient responds (or timeout) → mock-app publishes `fall/alert/<patient_id>` to MQTT
6. fall-dashboard receives confirmed alert → fans out SSE → Flutter dashboard shows red alert banner (if needs_help or no response)
7. mock-app writes `fall_events` point to MCS cloud InfluxDB
8. mock-app POSTs `/inference/{observation_id}/confirm` to Laptop 1
9. fall-dashboard `/api/falls` returns the event from InfluxDB

## MQTT topics

| Topic | Publisher | Subscriber | When |
|-------|-----------|------------|------|
| `fall/possible/<patient_id>` | mock-app | fall-dashboard | Immediately on fall detection (before patient confirms) |
| `fall/alert/<patient_id>` | mock-app | fall-dashboard | After patient confirmation or 10s timeout |

## Useful commands

Run all commands from `_6G_integration_v3_docker_focus/` as working directory.

```powershell
# Logs
docker compose logs -f mock-app
docker compose logs -f fall-dashboard

# Rebuild after code changes
docker compose build --no-cache
docker compose up -d

# Stop (volumes preserved)
docker compose down

# Full reset including volumes
docker compose down -v
```
