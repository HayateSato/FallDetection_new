# Caregiver Layer — Local Mock FOCUS Environment

Simulates the services that FOCUS hosts in their own network:
MQTT broker, InfluxDB (fall events), and fall-dashboard (MQTT subscriber + caregiver API).

Run this on the **second Windows laptop** when testing cross-machine communication.

## Services

| Service | Port | Purpose |
|---------|------|---------|
| mqtt | 1883 | MQTT broker — receives fall alerts from mobile app |
| influxdb | 8086 | Mock FOCUS InfluxDB — fall_events written here by mobile app |
| postgres | 5432 | participant_session table for fall-dashboard |
| fall-dashboard | 8002 | SSE feed + /api/falls + /api/patients |

## Two-laptop test setup

```
Laptop 1 (MCS)              Laptop 2 (FOCUS mock / this machine)
────────────────────         ──────────────────────────────────
deploy/                      caregiver_layer/
  inference-server :8001       mqtt            :1883
  ml-dashboard     :8004       influxdb        :8086
  server-health    :8006       fall-dashboard  :8002
  postgres         :5432       postgres        :5432
  mlflow           :5000
  minio            :9000
  prometheus       :9090
  grafana          :3000
```

Communication between laptops (same WiFi):

```
mock_app (Laptop 1) ──── HTTP POST /predict ────► inference-server (Laptop 1 :8001)
mock_app (Laptop 1) ──── MQTT publish ──────────► mqtt broker      (Laptop 2 :1883)
mock_app (Laptop 1) ──── HTTP write fall_events ► influxdb         (Laptop 2 :8086)
fall-dashboard (L2) ──── MQTT subscribe ────────► mqtt broker      (same machine)
fall-dashboard (L2) ──── query fall_events ─────► influxdb         (same machine)
FOCUS Flutter app   ──── GET /api/stream SSE ───► fall-dashboard   (Laptop 2 :8002)
```

## Quick start (Laptop 2)

### 1. Get the laptop's IP address

```powershell
ipconfig | Select-String "IPv4"
# Note down the IP, e.g. 192.168.1.50
```

### 2. Configure and start

```powershell
cd C:\...\FallDetection_new\_6G_Integration_v2_mqtt

# Copy env template (defaults work for local testing)
copy caregiver_layer\.env.example caregiver_layer\.env

# Start all services
docker compose -f caregiver_layer/docker-compose.yml --env-file caregiver_layer/.env up -d
```

### 3. Verify

```powershell
curl.exe http://localhost:8002/api/patients   # fall-dashboard
curl.exe http://localhost:8086/health         # InfluxDB
docker compose -f caregiver_layer/docker-compose.yml exec mqtt mosquitto_pub -h localhost -t test -m ping
```

## Configure Laptop 1 (MCS) to point to this machine

On the MCS laptop, update the local_dev `.env` so mock_app sends MQTT alerts and
fall events to this machine instead of localhost.

Find and update these lines in `_6G_Integration_v2_mqtt/.env`:

```ini
# Point to Laptop 2 IP (e.g. 192.168.1.50)
MQTT_BROKER_HOST=192.168.1.50
MQTT_BROKER_PORT=1883

# Point fall event writes to Laptop 2 InfluxDB
INFLUXDB_URL=http://192.168.1.50:8086
INFLUXDB_TOKEN=mock-focus-token
INFLUXDB_ORG=focus-mock
INFLUXDB_FALL_EVENTS_BUCKET=fall-events
```

Also tell server-health on Laptop 1 where fall-dashboard lives.
In `_6G_Integration_v2_mqtt/deploy/.env`:

```ini
FALL_DASHBOARD_URL=http://192.168.1.50:8002
```

Then restart mock_app on Laptop 1:

```powershell
python -m local_dev.mock_app.main
```

## Expected end-to-end flow

1. mock_app fetches ACC data from MCS cloud InfluxDB (unchanged)
2. mock_app POSTs to inference-server on Laptop 1 → gets `observation_id`
3. Fall detected → patient confirmation popup (10s)
4. mock_app publishes `fall/alert/<patient_id>` to **Laptop 2 MQTT broker**
5. fall-dashboard (Laptop 2) receives MQTT alert → fans out SSE
6. mock_app writes `fall_events` point to **Laptop 2 InfluxDB**
7. mock_app POSTs `/inference/{observation_id}/confirm` to inference-server (Laptop 1)
8. fall-dashboard `/api/falls` returns the event from Laptop 2 InfluxDB

## Useful commands

```powershell
# Logs
docker compose -f caregiver_layer/docker-compose.yml logs -f fall-dashboard
docker compose -f caregiver_layer/docker-compose.yml logs -f mqtt

# Stop (data preserved)
docker compose -f caregiver_layer/docker-compose.yml down

# Full reset
docker compose -f caregiver_layer/docker-compose.yml down -v
```
